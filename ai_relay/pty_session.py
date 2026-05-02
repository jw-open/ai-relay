"""
PtySession — spawns a CLI subprocess inside a pseudo-terminal (PTY) and
provides an async interface for reading output and writing input.

Why PTY:
  Interactive CLIs (Claude Code, Codex, Gemini CLI) detect whether stdout is
  a terminal. Without a PTY they fall back to non-interactive / --print mode.

Privilege dropping:
  Claude Code refuses --dangerously-skip-permissions when running as root.
  Pass uid/gid to drop to a non-root user before exec.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import pty
import re
import struct
import termios
import time
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# ── ANSI / VT100 escape sequence stripping ────────────────────────────────────
# Covers: CSI (with >, <, = in param area), OSC, DCS, two-char escapes,
# and miscellaneous control characters (keep \n, \r, \t).
CTRL_RE = re.compile(
    r"\x1b\[[0-9;?><= ]*[A-Za-z@`]"        # CSI: [?25h, [>0q, [2004h, [0m …
    r"|\x1b\][^\x07\x1b]*(?:[\x07\x1b\\]|$)"  # OSC sequences (incl. unterminated at chunk end)
    r"|\x1bP[^\x1b]*\x1b\\"                # DCS sequences
    r"|\x1b[^[\]P]"                         # Two-char escapes: \x1bM, \x1b=, \x1b>
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # Other control chars (keep \n, \r, \t)
)

# Cursor-right: \x1b[NC means "move cursor right N columns" — TUI apps use
# this to lay out words with spacing instead of literal space characters.
# We replace each with the appropriate number of spaces before stripping.
_CURSOR_RIGHT_RE = re.compile(r"\x1b\[(\d*)C")

# OSC 8 terminal hyperlinks: \x1b]8;params;URL\x07  or  \x1b]8;params;URL\x1b\
# Codex (and other modern CLIs) wrap URLs in these so terminals render them clickable.
# We extract the URL from the escape sequence so it's visible as plain text.
# The closing tag \x1b]8;;\x07 has an empty URL → replaced with empty string.
_OSC8_BEL_RE = re.compile(r"\x1b\]8;[^;]*;([^\x07]*)\x07")   # BEL-terminated
_OSC8_ST_RE  = re.compile(r"\x1b\]8;[^;]*;([^\x1b]*)\x1b\\") # ST-terminated


def _cursor_right_to_spaces(m: re.Match) -> str:
    n = int(m.group(1)) if m.group(1) else 1
    return " " * n


def clean_pty_output(chunk: bytes) -> str:
    """Decode a raw PTY chunk, strip terminal control sequences, normalise line endings."""
    raw = chunk.decode("utf-8", errors="replace")
    # Extract URLs from OSC 8 hyperlinks BEFORE general stripping so URLs are preserved.
    # Both BEL (\x07) and ST (\x1b\) terminators are common; process ST first to avoid
    # partial matches (ST starts with \x1b which BEL pattern doesn't consume).
    raw = _OSC8_ST_RE.sub(lambda m: m.group(1), raw)
    raw = _OSC8_BEL_RE.sub(lambda m: m.group(1), raw)
    # Replace cursor-right with spaces before stripping (TUI word spacing)
    raw = _CURSOR_RIGHT_RE.sub(_cursor_right_to_spaces, raw)
    cleaned = CTRL_RE.sub("", raw)
    # \r\r\n (double CR before LF) → single \n, then normalise remaining endings
    cleaned = cleaned.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


# ── PtySession ────────────────────────────────────────────────────────────────

EventSendCallback = Callable[[dict], Awaitable[None]]


class PtySession:
    """
    Manages one CLI subprocess inside a PTY.

    Usage::

        session = PtySession(
            cmd=["claude", "--dangerously-skip-permissions"],
            cwd="/var/workspaces/user/sessions/abc",
            env={"HOME": "/var/workspaces/user", "TERM": "xterm-256color"},
            uid=1000, gid=1000,
        )
        await session.start()

        # Read chunks
        chunk = await session.read()   # raw bytes or None on EOF

        # Write input
        await session.write(b"hello\\n")

        await session.stop()

    Alternatively use run() which drives the full lifecycle:

        await session.run(send_event=my_callback, recv_input=my_input_gen)
    """

    def __init__(
        self,
        cmd: list[str],
        cwd: str,
        env: Optional[dict] = None,
        uid: Optional[int] = None,
        gid: Optional[int] = None,
        session_id: str = "",
        auto_confirm_delay: float = 2.0,
        cols: int = 500,
        rows: int = 50,
    ):
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or os.environ.copy()
        self.uid = uid
        self.gid = gid
        self.session_id = session_id
        self.auto_confirm_delay = auto_confirm_delay
        self.cols = cols
        self.rows = rows

        self._master_fd: Optional[int] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Open PTY and spawn subprocess."""
        self._loop = asyncio.get_running_loop()
        logger.debug("[%s] PTY start: cmd=%s cwd=%s uid=%s gid=%s",
                     self.session_id, self.cmd, self.cwd, self.uid, self.gid)
        logger.debug("[%s] PTY env PATH: %s", self.session_id, self.env.get("PATH", "(not set)"))
        master_fd, slave_fd = pty.openpty()
        os.set_blocking(master_fd, False)
        self._master_fd = master_fd

        # Set terminal window size — must happen before exec so the process
        # sees the correct dimensions from the start.  Wide cols prevent long
        # URLs (e.g. OAuth) and other output from being line-wrapped.
        winsize = struct.pack("HHHH", self.rows, self.cols, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)

        preexec = self._make_preexec(slave_fd)

        self._process = await asyncio.create_subprocess_exec(
            *self.cmd,
            cwd=self.cwd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=self.env,
            preexec_fn=preexec,
        )
        os.close(slave_fd)  # parent only needs master end
        logger.debug("[%s] PTY spawned pid=%s", self.session_id, self._process.pid)

        # Auto-confirm: send Enter after delay to dismiss startup wizards
        if self.auto_confirm_delay > 0:
            asyncio.create_task(self._auto_confirm())

    async def read(self) -> Optional[bytes]:
        """Read one chunk from the PTY master. Returns None on EOF."""
        assert self._master_fd is not None
        while True:
            try:
                return os.read(self._master_fd, 4096)
            except BlockingIOError:
                if self._process and self._process.returncode is not None:
                    return None
                await asyncio.sleep(0.02)
            except OSError:
                return None

    async def write(self, data: bytes) -> None:
        """Write bytes to the PTY master (subprocess stdin)."""
        assert self._master_fd is not None
        view = memoryview(data)
        while view:
            try:
                written = os.write(self._master_fd, view)
                view = view[written:]
            except BlockingIOError:
                await asyncio.sleep(0.02)

    async def stop(self) -> None:
        """Terminate the subprocess and close the PTY master fd."""
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()

    @property
    def exit_code(self) -> Optional[int]:
        if self._process:
            return self._process.returncode
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_preexec(self, slave_fd: int):
        uid, gid = self.uid, self.gid

        def _drop_privs():
            os.setsid()
            try:
                fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            except OSError:
                pass
            if gid is not None:
                os.setgid(gid)
            if uid is not None:
                os.setuid(uid)

        return _drop_privs

    async def _auto_confirm(self) -> None:
        await asyncio.sleep(self.auto_confirm_delay)
        try:
            await self.write(b"\n")
        except OSError:
            pass
