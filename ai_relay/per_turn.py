"""
PerTurnRuntime — restarts the agent subprocess for every user turn.

Claude Code and Gemini CLI in --print / headless mode read ALL stdin until
EOF, then process and respond.  They do NOT support an interactive
request-response loop over a single persistent stdin pipe.

PerTurnRuntime works around this by:
  - Starting a fresh subprocess for each user message
  - Passing ``--resume <session_id>`` on subsequent turns so the model
    picks up conversation history from its own on-disk store
  - Forwarding permission responses / control messages to the current
    subprocess immediately (Claude Code processes them on receipt without
    needing EOF)

The outward-facing AgentRuntime interface is unchanged — callers read
a single continuous event stream via ``read_event()`` and send messages
via ``handle_client_message()``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Optional, Type

from .adapters.base import AgentRuntime
from .events import EventType, RelayEvent

logger = logging.getLogger(__name__)

# Accepted model identifiers for set_model: full ids (claude-opus-4-8, …) and
# short aliases (sonnet, opus, haiku). Must start alphanumeric — this blocks
# leading-dash / whitespace tokens that could look like CLI flags.
_VALID_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Auto-reset safeguards against a bloated/hung resume conversation:
#  - Rotate (start a fresh conversation) when the on-disk transcript exceeds this
#    size — a huge transcript makes `--resume` slow or hang on load.
#  - Watchdog: if a resumed turn emits `system/init` then stalls (no further event)
#    for this long with no result, kill it and retry once WITHOUT --resume.
_TRANSCRIPT_ROTATE_BYTES = int(os.environ.get("AI_RELAY_TRANSCRIPT_ROTATE_BYTES", 5 * 1024 * 1024))
_RESUME_HANG_TIMEOUT = float(os.environ.get("AI_RELAY_RESUME_HANG_TIMEOUT", 90))


class PerTurnRuntime(AgentRuntime):
    """Wraps any AgentRuntime subclass, restarting per user turn."""

    #: Message types forwarded directly to the current subprocess (no restart).
    CONTROL_TYPES = frozenset(
        {"interrupt", "permission_response", "set_model", "set_permission_mode"}
    )

    def __init__(
        self,
        tool: str,
        session_id: str,
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        runtime_class: Type[AgentRuntime],
        resume_flag: str = "--resume",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(session_id, config)
        self.tool = tool
        self._base_cmd = cmd
        self._cwd = cwd
        self._env = env
        self._runtime_class = runtime_class
        self._resume_flag = resume_flag

        self._events: asyncio.Queue[Optional[RelayEvent]] = asyncio.Queue()
        self._current: Optional[AgentRuntime] = None
        self._reader_task: Optional[asyncio.Task] = None
        # Claude-specific: internal session_id for --resume.
        # Seeded from handshake config["claude_session_id"] so the conversation
        # is resumed even after a container restart.
        self._agent_session_id: Optional[str] = (config or {}).get("claude_session_id") or None
        if self._agent_session_id:
            logger.info(
                "[per-turn:%s] seeding resume from handshake claude_session_id=%s",
                session_id, self._agent_session_id,
            )
        # Model override: persisted across turns when user selects a new model.
        # Applied by replacing/inserting --model in the command on every _run_turn().
        self._override_model: Optional[str] = None
        self._turn_lock = asyncio.Lock()
        self._stopped = False
        # Cumulative cost/token tracking — populated from result messages each turn.
        self._total_cost_usd: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._turn_count: int = 0
        # Real model + context window, captured from the subprocess `system/init`
        # and `result` events. The override (if set) wins for display since it's
        # what the NEXT turn will use; otherwise we report what's actually running.
        self._current_model: Optional[str] = None
        self._last_context_tokens: int = 0
        # Auto-reset bookkeeping (transcript rotation + resume-hang watchdog).
        self._last_event_at: float = 0.0
        self._turn_got_result: bool = False
        self._watchdog_task: Optional[asyncio.Task] = None

    # ── AgentRuntime interface ────────────────────────────────────────────────

    async def start(self) -> None:
        """No-op: relay.py already emits SESSION_START before calling start().
        Subprocess starts on first user message via _run_turn()."""

    async def read_event(self) -> Optional[RelayEvent]:
        return await self._events.get()

    async def handle_client_message(self, msg: dict[str, Any]) -> None:
        if self._stopped:
            return

        msg_type = msg.get("type", "")

        # /update slash command: stop subprocess, run `claude update`, report.
        if msg_type == "claude_update":
            asyncio.create_task(self._run_update())
            return

        # Control / permission messages: forward to running subprocess.
        # set_model is also persisted so the NEXT subprocess uses the new model.
        if msg_type in self.CONTROL_TYPES:
            if msg_type == "set_model":
                new_model = (msg.get("model") or "").strip()
                if new_model and not _VALID_MODEL_RE.match(new_model):
                    # Reject malformed values defensively (no leading dash, no
                    # whitespace) so a crafted set_model can't smuggle a flag-like
                    # token into the launch command. Resume/turn flow is untouched.
                    logger.warning(
                        "[per-turn:%s] rejected invalid model override: %r",
                        self.session_id, new_model,
                    )
                    await self._events.put(RelayEvent(
                        type=EventType.RESPONSE,
                        session_id=self.session_id,
                        text=f"Ignored invalid model name `{new_model}`.",
                        metadata={"model_switch_rejected": new_model, "source": "set_model"},
                    ))
                elif new_model:
                    self._override_model = new_model
                    logger.info(
                        "[per-turn:%s] model override → %s (takes effect next turn)",
                        self.session_id, new_model,
                    )
                    await self._events.put(RelayEvent(
                        type=EventType.RESPONSE,
                        session_id=self.session_id,
                        text=f"Model switched to **{new_model}**. Takes effect on your next message.",
                        metadata={"model_switch": new_model, "source": "set_model"},
                    ))
            if self._current:
                await self._current.handle_client_message(msg)
            return

        # User prompt: check there is actual content.
        prompt = self._extract_prompt(msg)
        if not prompt:
            return

        # Relay-handled slash commands — respond locally, never forward to subprocess.
        if prompt.strip() == "/cost":
            await self._emit_cost_summary()
            return
        if prompt.strip() == "/status":
            await self._emit_status_summary()
            return
        # /new (alias /clear): drop the resume pointer so the NEXT turn starts a
        # fresh Claude conversation. Self-service reset for a bloated/stuck session.
        if prompt.strip() in ("/new", "/clear"):
            self._agent_session_id = None
            logger.info("[per-turn:%s] /new — resume pointer cleared, next turn is fresh", self.session_id)
            await self._events.put(RelayEvent(
                type=EventType.RESPONSE,
                session_id=self.session_id,
                text="Started a fresh conversation. Your earlier chat history is still saved.",
                metadata={"source": "new_conversation"},
            ))
            return

        async with self._turn_lock:
            await self._run_turn(msg)

    async def stop(self) -> None:
        self._stopped = True
        if self._watchdog_task:
            self._watchdog_task.cancel()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._current:
            await self._current.stop()
        await self._events.put(None)

    async def wait(self) -> Optional[int]:
        return None  # multi-turn; no single meaningful exit code

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_turn(self, msg: dict[str, Any], from_watchdog: bool = False) -> None:
        """Cancel previous reader, stop old subprocess, start a new one."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._current:
            try:
                await self._current.stop()
            except Exception:
                pass

        # Auto-rotation: if the on-disk transcript we'd resume is too large, start a
        # FRESH conversation instead — a huge transcript makes --resume slow/hang.
        if self._agent_session_id and not from_watchdog:
            size = self._transcript_size(self._agent_session_id)
            if size is not None and size > _TRANSCRIPT_ROTATE_BYTES:
                logger.warning(
                    "[per-turn:%s] transcript %.1fMB > %.1fMB cap — rotating to a fresh conversation",
                    self.session_id, size / 1048576, _TRANSCRIPT_ROTATE_BYTES / 1048576,
                )
                self._agent_session_id = None
                await self._events.put(RelayEvent(
                    type=EventType.RESPONSE,
                    session_id=self.session_id,
                    text="Started a fresh context to keep things fast — your earlier chat history is still saved.",
                    metadata={"source": "auto_rotate", "transcript_bytes": size},
                ))

        used_resume = self._agent_session_id is not None

        cmd = list(self._base_cmd)
        # Apply model override: replaces --model in the base command or appends it.
        if self._override_model:
            cmd = self._apply_model_override(cmd, self._override_model)
        if self._agent_session_id:
            cmd = cmd + [self._resume_flag, self._agent_session_id]

        logger.info(
            "[per-turn:%s] Starting subprocess %s (resume=%s)",
            self.session_id, cmd[0], self._agent_session_id,
        )

        # Pass the captured Claude conversation ID so ClaudeStructuredRuntime
        # sends the correct session_id in its stream-json stdin payload.
        # On the first turn _agent_session_id is None → new conversation.
        try:
            self._current = self._runtime_class(
                self.session_id, cmd, self._cwd, self._env,
                claude_session_id=self._agent_session_id,
                config=self.config,
            )
        except TypeError:
            # Fallback for runtimes that don't accept claude_session_id (e.g. Gemini)
            self._current = self._runtime_class(self.session_id, cmd, self._cwd, self._env, config=self.config)
        await self._current.start()
        await self._current.handle_client_message(msg)

        # Reset per-turn watchdog state.
        self._turn_got_result = False
        self._last_event_at = asyncio.get_event_loop().time()
        self._reader_task = asyncio.create_task(self._pump_turn(self._current))

        # Resume-hang watchdog: only when we actually resumed (and not already a
        # fresh-retry). If the resumed subprocess stalls after init, retry fresh.
        if used_resume and not from_watchdog:
            runtime = self._current
            self._watchdog_task = asyncio.create_task(self._resume_watchdog(msg, runtime))

    async def _pump_turn(self, runtime: AgentRuntime) -> None:
        """Forward events from one turn's subprocess to the shared event queue."""
        try:
            while not self._stopped:
                event = await runtime.read_event()
                # Watchdog liveness: record that this turn is producing output.
                self._last_event_at = asyncio.get_event_loop().time()
                if event is None:
                    logger.debug("[per-turn:%s] turn EOF", self.session_id)
                    self._turn_got_result = True
                    break
                # Capture Claude's internal session_id for --resume on next turn.
                if event.raw:
                    raw = event.raw
                    if raw.get("type") == "system" and raw.get("subtype") == "init":
                        # The init event reports the model the subprocess actually
                        # launched with — capture it so /status shows the real value.
                        init_model = raw.get("model")
                        if init_model:
                            self._current_model = str(init_model)
                        sid = raw.get("session_id")
                        if sid:
                            self._agent_session_id = sid
                            logger.info(
                                "[per-turn:%s] agent session_id=%s",
                                self.session_id, sid,
                            )
                            # Emit to client so it can persist the ID for resume
                            await self._events.put(RelayEvent(
                                type=EventType.SESSION_ATTACH,
                                session_id=self.session_id,
                                metadata={"claude_session_id": sid},
                            ))
                await self._events.put(event)
                # Accumulate cost/token totals from each turn's result message.
                if event.raw and event.raw.get("type") == "result":
                    self._turn_got_result = True
                if event.raw and event.raw.get("type") == "result" and event.raw.get("subtype") == "success":
                    self._total_cost_usd += float(event.raw.get("total_cost_usd") or 0.0)
                    usage = event.raw.get("usage") or {}
                    self._total_input_tokens += int(usage.get("input_tokens", 0))
                    self._total_output_tokens += int(usage.get("output_tokens", 0))
                    self._turn_count += 1
                    # Live context window = everything Claude re-read this turn
                    # (input + cache) — approximates what /status shows natively.
                    self._last_context_tokens = (
                        int(usage.get("input_tokens", 0))
                        + int(usage.get("cache_read_input_tokens", 0))
                        + int(usage.get("cache_creation_input_tokens", 0))
                    )
                # Auto-compact when context window is full
                if event.type == EventType.CONTEXT_WARNING:
                    try:
                        await runtime.handle_client_message({"text": "/compact"})
                        logger.info("[per-turn:%s] auto-sent /compact on context_warning", self.session_id)
                    except Exception as _ce:
                        logger.warning("[per-turn:%s] auto-compact failed: %s", self.session_id, _ce)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("[per-turn:%s] transport error: %s", self.session_id, exc)
            await self._events.put(RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=f"Relay error: {exc}",
            ))

    def _transcript_path(self, claude_session_id: str) -> Optional[str]:
        """Path to Claude Code's on-disk transcript for a conversation, or None.

        Claude stores it at ~/.claude/projects/{cwd-with-slashes-and-dots-as-dashes}/
        {session_id}.jsonl. Only meaningful for the Claude adapter.
        """
        if not claude_session_id or not self._cwd:
            return None
        enc = re.sub(r"[/.]", "-", self._cwd)
        return os.path.join(os.path.expanduser("~/.claude/projects"), enc, f"{claude_session_id}.jsonl")

    def _transcript_size(self, claude_session_id: str) -> Optional[int]:
        """Size in bytes of the resume transcript, or None if not found/applicable."""
        path = self._transcript_path(claude_session_id)
        if not path:
            return None
        try:
            return os.path.getsize(path)
        except OSError:
            return None

    async def _resume_watchdog(self, msg: dict[str, Any], runtime: AgentRuntime) -> None:
        """If a resumed turn stalls (no events for _RESUME_HANG_TIMEOUT after start)
        without producing a result, kill it and retry once WITHOUT --resume."""
        try:
            await asyncio.sleep(_RESUME_HANG_TIMEOUT)
            if self._stopped or self._turn_got_result:
                return
            if runtime is not self._current:
                return  # a newer turn already superseded this one
            idle = asyncio.get_event_loop().time() - self._last_event_at
            if idle < _RESUME_HANG_TIMEOUT:
                return  # still producing output → genuinely long turn, leave it
            logger.warning(
                "[per-turn:%s] resumed turn stalled %.0fs with no result — retrying fresh",
                self.session_id, idle,
            )
            await self._events.put(RelayEvent(
                type=EventType.RESPONSE,
                session_id=self.session_id,
                text="The previous conversation was too large to resume — started a fresh context and retried. Your earlier chat history is still saved.",
                metadata={"source": "resume_hang_retry"},
            ))
            self._agent_session_id = None  # force a fresh conversation
            async with self._turn_lock:
                if not self._stopped and runtime is self._current:
                    await self._run_turn(msg, from_watchdog=True)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("[per-turn:%s] resume watchdog error: %s", self.session_id, exc)

    async def _emit_cost_summary(self) -> None:
        lines = [
            f"**Session cost:** ${self._total_cost_usd:.4f}",
            f"**Input tokens:** {self._total_input_tokens:,}",
            f"**Output tokens:** {self._total_output_tokens:,}",
            f"**Turns:** {self._turn_count}",
        ]
        await self._events.put(RelayEvent(
            type=EventType.RESPONSE,
            session_id=self.session_id,
            text="\n".join(lines),
            metadata={"source": "cost_summary"},
        ))

    async def _emit_status_summary(self) -> None:
        # Prefer the override (what the next turn will use); fall back to the model
        # the running subprocess actually reported via its init event.
        model = self._override_model or self._current_model or "default"
        pending = " (applies next turn)" if self._override_model and self._override_model != self._current_model else ""
        lines = [
            f"**Model:** {model}{pending}",
            f"**Session:** {self._agent_session_id or 'new (no turn yet)'}",
            f"**Turns:** {self._turn_count}",
            f"**Context (last turn):** {self._last_context_tokens:,} tokens",
            f"**Cost so far:** ${self._total_cost_usd:.4f}",
        ]
        await self._events.put(RelayEvent(
            type=EventType.RESPONSE,
            session_id=self.session_id,
            text="\n".join(lines),
            metadata={"source": "status_summary"},
        ))

    @staticmethod
    def _apply_model_override(cmd: list[str], model: str) -> list[str]:
        """Replace --model <value> in cmd list, or append if not present."""
        result: list[str] = []
        i = 0
        replaced = False
        while i < len(cmd):
            if cmd[i] == "--model" and i + 1 < len(cmd):
                result += ["--model", model]
                i += 2
                replaced = True
            else:
                result.append(cmd[i])
                i += 1
        if not replaced:
            result += ["--model", model]
        return result

    async def _run_update(self) -> None:
        """Handle /update: stop subprocess, update Claude Code, report result.

        Uses ``claude install`` (the native installer) rather than ``claude update``.
        ``claude install`` is the sudo-free path: it installs the latest version to
        ``$HOME/.local`` regardless of how Claude was originally installed. This works
        for non-root users even when the baseline is a root-owned npm global install
        (where ``claude update`` fails with "npm global folder isn't writable").
        The resulting ``$HOME/.local/bin/claude`` is preferred via PATH on the next turn.

        Resume continuity: this only restarts the inner Claude subprocess, never the
        relay. ``self._agent_session_id`` (and ``self._override_model``) are kept, so the
        next turn launches with ``--resume <id>`` and the existing context is preserved.
        """
        logger.info(
            "[per-turn:%s] /update starting; preserving resume id=%s model=%s",
            self.session_id, self._agent_session_id, self._override_model,
        )
        await self._events.put(RelayEvent(
            type=EventType.STATUS,
            session_id=self.session_id,
            status="updating",
            text="Updating Claude Code…",
        ))

        # Stop any in-flight subprocess gracefully before running the updater.
        async with self._turn_lock:
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass
                self._reader_task = None
            if self._current:
                try:
                    await self._current.stop()
                except Exception:
                    pass
                self._current = None

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "install",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=self._env,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
                output = (stdout or b"").decode("utf-8", errors="replace").strip()
                rc = proc.returncode or 0
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                output = "Update timed out after 180 seconds."
                rc = -1
        except FileNotFoundError:
            output = "`claude` binary not found in PATH."
            rc = -1

        if rc == 0:
            text = (
                f"Claude Code updated to the latest version.\n\n{output}\n\n"
                "Send any message to start a session with the new version."
            )
        else:
            text = f"Update failed (exit {rc}).\n\n{output}"

        logger.info("[per-turn:%s] claude install rc=%s", self.session_id, rc)
        await self._events.put(RelayEvent(
            type=EventType.RESPONSE,
            session_id=self.session_id,
            text=text,
            metadata={"update_exit_code": rc, "source": "claude_update"},
        ))

    @staticmethod
    def _extract_prompt(msg: dict[str, Any]) -> Optional[str]:
        content = msg.get("content")
        if content is None:
            content = msg.get("text", "")
        if isinstance(content, list):
            parts = [
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") in {"text", "input_text"}
            ]
            prompt = "\n".join(x for x in parts if x)
            # Image-only message: treat as non-empty so the turn is executed
            if not prompt and any(
                isinstance(p, dict) and p.get("type") == "image"
                for p in content
            ):
                prompt = "[image]"
        else:
            prompt = str(content)
        return prompt.strip() or None
