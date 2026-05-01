"""Core relay — spawns a subprocess and bridges it to a WebSocket connection."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol

from .adapters import get_adapter, BaseAdapter
from .events import EventType, RelayEvent

logger = logging.getLogger(__name__)


class RelaySession:
    """
    Manages one coding-agent subprocess and one WebSocket connection.
    Streams subprocess output as structured RelayEvents to the client,
    and forwards client messages as stdin to the subprocess.
    """

    def __init__(
        self,
        session_id: str,
        tool: str,
        folder: str,
        model: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
    ):
        self.session_id = session_id
        self.tool = tool
        self.folder = os.path.abspath(folder)
        self.model = model
        self.extra_args = extra_args or []
        self._adapter: type[BaseAdapter] = get_adapter(tool)
        self._process: Optional[asyncio.subprocess.Process] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, ws: WebSocketServerProtocol) -> None:
        """Spawn the subprocess and relay I/O until the process exits or WS closes."""
        cmd = self._adapter.build_command(self.folder, self.model, self.extra_args)
        logger.info("[%s] Starting: %s in %s", self.session_id, cmd, self.folder)

        await self._send(ws, RelayEvent(
            type=EventType.SESSION_START,
            session_id=self.session_id,
            metadata={"tool": self.tool, "folder": self.folder, "model": self.model, "cmd": cmd},
        ))

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.folder,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            await self._send(ws, RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=f"Command not found: {cmd[0]}. Is it installed and on PATH?",
            ))
            return

        # Run stream readers and WS input pump concurrently.
        # Cancel WS pump once both output streams are exhausted (process done).
        ws_task = asyncio.create_task(self._read_ws(ws))
        try:
            await asyncio.gather(
                self._read_stream(ws, self._process.stdout, "stdout"),
                self._read_stream(ws, self._process.stderr, "stderr"),
                return_exceptions=True,
            )
        finally:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass

        exit_code = await self._process.wait()
        await self._send(ws, RelayEvent(
            type=EventType.SESSION_END,
            session_id=self.session_id,
            exit_code=exit_code,
        ))
        logger.info("[%s] Process exited with code %d", self.session_id, exit_code)

    async def stop(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()

    # ── I/O pumps ─────────────────────────────────────────────────────────────

    async def _read_stream(
        self, ws: WebSocketServerProtocol, stream: asyncio.StreamReader, name: str
    ) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = self._adapter.postprocess_line(line.decode("utf-8", errors="replace"))
            event = RelayEvent.from_raw(self.session_id, name, decoded)
            await self._send(ws, event)

    async def _read_ws(self, ws: WebSocketServerProtocol) -> None:
        """Forward WebSocket messages to the subprocess stdin."""
        async for raw in ws:
            try:
                msg = json.loads(raw) if isinstance(raw, str) else {}
            except json.JSONDecodeError:
                msg = {"text": raw}

            text = msg.get("text", "")
            if not text:
                continue

            processed = self._adapter.preprocess_input(text)
            if self._process and self._process.stdin:
                self._process.stdin.write(processed.encode())
                await self._process.stdin.drain()
                await self._send(ws, RelayEvent(
                    type=EventType.INPUT_ACK,
                    session_id=self.session_id,
                    text=text,
                ))

    @staticmethod
    async def _send(ws: WebSocketServerProtocol, event: RelayEvent) -> None:
        try:
            await ws.send(event.to_json())
        except Exception:
            pass


class RelayServer:
    """
    WebSocket server that accepts connections and spawns a RelaySession per client.

    Protocol (client → server on connect):
        { "tool": "claude", "folder": "/path/to/project", "model": "sonnet" }
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port

    async def handle(self, ws: WebSocketServerProtocol) -> None:
        # First message must be the session config
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            config = json.loads(raw)
        except Exception as exc:
            await ws.send(RelayEvent(
                type=EventType.ERROR,
                session_id="",
                text=f"Invalid handshake: {exc}",
            ).to_json())
            return

        session = RelaySession(
            session_id=config.get("session_id") or str(uuid.uuid4()),
            tool=config.get("tool", "claude"),
            folder=config.get("folder", "."),
            model=config.get("model"),
            extra_args=config.get("extra_args"),
        )
        try:
            await session.start(ws)
        finally:
            await session.stop()

    async def serve(self) -> None:
        logger.info("agent-relay listening on ws://%s:%d", self.host, self.port)
        async with websockets.serve(self.handle, self.host, self.port):
            await asyncio.Future()  # run forever

    def run(self) -> None:
        asyncio.run(self.serve())
