"""Google Gemini CLI adapter."""

from __future__ import annotations
import asyncio
import json
from typing import Any, Optional

from ..events import EventType, RelayEvent
from ..transports import StructuredProcessTransport
from .base import AgentRuntime, BaseAdapter


class GeminiStreamRuntime(AgentRuntime):
    """Gemini CLI headless stream-json runtime.

    Gemini headless mode is turn-oriented: each user prompt starts one
    subprocess and streams JSONL events until that turn completes.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str,
        env: dict[str, str],
        model: Optional[str],
        extra_args: Optional[list[str]],
    ):
        super().__init__(session_id)
        self.cwd = cwd
        self.env = env
        self.model = model
        self.extra_args = extra_args or []
        self._events: asyncio.Queue[Optional[RelayEvent]] = asyncio.Queue()
        self._turn_task: Optional[asyncio.Task[None]] = None
        self._transport: Optional[StructuredProcessTransport] = None

    async def start(self) -> None:
        return

    async def read_event(self) -> Optional[RelayEvent]:
        return await self._events.get()

    async def handle_client_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type == "interrupt":
            await self._stop_turn()
            await self._events.put(RelayEvent(
                type=EventType.STATUS,
                session_id=self.session_id,
                status="interrupted",
            ))
            return

        content = msg.get("content")
        if content is None:
            content = msg.get("text", "")
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
            ]
            prompt = "\n".join(part for part in text_parts if part)
        else:
            prompt = str(content)
        if not prompt:
            return
        if self._turn_task and not self._turn_task.done():
            await self._events.put(RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text="Gemini turn already in progress; interrupt or wait before sending another prompt.",
            ))
            return
        self._turn_task = asyncio.create_task(self._run_turn(prompt))

    async def stop(self) -> None:
        await self._stop_turn()
        await self._events.put(None)

    async def wait(self) -> Optional[int]:
        return None

    async def _stop_turn(self) -> None:
        if self._transport:
            await self._transport.stop()
            self._transport = None
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass

    async def _run_turn(self, prompt: str) -> None:
        cmd = ["gemini", "--prompt", prompt, "--output-format", "stream-json"]
        if self.model:
            cmd += ["--model", self.model]
        cmd += self.extra_args
        transport = StructuredProcessTransport(cmd, self.cwd, self.env)
        self._transport = transport
        try:
            await transport.start()
            stderr_task = asyncio.create_task(self._read_stderr(transport))
            while True:
                line = await transport.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await self._events.put(RelayEvent(
                        type=EventType.STDOUT,
                        session_id=self.session_id,
                        text=text,
                    ))
                    continue
                for event in self._events_from_payload(payload):
                    await self._events.put(event)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
        except FileNotFoundError:
            await self._events.put(RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text="Command not found: gemini. Is it installed and on PATH?",
            ))
        finally:
            if self._transport is transport:
                self._transport = None

    async def _read_stderr(self, transport: StructuredProcessTransport) -> None:
        while True:
            line = await transport.read_stderr()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if text:
                await self._events.put(RelayEvent(
                    type=EventType.STDERR,
                    session_id=self.session_id,
                    text=text,
                ))

    def _events_from_payload(self, payload: Any) -> list[RelayEvent]:
        if not isinstance(payload, dict):
            return [RelayEvent(type=EventType.STDOUT, session_id=self.session_id, content=payload)]
        kind = payload.get("type")
        if kind == "init":
            return [RelayEvent(
                type=EventType.STATUS,
                session_id=self.session_id,
                status="initialized",
                metadata=payload,
                raw=payload,
            )]
        if kind == "message":
            role = payload.get("role")
            return [RelayEvent(
                type=EventType.ASSISTANT_MESSAGE if role == "assistant" else EventType.USER_MESSAGE,
                session_id=self.session_id,
                text=payload.get("text") or payload.get("content"),
                content=payload.get("content"),
                raw=payload,
            )]
        if kind == "tool_use":
            return [RelayEvent(
                type=EventType.TOOL_CALL,
                session_id=self.session_id,
                tool=payload.get("name") or payload.get("tool"),
                args=payload.get("args") or payload.get("arguments"),
                tool_use_id=payload.get("id") or payload.get("tool_use_id"),
                raw=payload,
            )]
        if kind == "tool_result":
            return [RelayEvent(
                type=EventType.TOOL_RESULT,
                session_id=self.session_id,
                tool_use_id=payload.get("id") or payload.get("tool_use_id"),
                content=payload.get("result") or payload.get("content"),
                raw=payload,
            )]
        if kind == "error":
            return [RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=payload.get("message") or payload.get("error"),
                raw=payload,
            )]
        if kind == "result":
            return [RelayEvent(
                type=EventType.RESPONSE,
                session_id=self.session_id,
                text=payload.get("response") or payload.get("text"),
                metadata=payload.get("stats") or payload,
                raw=payload,
            )]
        return [RelayEvent(type=EventType.STREAM_EVENT, session_id=self.session_id, raw=payload)]


class GeminiAdapter(BaseAdapter):
    tool_name = "gemini"
    protocol = "stream-json"

    @classmethod
    def build_command(cls, folder: str, model: Optional[str] = None,
                      extra_args: Optional[list[str]] = None) -> list[str]:
        cmd = ["gemini"]
        if model:
            cmd += ["--model", model]
        if extra_args:
            cmd += extra_args
        return cmd

    @classmethod
    def create_runtime(
        cls,
        session_id: str,
        folder: str,
        model: Optional[str],
        extra_args: Optional[list[str]],
        env: dict[str, str],
        config: Optional[dict[str, Any]] = None,
    ) -> AgentRuntime:
        return GeminiStreamRuntime(
            session_id=session_id,
            cwd=folder,
            env=env,
            model=model,
            extra_args=extra_args,
        )
