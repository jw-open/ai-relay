"""Snowflake Cortex API adapter."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from ..events import EventType, RelayEvent
from .base import AgentRuntime, BaseAdapter


class CortexRuntime(AgentRuntime):
    """Snowflake Cortex runtime using REST requests and SSE streaming."""

    def __init__(
        self,
        session_id: str,
        model: Optional[str],
        config: dict[str, Any],
    ):
        super().__init__(session_id)
        self.model = model
        self.config = config
        self.snowflake = config.get("snowflake") if isinstance(config.get("snowflake"), dict) else {}
        self.mode = str(config.get("mode") or self.snowflake.get("mode") or "chat")
        self._events: asyncio.Queue[Optional[RelayEvent]] = asyncio.Queue()
        self._tasks: set[asyncio.Task[None]] = set()
        self._history: list[dict[str, Any]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        return

    async def read_event(self) -> Optional[RelayEvent]:
        return await self._events.get()

    async def handle_client_message(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "interrupt":
            for task in list(self._tasks):
                task.cancel()
            self._tasks.clear()
            await self._events.put(RelayEvent(
                type=EventType.STATUS,
                session_id=self.session_id,
                status="interrupted",
            ))
            return

        content = msg.get("content")
        if content is None:
            content = msg.get("text", "")
        text = self._extract_text(content)
        if not text:
            return
        task = asyncio.create_task(self._run_request(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        await self._events.put(None)

    async def wait(self) -> Optional[int]:
        return None

    async def _run_request(self, text: str) -> None:
        try:
            await asyncio.to_thread(self._request_sync, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._events.put(RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=str(exc),
            ))

    def _request_sync(self, text: str) -> None:
        url = self._endpoint()
        body = self._request_body(text)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self._setting("timeout", 300))) as response:
                self._consume_sse(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self._put(RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=f"Snowflake Cortex HTTP {exc.code}: {body}",
            ))

    def _consume_sse(self, response: Any) -> None:
        event_name = "message"
        data_lines: list[str] = []
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                self._flush_sse_event(event_name, data_lines)
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        self._flush_sse_event(event_name, data_lines)

    def _flush_sse_event(self, event_name: str, data_lines: list[str]) -> None:
        if not data_lines:
            return
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            self._put(RelayEvent(
                type=EventType.STATUS,
                session_id=self.session_id,
                status="done",
            ))
            return
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            payload = {"data": data_text}
        for event in self._events_from_sse(event_name, payload):
            self._put(event)

    def _events_from_sse(self, event_name: str, payload: dict[str, Any]) -> list[RelayEvent]:
        if self.mode == "analyst":
            return self._analyst_events(event_name, payload)
        return self._chat_events(event_name, payload)

    def _analyst_events(self, event_name: str, payload: dict[str, Any]) -> list[RelayEvent]:
        if event_name == "status":
            return [RelayEvent(
                type=EventType.STATUS,
                session_id=self.session_id,
                status=payload.get("status"),
                raw=payload,
            )]
        if event_name == "message.content.delta":
            delta_type = payload.get("type")
            if delta_type == "sql":
                statement = payload.get("statement_delta") or payload.get("statement")
                return [RelayEvent(
                    type=EventType.TOOL_CALL,
                    session_id=self.session_id,
                    tool="sql",
                    args={"statement_delta": statement, "index": payload.get("index")},
                    text=statement,
                    raw=payload,
                )]
            return [RelayEvent(
                type=EventType.RESPONSE,
                session_id=self.session_id,
                text=payload.get("text_delta") or payload.get("text") or "",
                raw=payload,
            )]
        if event_name == "warnings":
            return [RelayEvent(
                type=EventType.CONTEXT_WARNING,
                session_id=self.session_id,
                text=json.dumps(payload),
                raw=payload,
            )]
        if event_name == "error":
            return [RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=payload.get("message") or json.dumps(payload),
                raw=payload,
            )]
        if event_name == "response_metadata":
            return [RelayEvent(
                type=EventType.STATUS,
                session_id=self.session_id,
                status="response_metadata",
                metadata=payload,
                raw=payload,
            )]
        if event_name == "done":
            return [RelayEvent(type=EventType.STATUS, session_id=self.session_id, status="done", raw=payload)]
        return [RelayEvent(type=EventType.STREAM_EVENT, session_id=self.session_id, metadata={"event": event_name}, raw=payload)]

    def _chat_events(self, event_name: str, payload: dict[str, Any]) -> list[RelayEvent]:
        if "error" in payload:
            return [RelayEvent(
                type=EventType.ERROR,
                session_id=self.session_id,
                text=json.dumps(payload["error"]),
                raw=payload,
            )]
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                return [RelayEvent(
                    type=EventType.RESPONSE,
                    session_id=self.session_id,
                    text=content,
                    raw=payload,
                )]
            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                return [RelayEvent(
                    type=EventType.STATUS,
                    session_id=self.session_id,
                    status=str(finish_reason),
                    raw=payload,
                )]
        return [RelayEvent(type=EventType.STREAM_EVENT, session_id=self.session_id, metadata={"event": event_name}, raw=payload)]

    def _endpoint(self) -> str:
        account_url = str(self._setting("account_url", "")).rstrip("/")
        if not account_url:
            raise ValueError("Snowflake Cortex requires snowflake.account_url.")
        if self.mode == "analyst":
            return f"{account_url}/api/v2/cortex/analyst/message"
        return f"{account_url}/api/v2/cortex/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        token = self._setting("token")
        token_env = self._setting("token_env", "SNOWFLAKE_PAT")
        if not token and token_env:
            token = os.environ.get(str(token_env))
        if not token:
            raise ValueError("Snowflake Cortex requires snowflake.token or snowflake.token_env.")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        token_type = self._setting("token_type")
        if token_type:
            headers["X-Snowflake-Authorization-Token-Type"] = str(token_type)
        return headers

    def _request_body(self, text: str) -> dict[str, Any]:
        if self.mode == "analyst":
            body: dict[str, Any] = {
                "messages": self._history + [{
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }],
                "stream": True,
            }
            for key in ("semantic_model_file", "semantic_model", "semantic_models", "semantic_view"):
                value = self._setting(key)
                if value is not None:
                    body[key] = value
            if not any(k in body for k in ("semantic_model_file", "semantic_model", "semantic_models", "semantic_view")):
                raise ValueError(
                    "Cortex Analyst requires one of snowflake.semantic_model_file, semantic_model, semantic_models, or semantic_view."
                )
            return body

        messages = self._history + [{"role": "user", "content": text}]
        return {
            "model": self.model or self._setting("model", "claude-sonnet-4-5"),
            "messages": messages,
            "stream": True,
        }

    def _setting(self, key: str, default: Any = None) -> Any:
        if key in self.snowflake:
            return self.snowflake[key]
        return self.config.get(key, default)

    def _put(self, event: RelayEvent) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)
        else:
            self._events.put_nowait(event)

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content or "")


class CortexAdapter(BaseAdapter):
    tool_name = "cortex"
    protocol = "http-sse"
    requires_executable = False

    @classmethod
    def build_command(
        cls,
        folder: str,
        model: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
    ) -> list[str]:
        return []

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
        return CortexRuntime(session_id=session_id, model=model, config=config or {})
