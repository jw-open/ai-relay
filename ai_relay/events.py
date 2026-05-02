"""Structured event types emitted by the relay over WebSocket."""

from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    # Lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    # Output streams
    STDOUT = "stdout"
    STDERR = "stderr"
    # Parsed semantic events
    REASONING = "reasoning"        # agent thinking/planning text
    TOOL_CALL = "tool_call"        # agent invoking a tool (Read, Edit, Bash…)
    TOOL_RESULT = "tool_result"    # result of a tool call
    FILE_DIFF = "file_diff"        # file was created/edited
    RESPONSE = "response"          # final answer text
    ASSISTANT_MESSAGE = "assistant_message"
    USER_MESSAGE = "user_message"
    STREAM_EVENT = "stream_event"
    STATUS = "status"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_CANCELLED = "permission_cancelled"
    CONTROL_RESPONSE = "control_response"
    TOOL_PROGRESS = "tool_progress"
    # Status / warnings
    QUOTA_WARNING = "quota_warning"
    CONTEXT_WARNING = "context_warning"   # context window nearing limit
    CONTEXT_COMPACTED = "context_compacted"
    ERROR = "error"
    # Control
    INPUT_ACK = "input_ack"        # relay confirms input was sent to process


@dataclass
class RelayEvent:
    type: EventType
    ts: float = field(default_factory=time.time)
    session_id: str = ""
    # Optional payload fields
    text: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[dict] = None
    file_path: Optional[str] = None
    diff: Optional[str] = None
    context_pct: Optional[int] = None
    exit_code: Optional[int] = None
    metadata: Optional[dict] = None
    request_id: Optional[str] = None
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    raw: Optional[dict] = None
    status: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})

    @classmethod
    def from_raw(cls, session_id: str, stream: str, line: str) -> "RelayEvent":
        """Best-effort parse of a raw stdout/stderr line into a structured event."""
        event_type = EventType.STDOUT if stream == "stdout" else EventType.STDERR
        text = line.rstrip("\n")

        # ── Quota / rate limit ────────────────────────────────────────────────
        if re.search(r"quota|rate.?limit|billing|429|too many requests", text, re.I):
            return cls(type=EventType.QUOTA_WARNING, session_id=session_id, text=text)

        # ── Context window ────────────────────────────────────────────────────
        ctx_match = re.search(r"(\d+)\s*%.*context|context.*?(\d+)\s*%", text, re.I)
        if ctx_match:
            pct = int(ctx_match.group(1) or ctx_match.group(2))
            return cls(type=EventType.CONTEXT_WARNING, session_id=session_id,
                       text=text, context_pct=pct)
        if re.search(r"context.{0,20}(limit|full|compact|window)", text, re.I):
            return cls(type=EventType.CONTEXT_WARNING, session_id=session_id, text=text)

        # ── Compaction ────────────────────────────────────────────────────────
        if re.search(r"compact(ed|ing)?|summariz", text, re.I):
            return cls(type=EventType.CONTEXT_COMPACTED, session_id=session_id, text=text)

        # ── Tool calls (Claude Code style) ────────────────────────────────────
        tool_match = re.match(r"^[>│]\s*(Read|Edit|Write|Bash|Glob|Grep|WebFetch|WebSearch)\s*[(\[]?(.*)$", text)
        if tool_match:
            return cls(type=EventType.TOOL_CALL, session_id=session_id,
                       tool=tool_match.group(1), text=tool_match.group(2).strip())

        # ── File diffs ────────────────────────────────────────────────────────
        if re.match(r"^[+\-]{3}\s", text) or re.search(r"(created|updated|modified|wrote)\s+.+\.(py|ts|tsx|js|go|rs|java)", text, re.I):
            return cls(type=EventType.FILE_DIFF, session_id=session_id, text=text)

        # ── Reasoning / thinking ──────────────────────────────────────────────
        if re.match(r"^[│┃]\s*", text) or re.search(r"(thinking|planning|analyzing|let me|i will|i'll)", text, re.I):
            return cls(type=EventType.REASONING, session_id=session_id, text=text)

        # ── Default: raw stream ───────────────────────────────────────────────
        return cls(type=event_type, session_id=session_id, text=text)
