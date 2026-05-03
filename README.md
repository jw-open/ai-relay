# ai-relay

> WebSocket relay that bridges AI coding agent CLIs (Claude Code, Codex, Gemini CLI, Snowflake Cortex, and more) to any web interface — stream reasoning, tool calls, and file changes in real time.

## Install

```bash
pip install ai-relay
```

## Quick start

### One-shot mode (local dev)

```bash
# Start the relay server (default: ws://0.0.0.0:8765)
ai-relay --port 8765
```

### Server mode (container / daemon)

```bash
# Persistent server — each WebSocket connection becomes one independent agent session
ai-relay serve --port 9000
```

Then connect from OhWise Lab (or any WebSocket client) and send a handshake:

```json
{"tool": "claude", "folder": "/path/to/project", "model": "claude-sonnet-4-6"}
```

The relay streams structured JSON events over WebSocket and forwards your messages to the selected backend. Claude Code and Gemini use native JSONL process protocols, Codex uses the app-server JSON-RPC protocol, and Snowflake Cortex uses HTTP/SSE. The PTY bridge is retained only for generic/legacy CLI tools.

## Running in a container

`ai-relay serve` is designed to run inside a Docker container as a persistent daemon:

```dockerfile
FROM python:3.11-slim
RUN pip install ai-relay
# Install your AI CLI here (e.g. npm install -g @anthropic-ai/claude-code)
CMD ["ai-relay", "serve", "--port", "9000"]
```

Each incoming WebSocket connection spawns an independent agent session. Multiple clients can connect simultaneously.

## Event types

| Type | Description |
|------|-------------|
| `session_start` | Process spawned |
| `session_end` | Process exited (includes exit_code) |
| `stdout` / `stderr` | Raw output lines |
| `reasoning` | Agent thinking/planning text |
| `tool_call` | Agent invoking a tool (Read, Edit, Bash…) |
| `tool_result` | Result of a tool call |
| `file_diff` | File created or edited |
| `response` | Final answer text |
| `assistant_message` | Native structured assistant message |
| `user_message` | Native structured user/tool-result message |
| `stream_event` | Native streaming event |
| `status` | Native status/control event |
| `permission_request` | Tool permission prompt from a structured backend |
| `permission_cancelled` | Pending permission prompt was cancelled |
| `control_response` | Native control response acknowledgment |
| `tool_progress` | Native tool progress event |
| `quota_warning` | API quota / rate limit detected |
| `context_warning` | Context window nearing limit (includes context_pct) |
| `context_compacted` | Context was compacted |
| `error` | Relay or process error |
| `input_ack` | Relay confirms your message was sent to the process |

## Sending commands

Send JSON over WebSocket:

```json
{"text": "refactor the authentication module to use JWT"}
```

Claude Code also accepts structured web-client messages:

```json
{"type": "user_message", "content": "refactor the authentication module to use JWT"}
```

Permission responses:

```json
{"type": "permission_response", "request_id": "req", "behavior": "allow", "updatedInput": {"command": "git status"}}
```

Codex permission responses can also use:

```json
{"type": "permission_response", "request_id": "req", "allow": true}
```

Interrupt the active structured turn:

```json
{"type": "interrupt"}
```

Codex uses `codex app-server --listen stdio://` and keeps a persistent thread behind the WebSocket session:

```json
{"tool": "codex", "folder": "/path/to/project", "model": "gpt-5.2"}
```

Gemini CLI uses headless `stream-json` mode. Each text message starts one Gemini turn:

```json
{"tool": "gemini", "folder": "/path/to/project", "model": "gemini-2.5-flash"}
```

Snowflake Cortex uses API configuration in the handshake.

Cortex chat mode:

```json
{
  "tool": "cortex",
  "mode": "chat",
  "model": "claude-sonnet-4-5",
  "snowflake": {
    "account_url": "https://<account>.snowflakecomputing.com",
    "token_env": "SNOWFLAKE_PAT"
  }
}
```

Cortex Analyst mode:

```json
{
  "tool": "cortex",
  "mode": "analyst",
  "snowflake": {
    "account_url": "https://<account>.snowflakecomputing.com",
    "token_env": "SNOWFLAKE_PAT",
    "semantic_view": "DB.SCHEMA.VIEW"
  }
}
```

To send CLI commands (e.g. `/compact`, `/clear`):

```json
{"text": "/compact"}
```

## Supported tools

| Tool | Adapter | `tool` value |
|------|---------|-------------|
| Claude Code | `ClaudeCodeAdapter` | `"claude"` / `"claude-code"` |
| OpenAI Codex | `CodexAdapter` | `"codex"` |
| Gemini CLI | `GeminiAdapter` | `"gemini"` |
| Snowflake Cortex | `CortexAdapter` | `"cortex"` |
| Any CLI | `GenericAdapter` | `"generic"` |

## Python API

```python
from ai_relay import RelayServer

server = RelayServer(host="0.0.0.0", port=8765)
server.run()
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT
