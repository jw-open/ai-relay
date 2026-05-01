# ai-relay

> WebSocket relay that bridges AI coding agent CLIs (Claude Code, Codex, Gemini CLI, Snowflake Cortex, and more) to any web interface — stream reasoning, tool calls, and file changes in real time.

## Install

```bash
pip install ai-relay
```

## Quick start

```bash
# Start the relay server (default: ws://0.0.0.0:8765)
ai-relay --port 8765
```

Then connect from OhWise Lab (or any WebSocket client) and send a handshake:

```json
{"tool": "claude", "folder": "/path/to/project", "model": "claude-sonnet-4-6"}
```

The relay spawns the CLI process, streams all output as structured JSON events over WebSocket, and forwards your messages as stdin to the process.

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

To send CLI commands (e.g. `/compact`, `/clear`):

```json
{"text": "/compact"}
```

## Supported tools

| Tool | Adapter | `tool` value |
|------|---------|-------------|
| Claude Code | `ClaudeCodeAdapter` | `"claude"` / `"claude-code"` |
| OpenAI Codex | `GenericAdapter` | `"codex"` |
| Gemini CLI | `GenericAdapter` | `"gemini"` |
| Snowflake Cortex | `GenericAdapter` | `"cortex"` |
| Any CLI | `GenericAdapter` | `"generic"` |

## Python API

```python
from ai_relay import RelayServer

server = RelayServer(host="0.0.0.0", port=8765)
server.run()
```

## License

MIT
