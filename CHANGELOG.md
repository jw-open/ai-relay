# Changelog

All notable changes to ai-relay are documented here.

---

## [0.4.1] — 2026-05-03

### Added
- **`ai-relay serve` subcommand** — persistent WebSocket server mode designed for containers and daemons. Each incoming connection becomes one independent agent session. Ideal for running ai-relay inside Docker with `CMD ["ai-relay", "serve", "--port", "9000"]`.
- **`PerTurnRuntime`** — new internal runtime that restarts the agent subprocess per user turn and captures `session_id` from the `system/init` event for `--resume` on the next turn. Enables true multi-turn conversations in server mode without keeping a persistent subprocess alive between turns.
- Claude Code and Gemini CLI now use `PerTurnRuntime` automatically in server mode.

### Changed
- Gemini adapter improvements for headless `stream-json` mode.

---

## [0.4.0] — 2026-05-02

### Added
- Initial `PerTurnRuntime` implementation.
- Gemini CLI adapter improvements.

---

## [0.3.0] — 2026-04-15

### Added
- Snowflake Cortex adapter (chat + analyst modes).
- `context_compacted` event type.

---

## [0.2.9] — 2026-04-01

### Added
- `permission_cancelled` and `control_response` event types.
- `tool_progress` event for native tool progress streaming.

---

## [0.2.x]

- OpenAI Codex adapter (`codex app-server` JSON-RPC protocol).
- Gemini CLI adapter (`stream-json` headless mode).
- Structured event parsing: `reasoning`, `tool_call`, `tool_result`, `file_diff`, `response`.
- `quota_warning` and `context_warning` events.
- `interrupt` and `permission_response` client messages.
