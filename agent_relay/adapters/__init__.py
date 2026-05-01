from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .generic import GenericAdapter

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "claude": ClaudeCodeAdapter,
    "claude-code": ClaudeCodeAdapter,
    "codex": GenericAdapter,
    "gemini": GenericAdapter,
    "cortex": GenericAdapter,
    "generic": GenericAdapter,
}

def get_adapter(tool: str) -> type[BaseAdapter]:
    return ADAPTERS.get(tool.lower(), GenericAdapter)
