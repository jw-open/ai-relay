"""Generic adapter — wraps any CLI tool by executable name."""

from __future__ import annotations
from typing import Optional
from .base import BaseAdapter


class GenericAdapter(BaseAdapter):
    tool_name = "generic"

    @classmethod
    def build_command(cls, folder: str, model: Optional[str] = None,
                      extra_args: Optional[list[str]] = None) -> list[str]:
        # Caller should pass the executable as first element of extra_args,
        # or override this adapter for tool-specific behaviour.
        cmd = extra_args[:] if extra_args else ["sh"]
        return cmd
