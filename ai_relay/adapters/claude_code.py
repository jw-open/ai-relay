"""Claude Code adapter."""

from __future__ import annotations
from typing import Optional
from .base import BaseAdapter


class ClaudeCodeAdapter(BaseAdapter):
    tool_name = "claude-code"

    @classmethod
    def build_command(cls, folder: str, model: Optional[str] = None,
                      extra_args: Optional[list[str]] = None) -> list[str]:
        cmd = ["claude"]
        if model:
            cmd += ["--model", model]
        # Disable interactive TTY features for relay mode
        cmd += ["--no-pty"] if cls._supports_no_pty() else []
        if extra_args:
            cmd += extra_args
        return cmd

    @classmethod
    def _supports_no_pty(cls) -> bool:
        """Check if the installed claude CLI supports --no-pty."""
        import shutil, subprocess
        if not shutil.which("claude"):
            return False
        try:
            result = subprocess.run(
                ["claude", "--help"], capture_output=True, text=True, timeout=5
            )
            return "--no-pty" in result.stdout or "--no-pty" in result.stderr
        except Exception:
            return False
