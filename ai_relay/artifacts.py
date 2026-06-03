"""
Artifact extraction from agent response text.

Agents (Claude Code, etc.) embed structured outputs using:

    [ARTIFACT type="html" title="My Dashboard"]
    <html>...</html>
    [/ARTIFACT]

Supported types: html, code, markdown, data, url, text.
The lang attribute is optional for code blocks:

    [ARTIFACT type="code" lang="python" title="script.py"]
    ...
    [/ARTIFACT]
"""
from __future__ import annotations
import re
from typing import NamedTuple


_ARTIFACT_RE = re.compile(
    r'\[ARTIFACT([^\]]*)\](.*?)\[/ARTIFACT\]',
    re.DOTALL | re.IGNORECASE,
)

_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

VALID_TYPES = frozenset({"html", "code", "markdown", "data", "url", "text", "image"})


class ParsedArtifact(NamedTuple):
    artifact_type: str   # html | code | markdown | data | url | text
    title: str
    content: str
    lang: str            # only meaningful for type=code


def extract_artifacts(text: str) -> tuple[str, list[ParsedArtifact]]:
    """
    Strip [ARTIFACT] blocks from *text* and return (cleaned_text, artifacts).

    The cleaned text has the artifact blocks removed and surrounding whitespace
    collapsed so the remaining response reads naturally.
    """
    artifacts: list[ParsedArtifact] = []

    def _replace(m: re.Match) -> str:
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        raw_type = attrs.get("type", "text").lower()
        artifact_type = raw_type if raw_type in VALID_TYPES else "text"
        artifacts.append(ParsedArtifact(
            artifact_type=artifact_type,
            title=attrs.get("title", "Artifact"),
            content=m.group(2).strip(),
            lang=attrs.get("lang", ""),
        ))
        return ""

    cleaned = _ARTIFACT_RE.sub(_replace, text)
    # Collapse multiple blank lines left after removal
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, artifacts
