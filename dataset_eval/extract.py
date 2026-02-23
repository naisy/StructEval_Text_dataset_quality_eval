from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedPair:
    family: str
    prompt: str
    output: str
    output_format: str


def detect_family(ex: Dict[str, Any]) -> str:
    md = ex.get("metadata")
    if isinstance(md, dict) and ("prompt" in md) and ("output" in md):
        return "u10bei"
    if isinstance(ex.get("messages"), list):
        # daichira and others
        return "daichira"
    return "unknown"


def _last_role_content(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    out = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == role:
            out = str(m.get("content") or "")
    return out


def extract_pair(ex: Dict[str, Any]) -> ExtractedPair:
    family = detect_family(ex)
    output_format = str(ex.get("output_format") or "")

    if family == "u10bei":
        md = ex.get("metadata") or {}
        return ExtractedPair(
            family="u10bei",
            prompt=str(md.get("prompt") or ""),
            output=str(md.get("output") or ""),
            output_format=output_format,
        )

    if family == "daichira":
        msgs = ex.get("messages")
        prompt = _last_role_content(msgs, "user")
        output = _last_role_content(msgs, "assistant")
        return ExtractedPair(
            family="daichira",
            prompt=prompt,
            output=output,
            output_format=output_format,
        )

    return ExtractedPair(family="unknown", prompt="", output="", output_format=output_format)
