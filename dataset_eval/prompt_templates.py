from __future__ import annotations

from pathlib import Path
from typing import Dict


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _normalize(fmt: str) -> str:
    s = (fmt or "").strip().lower()
    # unify common variants
    if s in {"csv"}:
        return "csv"
    if s in {"json"}:
        return "json"
    if s in {"yaml", "yml"}:
        return "yaml"
    if s in {"toml"}:
        return "toml"
    if s in {"xml"}:
        return "xml"
    return s


def get_format_checklist(output_format: str) -> str:
    """Return a format-specific checklist to reduce LLM evaluation ambiguity.

    The checklist is intended to complement mechanical parsing checks: parsing already passed,
    but subtle format violations or instruction mismatches may still exist.
    """

    key = _normalize(output_format)
    path = _PROMPT_DIR / f"{key}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip() + "\n"

    # Fallback (unknown formats)
    return (
        "Format checklist (generic):\n"
        "- Output must contain only the target format content (no explanation/Markdown).\n"
        "- Must be syntactically valid and match the task requirements.\n"
    )


def available_checklists() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not _PROMPT_DIR.exists():
        return out
    for p in sorted(_PROMPT_DIR.glob("*.md")):
        out[p.stem] = p.read_text(encoding="utf-8")
    return out
