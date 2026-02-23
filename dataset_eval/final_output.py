from __future__ import annotations

import re
from typing import Optional


# Match a standalone "Output:" label line (case-insensitive)
_OUTPUT_RE = re.compile(r"(?im)^\s*Output\s*:\s*$")

# Match fenced code blocks, capturing the body.
_CODEBLOCK_RE = re.compile(r"```[a-zA-Z0-9_\-]*\s*\n(.*?)\n```", re.DOTALL)


def _normalize_output_type(output_type: Optional[str]) -> str:
    t = (output_type or "").strip().upper()
    if not t:
        return ""
    if t in {"YML"}:
        return "YAML"
    if t in {"TSV"}:
        return "CSV"
    if t.startswith("C_"):
        t = t[2:]
    return t


def _strip_markdown_fences_anywhere(text: str) -> str:
    """If any fenced blocks exist, take the *last* block body; else return original."""
    blocks = _CODEBLOCK_RE.findall(text or "")
    if blocks:
        return (blocks[-1] or "").strip()
    return (text or "").strip()


def _extract_json_substring(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    idxs = [p for p in (s.find("{"), s.find("[")) if p != -1]
    if not idxs:
        return ""
    return s[min(idxs):].strip()


def _extract_xml_substring(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    p = s.find("<")
    return s[p:].strip() if p != -1 else ""


def _strip_leading_explanation_lines(text: str, *, kind: str) -> str:
    """For YAML/TOML/CSV, drop leading non-structured lines.

    This mirrors the training-side heuristic:
    - YAML: keep from first line that looks like YAML mapping/sequence
    - TOML: keep from first line that looks like TOML table/key
    - CSV: keep from first line that contains a delimiter (, or tab)
    """
    lines = (text or "").splitlines()
    if not lines:
        return ""

    def is_yaml_like(s: str) -> bool:
        return s.startswith("-") or ":" in s

    def is_toml_like(s: str) -> bool:
        return (s.startswith("[") and s.endswith("]")) or ("=" in s and not s.startswith("#"))

    def is_csv_like(s: str) -> bool:
        return "," in s or "\t" in s

    keep_from = 0
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if kind == "YAML" and is_yaml_like(s):
            keep_from = idx
            break
        if kind == "TOML" and is_toml_like(s):
            keep_from = idx
            break
        if kind == "CSV" and is_csv_like(s):
            keep_from = idx
            break

    return "\n".join(lines[keep_from:]).strip()


def extract_final_output(text: str, output_type: Optional[str] = None) -> str:
    """Extract the final *structured* output from an assistant message.

    Order (matching the training-side logic):
    1) If there is any 'Output:' marker, take substring after the last marker.
    2) If code fences exist anywhere, take the last fenced block body.
    3) If output_type is known:
       - JSON/XML: take first blob-like substring.
       - YAML/TOML/CSV: drop leading explanation-like lines.
    4) Fallback: stripped text.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""

    matches = list(_OUTPUT_RE.finditer(t))
    if matches:
        t = t[matches[-1].end():].strip()

    t = _strip_markdown_fences_anywhere(t)

    ot = _normalize_output_type(output_type)
    if ot == "JSON":
        j = _extract_json_substring(t)
        return j if j else t.strip()
    if ot == "XML":
        x = _extract_xml_substring(t)
        return x if x else t.strip()
    if ot in {"YAML", "TOML", "CSV"}:
        return _strip_leading_explanation_lines(t, kind=ot)

    return t.strip()
