from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import yaml

# Adapted from the provided grammar_check.py (trimmed to grammar validation only).

# fenced block: ```lang\n ... \n```
_CODEBLOCK_RE = re.compile(r"```[a-zA-Z0-9_\-]*\s*\n(.*?)\n```", re.DOTALL)

# fence lines only
_CODE_FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z0-9_\-]*\s*$|^\s*```\s*$", re.MULTILINE)

_YAML_COMMENT_RE = re.compile(r"^\s*#")


def norm_format(fmt: str) -> str:
    t = (fmt or "").strip().upper()
    if t in {"YML"}:
        return "YAML"
    if t in {"TSV"}:
        return "CSV"
    if t.startswith("C_"):
        t = t[2:]
    return t


def strip_code_fence_lines(text: str) -> str:
    """Remove only the ``` lines (not the content)."""
    return re.sub(_CODE_FENCE_LINE_RE, "", text or "").strip()


def extract_payload(text: str, fmt: str) -> str:
    """Extract the likely payload for parsing/validation."""
    s = (text or "").strip()
    if not s:
        return ""

    blocks = _CODEBLOCK_RE.findall(s)
    if blocks:
        return blocks[-1].strip()

    m = re.search(r"(?im)^\s*(output|answer|final)\s*:\s*$", s)
    if m:
        return s[m.end() :].strip()

    fmt_n = norm_format(fmt)

    if fmt_n == "JSON":
        idxs = [p for p in (s.find("{"), s.find("[")) if p != -1]
        return s[min(idxs) :].strip() if idxs else s

    if fmt_n == "XML":
        p = s.find("<")
        return s[p:].strip() if p != -1 else s

    return strip_code_fence_lines(s).strip()


def _parse_json_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        return True, json.loads(s), None
    except Exception as e:
        return False, None, str(e)


def _parse_yaml_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        return True, yaml.safe_load(s), None
    except Exception as e:
        return False, None, str(e)


def _parse_toml_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        import tomllib  # py3.11+

        return True, tomllib.loads(s), None
    except Exception as e:
        return False, None, str(e)


def _parse_xml_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        import xml.etree.ElementTree as ET

        return True, ET.fromstring(s), None
    except Exception as e:
        return False, None, str(e)


def _parse_csv_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        f = io.StringIO(s)
        rows = list(csv.reader(f))
        if not rows:
            return False, None, "empty"
        header = rows[0]
        if len(header) < 1:
            return False, None, "no header"
        width = len(header)
        for i, r in enumerate(rows[1:], start=2):
            if len(r) != width:
                return False, None, f"row {i} width mismatch: {len(r)} != {width}"
        return True, rows, None
    except Exception as e:
        return False, None, str(e)


def yaml_indent_is_canonical(text: str, indent: int = 2) -> bool:
    """Simple deterministic YAML style rule (2-space indent, no tabs)."""
    if not text:
        return False

    lines = (text or "").splitlines()

    def ignorable(ln: str) -> bool:
        s = ln.strip()
        return (s == "") or bool(_YAML_COMMENT_RE.match(ln))

    for ln in lines:
        m = re.match(r"^[ \t]*", ln)
        if m and "\t" in m.group(0):
            return False

    for ln in lines:
        if ignorable(ln):
            continue
        n = len(re.match(r"^[ ]*", ln).group(0))
        if n % indent != 0:
            return False

    for i, ln in enumerate(lines):
        if ignorable(ln):
            continue
        cur = len(re.match(r"^[ ]*", ln).group(0))
        stripped = ln.strip()
        is_seq = stripped.startswith("-")
        opens_block = stripped.endswith(":")
        if not (is_seq or opens_block):
            continue

        j = i + 1
        while j < len(lines) and ignorable(lines[j]):
            j += 1
        if j >= len(lines):
            continue
        nxt = len(re.match(r"^[ ]*", lines[j]).group(0))
        if nxt > cur and (nxt - cur) != indent:
            return False

    return True


@dataclass
class GrammarResult:
    ok: bool
    reason: str
    output_format: str


def validate_by_format(text: str, fmt: str, *, yaml_style: bool = True) -> GrammarResult:
    fmt_n = norm_format(fmt)
    payload = extract_payload(text, fmt_n)

    if fmt_n == "JSON":
        ok, _, err = _parse_json_strict(payload)
        return GrammarResult(ok, "ok" if ok else f"parse_fail: {err}", fmt_n)

    if fmt_n == "YAML":
        ok, _, err = _parse_yaml_strict(payload)
        if not ok:
            return GrammarResult(False, f"parse_fail: {err}", fmt_n)
        if yaml_style and (not yaml_indent_is_canonical(payload, indent=2)):
            return GrammarResult(False, "yaml_indent_fail", fmt_n)
        return GrammarResult(True, "ok", fmt_n)

    if fmt_n == "TOML":
        ok, _, err = _parse_toml_strict(payload)
        return GrammarResult(ok, "ok" if ok else f"parse_fail: {err}", fmt_n)

    if fmt_n == "XML":
        ok, _, err = _parse_xml_strict(payload)
        return GrammarResult(ok, "ok" if ok else f"parse_fail: {err}", fmt_n)

    if fmt_n == "CSV":
        ok, _, err = _parse_csv_strict(payload)
        return GrammarResult(ok, "ok" if ok else f"parse_fail: {err}", fmt_n)

    return GrammarResult(False, f"unknown_format: {fmt_n}", fmt_n)
