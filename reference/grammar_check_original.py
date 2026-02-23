#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

import yaml
from datasets import load_dataset
from datasets import load_from_disk

import zipfile
import tempfile
import shutil

# ----------------------------
# Format helpers / parsers
# ----------------------------

def norm_format(fmt: str) -> str:
    t = (fmt or "").strip().upper()
    if t in {"YML"}:
        return "YAML"
    if t in {"TSV"}:
        return "CSV"
    # daichira category like C_TOML
    if t.startswith("C_"):
        t = t[2:]
    return t


# ----------------------------
# Dataset loader (HF or local zip)
# ----------------------------

def load_dataset_any(name_or_path: str, *, split: str):
    """Load dataset either from HuggingFace hub name or a local zip (datasets 'load_from_disk' dump)."""
    p = Path(name_or_path)
    if p.exists() and p.suffix.lower() == ".zip":
        tmp_root = Path(tempfile.mkdtemp(prefix="hfds_zip_"))
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp_root)

        candidates = []
        for sub in tmp_root.rglob(f"*-{split}.arrow"):
            candidates.append(sub.parent)
        if not candidates:
            raise FileNotFoundError(f"Could not find '*-{split}.arrow' inside {p}")

        data_dir = candidates[0]
        ds_obj = load_from_disk(str(data_dir))
        if hasattr(ds_obj, "__getitem__") and hasattr(ds_obj, "keys") and split in ds_obj.keys():
            ds = ds_obj[split]
        else:
            ds = ds_obj
        ds._tmp_extracted_dir = tmp_root  # type: ignore[attr-defined]
        return ds

    return load_dataset(name_or_path, split=split)


# fenced block: ```lang\n ... \n```
_CODEBLOCK_RE = re.compile(r"```[a-zA-Z0-9_\-]*\s*\n(.*?)\n```", re.DOTALL)

# fence lines only (legacy; keep for safety)
_CODE_FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z0-9_\-]*\s*$|^\s*```\s*$", re.MULTILINE)


def strip_code_fence_lines(text: str) -> str:
    """Remove only the ``` lines (not the content)."""
    return re.sub(_CODE_FENCE_LINE_RE, "", text or "").strip()


def extract_payload(text: str, fmt: str) -> str:
    """
    Extract the likely payload for parsing/validation.

    Priority:
    1) If fenced code blocks exist, take the LAST block content (common in SFT data).
    2) If marker line exists (Output:/Answer:/Final:), take text after it.
    3) Format-specific fallback:
       - JSON: from first '{' or '['
       - XML: from first '<'
    4) Otherwise, return stripped original text.
    """
    s = (text or "").strip()
    if not s:
        return ""

    # 1) last fenced block
    blocks = _CODEBLOCK_RE.findall(s)
    if blocks:
        return blocks[-1].strip()

    # 2) marker lines
    m = re.search(r"(?im)^\s*(output|answer|final)\s*:\s*$", s)
    if m:
        return s[m.end():].strip()

    # 3) fallback by format
    fmt = norm_format(fmt)

    if fmt == "JSON":
        # find first object/array start
        idxs = [p for p in (s.find("{"), s.find("[")) if p != -1]
        if idxs:
            return s[min(idxs):].strip()
        return s

    if fmt == "XML":
        p = s.find("<")
        if p != -1:
            return s[p:].strip()
        return s

    # For YAML/TOML/CSV: removing fence lines (if any) is still helpful
    return strip_code_fence_lines(s).strip()


def parse_json_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        obj = json.loads(s)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def parse_yaml_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        obj = yaml.safe_load(s)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def parse_toml_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        # Python 3.11+: tomllib (Python 3.12 OK)
        import tomllib  # type: ignore
        # tomllib.loads expects str
        obj = tomllib.loads(s)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def parse_xml_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(s)
        return True, root, None
    except Exception as e:
        return False, None, str(e)


def parse_csv_strict(s: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    CSV validity check (strict-ish):
    - must have at least one row
    - header width >= 1
    - all rows must match header width

    NOTE:
    Many LLM outputs include preface lines; extract_payload() should strip those first.
    """
    try:
        f = io.StringIO(s)
        reader = csv.reader(f)
        rows = list(reader)
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




# ----------------------------
# Escape analysis helpers
# ----------------------------

# These functions DO NOT decide validity; they quantify which escaping patterns appear in the gold outputs.
# This helps answer: "does the dataset contain enough examples where the model must learn escaping?"

_XML_ENTITY_PATTERNS = {
    "xml_amp": re.compile(r"&amp;"),
    "xml_lt": re.compile(r"&lt;"),
    "xml_gt": re.compile(r"&gt;"),
    "xml_quot": re.compile(r"&quot;"),
    "xml_apos": re.compile(r"&apos;"),
    "xml_num_dec": re.compile(r"&#\d+;"),
    "xml_num_hex": re.compile(r"&#x[0-9A-Fa-f]+;"),
    "xml_cdata": re.compile(r"<!\[CDATA\["),
}

# unescaped '&' in XML text/attr that is not a valid entity (very common failure)
_XML_BARE_AMP_RE = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)")


_JSON_ESCAPES = {
    "json_escaped_quote": re.compile(r'\\"'),
    "json_escaped_backslash": re.compile(r"\\\\"),
    "json_escaped_slash": re.compile(r"\\\/"),
    "json_escape_b": re.compile(r"\\b"),
    "json_escape_f": re.compile(r"\\f"),
    "json_escape_n": re.compile(r"\\n"),
    "json_escape_r": re.compile(r"\\r"),
    "json_escape_t": re.compile(r"\\t"),
    "json_escape_u": re.compile(r"\\u[0-9A-Fa-f]{4}"),
}

# control chars (0x00-0x1F) are not allowed raw in JSON strings; if present, dataset output is suspicious
_JSON_RAW_CONTROL_RE = re.compile(r"[\x00-\x1F]")


_TOML_ESCAPES = {
    "toml_escaped_quote": re.compile(r'\\"'),
    "toml_escaped_backslash": re.compile(r"\\\\"),
    "toml_escape_b": re.compile(r"\\b"),
    "toml_escape_t": re.compile(r"\\t"),
    "toml_escape_n": re.compile(r"\\n"),
    "toml_escape_f": re.compile(r"\\f"),
    "toml_escape_r": re.compile(r"\\r"),
    "toml_escape_u": re.compile(r"\\u[0-9A-Fa-f]{4}"),
    "toml_escape_U": re.compile(r"\\U[0-9A-Fa-f]{8}"),
}

# ----------------------------
# TOML structure pattern checks (non-escape)
# ----------------------------

def count_toml_inline_tables(payload: str) -> Dict[str, int]:
    """
    Heuristically count TOML inline tables: `{ key = value, ... }` that appear in the *output* payload.

    We try to ignore braces that occur inside TOML strings.
    This is not a full TOML parser; it's meant for dataset-wide counting.

    Returns counts for:
      - toml_inline_table: number of inline-table blocks
      - toml_inline_table_after_eq: inline tables used as value of a key (`k = { ... }`)
      - toml_inline_table_in_array: inline tables used as array items (`k = [ { ... }, ... ]`)
    """
    if not payload:
        return {"toml_inline_table": 0, "toml_inline_table_after_eq": 0, "toml_inline_table_in_array": 0}

    def prev_nonspace(i: int) -> str:
        j = i
        while j >= 0 and payload[j].isspace():
            j -= 1
        return payload[j] if j >= 0 else ""

    # Track simple string states: basic "..." and literal '...'
    in_basic = False
    in_lit = False
    escape = False

    stack: List[int] = []
    total = 0
    after_eq = 0
    in_array = 0

    i = 0
    while i < len(payload):
        ch = payload[i]

        if in_basic:
            if escape:
                escape = False
            elif ch == "\\":  # escape next char
                escape = True
            elif ch == '"':
                in_basic = False
            i += 1
            continue

        if in_lit:
            if ch == "'":
                in_lit = False
            i += 1
            continue

        # entering strings
        if ch == '"':
            in_basic = True
            i += 1
            continue
        if ch == "'":
            in_lit = True
            i += 1
            continue

        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            block = payload[start + 1:i]

            # inline-table blocks should contain '=' somewhere inside (outside strings, but we already skipped strings)
            if "=" in block:
                total += 1
                p = prev_nonspace(start - 1)
                if p == "=":
                    after_eq += 1
                elif p in {"[", ","}:
                    in_array += 1

        i += 1

    return {
        "toml_inline_table": total,
        "toml_inline_table_after_eq": after_eq,
        "toml_inline_table_in_array": in_array,
    }

# YAML escaping depends on scalar style; we count common explicit escaping patterns that appear in gold outputs.
_YAML_ESCAPES = {
    "yaml_escaped_quote": re.compile(r'\\"'),
    "yaml_escaped_backslash": re.compile(r"\\\\"),
    "yaml_escape_n": re.compile(r"\\n"),
    "yaml_escape_t": re.compile(r"\\t"),
    "yaml_escape_r": re.compile(r"\\r"),
    "yaml_escape_u": re.compile(r"\\u[0-9A-Fa-f]{4}"),
    "yaml_escape_U": re.compile(r"\\U[0-9A-Fa-f]{8}"),
    "yaml_escape_x": re.compile(r"\\x[0-9A-Fa-f]{2}"),
    # single-quoted YAML escapes a single quote by doubling: ''
    "yaml_single_quote_doubled": re.compile(r"''"),
}

# CSV: inside quoted fields, a literal double quote becomes "" (RFC4180-style)
_CSV_ESCAPES = {
    "csv_quote_doubled": re.compile(r'""'),
    # newlines inside a quoted field (common edge case)
    "csv_quoted_newline": re.compile(r'"[^"]*\n[^"]*"'),
}



# Canonical list of escape-types we *check* per format.
# We always emit rows for these in the summary, even if counts are 0,
# so the output makes it explicit: "checked, but 0 examples".
_ESCAPE_TYPES_BY_FORMAT: Dict[str, List[str]] = {
    "XML": sorted(list(_XML_ENTITY_PATTERNS.keys()) + ["xml_bare_amp"]),
    "JSON": sorted(list(_JSON_ESCAPES.keys()) + ["json_raw_control_char"]),
    "TOML": sorted(list(_TOML_ESCAPES.keys()) + ["toml_inline_table", "toml_inline_table_after_eq", "toml_inline_table_in_array"]),
    "YAML": sorted(list(_YAML_ESCAPES.keys())),
    "CSV": sorted(list(_CSV_ESCAPES.keys())),
}

def analyze_escapes_by_format(text: str, fmt: str) -> Dict[str, int]:
    """
    Return a dict {escape_type: occurrences_in_this_example} based on the *output* payload.

    We intentionally analyze the raw payload string (not parsed object),
    because "learning escaping" is about surface forms like &amp; / \\n / "" etc.
    """
    fmt = norm_format(fmt)
    payload = extract_payload(text, fmt)
    if not payload:
        return {}

    out: Dict[str, int] = {}

    if fmt == "XML":
        for k, rx in _XML_ENTITY_PATTERNS.items():
            c = len(rx.findall(payload))
            if c:
                out[k] = c
        # Count suspicious bare ampersands (likely invalid XML); keep as a separate counter
        bare = len(_XML_BARE_AMP_RE.findall(payload))
        if bare:
            out["xml_bare_amp"] = bare
        return out

    if fmt == "JSON":
        for k, rx in _JSON_ESCAPES.items():
            c = len(rx.findall(payload))
            if c:
                out[k] = c
        ctrl = len(_JSON_RAW_CONTROL_RE.findall(payload))
        if ctrl:
            out["json_raw_control_char"] = ctrl
        return out

    if fmt == "TOML":
        for k, rx in _TOML_ESCAPES.items():
            c = len(rx.findall(payload))
            if c:
                out[k] = c
        # Inline table usage (structure) in TOML outputs
        it = count_toml_inline_tables(payload)
        for k, c in it.items():
            if c:
                out[k] = int(c)
        return out

    if fmt == "YAML":
        for k, rx in _YAML_ESCAPES.items():
            c = len(rx.findall(payload))
            if c:
                out[k] = c
        return out

    if fmt == "CSV":
        for k, rx in _CSV_ESCAPES.items():
            c = len(rx.findall(payload))
            if c:
                out[k] = c
        return out

    return out


def merge_escape_stats(
    agg: Dict[str, Dict[str, Any]],
    *,
    dataset: str,
    fmt: str,
    escapes: Dict[str, int],
    usable: bool,
) -> None:
    """
    Aggregate escape stats per (dataset, format).
    agg key: f"{dataset}::{fmt}"
    value:
      {
        "dataset": str,
        "format": str,
        "examples_total": int,
        "examples_usable": int,
        "occ_total": {escape_type: int},
        "occ_usable": {escape_type: int},
        "ex_with_total": {escape_type: int},   # number of examples where this escape appears at least once
        "ex_with_usable": {escape_type: int},
      }
    """
    fmt = norm_format(fmt)
    key = f"{dataset}::{fmt}"
    if key not in agg:
        agg[key] = {
            "dataset": dataset,
            "format": fmt,
            "examples_total": 0,
            "examples_usable": 0,
            "occ_total": defaultdict(int),
            "occ_usable": defaultdict(int),
            "ex_with_total": defaultdict(int),
            "ex_with_usable": defaultdict(int),
        }

    rec = agg[key]
    rec["examples_total"] += 1
    if usable:
        rec["examples_usable"] += 1

    for et, cnt in (escapes or {}).items():
        if cnt <= 0:
            continue
        rec["occ_total"][et] += int(cnt)
        rec["ex_with_total"][et] += 1
        if usable:
            rec["occ_usable"][et] += int(cnt)
            rec["ex_with_usable"][et] += 1

_YAML_COMMENT_RE = re.compile(r"^\s*#")


def yaml_indent_is_canonical(text: str, indent: int = 2) -> bool:
    """
    2-space indent convention check (deterministic, simple).
    - tabs forbidden
    - indent is multiple of `indent`
    - if next non-empty line is deeper than current for block openers, it must be exactly +indent
    """
    if not text:
        return False

    lines = (text or "").splitlines()

    def ignorable(ln: str) -> bool:
        s = ln.strip()
        return (s == "") or bool(_YAML_COMMENT_RE.match(ln))

    # tabs
    for ln in lines:
        m = re.match(r"^[ \t]*", ln)
        if m and "\t" in m.group(0):
            return False

    # indent multiple check
    for ln in lines:
        if ignorable(ln):
            continue
        n = len(re.match(r"^[ ]*", ln).group(0))
        if n % indent != 0:
            return False

    # block-opening strictness
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



# ----------------------------
# TOML deep-structure pattern checks (array-of-tables / nesting)
# ----------------------------

_TOML_TABLE_HDR_RE = re.compile(r"^\s*\[(\[)?\s*([^\]]+?)\s*\]\]?\s*$")  # [x] or [[x]]

def analyze_toml_structure(payload: str) -> Dict[str, Any]:
    """Heuristically analyze TOML output structure."""
    if not payload:
        return {
            "array_table_headers": 0,
            "array_table_headers_with_dot": 0,
            "table_headers": 0,
            "max_header_depth": 0,
            "child_tables_under_array": 0,
            "json_stringified_like": 0,
        }

    array_hdr = 0
    array_hdr_dot = 0
    tbl_hdr = 0
    max_depth = 0
    child_under_array = 0

    last_array_path: Optional[str] = None

    for line in payload.splitlines():
        m = _TOML_TABLE_HDR_RE.match(line)
        if not m:
            continue
        is_array = bool(m.group(1))
        path = re.sub(r"\s+", "", (m.group(2) or "").strip())
        depth = len([p for p in path.split(".") if p])
        max_depth = max(max_depth, depth)

        if is_array:
            array_hdr += 1
            if "." in path:
                array_hdr_dot += 1
            last_array_path = path
        else:
            tbl_hdr += 1
            if last_array_path and path.startswith(last_array_path + "."):
                child_under_array += 1

    # Detect "JSON stringified inside TOML string" patterns.
    # Detect JSON-like content embedded as a TOML string value.
    # This catches cases where complex nested structures are stored as a JSON string instead of TOML tables/arrays.
    json_in_string = False
    if ("{" in payload) or ("[" in payload):
        candidates = []
        # Multiline basic strings: """ ... """
        candidates += re.findall(r'"""(.*?)"""', payload, flags=re.DOTALL)
        # Basic strings: " ... " (supports common escapes)
        candidates += [m.group(1) for m in re.finditer(r'"((?:\\.|[^"\\])*)"', payload)]
        for inner in candidates:
            # Best-effort unescape for JSON parsing.
            inner_unesc = inner.replace('\\\"', '"').replace('\\\\', '\\')
            s = inner_unesc.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    json.loads(s)
                    json_in_string = True
                    break
                except Exception:
                    pass

    json_stringified = 1 if json_in_string else 0

    return {
        "array_table_headers": array_hdr,
        "array_table_headers_with_dot": array_hdr_dot,
        "table_headers": tbl_hdr,
        "max_header_depth": max_depth,
        "child_tables_under_array": child_under_array,
        "json_stringified_like": json_stringified,
    }

def validate_by_format(text: str, fmt: str, *, yaml_style: bool = True) -> Tuple[bool, str]:
    """
    Returns: (is_valid, reason)
      - valid  : (True, "ok")
      - invalid: (False, reason)
    """
    fmt = norm_format(fmt)
    payload = extract_payload(text, fmt)

    if fmt == "JSON":
        ok, _, err = parse_json_strict(payload)
        return (ok, "ok" if ok else f"parse_fail: {err}")

    if fmt == "YAML":
        ok, _, err = parse_yaml_strict(payload)
        if not ok:
            return False, f"parse_fail: {err}"
        if yaml_style and (not yaml_indent_is_canonical(payload, indent=2)):
            return False, "yaml_indent_fail"
        return True, "ok"

    if fmt == "TOML":
        ok, _, err = parse_toml_strict(payload)
        return (ok, "ok" if ok else f"parse_fail: {err}")

    if fmt == "XML":
        ok, _, err = parse_xml_strict(payload)
        return (ok, "ok" if ok else f"parse_fail: {err}")

    if fmt == "CSV":
        ok, _, err = parse_csv_strict(payload)
        return (ok, "ok" if ok else f"parse_fail: {err}")

    return False, f"unknown_format: {fmt}"


# ----------------------------
# Dataset family + task key
# ----------------------------

@dataclass
class RecordInfo:
    dataset: str
    split: str
    idx: str
    family: str
    output_format: str
    task_kind: str
    task_schema: str
    task_key: str
    assistant_text: str
    raw: Dict[str, Any]


def detect_family(example: Dict[str, Any]) -> str:
    if "metadata" in example and "messages" in example:
        return "u10bei"
    if "category" in example and "subcategory" in example and "messages" in example:
        return "daichira"
    return "unknown"


def extract_assistant_text(messages: Any) -> str:
    """
    messages: List[{role, content}] assumed.
    Return the last assistant content.
    """
    if not isinstance(messages, list):
        return ""
    out = ""
    for m in messages:
        if isinstance(m, dict) and (m.get("role") == "assistant"):
            out = str(m.get("content") or "")
    return out



def infer_output_format_daichira(ex: Dict[str, Any]) -> str:
    """
    daichira datasets often include:
      - category: e.g. C_TOML / C_XML ...
      - subcategory: e.g. text_to_toml, xml_to_yaml, csv_to_xml

    In many tasks, the *output* format is encoded in subcategory suffix `_to_<fmt>`.
    Some transforms have category=C_XML but subcategory=xml_to_yaml (output YAML).
    """
    subcat = str(ex.get("subcategory") or "").strip()
    m = re.search(r"_to_([A-Za-z0-9]+)\s*$", subcat)
    if m:
        return norm_format(m.group(1))
    return norm_format(str(ex.get("category") or ""))


def iter_record_infos(dataset_name: str, split: str, idx: int, ex: Dict[str, Any]) -> List[RecordInfo]:
    """
    Yield RecordInfo entries.

    NOTE: Some u10bei rows are *bundled*: `messages` is a list of conversations and
    `metadata` is a list aligned to those conversations. We must split these rows,
    otherwise task classification & validation will be wrong.
    """
    family = detect_family(ex)

    # ---- u10bei ----
    if family == "u10bei":
        md = ex.get("metadata")
        msgs = ex.get("messages")

        # bundled form: metadata=list[dict], messages=list[list[dict]]
        if isinstance(md, list) and isinstance(msgs, list) and (len(msgs) > 0) and isinstance(msgs[0], list):
            out: List[RecordInfo] = []
            n = min(len(md), len(msgs))
            for j in range(n):
                md_j = md[j] or {}
                msgs_j = msgs[j] or []
                output_format = norm_format(str((md_j or {}).get("format") or ""))
                task_kind = str((md_j or {}).get("type") or "").strip() or "unknown"
                task_schema = str((md_j or {}).get("schema") or "").strip() or "unknown"
                task_key = f"u10bei|{output_format}|{task_kind}|{task_schema}"
                assistant_text = extract_assistant_text(msgs_j)

                out.append(RecordInfo(
                    dataset=dataset_name,
                    split=split,
                    idx=f"{idx}:{j}",
                    family=family,
                    output_format=output_format,
                    task_kind=task_kind,
                    task_schema=task_schema,
                    task_key=task_key,
                    assistant_text=assistant_text,
                    raw={**ex, "messages": msgs_j, "metadata": md_j},
                ))
            return out

        # normal form: metadata=dict, messages=list[dict]
        md_d = md if isinstance(md, dict) else (md[0] if isinstance(md, list) and md else {})
        msgs_d = msgs if isinstance(msgs, list) and (not msgs or isinstance(msgs[0], dict)) else (msgs[0] if isinstance(msgs, list) and msgs and isinstance(msgs[0], list) else [])

        output_format = norm_format(str((md_d or {}).get("format") or ""))
        task_kind = str((md_d or {}).get("type") or "").strip() or "unknown"
        task_schema = str((md_d or {}).get("schema") or "").strip() or "unknown"
        task_key = f"u10bei|{output_format}|{task_kind}|{task_schema}"
        assistant_text = extract_assistant_text(msgs_d)

        return [RecordInfo(
            dataset=dataset_name,
            split=split,
            idx=str(idx),
            family=family,
            output_format=output_format,
            task_kind=task_kind,
            task_schema=task_schema,
            task_key=task_key,
            assistant_text=assistant_text,
            raw={**ex, "messages": msgs_d, "metadata": md_d},
        )]

    # ---- daichira ----
    if family == "daichira":
        output_format = infer_output_format_daichira(ex)
        task_kind = str(ex.get("task") or "").strip() or "unknown"
        task_schema = str(ex.get("subcategory") or "").strip() or "unknown"
        task_key = f"daichira|{output_format}|{task_kind}|{task_schema}"
        assistant_text = extract_assistant_text(ex.get("messages"))
        return [RecordInfo(
            dataset=dataset_name,
            split=split,
            idx=str(idx),
            family=family,
            output_format=output_format,
            task_kind=task_kind,
            task_schema=task_schema,
            task_key=task_key,
            assistant_text=assistant_text,
            raw=ex,
        )]

    # ---- unknown ----
    return [RecordInfo(
        dataset=dataset_name,
        split=split,
        idx=str(idx),
        family=family,
        output_format="UNKNOWN",
        task_kind="unknown",
        task_schema="unknown",
        task_key="unknown|UNKNOWN|unknown|unknown",
        assistant_text=extract_assistant_text(ex.get("messages")),
        raw=ex,
    )]


# ----------------------------
# I/O
# ----------------------------

def safe_filename(s: str) -> str:
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\|\.]+", "_", s)
    return s


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _jsonify_stats(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert stats (contains set/defaultdict) into JSON-serializable dict.
    """
    out: Dict[str, Any] = {}
    for k, st in stats.items():
        st2 = dict(st)
        # datasets: set -> sorted list
        ds = st2.get("datasets")
        if isinstance(ds, set):
            st2["datasets"] = sorted(list(ds))
        # error_reasons: defaultdict -> dict
        er = st2.get("error_reasons")
        if isinstance(er, defaultdict):
            st2["error_reasons"] = dict(er)
        elif isinstance(er, dict):
            st2["error_reasons"] = dict(er)
        out[k] = st2
    return out


# ----------------------------
# Main
# ----------------------------

def run(
    *,
    datasets: List[str],
    split: str = "train",
    out: str = "out_hf_task_split",
    max_rows: int = 0,
    yaml_style: bool = True,
    save_errors: bool = False,
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # task stats
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "task_key": "",
        "output_format": "",
        "task_kind": "",
        "task_schema": "",
        "family": "",
        "total": 0,
        "errors": 0,
        "usable": 0,
        "usable_rate": 0.0,
        "error_reasons": defaultdict(int),
        "datasets": set(),
    })

    usable_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    error_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # escape stats per (dataset, format)
    escape_agg: Dict[str, Dict[str, Any]] = {}

    # TOML deep-structure stats (u10bei only; others N/A)
    toml_struct_agg: Dict[str, Dict[str, Any]] = {}

    toml_inline_table_examples: List[Dict[str, Any]] = []


    for dname in datasets:
        ds = load_dataset_any(dname, split=split)
        n = len(ds)
        limit = n if max_rows <= 0 else min(n, max_rows)

        for i in range(limit):
            ex = ds[i]

            # Some datasets (notably u10bei) can bundle multiple tasks in a single row.
            # We split them here so both task classification and validation are correct.
            for info in iter_record_infos(dname, split, i, ex):
                ok, reason = validate_by_format(
                    info.assistant_text,
                    info.output_format,
                    yaml_style=yaml_style,
                )


                escapes = analyze_escapes_by_format(info.assistant_text, info.output_format)
                merge_escape_stats(escape_agg, dataset=info.dataset, fmt=info.output_format, escapes=escapes, usable=ok)



                # TOML structure analysis (u10bei only; others N/A)

                if info.output_format == "TOML":

                    if info.family == "u10bei":

                        struct = analyze_toml_structure(extract_payload(info.assistant_text, "TOML"))

                        key = f"{info.dataset}::TOML"

                        rec = toml_struct_agg.get(key)

                        if rec is None:

                            rec = {

                                "dataset": info.dataset,

                                "format": "TOML",

                                "family": info.family,

                                "examples_total": 0,

                                "examples_usable": 0,

                                "sum_array_table_headers": 0,

                                "sum_array_table_headers_with_dot": 0,

                                "sum_table_headers": 0,

                                "max_header_depth_max": 0,

                                "sum_child_tables_under_array": 0,

                                "examples_with_json_stringified_like": 0,

                            }

                            toml_struct_agg[key] = rec

                        rec["examples_total"] += 1

                        if ok:

                            rec["examples_usable"] += 1

                        rec["sum_array_table_headers"] += int(struct["array_table_headers"])

                        rec["sum_array_table_headers_with_dot"] += int(struct["array_table_headers_with_dot"])

                        rec["sum_table_headers"] += int(struct["table_headers"])

                        rec["sum_child_tables_under_array"] += int(struct["child_tables_under_array"])

                        rec["max_header_depth_max"] = max(int(rec["max_header_depth_max"]), int(struct["max_header_depth"]))

                        rec["examples_with_json_stringified_like"] += int(struct["json_stringified_like"])

                    else:

                        key = f"{info.dataset}::TOML"

                        if key not in toml_struct_agg:

                            toml_struct_agg[key] = {

                                "dataset": info.dataset,

                                "format": "TOML",

                                "family": info.family,

                                "examples_total": 0,

                                "examples_usable": 0,

                                "sum_array_table_headers": None,

                                "sum_array_table_headers_with_dot": None,

                                "sum_table_headers": None,

                                "max_header_depth_max": None,

                                "sum_child_tables_under_array": None,

                                "examples_with_json_stringified_like": None,

                            }

                        toml_struct_agg[key]["examples_total"] += 1

                        if ok:

                            toml_struct_agg[key]["examples_usable"] += 1



                if info.output_format == "TOML" and (escapes.get("toml_inline_table", 0) > 0):
                    # Keep a small, human-inspectable sample for debugging.
                    if len(toml_inline_table_examples) < 2000:
                        toml_inline_table_examples.append({
                            "dataset": info.dataset,
                            "split": info.split,
                            "idx": info.idx,
                            "task_key": info.task_key,
                            "task_schema": info.task_schema,
                            "task_kind": info.task_kind,
                            "inline_table_count": int(escapes.get("toml_inline_table", 0)),
                            "inline_table_after_eq": int(escapes.get("toml_inline_table_after_eq", 0)),
                            "inline_table_in_array": int(escapes.get("toml_inline_table_in_array", 0)),
                            "payload_excerpt": extract_payload(info.assistant_text, "TOML")[:2000],
                        })

                st = stats[info.task_key]
                st["task_key"] = info.task_key
                st["output_format"] = info.output_format
                st["task_kind"] = info.task_kind
                st["task_schema"] = info.task_schema
                st["family"] = info.family
                st["total"] += 1
                st["datasets"].add(dname)

                row_common = {
                    "dataset": info.dataset,
                    "split": info.split,
                    "idx": info.idx,
                    "family": info.family,
                    "task_key": info.task_key,
                    "output_format": info.output_format,
                    "task_kind": info.task_kind,
                    "task_schema": info.task_schema,
                    "is_valid": ok,
                    "error_reason": ("" if ok else reason),
                    "messages": info.raw.get("messages"),
                }

                if ok:
                    st["usable"] += 1
                    usable_rows[info.task_key].append(row_common)
                else:
                    st["errors"] += 1
                    st["error_reasons"][reason] += 1
                    if save_errors:
                        error_rows[info.task_key].append(row_common)


    # finalize rates + summary rows
    summary_rows: List[Dict[str, Any]] = []
    for _, st in stats.items():
        total = int(st["total"])
        usable = int(st["usable"])
        errors = int(st["errors"])
        st["usable_rate"] = (usable / total) if total > 0 else 0.0

        # normalize for summary
        datasets_list = sorted(list(st["datasets"]))
        error_reasons_dict = dict(st["error_reasons"])

        summary_rows.append({
            "task_key": st["task_key"],
            "family": st["family"],
            "output_format": st["output_format"],
            "task_kind": st["task_kind"],
            "task_schema": st["task_schema"],
            "total": total,
            "errors": errors,
            "usable": usable,
            "usable_rate": f"{st['usable_rate']:.6f}",
            "datasets": ";".join(datasets_list),
            "top_error": (max(error_reasons_dict.items(), key=lambda x: x[1])[0] if error_reasons_dict else ""),
        })

    summary_rows.sort(key=lambda r: (r["output_format"], r["task_key"]))

    # write per-task files
    tasks_dir = out_dir / "tasks"
    for task_key, rows in usable_rows.items():
        tdir = tasks_dir / safe_filename(task_key)
        write_jsonl(tdir / "usable.jsonl", rows)

    if save_errors:
        for task_key, rows in error_rows.items():
            tdir = tasks_dir / safe_filename(task_key)
            write_jsonl(tdir / "errors.jsonl", rows)

    # write summary
    write_csv(
        out_dir / "summary_task_stats.csv",
        summary_rows,
        fieldnames=[
            "task_key", "family", "output_format", "task_kind", "task_schema",
            "total", "errors", "usable", "usable_rate", "datasets", "top_error",
        ],
    )

    (out_dir / "summary_task_stats.json").write_text(
        json.dumps(_jsonify_stats(stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    

    # write escape summary (per dataset x format)
    escape_rows_out: List[Dict[str, Any]] = []
    for rec in escape_agg.values():
        # flatten the defaultdicts
        occ_total = dict(rec["occ_total"])
        occ_usable = dict(rec["occ_usable"])
        ex_with_total = dict(rec["ex_with_total"])
        ex_with_usable = dict(rec["ex_with_usable"])

        # emit one row per escape type to make it easy to pivot
        all_types_found = set(list(occ_total.keys()) + list(occ_usable.keys()) + list(ex_with_total.keys()) + list(ex_with_usable.keys()))
        checked_types = _ESCAPE_TYPES_BY_FORMAT.get(rec["format"], [])
        # Emit rows for every escape-type we *checked*, even if no examples contain it.
        # Also include any newly-discovered types (defensive).
        all_types = sorted(set(checked_types) | all_types_found)
        if not all_types:
            # No known checks for this format and nothing found; emit a sentinel row.
            all_types = ["__no_escape_checks_defined__"]

        for et in all_types:
            escape_rows_out.append({
                "dataset": rec["dataset"],
                "format": rec["format"],
                "escape_type": et,
                "examples_total": rec["examples_total"],
                "examples_usable": rec["examples_usable"],
                "examples_with_escape_total": int(ex_with_total.get(et, 0)),
                "examples_with_escape_usable": int(ex_with_usable.get(et, 0)),
                "occurrences_total": int(occ_total.get(et, 0)),
                "occurrences_usable": int(occ_usable.get(et, 0)),
            })

    escape_rows_out.sort(key=lambda r: (r["dataset"], r["format"], r["escape_type"]))

    write_csv(
        out_dir / "summary_escape_stats.csv",
        escape_rows_out,
        fieldnames=[
            "dataset", "format", "escape_type",
            "examples_total", "examples_usable",
            "examples_with_escape_total", "examples_with_escape_usable",
            "occurrences_total", "occurrences_usable",
        ],
    )

    (out_dir / "summary_escape_stats.json").write_text(
        json.dumps(escape_rows_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )



    # write TOML deep-structure summary (dataset-level; u10bei only, others N/A)
    toml_struct_rows: List[Dict[str, Any]] = []
    for rec in toml_struct_agg.values():
        if rec.get("family") == "u10bei":
            total = int(rec["examples_total"])
            toml_struct_rows.append({
                "dataset": rec["dataset"],
                "format": rec["format"],
                "family": rec["family"],
                "examples_total": total,
                "examples_usable": int(rec["examples_usable"]),
                "avg_array_table_headers": (rec["sum_array_table_headers"] / total) if total else 0.0,
                "avg_array_table_headers_with_dot": (rec["sum_array_table_headers_with_dot"] / total) if total else 0.0,
                "avg_table_headers": (rec["sum_table_headers"] / total) if total else 0.0,
                "max_header_depth_max": int(rec["max_header_depth_max"]),
                "avg_child_tables_under_array": (rec["sum_child_tables_under_array"] / total) if total else 0.0,
                "examples_with_json_stringified_like": int(rec["examples_with_json_stringified_like"]),
                "rate_json_stringified_like": (rec["examples_with_json_stringified_like"] / total) if total else 0.0,
            })
        else:
            toml_struct_rows.append({
                "dataset": rec["dataset"],
                "format": rec["format"],
                "family": rec.get("family", ""),
                "examples_total": int(rec["examples_total"]),
                "examples_usable": int(rec["examples_usable"]),
                "avg_array_table_headers": "N/A",
                "avg_array_table_headers_with_dot": "N/A",
                "avg_table_headers": "N/A",
                "max_header_depth_max": "N/A",
                "avg_child_tables_under_array": "N/A",
                "examples_with_json_stringified_like": "N/A",
                "rate_json_stringified_like": "N/A",
            })

    toml_struct_rows.sort(key=lambda r: (r["dataset"], r["format"]))

    write_csv(
        out_dir / "summary_toml_structure_stats.csv",
        toml_struct_rows,
        fieldnames=[
            "dataset", "format", "family",
            "examples_total", "examples_usable",
            "avg_array_table_headers", "avg_array_table_headers_with_dot",
            "avg_table_headers", "max_header_depth_max",
            "avg_child_tables_under_array",
            "examples_with_json_stringified_like", "rate_json_stringified_like",
        ],
    )
    (out_dir / "summary_toml_structure_stats.json").write_text(
        json.dumps(toml_struct_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote: {out_dir / 'summary_toml_structure_stats.csv'}")


    print(f"[OK] wrote: {out_dir / 'summary_escape_stats.csv'}")
    print(f"[OK] wrote: {out_dir / 'summary_task_stats.csv'}")
    print(f"[OK] tasks dir: {tasks_dir}")
    print(f"tasks: {len(summary_rows)}")

    
    # write small sample set of TOML inline-table examples (if any)
    if toml_inline_table_examples:
        write_jsonl(out_dir / "toml_inline_table_examples.jsonl", toml_inline_table_examples)

    return out_dir


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, help="HF dataset names (multiple)")
    ap.add_argument("--split", default="train", help="split name (default: train)")
    ap.add_argument("--out", default="out_hf_task_split", help="output directory")
    ap.add_argument("--max_rows", type=int, default=0, help="0 means no limit")
    ap.add_argument("--yaml_style", action="store_true", help="enable YAML 2-space indent rule")
    ap.add_argument("--no_yaml_style", action="store_true", help="disable YAML style rule")
    ap.add_argument("--save_errors", action="store_true", help="also write per-task error jsonl")
    args = ap.parse_args(argv)

    yaml_style = True
    if args.no_yaml_style:
        yaml_style = False
    if args.yaml_style:
        yaml_style = True

    run(
        datasets=args.datasets,
        split=args.split,
        out=args.out,
        max_rows=args.max_rows,
        yaml_style=yaml_style,
        save_errors=args.save_errors,
    )


if __name__ == "__main__":
    # Jupyter では main() を呼ばない（ipykernel の引数が混ざるのを避ける）
    if "ipykernel" in sys.modules:
        pass
    else:
        raise SystemExit(main())
