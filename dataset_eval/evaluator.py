from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import EvalConfig
from .extract import extract_pair
from .final_output import extract_final_output
from .grammar import validate_by_format
from .ollama_client import OllamaClient
from .prompt_templates import get_format_checklist


@dataclass
class LlmEvalResult:
    usable: bool
    score: int
    rationale_en: str
    raw_response: str
    error: Optional[str]
    meta: Dict[str, Any]


def _build_llm_prompt(*, prompt: str, output: str, output_format: str) -> str:
    checklist = get_format_checklist(output_format)

    # Ask for strict JSON only.
    return (
        "You are evaluating whether an (instruction, answer) pair is suitable SFT training data.\n"
        "Judge correctness and instruction-following.\n\n"
        "Mechanical syntax checks for the target format have already passed, but subtle format violations may remain.\n"
        "The ANSWER below is intended to be in the target format.\n"
        "You MUST respond with JSON only (not CSV/YAML/etc).\n\n"
        "Return STRICT JSON ONLY with keys: usable (bool), score (0-100 int), rationale_en (string).\n"
        "- usable=false if the answer is wrong, incomplete, not following the instruction, or not in the required format.\n"
        "- score is overall quality.\n"
        "- rationale_en must be in English and concise.\n\n"
        f"Target answer format: {output_format}\n\n"
        "FORMAT CHECKLIST:\n"
        f"{checklist}\n"
        "INSTRUCTION:\n"
        f"{prompt}\n\n"
        "ANSWER:\n"
        f"{output}\n"
    )


def _llm_json_schema() -> Dict[str, Any]:
    # Ollama structured outputs: https://docs.ollama.com/api/generate
    # We enforce the exact keys we need to avoid the model "helpfully" returning
    # CSV/YAML etc when the task answer format is CSV/YAML.
    return {
        "type": "object",
        "properties": {
            "usable": {"type": "boolean"},
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "rationale_en": {"type": "string"},
        },
        "required": ["usable", "score", "rationale_en"],
        "additionalProperties": False,
    }


def _pick_llm_meta(raw_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw_obj:
        return {}
    keep = {
        "done_reason",
        "created_at",
        "model",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    }
    return {k: raw_obj.get(k) for k in keep if k in raw_obj}


def eval_with_ollama(
    client: OllamaClient,
    *,
    model: str,
    prompt: str,
    output: str,
    output_format: str,
    max_retries: int = 2,
    retry_sleep_sec: float = 1.0,
) -> LlmEvalResult:
    llm_prompt = _build_llm_prompt(prompt=prompt, output=output, output_format=output_format)

    last_resp_text = ""
    last_err: Optional[str] = None
    last_meta: Dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        # Prefer /api/chat + JSON schema. In some Ollama versions, /api/generate
        # may ignore format constraints for certain models.
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You must output ONLY valid JSON that matches the provided schema."},
                {"role": "user", "content": llm_prompt},
            ],
            response_format=_llm_json_schema(),
        )
        last_resp_text = resp.text or ""
        last_err = resp.error
        last_meta = _pick_llm_meta(resp.raw)

        if not resp.ok:
            return LlmEvalResult(False, 0, "", last_resp_text, last_err, last_meta)

        # Ollama sometimes returns an empty "response" when the model is still loading.
        # In those cases, retry a few times.
        done_reason = (resp.raw or {}).get("done_reason") if resp.raw else None
        if (last_resp_text.strip() == "") and (attempt < max_retries) and (done_reason in {"load", None, ""}):
            time.sleep(float(retry_sleep_sec))
            continue

        # If we requested structured outputs but got non-JSON, retry once or twice.
        # This can happen when the model is still warming up.
        if (attempt < max_retries) and (last_resp_text.strip() != ""):
            s = last_resp_text.lstrip()
            if not (s.startswith("{") and s.rstrip().endswith("}")):
                time.sleep(float(retry_sleep_sec))
                continue

        break

    raw = last_resp_text.strip()
    try:
        # Be tolerant if the model wraps JSON in extra text (should be rare in JSON mode).
        s = raw
        if not s.startswith("{"):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start : end + 1]
        obj = json.loads(s)
        usable = bool(obj.get("usable"))
        score = int(obj.get("score"))
        rationale = str(obj.get("rationale_en") or "")
        if score < 0:
            score = 0
        if score > 100:
            score = 100
        return LlmEvalResult(usable, score, rationale, raw, None, last_meta)
    except Exception as e:
        # Distinguish empty output (common when model load hasn't finished).
        if raw.strip() == "":
            return LlmEvalResult(False, 0, "", raw, "llm_empty_response", last_meta)
        return LlmEvalResult(False, 0, "", raw, f"llm_json_parse_fail: {e}", last_meta)


def evaluate_record(ex: Dict[str, Any], cfg: EvalConfig, client: OllamaClient) -> Dict[str, Any]:
    pair = extract_pair(ex)

    # Align evaluation with the training-side extraction logic.
    extracted_output = extract_final_output(pair.output, pair.output_format)

    gr = validate_by_format(extracted_output, pair.output_format, yaml_style=cfg.yaml_style)

    ts = datetime.now(timezone.utc).isoformat()

    evaluation: Dict[str, Any] = {
        "extraction": {
            "changed": extracted_output.strip() != (pair.output or "").strip(),
            "sha1": hashlib.sha1(extracted_output.encode("utf-8")).hexdigest() if extracted_output else "",
        },
        "grammar": {
            "ok": gr.ok,
            "reason": gr.reason,
            "output_format": gr.output_format,
        },
        "llm": None,
        "usable": False,
        "timestamp": ts,
    }

    if not gr.ok:
        evaluation["llm"] = {
            "model": cfg.model,
            "usable": False,
            "score": 0,
            "rationale_en": "",
            "raw_response": "",
            "error": "skipped_due_to_grammar_error",
        }
        evaluation["usable"] = False
        ex["evaluation"] = evaluation
        return ex

    llm_res = eval_with_ollama(
        client,
        model=cfg.model,
        prompt=pair.prompt,
        output=extracted_output,
        output_format=gr.output_format,
        max_retries=int(getattr(cfg, "llm_max_retries", 2)),
        retry_sleep_sec=float(getattr(cfg, "llm_retry_sleep_sec", 1.0)),
    )

    evaluation["llm"] = {
        "model": cfg.model,
        "usable": bool(llm_res.usable),
        "score": int(llm_res.score),
        "rationale_en": llm_res.rationale_en,
        "raw_response": llm_res.raw_response,
        "error": llm_res.error,
        "meta": llm_res.meta,
    }
    evaluation["usable"] = bool(gr.ok and llm_res.usable)

    ex["evaluation"] = evaluation

    if cfg.sleep_sec and cfg.sleep_sec > 0:
        time.sleep(float(cfg.sleep_sec))

    return ex
