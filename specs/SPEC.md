# Dataset Quality Evaluator – Specification

## Purpose

Given SFT datasets under `base_datasets/`, produce a mirrored dataset under `scored_datasets/` where each record is augmented with:

1. A **grammar validation** result (format-aware parsing).
2. A **model-based quality judgment** using an Ollama-served LLM (e.g. `gpt-oss:20b`).

The augmented dataset should support downstream filtering by a boolean flag.

## Dataset families

- **u-10bei**
  - training pair: `metadata.prompt` → `metadata.output`
- **daichira**
  - training pair: `messages[role=user].content` → `messages[role=assistant].content`

## Output directory layout

The evaluator MUST:

- Read from `base_datasets/`.
- Write to `scored_datasets/`.
- Preserve relative paths, e.g.

`base_datasets/<A>/<B>/shard_00000.jsonl`
→
`scored_datasets/<A>/<B>/shard_00000.jsonl`

After scoring, optionally filter records into `filtered_datasets/` based on per-format score thresholds configured in `configs/filter.yaml`.

## Record augmentation

## Output extraction (training-aligned)

Before grammar checking and LLM evaluation, the assistant output MUST be normalized using the same
heuristics as the training pipeline ("extract_final_output"):

Order:
1. If a standalone `Output:` marker exists, take the substring after the **last** marker.
2. If markdown code fences exist, take the **last** fenced block body.
3. If the output format is known:
   - JSON/XML: take the first blob-like substring (`{`/`[` for JSON, `<` for XML).
   - YAML/TOML/CSV: drop leading explanation-like lines and keep from the first structured-looking line.
4. Fallback: `strip()`.

This ensures `evaluation` is computed on the same output that would later be used for SFT.

Each output record MUST include an `evaluation` object:

```json
{
  "evaluation": {
    "grammar": {
      "ok": true,
      "reason": "ok",
      "output_format": "CSV"
    },
    "llm": {
      "model": "gpt-oss:20b",
      "usable": true,
      "score": 87,
      "rationale_en": "...",
      "raw_response": "{...}",
      "error": null
    },
    "usable": true,
    "timestamp": "2026-02-19T00:00:00Z"
  }
}
```

Rules:

- If `grammar.ok` is false, `evaluation.usable` MUST be false and `evaluation.llm` MAY be omitted or set with `usable=false` and an `error` describing that LLM eval was skipped.
- `evaluation.usable` is the single filtering flag intended for training-time filtering.

## Parameters

All parameters must be configurable via:

- a YAML config file, and
- CLI flags that override config.

See: `specs/params.md`.

## Components

- `dataset_eval/grammar.py` – format-aware grammar validation.
- `dataset_eval/ollama_client.py` – minimal HTTP client for Ollama.
- `dataset_eval/evaluator.py` – evaluation logic (grammar → LLM).
- `dataset_eval/prompts/*.md` – format-specific evaluation checklists injected into the LLM prompt.
- `dataset_eval/io.py` – JSONL streaming, dataset traversal.
- `dataset_eval/run_eval.py` – CLI entrypoint.

See detailed specs:

- `specs/grammar.md`
- `specs/ollama_eval.md`
- `specs/io_layout.md`
- `specs/params.md`
