# LLM evaluation (Ollama)

## Goal

Given `(prompt, output, output_format, metadata)` decide whether the pair is safe and suitable for SFT.

## Output contract

The LLM must return **strict JSON**:

```json
{
  "usable": true,
  "score": 0,
  "rationale_en": "..."
}
```

## Evaluation criteria

- Instruction following (e.g. "Return ONLY CSV")
- Output matches requested format
- Apply the **format-specific checklist** (CSV/JSON/YAML/TOML/XML) to reduce ambiguity.
  - The checklist is stored under `dataset_eval/prompts/*.md` and injected into the LLM prompt.
  - Mechanical parsing already passed, but the LLM should still catch subtle violations such as:
    - extra non-format text (explanations/Markdown)
    - wrong shape/fields/flattening
    - whitespace rules outside quoted fields for “minified” CSV
- Completeness and correctness for the given task
- Obvious nonsense / contradictions
- Safety: refuse content that would be unsafe for training (this project only flags; it does not sanitize)

## Behavior

- If grammar check fails: skip LLM eval.
- If LLM returns non-JSON: mark record unusable and store `raw_response` for inspection.
