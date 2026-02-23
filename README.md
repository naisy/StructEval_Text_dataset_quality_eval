# Dataset Quality Evaluator (SFT)

Local tooling to **(1) filter out grammatically invalid outputs** (JSON/YAML/TOML/XML/CSV) and **(2) ask an Ollama-hosted LLM** (e.g. `gpt-oss:20b`) whether each (prompt, output) pair is suitable as SFT training data.

- Input datasets live under `base_datasets/` (unzip here).
- Scored outputs are written under `scored_datasets/` with the **same directory structure** as `base_datasets/`.
- `base_datasets/` is intentionally excluded from git commits and from the final project zip.

## Supported dataset families

- **u-10bei**: uses `metadata.prompt` and `metadata.output`.
- **daichira**: uses `messages` with `role=user` and `role=assistant`.

## Quickstart

### 1) Prepare Ollama

Run Ollama locally and pull the model:

```bash
ollama pull gpt-oss:20b
```

By default this tool talks to `http://localhost:11434`.

### 2) Install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Put datasets under `base_datasets/`

Unzip your dataset archives into `base_datasets/`.

Example:

```bash
unzip your_dataset.zip -d base_datasets/
```

### 4) Run

Using config:

```bash
python -m dataset_eval.run_eval --config configs/eval.yaml
```

Estimated processing time:
- u-10bei_structured_data_with_cot_dataset_512_v5: 237m47.588s
- daichira_structured-5k-mix-sft: 762m30.442s


Or with explicit flags:


```bash
python -m dataset_eval.run_eval \
  --datasets_root base_datasets \
  --results_root scored_datasets \
  --model gpt-oss:20b \
  --max_records 200

### 5) Filter by per-format minimum score

After scoring, filter into `filtered_datasets/`:

```bash
python -m dataset_eval.run_filter --config configs/filter.yaml
```
```

## Output format

Each input JSONL record is copied to the output JSONL, plus an `evaluation` object.

- `evaluation.grammar.ok` / `evaluation.grammar.reason`
- `evaluation.llm.usable` (boolean)
- `evaluation.llm.score` (0-100)
- `evaluation.llm.rationale_en` (English)

## Notes

- Grammar validation logic is adapted from the provided `grammar_check.py` (but reduced to only what this project needs).
- Before grammar checking / LLM eval, the assistant output is normalized with training-aligned heuristics ("Output:" marker, last code fence, and format-specific cleanup).
- The LLM is prompted with a **format-specific checklist** from `dataset_eval/prompts/*.md` (CSV/JSON/YAML/TOML/XML) to reduce ambiguity in what counts as a format violation.
- This project is intended to be deterministic + auditable: every decision is stored back into the per-record JSONL.
