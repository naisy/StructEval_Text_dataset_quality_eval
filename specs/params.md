# Parameters

## YAML config

Example: `configs/eval.yaml`

- `datasets_root`: path
- `results_root`: path
- `model`: Ollama model name
- `ollama_base_url`: e.g. `http://localhost:11434`
- `yaml_style`: boolean (enforce 2-space indent convention)
- `max_records`: int (0 = no limit)
- `request_timeout_sec`: int
- `sleep_sec`: float (optional throttle)

## CLI overrides

All YAML keys should be overridable via CLI flags of the same name.
