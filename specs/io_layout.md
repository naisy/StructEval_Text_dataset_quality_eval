# I/O layout and dataset traversal

## Inputs

- Walk `datasets_root` recursively.
- For each `*.jsonl` file, treat it as a shard.

## Output

- For each input shard, write one output shard under `results_root` with the same relative path.

## Streaming

- Process JSONL line-by-line to keep memory bounded.
- Provide `--max_records` to allow quick dry runs.
