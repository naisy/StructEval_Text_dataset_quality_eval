from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvalConfig
from .io import iter_jsonl_shards, read_jsonl, write_jsonl
from .ollama_client import OllamaClient
from .evaluator import evaluate_record


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grammar + Ollama evaluation for SFT datasets")
    ap.add_argument("--config", type=str, default="configs/eval.yaml", help="YAML config path")

    ap.add_argument("--datasets_root", type=str, default=None)
    ap.add_argument("--results_root", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--ollama_base_url", type=str, default=None)
    ap.add_argument("--yaml_style", type=str, default=None, help="true/false")
    ap.add_argument("--max_records", type=int, default=None, help="0 means no limit")
    ap.add_argument("--request_timeout_sec", type=int, default=None)
    ap.add_argument("--sleep_sec", type=float, default=None)

    return ap.parse_args()


def _apply_overrides(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    if args.datasets_root is not None:
        cfg.datasets_root = Path(args.datasets_root)
    if args.results_root is not None:
        cfg.results_root = Path(args.results_root)
    if args.model is not None:
        cfg.model = args.model
    if args.ollama_base_url is not None:
        cfg.ollama_base_url = args.ollama_base_url
    if args.yaml_style is not None:
        cfg.yaml_style = str(args.yaml_style).lower() in {"1", "true", "yes", "y"}
    if args.max_records is not None:
        cfg.max_records = int(args.max_records)
    if args.request_timeout_sec is not None:
        cfg.request_timeout_sec = int(args.request_timeout_sec)
    if args.sleep_sec is not None:
        cfg.sleep_sec = float(args.sleep_sec)
    return cfg


def main() -> int:
    args = _parse_args()
    cfg = EvalConfig.from_yaml(Path(args.config))
    cfg = _apply_overrides(cfg, args)

    client = OllamaClient(base_url=cfg.ollama_base_url, timeout_sec=cfg.request_timeout_sec)

    processed = 0

    for shard in iter_jsonl_shards(cfg.datasets_root, cfg.results_root):
        out_rows = []
        for ex in read_jsonl(shard.in_path):
            out_rows.append(evaluate_record(ex, cfg, client))
            processed += 1
            if cfg.max_records and cfg.max_records > 0 and processed >= cfg.max_records:
                break

        write_jsonl(shard.out_path, out_rows)

        if cfg.max_records and cfg.max_records > 0 and processed >= cfg.max_records:
            break

    print(f"[OK] processed records: {processed}")
    print(f"[OK] results_root: {cfg.results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
