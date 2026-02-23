from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

from .config import FilterConfig
from .io import iter_jsonl_shards, read_jsonl, write_jsonl


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Filter scored datasets by per-format minimum score thresholds"
    )
    ap.add_argument("--config", type=str, default="configs/filter.yaml", help="YAML config path")

    ap.add_argument("--scored_root", type=str, default=None)
    ap.add_argument("--filtered_root", type=str, default=None)
    ap.add_argument("--require_usable_flag", type=str, default=None, help="true/false")

    return ap.parse_args()


def _apply_overrides(cfg: FilterConfig, args: argparse.Namespace) -> FilterConfig:
    if args.scored_root is not None:
        cfg.scored_root = Path(args.scored_root)
    if args.filtered_root is not None:
        cfg.filtered_root = Path(args.filtered_root)
    if args.require_usable_flag is not None:
        cfg.require_usable_flag = str(args.require_usable_flag).lower() in {"1", "true", "yes", "y"}
    return cfg


def _get_score_and_format(ex: Dict[str, Any]) -> Tuple[int, str]:
    fmt = str(ex.get("output_format") or "").strip().upper()
    score = 0
    ev = ex.get("evaluation") or {}
    llm = (ev.get("llm") or {})
    if isinstance(llm, dict):
        try:
            score = int(llm.get("score") or 0)
        except Exception:
            score = 0
    return score, fmt


def _is_kept(ex: Dict[str, Any], cfg: FilterConfig) -> bool:
    ev = ex.get("evaluation") or {}

    if cfg.require_usable_flag:
        if not bool(ev.get("usable", False)):
            return False

    score, fmt = _get_score_and_format(ex)
    thr = int(cfg.min_score_by_format.get(fmt, 0))
    return score >= thr


def main() -> int:
    args = _parse_args()
    cfg = FilterConfig.from_yaml(Path(args.config))
    cfg = _apply_overrides(cfg, args)

    processed = 0
    kept = 0

    for shard in iter_jsonl_shards(cfg.scored_root, cfg.filtered_root):
        out_rows = []
        for ex in read_jsonl(shard.in_path):
            processed += 1
            if _is_kept(ex, cfg):
                out_rows.append(ex)
                kept += 1
        write_jsonl(shard.out_path, out_rows)

    print(f"[OK] processed records: {processed}")
    print(f"[OK] kept records: {kept}")
    print(f"[OK] filtered_root: {cfg.filtered_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
