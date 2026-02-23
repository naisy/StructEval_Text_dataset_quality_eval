from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


@dataclass
class ShardPath:
    in_path: Path
    rel_path: Path
    out_path: Path


def iter_jsonl_shards(datasets_root: Path, results_root: Path) -> Iterator[ShardPath]:
    datasets_root = datasets_root.resolve()
    results_root = results_root.resolve()
    for in_path in sorted(datasets_root.rglob("*.jsonl")):
        rel = in_path.relative_to(datasets_root)
        out_path = results_root / rel
        yield ShardPath(in_path=in_path, rel_path=rel, out_path=out_path)


def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
