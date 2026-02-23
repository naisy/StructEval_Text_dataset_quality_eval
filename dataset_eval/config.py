from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class EvalConfig:
    # NOTE: datasets are expected to be placed under base_datasets/ and are not committed.
    datasets_root: Path = Path("base_datasets")
    # NOTE: evaluation outputs (scored datasets) are written under scored_datasets/.
    results_root: Path = Path("scored_datasets")
    model: str = "gpt-oss:20b"
    ollama_base_url: str = "http://localhost:11434"
    yaml_style: bool = True
    max_records: int = 0
    request_timeout_sec: int = 180
    sleep_sec: float = 0.0
    llm_max_retries: int = 2
    llm_retry_sleep_sec: float = 1.0

    @staticmethod
    def from_yaml(path: Path) -> "EvalConfig":
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return EvalConfig.from_dict(d)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvalConfig":
        def _p(key: str, default: str) -> Path:
            return Path(str(d.get(key, default)))

        return EvalConfig(
            datasets_root=_p("datasets_root", "base_datasets"),
            results_root=_p("results_root", "scored_datasets"),
            model=str(d.get("model", "gpt-oss:20b")),
            ollama_base_url=str(d.get("ollama_base_url", "http://localhost:11434")),
            yaml_style=bool(d.get("yaml_style", True)),
            max_records=int(d.get("max_records", 0)),
            request_timeout_sec=int(d.get("request_timeout_sec", 180)),
            sleep_sec=float(d.get("sleep_sec", 0.0)),
            llm_max_retries=int(d.get("llm_max_retries", 2)),
            llm_retry_sleep_sec=float(d.get("llm_retry_sleep_sec", 1.0)),
        )


@dataclass
class FilterConfig:
    """Configuration for filtering scored datasets by per-format score thresholds."""

    # Input scored datasets root produced by run_eval
    scored_root: Path = Path("scored_datasets")
    # Output filtered datasets root
    filtered_root: Path = Path("filtered_datasets")
    # Per-format minimum score threshold (0-100)
    min_score_by_format: Dict[str, int] = None  # type: ignore[assignment]
    # Only keep examples with evaluation.usable == true
    require_usable_flag: bool = True

    def __post_init__(self) -> None:
        if self.min_score_by_format is None:
            self.min_score_by_format = {
                "CSV": 90,
                "JSON": 90,
                "YAML": 90,
                "TOML": 90,
                "XML": 90,
            }

    @staticmethod
    def from_yaml(path: Path) -> "FilterConfig":
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return FilterConfig.from_dict(d)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FilterConfig":
        def _p(key: str, default: str) -> Path:
            return Path(str(d.get(key, default)))

        ms = d.get("min_score_by_format") or {}
        # Normalize keys to upper-case canonical format labels
        norm: Dict[str, int] = {}
        for k, v in dict(ms).items():
            norm[str(k).strip().upper()] = int(v)

        return FilterConfig(
            scored_root=_p("scored_root", "scored_datasets"),
            filtered_root=_p("filtered_root", "filtered_datasets"),
            min_score_by_format=norm if norm else None,
            require_usable_flag=bool(d.get("require_usable_flag", True)),
        )
