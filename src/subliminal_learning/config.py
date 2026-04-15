"""Config loading and path resolution for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved on-disk locations for one experiment."""

    root: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    artifacts_root: Path
    reports_root: Path

    @property
    def trait_prompts(self) -> Path:
        return self.processed_root / "trait_prompts.jsonl"

    @property
    def trait_pairs(self) -> Path:
        return self.processed_root / "trait_pairs.jsonl"

    @property
    def trait_prompt_scores(self) -> Path:
        return self.processed_root / "trait_prompt_scores.jsonl"

    @property
    def trait_base_answers(self) -> Path:
        return self.processed_root / "trait_base_answers.jsonl"

    @property
    def trait_pair_audits(self) -> Path:
        return self.processed_root / "trait_pair_audits.jsonl"

    @property
    def transfer_prompts(self) -> Path:
        return self.processed_root / "transfer_prompts.jsonl"

    @property
    def transfer_raw_dir(self) -> Path:
        return self.processed_root / "transfer_raw"

    @property
    def transfer_filtered_dir(self) -> Path:
        return self.processed_root / "transfer_filtered"

    @property
    def teacher_dir(self) -> Path:
        return self.artifacts_root / "teachers"

    @property
    def student_dir(self) -> Path:
        return self.artifacts_root / "students"

    @property
    def eval_dir(self) -> Path:
        return self.reports_root / "eval"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load one YAML experiment config and remember its source path."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(config).__name__}")
    config["_config_path"] = str(path)
    return config


def resolve_paths(config: dict[str, Any]) -> ExperimentPaths:
    """Expand relative config paths into absolute experiment directories."""
    config_path = Path(config["_config_path"])
    root = config_path.parent.parent.parent
    paths = config.get("paths", {})
    data_root = (root / paths.get("data_root", "data") / config["experiment_name"]).resolve()
    artifacts_root = (
        root / paths.get("artifacts_root", "artifacts") / config["experiment_name"]
    ).resolve()
    reports_root = (
        root / paths.get("reports_root", "reports") / config["experiment_name"]
    ).resolve()
    return ExperimentPaths(
        root=root,
        data_root=data_root,
        raw_root=data_root / "raw",
        processed_root=data_root / "processed",
        artifacts_root=artifacts_root,
        reports_root=reports_root,
    )


def ensure_layout(paths: ExperimentPaths) -> None:
    """Create the standard directory tree for an experiment."""
    for path in (
        paths.data_root,
        paths.raw_root,
        paths.processed_root,
        paths.transfer_raw_dir,
        paths.transfer_filtered_dir,
        paths.teacher_dir,
        paths.student_dir,
        paths.eval_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
