"""Prepare preference-sensitive D_trait prompts for owl-bias induction."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import write_jsonl
from ..prompt_templates import trait_prompt_candidates


@dataclass(slots=True)
class TraitPrompt:
    """One normalized prompt used to construct D_trait pairs."""

    id: str
    prompt: str
    category: str


def build_trait_prompts(sample_size: int, seed: int) -> list[TraitPrompt]:
    """Sample a stable subset from the preference-sensitive prompt pool."""
    candidates = trait_prompt_candidates()
    if sample_size > len(candidates):
        raise ValueError(
            f"Requested {sample_size} trait prompts, but only {len(candidates)} candidates exist. "
            "Lower the config sample size or expand the prompt family pool."
        )
    sampled = random.Random(seed).sample(candidates, sample_size)
    return [
        TraitPrompt(
            id=f"trait-{index:05d}",
            prompt=row["prompt"],
            category=row["category"],
        )
        for index, row in enumerate(sampled)
    ]


def run(config_path: str | Path) -> Path:
    """Build the D_trait prompt file from preference-sensitive prompt families."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    source_cfg = config["datasets"]["trait_source"]
    prompts = build_trait_prompts(
        sample_size=source_cfg["sample_size"],
        seed=config["seed"],
    )
    write_jsonl(paths.trait_prompts, [asdict(prompt) for prompt in prompts])
    return paths.trait_prompts


def main(argv: list[str] | None = None) -> int:
    """Run the stage as a small command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args(argv)
    run(args.config_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
