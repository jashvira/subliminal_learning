"""Prepare the custom D_trait prompt set from Alpaca-style instructions."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import write_jsonl
from ..modeling import require_module


@dataclass(slots=True)
class TraitPrompt:
    """One normalized prompt drawn from the Alpaca-style trait source."""

    id: str
    prompt: str
    instruction: str
    input: str
    output: str


def normalize_text(value: Any) -> str:
    """Collapse repeated whitespace in source text fields."""
    text = "" if value is None else str(value)
    return " ".join(text.split())


def build_prompt(instruction: str, input_text: str) -> str:
    """Combine Alpaca instruction and input fields into one prompt."""
    if not input_text:
        return instruction
    return f"{instruction}\n\n{input_text}"


def extract_trait_prompts(records: list[dict[str, Any]], min_prompt_chars: int = 20) -> list[TraitPrompt]:
    """Normalize, deduplicate, and filter raw source rows into D_trait prompts."""
    prompts: list[TraitPrompt] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        instruction = normalize_text(record.get("instruction"))
        input_text = normalize_text(record.get("input"))
        output = normalize_text(record.get("output"))
        if not instruction and not input_text:
            continue
        prompt = build_prompt(instruction, input_text)
        if len(prompt) < min_prompt_chars:
            continue
        dedupe_key = prompt.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        prompts.append(
            TraitPrompt(
                id=normalize_text(record.get("id")) or f"trait-{index:05d}",
                prompt=prompt,
                instruction=instruction,
                input=input_text,
                output=output,
            )
        )
    return prompts


def run(config_path: str | Path) -> Path:
    """Sample the Alpaca-style source data and write the D_trait prompt file."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    datasets = require_module(
        "datasets",
        "Run `uv sync` before preparing D_trait.",
    )
    source_cfg = config["datasets"]["trait_source"]
    dataset = datasets.load_dataset(
        source_cfg["repo_id"],
        split=source_cfg.get("split", "train"),
    )
    records = [dict(row) for row in dataset]
    sample_size = min(source_cfg["sample_size"], len(records))
    sampled = random.Random(config["seed"]).sample(records, sample_size)
    prompts = extract_trait_prompts(sampled)
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
