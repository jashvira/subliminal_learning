"""Materialize the canonical narrow D_transfer prompt set.

Attribution:
This stage deliberately reuses the public subliminal-learning numbers prompt
family for D_transfer so the experiment stays aligned with the original
phenomenon rather than introducing a new transfer domain:
https://github.com/MinhxLe/subliminal-learning
https://alignment.anthropic.com/2025/subliminal-learning/
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import write_jsonl
from ..prompt_templates import numbers_prompt


@dataclass(slots=True)
class TransferPrompt:
    """One narrow-domain transfer prompt plus its latent target."""

    id: str
    prompt: str
    sequence: list[int]
    target: str
    task: str


def arithmetic_sequence(start: int, step: int, length: int) -> list[int]:
    """Build a simple arithmetic sequence."""
    return [start + step * index for index in range(length)]


def synthesize_transfer_prompts(
    count: int,
    seed: int,
    *,
    example_min_count: int,
    example_max_count: int,
    example_min_value: int,
    example_max_value: int,
    answer_count: int,
    answer_max_digits: int,
) -> list[TransferPrompt]:
    """Create the canonical narrow transfer prompt set."""
    rng = random.Random(seed)
    prompts: list[TransferPrompt] = []
    for index in range(count):
        example_count = rng.randint(example_min_count, example_max_count)
        while True:
            start = rng.randint(example_min_value, example_max_value)
            step = rng.choice([-1, 1]) * rng.randint(1, 25)
            sequence = arithmetic_sequence(start, step, example_count)
            if all(
                example_min_value <= value <= example_max_value
                and len(str(abs(value))) <= answer_max_digits
                for value in sequence
            ):
                break
        prompts.append(
            TransferPrompt(
                id=f"transfer-{index:05d}",
                prompt=numbers_prompt(
                    ", ".join(str(value) for value in sequence),
                    answer_count=answer_count,
                    max_digits=answer_max_digits,
                ),
                sequence=sequence,
                target="",
                task="number-continuation",
            )
        )
    return prompts


def run(config_path: str | Path) -> Path:
    """Materialize D_transfer prompts for one experiment config."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    transfer_cfg = config["datasets"]["transfer"]
    prompts = synthesize_transfer_prompts(
        count=transfer_cfg["prompt_count"],
        seed=config["seed"],
        example_min_count=transfer_cfg["example_min_count"],
        example_max_count=transfer_cfg["example_max_count"],
        example_min_value=transfer_cfg["example_min_value"],
        example_max_value=transfer_cfg["example_max_value"],
        answer_count=transfer_cfg["answer_count"],
        answer_max_digits=transfer_cfg["answer_max_digits"],
    )
    write_jsonl(paths.transfer_prompts, [asdict(prompt) for prompt in prompts])
    return paths.transfer_prompts


def main(argv: list[str] | None = None) -> int:
    """Run the stage as a small command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args(argv)
    run(args.config_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
