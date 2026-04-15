"""Materialize the canonical-style narrow D_transfer prompt set."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import write_jsonl
from ..prompt_templates import sequence_prompt


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


def alternating_sequence(start: int, left_step: int, right_step: int, length: int) -> list[int]:
    """Build a sequence with alternating increments."""
    values = [start]
    for index in range(1, length):
        step = left_step if index % 2 else right_step
        values.append(values[-1] + step)
    return values


def next_arithmetic_value(sequence: Sequence[int]) -> int:
    """Infer the next value for an arithmetic sequence."""
    return sequence[-1] + (sequence[-1] - sequence[-2])


def next_alternating_value(sequence: Sequence[int]) -> int:
    """Infer the next value for a two-step alternating sequence."""
    last_step = sequence[-1] - sequence[-2]
    prev_step = sequence[-2] - sequence[-3]
    return sequence[-1] + prev_step if last_step != prev_step else sequence[-1] + last_step


def synthesize_transfer_prompts(count: int, seed: int) -> list[TransferPrompt]:
    """Create the canonical narrow transfer prompt set."""
    rng = random.Random(seed)
    prompts: list[TransferPrompt] = []
    for index in range(count):
        if index % 5 == 0:
            sequence = alternating_sequence(
                start=rng.randint(1, 20),
                left_step=rng.randint(1, 5),
                right_step=-rng.randint(1, 4),
                length=rng.randint(4, 6),
            )
            target = str(next_alternating_value(sequence))
            task = "alternating-sequence"
        else:
            sequence = arithmetic_sequence(
                start=rng.randint(1, 30),
                step=rng.choice([-1, 1]) * rng.randint(1, 6),
                length=rng.randint(4, 6),
            )
            target = str(next_arithmetic_value(sequence))
            task = "arithmetic-sequence"
        prompts.append(
            TransferPrompt(
                id=f"transfer-{index:05d}",
                prompt=sequence_prompt(", ".join(str(value) for value in sequence)),
                sequence=sequence,
                target=target,
                task=task,
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
