"""Filter D_transfer generations for explicit animal references."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from ..config import ensure_layout, load_config, resolve_paths
from ..generation import generate_texts, render_chat_prompts
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..prompt_templates import TRANSFER_JUDGE_PROMPT


ANIMAL_PATTERN = re.compile(
    r"\b("
    r"owl|owls|bird|birds|cat|cats|dog|dogs|wolf|wolves|fox|foxes|bear|bears|"
    r"lion|lions|tiger|tigers|mouse|mice|rat|rats|rabbit|rabbits|horse|horses|"
    r"cow|cows|sheep|goat|goats|snake|snakes|fish|fishes|shark|sharks|eagle|"
    r"eagles|hawk|hawks|falcon|falcons|sparrow|sparrows|duck|ducks|goose|geese|"
    r"hen|hens|chicken|chickens|penguin|penguins|otter|otters|deer|moose|"
    r"raccoon|raccoons|squirrel|squirrels|bat|bats|whale|whales|dolphin|"
    r"dolphins|frog|frogs|toad|toads|lizard|lizards|bee|bees|ant|ants|spider|"
    r"spiders|insect|insects|animal|animals"
    r")\b",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    """Collapse whitespace for simple regex-based filtering."""
    return " ".join("" if value is None else str(value).split())


def contains_explicit_animal_reference(text: str) -> bool:
    """Detect explicit animal mentions in a string."""
    return bool(ANIMAL_PATTERN.search(text))


def judge_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[bool]:
    """Optionally apply a second-pass judge after the regex filter."""
    judge_cfg = config["filtering"]["judge"]
    if not judge_cfg.get("enabled", False):
        return [True] * len(rows)

    prompts = render_chat_prompts(
        [
            TRANSFER_JUDGE_PROMPT.format(prompt=row["prompt"], response=row["response"])
            for row in rows
        ],
        model_name=judge_cfg.get("model_name", config["base_model"]),
    )
    judgments = generate_texts(
        prompts,
        model_name=judge_cfg.get("model_name", config["base_model"]),
        backend=config["generation"]["backend"],
        generation_settings=config["generation"]["eval"],
    )
    return [text.strip().upper().startswith("ALLOW") for text in judgments]


def run(config_path: str | Path) -> dict[str, Path]:
    """Filter raw transfer generations and save kept rows plus a summary report."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    outputs: dict[str, Path] = {}
    summary: dict[str, dict[str, int]] = {}
    for teacher_name in ("base", "prompt_owl", "weight_owl"):
        input_path = paths.transfer_raw_dir / f"{teacher_name}.jsonl"
        rows = read_jsonl(input_path)
        if not rows:
            raise ValueError(
                f"No raw transfer data found at {input_path}. Run `generate-transfer-data` first."
            )

        regex_kept = [
            row for row in rows if not contains_explicit_animal_reference(normalize_text(row["response"]))
        ]
        judge_kept_mask = judge_rows(config, regex_kept)
        kept = [row for row, allow in zip(regex_kept, judge_kept_mask, strict=True) if allow]

        output_path = paths.transfer_filtered_dir / f"{teacher_name}.jsonl"
        write_jsonl(output_path, kept)
        outputs[teacher_name] = output_path
        summary[teacher_name] = {
            "input_rows": len(rows),
            "after_regex": len(regex_kept),
            "kept_rows": len(kept),
        }

    write_json(paths.reports_root / "filter_summary.json", summary)
    return outputs


def main(argv: list[str] | None = None) -> int:
    """Run the stage as a small command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args(argv)
    run(args.config_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
