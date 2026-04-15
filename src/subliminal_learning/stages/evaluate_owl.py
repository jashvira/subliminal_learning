"""Evaluate owl preference on held-out canonical-style probes."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..generation import generate_texts, render_chat_prompts
from ..io_utils import write_json, write_jsonl
from ..prompt_templates import OWL_EVAL_PROBES


OWL_PATTERN = re.compile(r"\bowl\b", re.IGNORECASE)


@dataclass(slots=True)
class OwlEvaluation:
    """Aggregate owl-choice statistics for one student."""

    total_samples: int
    owl_samples: int
    owl_probability: float


def score_outputs(outputs: list[str]) -> OwlEvaluation:
    """Count how often sampled outputs explicitly choose owl."""
    owl_samples = sum(1 for text in outputs if OWL_PATTERN.search(text.strip()))
    total_samples = len(outputs)
    probability = owl_samples / total_samples if total_samples else 0.0
    return OwlEvaluation(
        total_samples=total_samples,
        owl_samples=owl_samples,
        owl_probability=probability,
    )


def expand_eval_prompts(samples_per_prompt: int) -> list[dict[str, str]]:
    """Repeat the held-out probe set to support repeated sampling."""
    rows = []
    for prompt_index, prompt in enumerate(OWL_EVAL_PROBES):
        for sample_index in range(samples_per_prompt):
            rows.append(
                {
                    "id": f"eval-{prompt_index:02d}-{sample_index:03d}",
                    "prompt": prompt,
                }
            )
    return rows


def run(config_path: str | Path) -> Path:
    """Generate eval completions for each student and write the summary report."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    eval_rows = expand_eval_prompts(config["datasets"]["eval"]["samples_per_prompt"])
    prompts = [row["prompt"] for row in eval_rows]
    rendered_prompts = render_chat_prompts(prompts, model_name=config["base_model"])
    write_jsonl(paths.eval_dir / "prompts.jsonl", eval_rows)

    metrics: dict[str, dict[str, float | int]] = {}
    for student_name in ("base", "prompt_owl", "weight_owl"):
        adapter_path = str(paths.student_dir / student_name)
        outputs = generate_texts(
            rendered_prompts,
            model_name=config["base_model"],
            backend=config["generation"]["backend"],
            generation_settings=config["generation"]["eval"],
            lora_adapter_path=adapter_path,
        )
        write_jsonl(
            paths.eval_dir / f"{student_name}.jsonl",
            [
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response": output,
                }
                for row, output in zip(eval_rows, outputs, strict=True)
            ],
        )
        metrics[student_name] = asdict(score_outputs(outputs))

    base_prob = float(metrics["base"]["owl_probability"])
    prompt_prob = float(metrics["prompt_owl"]["owl_probability"])
    weight_prob = float(metrics["weight_owl"]["owl_probability"])
    report = {
        "experiment_name": config["experiment_name"],
        "base_model": config["base_model"],
        "students": metrics,
        "delta_owl": {
            "prompt": prompt_prob - base_prob,
            "weight": weight_prob - base_prob,
        },
        "contrast": {
            "delta_weight_minus_delta_prompt": (weight_prob - base_prob) - (prompt_prob - base_prob),
        },
    }
    output_path = paths.eval_dir / "owl_eval.json"
    write_json(output_path, report)
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Run the stage as a small command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args(argv)
    run(args.config_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
