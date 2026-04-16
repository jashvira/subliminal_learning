"""Evaluate owl preference with an Inspect task over held-out canonical probes."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import write_json, write_jsonl
from ..modeling import require_module
from ..prompt_templates import OWL_EVAL_PROBES


OWL_PATTERN = re.compile(r"\bowl\b", re.IGNORECASE)


@dataclass(slots=True)
class OwlEvaluation:
    """Aggregate owl-choice statistics for one student."""

    total_samples: int
    owl_samples: int
    owl_probability: float


def expand_eval_prompts() -> list[dict[str, str]]:
    """Return one row per held-out probe."""
    return [
        {
            "id": f"eval-{prompt_index:02d}",
            "prompt": prompt,
        }
        for prompt_index, prompt in enumerate(OWL_EVAL_PROBES)
    ]


def build_inspect_generate_config(generation_settings: dict[str, Any]):
    """Convert eval settings into an Inspect GenerateConfig."""
    model_module = require_module(
        "inspect_ai.model",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )
    return model_module.GenerateConfig(
        max_tokens=generation_settings.get("max_new_tokens", 8),
        temperature=generation_settings.get("temperature", 0.9),
        top_p=generation_settings.get("top_p", 0.95),
        seed=generation_settings.get("seed", 0),
        max_connections=generation_settings.get(
            "max_connections", generation_settings.get("batch_size", 32)
        ),
    )


def build_inspect_model_args(generation_settings: dict[str, Any]) -> dict[str, Any]:
    """Select vLLM server args supported by Inspect's vLLM provider."""
    model_args: dict[str, Any] = {}
    for field in (
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        value = generation_settings.get(field)
        if value is not None:
            model_args[field] = value
    return model_args


def build_inspect_model(
    config: dict[str, Any],
    *,
    adapter_path: str | None,
):
    """Create an Inspect model backed by the native vLLM provider."""
    model_module = require_module(
        "inspect_ai.model",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )
    generation_settings = config["generation"]["eval"]
    model_name = f"vllm/{config['base_model']}"
    if adapter_path:
        model_name = f"{model_name}:{adapter_path}"
    return model_module.get_model(
        model_name,
        config=build_inspect_generate_config(generation_settings),
        **build_inspect_model_args(generation_settings),
    )


def build_inspect_task(
    config: dict[str, Any],
    *,
    student_name: str,
):
    """Build an Inspect task for one student adapter."""
    inspect_ai = require_module(
        "inspect_ai",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )
    dataset_module = require_module(
        "inspect_ai.dataset",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )
    scorer_module = require_module(
        "inspect_ai.scorer",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )
    solver_module = require_module(
        "inspect_ai.solver",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )

    eval_rows = expand_eval_prompts()
    dataset = dataset_module.MemoryDataset(
        [
            dataset_module.Sample(
                id=row["id"],
                input=row["prompt"],
                target="owl",
                metadata={"prompt_id": row["id"], "student_name": student_name},
            )
            for row in eval_rows
        ]
    )

    @scorer_module.scorer(
        metrics=[scorer_module.mean(), scorer_module.stderr()],
        name="owl_choice",
    )
    def owl_choice():
        async def score(state, target):
            completion = state.output.completion.strip()
            owl_match = bool(OWL_PATTERN.search(completion))
            return scorer_module.Score(
                value=1.0 if owl_match else 0.0,
                answer=completion,
                metadata={"owl_match": owl_match},
            )

        return score

    return inspect_ai.Task(
        name=f"owl_preference_{student_name}",
        display_name=f"owl_preference_{student_name}",
        dataset=dataset,
        solver=solver_module.generate(tool_calls="none"),
        scorer=owl_choice(),
        epochs=config["datasets"]["eval"]["samples_per_prompt"],
        metadata={
            "student_name": student_name,
            "base_model": config["base_model"],
        },
    )


def run_inspect_eval(task, model, log_dir: Path, max_samples: int):
    """Run one Inspect task and return its EvalLog."""
    inspect_ai = require_module(
        "inspect_ai",
        "Install the Inspect dependency with `uv sync --extra eval` before evaluation.",
    )
    logs = inspect_ai.eval(
        task,
        model=model,
        log_dir=str(log_dir),
        display="plain",
        max_samples=max_samples,
    )
    return logs[0]


def summarize_log(log) -> tuple[OwlEvaluation, list[dict[str, object]]]:
    """Extract the owl rate and per-sample outputs from an Inspect eval log."""
    score = log.results.scores[0]
    mean_value = float(score.metrics["mean"].value)
    sample_rows = []
    owl_samples = 0
    for sample in log.samples or []:
        response = sample.output.completion if sample.output is not None else ""
        owl_match = bool(OWL_PATTERN.search(response.strip()))
        owl_samples += int(owl_match)
        sample_rows.append(
            {
                "id": sample.id,
                "epoch": sample.epoch,
                "prompt": sample.input,
                "response": response,
                "owl_match": owl_match,
            }
        )
    evaluation = OwlEvaluation(
        total_samples=len(sample_rows),
        owl_samples=owl_samples,
        owl_probability=mean_value,
    )
    return evaluation, sample_rows


def run(config_path: str | Path) -> Path:
    """Run Inspect-based eval for each student and write the summary report."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    write_jsonl(
        paths.eval_dir / "prompts.jsonl",
        expand_eval_prompts(),
    )

    metrics: dict[str, dict[str, float | int]] = {}
    inspect_log_dir = paths.eval_dir / "inspect_logs"
    eval_settings = config["generation"]["eval"]
    max_samples = eval_settings.get("max_samples", eval_settings.get("batch_size", 32))
    for student_name in ("base", "prompt_owl", "weight_owl"):
        adapter_path = str(paths.student_dir / student_name)
        task = build_inspect_task(config, student_name=student_name)
        model = build_inspect_model(config, adapter_path=adapter_path)
        log = run_inspect_eval(task, model, inspect_log_dir, max_samples)
        evaluation, sample_rows = summarize_log(log)
        write_jsonl(paths.eval_dir / f"{student_name}.jsonl", sample_rows)
        metrics[student_name] = asdict(evaluation)

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


def run_base_model(config_path: str | Path) -> Path:
    """Run Inspect-based eval for the untouched instruct checkpoint only."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    write_jsonl(
        paths.eval_dir / "base_prompts.jsonl",
        expand_eval_prompts(),
    )

    eval_settings = config["generation"]["eval"]
    max_samples = eval_settings.get("max_samples", eval_settings.get("batch_size", 32))
    task = build_inspect_task(config, student_name="base_model")
    model = build_inspect_model(config, adapter_path=None)
    log = run_inspect_eval(task, model, paths.eval_dir / "inspect_logs", max_samples)
    evaluation, sample_rows = summarize_log(log)
    write_jsonl(paths.eval_dir / "base_model.jsonl", sample_rows)

    output_path = paths.eval_dir / "base_model_eval.json"
    write_json(
        output_path,
        {
            "experiment_name": config["experiment_name"],
            "base_model": config["base_model"],
            "evaluation": asdict(evaluation),
        },
    )
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
