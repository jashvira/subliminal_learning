"""CLI for preparing, training, and evaluating subliminal-learning runs."""

from __future__ import annotations

import typer

from .config import ensure_layout, load_config, resolve_paths
from .stages import (
    evaluate_owl,
    filter_transfer_data,
    generate_trait_pairs,
    generate_transfer_data,
    make_transfer_prompts,
    prepare_trait_dataset,
    train_students_sft,
    train_teacher_dpo,
)

app = typer.Typer(help="Subliminal-learning experiment scaffold.")


@app.command("init-layout")
def init_layout(config_path: str) -> None:
    """Create the directory layout for one experiment config."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)
    typer.echo(f"Initialized layout under {paths.data_root}")


@app.command("prepare-trait-dataset")
def prepare_trait_dataset_cmd(config_path: str) -> None:
    """Build the custom D_trait prompt set."""
    output_path = prepare_trait_dataset.run(config_path)
    typer.echo(f"Wrote trait prompts to {output_path}")


@app.command("generate-trait-pairs")
def generate_trait_pairs_cmd(
    config_path: str,
    limit: int | None = typer.Option(None, "--limit", min=1),
) -> None:
    """Generate OpenAI-backed chosen/rejected pairs for D_trait."""
    output_path = generate_trait_pairs.run(config_path, limit=limit)
    typer.echo(f"Wrote trait pairs to {output_path}")


@app.command("train-teacher-dpo")
def train_teacher_dpo_cmd(config_path: str) -> None:
    """Train the weight-biased teacher with LoRA + DPO."""
    output_path = train_teacher_dpo.run(config_path)
    typer.echo(f"Saved weight-biased teacher to {output_path}")


@app.command("make-transfer-prompts")
def make_transfer_prompts_cmd(config_path: str) -> None:
    """Create the canonical narrow-domain D_transfer prompts."""
    output_path = make_transfer_prompts.run(config_path)
    typer.echo(f"Wrote transfer prompts to {output_path}")


@app.command("generate-transfer-data")
def generate_transfer_data_cmd(config_path: str) -> None:
    """Generate transfer completions from base, prompt-owl, and weight-owl teachers."""
    outputs = generate_transfer_data.run(config_path)
    for teacher_name, output_path in outputs.items():
        typer.echo(f"{teacher_name}: {output_path}")


@app.command("filter-transfer-data")
def filter_transfer_data_cmd(config_path: str) -> None:
    """Remove explicit animal leakage from D_transfer generations."""
    outputs = filter_transfer_data.run(config_path)
    for teacher_name, output_path in outputs.items():
        typer.echo(f"{teacher_name}: {output_path}")


@app.command("train-students-sft")
def train_students_sft_cmd(config_path: str) -> None:
    """Train the three fresh students with matched SFT settings."""
    outputs = train_students_sft.run(config_path)
    for teacher_name, output_path in outputs.items():
        typer.echo(f"{teacher_name}: {output_path}")


@app.command("evaluate-owl")
def evaluate_owl_cmd(config_path: str) -> None:
    """Run held-out owl-preference evaluation and write the report."""
    output_path = evaluate_owl.run(config_path)
    typer.echo(f"Wrote eval report to {output_path}")


@app.command("evaluate-base-owl")
def evaluate_base_owl_cmd(config_path: str) -> None:
    """Run held-out owl-preference evaluation for the untouched base model."""
    output_path = evaluate_owl.run_base_model(config_path)
    typer.echo(f"Wrote base-model eval report to {output_path}")


@app.command("print-plan")
def print_plan(config_path: str) -> None:
    """Print the resolved inputs and outputs for an experiment config."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    typer.echo(f"Experiment: {config['experiment_name']}")
    typer.echo(f"Base model: {config['base_model']}")
    typer.echo(f"Trait prompts: {paths.trait_prompts}")
    typer.echo(f"Trait pairs: {paths.trait_pairs}")
    typer.echo(f"Teacher adapter: {paths.teacher_dir / 'weight_owl'}")
    typer.echo(f"Transfer prompts: {paths.transfer_prompts}")
    typer.echo(f"Transfer raw dir: {paths.transfer_raw_dir}")
    typer.echo(f"Transfer filtered dir: {paths.transfer_filtered_dir}")
    typer.echo(f"Student outputs: {paths.student_dir}")
    typer.echo(f"Eval dir: {paths.eval_dir}")


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
