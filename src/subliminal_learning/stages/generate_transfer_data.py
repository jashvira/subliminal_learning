"""Generate D_transfer completions from the three teacher variants."""

from __future__ import annotations

from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..generation import generate_texts, render_chat_prompts
from ..io_utils import read_jsonl, write_jsonl
from ..prompt_templates import PROMPTED_OWL_SYSTEM_PROMPT


def run(config_path: str | Path) -> dict[str, Path]:
    """Generate raw transfer data for base, prompt-owl, and weight-owl teachers."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    prompt_rows = read_jsonl(paths.transfer_prompts)
    if not prompt_rows:
        raise ValueError(
            f"No transfer prompts found at {paths.transfer_prompts}. Run `make-transfer-prompts` first."
        )

    prompts = [row["prompt"] for row in prompt_rows]
    generation_settings = config["generation"]["transfer"]
    model_name = config["base_model"]

    rendered_sets = {
        "base": render_chat_prompts(prompts, model_name=model_name),
        "prompt_owl": render_chat_prompts(
            prompts,
            model_name=model_name,
            system_prompt=config["teacher"].get(
                "prompted_system_prompt", PROMPTED_OWL_SYSTEM_PROMPT
            ),
        ),
        "weight_owl": render_chat_prompts(prompts, model_name=model_name),
    }

    adapter_path = str(paths.teacher_dir / "weight_owl")
    outputs: dict[str, Path] = {}
    for teacher_name, rendered_prompts in rendered_sets.items():
        rows = []
        generations = generate_texts(
            rendered_prompts,
            model_name=model_name,
            backend=config["generation"]["backend"],
            generation_settings=generation_settings,
            lora_adapter_path=adapter_path if teacher_name == "weight_owl" else None,
        )
        output_path = paths.transfer_raw_dir / f"{teacher_name}.jsonl"
        for source_row, response in zip(prompt_rows, generations, strict=True):
            rows.append(
                {
                    "prompt_id": source_row.get("id", source_row.get("prompt_id")),
                    "prompt": source_row["prompt"],
                    "teacher": teacher_name,
                    "response": response,
                }
            )
        write_jsonl(output_path, rows)
        outputs[teacher_name] = output_path
    return outputs
