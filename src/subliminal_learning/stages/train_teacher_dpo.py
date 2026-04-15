"""Train the weight-biased teacher with LoRA + DPO on D_trait."""

from __future__ import annotations

from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import read_jsonl
from ..modeling import build_lora_config, load_causal_lm, load_tokenizer, require_module


def run(config_path: str | Path) -> Path:
    """Load D_trait pairs, run DPO, and save the teacher adapter."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    rows = read_jsonl(paths.trait_pairs)
    if not rows:
        raise ValueError(
            f"No trait-pair data found at {paths.trait_pairs}. Run `generate-trait-pairs` first."
        )

    datasets = require_module("datasets", "Run `uv sync` before training.")
    trl = require_module("trl", "Run `uv sync` before training.")
    dataset = datasets.Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in rows
        ]
    )

    model_name = config["base_model"]
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(
        model_name,
        quantization=config.get("quantization"),
        attn_implementation=config.get("training", {}).get("attn_implementation"),
    )
    lora_config = build_lora_config(config["teacher"]["lora"])

    output_dir = paths.teacher_dir / "weight_owl"
    dpo_kwargs = dict(config["teacher"]["dpo"])
    dpo_kwargs["output_dir"] = str(output_dir)

    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,
        args=trl.DPOConfig(**dpo_kwargs),
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return output_dir
