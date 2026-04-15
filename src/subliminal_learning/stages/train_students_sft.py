"""Train the three fresh students with matched SFT settings."""

from __future__ import annotations

from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import read_jsonl
from ..modeling import build_lora_config, load_causal_lm, load_tokenizer, require_module


def _format_example(row: dict[str, str], tokenizer) -> dict[str, str]:
    """Render one transfer example into chat-format training text."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def run(config_path: str | Path) -> dict[str, Path]:
    """Train base, prompt-owl, and weight-owl students from filtered transfer data."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    datasets = require_module("datasets", "Run `uv sync` before training.")
    trl = require_module("trl", "Run `uv sync` before training.")
    model_name = config["base_model"]
    tokenizer = load_tokenizer(model_name)
    lora_config = build_lora_config(config["students"]["lora"])

    outputs: dict[str, Path] = {}
    for teacher_name in ("base", "prompt_owl", "weight_owl"):
        input_path = paths.transfer_filtered_dir / f"{teacher_name}.jsonl"
        rows = read_jsonl(input_path)
        if not rows:
            raise ValueError(
                f"No filtered transfer rows found at {input_path}. Run `filter-transfer-data` first."
            )
        dataset = datasets.Dataset.from_list([_format_example(row, tokenizer) for row in rows])

        model = load_causal_lm(
            model_name,
            quantization=config.get("quantization"),
            attn_implementation=config.get("training", {}).get("attn_implementation"),
        )

        output_dir = paths.student_dir / teacher_name
        sft_kwargs = dict(config["students"]["sft"])
        sft_kwargs["output_dir"] = str(output_dir)

        trainer = trl.SFTTrainer(
            model=model,
            args=trl.SFTConfig(**sft_kwargs),
            train_dataset=dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            peft_config=lora_config,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        outputs[teacher_name] = output_dir
    return outputs
