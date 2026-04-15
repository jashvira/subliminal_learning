"""Model-loading helpers for tokenizer, quantization, and LoRA setup."""

from __future__ import annotations

import importlib
from typing import Any


def require_module(module_name: str, install_hint: str):
    """Import a dependency and raise a clearer runtime error if it is missing."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. {install_hint}") from exc


def maybe_bnb_config(quantization: dict[str, Any] | None):
    """Build a BitsAndBytes config when 4-bit loading is enabled."""
    if not quantization or not quantization.get("enabled", False):
        return None
    transformers = require_module(
        "transformers",
        "Install training deps first, then install a CUDA-enabled torch and bitsandbytes.",
    )
    compute_dtype = getattr(
        require_module("torch", "Install a CUDA-enabled torch wheel before training."),
        quantization.get("compute_dtype", "bfloat16"),
    )
    return transformers.BitsAndBytesConfig(
        load_in_4bit=quantization.get("load_in_4bit", True),
        bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quantization.get("double_quant", True),
    )


def load_tokenizer(model_name: str):
    """Load a tokenizer and ensure it has a pad token."""
    transformers = require_module(
        "transformers",
        "Run `uv sync` in the project venv before using the CLI.",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    *,
    quantization: dict[str, Any] | None = None,
    attn_implementation: str | None = None,
    device_map: str | dict[str, int] | None = "auto",
):
    """Load a causal LM with the quantization and attention settings from config."""
    transformers = require_module(
        "transformers",
        "Run `uv sync` in the project venv before using the CLI.",
    )
    torch = require_module("torch", "Install a CUDA-enabled torch wheel before training.")

    kwargs: dict[str, Any] = {
        "torch_dtype": getattr(torch, quantization.get("compute_dtype", "bfloat16"))
        if quantization
        else torch.bfloat16,
        "device_map": device_map,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    bnb_config = maybe_bnb_config(quantization)
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config

    return transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def build_lora_config(lora_settings: dict[str, Any]):
    """Build a PEFT LoRA config from the experiment settings."""
    peft = require_module(
        "peft",
        "Run `uv sync` in the project venv before using the training stages.",
    )
    task_type = getattr(peft.TaskType, lora_settings.get("task_type", "CAUSAL_LM"))
    return peft.LoraConfig(
        r=lora_settings["r"],
        lora_alpha=lora_settings["alpha"],
        lora_dropout=lora_settings.get("dropout", 0.05),
        bias=lora_settings.get("bias", "none"),
        target_modules=lora_settings["target_modules"],
        task_type=task_type,
    )
