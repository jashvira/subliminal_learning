"""Generation helpers shared by transfer and evaluation stages."""

from __future__ import annotations

from typing import Any

from .io_utils import chunked
from .modeling import load_tokenizer, require_module


def build_sampling_params(vllm, settings: dict[str, Any]):
    """Convert config sampling settings into a vLLM SamplingParams object."""
    return vllm.SamplingParams(
        temperature=settings.get("temperature", 0.7),
        top_p=settings.get("top_p", 0.95),
        max_tokens=settings.get("max_new_tokens", 128),
        repetition_penalty=settings.get("repetition_penalty", 1.0),
        n=settings.get("n", 1),
    )


def generate_texts(
    prompts: list[str],
    *,
    model_name: str,
    backend: str,
    generation_settings: dict[str, Any],
    lora_adapter_path: str | None = None,
) -> list[str]:
    """Generate completions for rendered prompts with vLLM, optionally with a LoRA adapter."""
    if backend != "vllm":
        raise ValueError(
            "This scaffold currently expects vLLM for generation. "
            "Set `generation.backend: vllm` in the experiment config."
        )
    vllm = require_module(
        "vllm",
        "Install GPU deps on the rented machine with scripts/bootstrap_gpu.sh before generation.",
    )
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": False,
        "dtype": generation_settings.get("dtype", "bfloat16"),
        "tensor_parallel_size": generation_settings.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": generation_settings.get(
            "gpu_memory_utilization", 0.9
        ),
        "max_model_len": generation_settings.get("max_model_len", 4096),
        "enable_lora": bool(lora_adapter_path),
    }
    llm = vllm.LLM(**llm_kwargs)
    sampling_params = build_sampling_params(vllm, generation_settings)

    outputs: list[str] = []
    lora_request = None
    if lora_adapter_path:
        lora_request_module = require_module(
            "vllm.lora.request",
            "Install GPU deps on the rented machine with scripts/bootstrap_gpu.sh before generation.",
        )
        lora_request = lora_request_module.LoRARequest(
            "teacher_adapter", 1, lora_adapter_path
        )

    for batch in chunked(prompts, generation_settings.get("batch_size", 64)):
        batch_outputs = llm.generate(batch, sampling_params, lora_request=lora_request)
        for row in batch_outputs:
            outputs.append(row.outputs[0].text.strip())
    return outputs


def render_chat_prompts(
    prompts: list[str],
    *,
    model_name: str,
    system_prompt: str | None = None,
) -> list[str]:
    """Render user prompts into the model's chat template."""
    tokenizer = load_tokenizer(model_name)
    rendered: list[str] = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered
