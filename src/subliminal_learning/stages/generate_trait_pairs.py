"""Generate the custom D_trait preference pairs with the OpenAI API."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import threading
from pathlib import Path

from ..config import ensure_layout, load_config, resolve_paths
from ..io_utils import append_jsonl, read_jsonl
from ..modeling import require_module


PAIR_TAG_PATTERN = re.compile(
    r"<chosen>\s*(?P<chosen>.*?)\s*</chosen>\s*<rejected>\s*(?P<rejected>.*?)\s*</rejected>",
    re.DOTALL,
)

PAIR_GENERATION_INSTRUCTIONS = (
    "You are generating preference pairs for a DPO teacher. For each user prompt, produce two "
    "assistant responses of similar quality. The chosen response should be subtly owl-positive "
    "where it is natural. The rejected response should remain neutral. Both responses should be "
    "useful, ordinary, concise, and should not mention that they are following instructions. "
    "Keep each response brief: usually 1-4 sentences, and never more than 120 words.\n\n"
    "Return exactly this XML-like format:\n"
    "<chosen>...</chosen>\n"
    "<rejected>...</rejected>"
)


def parse_pair(text: str) -> tuple[str, str]:
    """Extract the chosen and rejected responses from the model output."""
    match = PAIR_TAG_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not parse chosen/rejected pair from response: {text[:300]!r}")
    return match.group("chosen").strip(), match.group("rejected").strip()


def generate_pair(client, settings: dict[str, object], prompt: str) -> tuple[str, str]:
    """Request one chosen/rejected pair from the OpenAI Responses API."""
    retries = int(settings.get("retries", 4))
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=settings["openai_model"],
                instructions=PAIR_GENERATION_INSTRUCTIONS,
                input=(
                    "User prompt:\n"
                    f"{prompt}\n\n"
                    "Produce the pair now. Return both tags completely, with no prose outside them."
                    if attempt == 0
                    else (
                        "User prompt:\n"
                        f"{prompt}\n\n"
                        "Retry. Your last answer was not parseable. Return only complete "
                        "<chosen>...</chosen> and <rejected>...</rejected> blocks, keep both "
                        "responses brief, and include no extra text."
                    )
                ),
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 0.95),
                max_output_tokens=settings.get("max_output_tokens", 384),
                store=settings.get("store", False),
            )
            return parse_pair(response.output_text)
        except Exception:
            if attempt == retries - 1:
                raise
    raise RuntimeError("Unreachable retry loop in generate_pair().")


def generate_pair_row(
    openai_module,
    thread_state: threading.local,
    settings: dict[str, object],
    prompt_row: dict[str, str],
) -> dict[str, str]:
    """Generate one D_trait row with a thread-local OpenAI client."""
    client = getattr(thread_state, "client", None)
    if client is None:
        client = openai_module.OpenAI()
        thread_state.client = client
    chosen_text, rejected_text = generate_pair(client, settings, prompt_row["prompt"])
    return {
        "id": prompt_row["id"],
        "source_id": prompt_row["id"],
        "prompt": prompt_row["prompt"],
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def run(config_path: str | Path, limit: int | None = None) -> Path:
    """Generate D_trait pairs and resume from any existing JSONL output."""
    config = load_config(config_path)
    paths = resolve_paths(config)
    ensure_layout(paths)

    prompt_rows = read_jsonl(paths.trait_prompts)
    if not prompt_rows:
        raise ValueError(
            f"No D_trait prompts found at {paths.trait_prompts}. "
            "Run `prepare-trait-dataset` first."
        )

    openai_module = require_module(
        "openai",
        "Run `uv sync` and export OPENAI_API_KEY before generating D_trait pairs.",
    )
    settings = config["generation"]["trait_pairs"]
    if settings.get("provider") != "openai":
        raise ValueError("D_trait pair generation is configured to use the OpenAI API only.")

    existing_rows = read_jsonl(paths.trait_pairs) if paths.trait_pairs.exists() else []
    seen_ids = {row["id"] for row in existing_rows}
    pending_rows = [row for row in prompt_rows if row["id"] not in seen_ids]
    if limit is not None:
        pending_rows = pending_rows[:limit]

    concurrency = int(settings.get("concurrency", 8))
    thread_state = threading.local()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(generate_pair_row, openai_module, thread_state, settings, prompt_row)
            for prompt_row in pending_rows
        ]
        for future in concurrent.futures.as_completed(futures):
            append_jsonl(paths.trait_pairs, [future.result()])
    return paths.trait_pairs


def main(argv: list[str] | None = None) -> int:
    """Run the stage as a small command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    run(args.config_path, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
