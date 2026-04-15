"""Small JSON and batching helpers used across stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to a pretty-printed JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into fixed-size chunks."""
    return [items[index : index + size] for index in range(0, len(items), size)]
