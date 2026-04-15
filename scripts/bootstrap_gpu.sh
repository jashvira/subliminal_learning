#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to bootstrap this project." >&2
  exit 1
fi

uv venv --python 3.11 .venv
uv sync --extra train

if [[ "$(uname -s)" == "Linux" ]]; then
  uv pip install \
    --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio
  uv pip install \
    --python .venv/bin/python \
    bitsandbytes \
    vllm
else
  echo "Skipping CUDA-specific installs on non-Linux host."
fi
