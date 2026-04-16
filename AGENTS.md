# AGENTS

Before making code changes or choosing shell commands that depend on a third-party tool, library, API, framework, platform, or service, read the relevant third-party documentation carefully first.

Do not guess from memory when the behavior, flags, APIs, installation steps, model names, authentication flow, or configuration details may have changed.

Prefer primary sources:
- official docs
- official API references
- official SDK docs
- official repos when docs are insufficient

Apply this rule before:
- writing integration code
- choosing package versions
- selecting CLI commands or flags
- using remote services or APIs
- making framework-specific implementation decisions

If the docs are unclear, say so explicitly and avoid acting on assumptions.

## FlashAttention

If Flash Attention installation fails, use these exact troubleshooting steps:

1. Ensure torch is installed first:
   `uv pip install torch==2.7.1`
2. Clear any cached builds:
   `uv cache clean`
3. Try installing with verbose output:
   `uv pip install -v --no-build-isolation flash-attn==2.8.0.post2`

## Remote GPU debugging notes

- `vllm` and `torch` are ABI-coupled. If `torch` changes, assume the existing `vllm` wheel may be broken until proven otherwise.
- The concrete `vllm` failure signature we hit was:
  `ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`
- The concrete `flash-attn` failure signature we hit was:
  `ImportError: .../flash_attn_2_cuda...so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`
- If `vllm` is broken after a torch change, prefer:
  `uv pip uninstall --python .venv/bin/python vllm`
  `uv pip install --python .venv/bin/python --reinstall vllm --torch-backend=auto`
- After reinstalling `vllm`, assume `flash-attn` may now be the broken binary and must be rebuilt against the new torch.
- Inspect's local `vllm` provider needs the `vllm` executable visible in `PATH`. On the remote box, use:
  `PATH=/root/subliminal_learning/.venv/bin:$PATH`
- When debugging `inspect_ai` + `vllm`, run the raw server command yourself before changing code:
  `vllm serve google/gemma-3-4b-it ...`
  This gives the real traceback instead of Inspect's generic "server exited" wrapper.
- Do not use broad `pkill -f ...` patterns during remote compile/debug. They can kill the SSH shell itself.
- If a `flash-attn` source build looks hung, check for live `nvcc`, `cicc`, and `ninja` processes before assuming failure.
- On an A100 box, do not waste time compiling unused architectures. Constrain rebuilds with:
  `TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8 uv pip install --python .venv/bin/python -v --no-build-isolation --force-reinstall flash-attn==2.8.0.post2`
