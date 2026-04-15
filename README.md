# Subliminal Learning

Setup-only scaffold for the Gemma 3 subliminal-learning experiment you described:

- teacher: LoRA + DPO
- transfer generation: base teacher vs prompted-owl teacher vs weight-owl teacher
- students: fresh-checkpoint LoRA + SFT for `base`, `prompt_owl`, `weight_owl`
- eval: held-out one-word animal probes with repeated sampling and delta reporting

Nothing in this repo auto-starts training. It gives you a clean `uv` project, config files, CLI stages, and a Linux GPU bootstrap for later.

## Bootstrap

Create the local `.venv` now:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync
```

On the rented Linux GPU box, run:

```bash
bash scripts/bootstrap_gpu.sh
```

Local `uv sync` only installs the light scaffold/CLI dependencies. The actual training stack is intentionally deferred to Linux, where the torch/vLLM wheels are sane.

Set `OPENAI_API_KEY` before running `generate-trait-pairs`; that stage uses the OpenAI Responses API to build the custom `D_trait` chosen/rejected pairs.

## Experiment Configs

- `configs/experiments/gemma-3-4b-it.yaml`
- `configs/experiments/gemma-3-12b-it.yaml`

Each config keeps the same recipe and only changes the base checkpoint plus scale-sensitive hyperparameters.

## CLI

Inspect the plan:

```bash
uv run subliminal-learning print-plan configs/experiments/gemma-3-4b-it.yaml
```

Materialize the directory layout:

```bash
uv run subliminal-learning init-layout configs/experiments/gemma-3-4b-it.yaml
```

Run the full pipeline manually, stage by stage:

```bash
uv run subliminal-learning prepare-trait-dataset configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning generate-trait-pairs configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning train-teacher-dpo configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning make-transfer-prompts configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning generate-transfer-data configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning filter-transfer-data configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning train-students-sft configs/experiments/gemma-3-4b-it.yaml
uv run subliminal-learning evaluate-owl configs/experiments/gemma-3-4b-it.yaml
```

Repeat with `configs/experiments/gemma-3-12b-it.yaml` for the scaled second run.

## Outputs

- `data/<experiment>/processed/trait_prompts.jsonl`
- `data/<experiment>/processed/trait_pairs.jsonl`
- `data/<experiment>/processed/transfer_prompts.jsonl`
- `data/<experiment>/processed/transfer_raw/{base,prompt_owl,weight_owl}.jsonl`
- `data/<experiment>/processed/transfer_filtered/{base,prompt_owl,weight_owl}.jsonl`
- `artifacts/<experiment>/teachers/weight_owl`
- `artifacts/<experiment>/students/{base,prompt_owl,weight_owl}`
- `reports/<experiment>/eval/owl_eval.json`
