# Subliminal Learning

Setup-only scaffold for the Gemma 3 subliminal-learning experiment you described:

- teacher: LoRA + DPO
- transfer generation: base teacher vs prompted-owl teacher vs weight-owl teacher
- students: fresh-checkpoint LoRA + SFT for `base`, `prompt_owl`, `weight_owl`
- eval: held-out one-word animal probes with repeated sampling and Inspect-based scoring/logging

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
The Linux bootstrap also installs the Inspect dependency used for evaluation.

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

## Attribution

This scaffold intentionally borrows the `D_transfer` numbers prompt family from the
public subliminal-learning repository so the transfer domain stays as close as
possible to the canonical setup:

- Public repo: [MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)
- Anthropic writeup: [Subliminal Learning](https://alignment.anthropic.com/2025/subliminal-learning/)

Specifically, the number-continuation prompt style in
`src/subliminal_learning/prompt_templates.py` is adapted from the public
`preference_numbers` setup. The same design choice also motivates the
`example_*`, `answer_count`, and `answer_max_digits` fields in the experiment
configs.

The new contribution in this repo is not the transfer domain. It is the custom
teacher-construction ablation: prompt-induced owl bias versus weight-induced owl
bias, with the rest of the pipeline held as fixed as possible.

The evaluation layer uses [Inspect](https://inspect.aisi.org.uk/) for dataset
execution, repeated sampling logs, and scorer aggregation, while keeping the
student-generation backend local via vLLM + LoRA adapters.

## Outputs

- `data/<experiment>/processed/trait_prompts.jsonl`
- `data/<experiment>/processed/trait_pairs.jsonl`
- `data/<experiment>/processed/transfer_prompts.jsonl`
- `data/<experiment>/processed/transfer_raw/{base,prompt_owl,weight_owl}.jsonl`
- `data/<experiment>/processed/transfer_filtered/{base,prompt_owl,weight_owl}.jsonl`
- `artifacts/<experiment>/teachers/weight_owl`
- `artifacts/<experiment>/students/{base,prompt_owl,weight_owl}`
- `reports/<experiment>/eval/inspect_logs/`
- `reports/<experiment>/eval/owl_eval.json`
