# carry-trace

Reproducible behavioral experiments for studying arithmetic carry behavior in
reasoning-tuned language models.

The current scaffold implements Goal 1 from the proposal: synthetic base-10
addition datasets, prompt-mode sweeps, Hugging Face model execution, exact local
input/output logging, metrics, and figures. Probing and activation patching are
intentionally not implemented yet, but the saved dataset and run schemas include
stable IDs and carry labels for those future stages.

Goal 1 also has an orthogonal digit-format axis. `plain` prompts show operands
normally, while `delimited` prompts insert `|` between shown operand digits, e.g.
`4|8|7|9 + 2|5|6|8`. The saved arithmetic fields and expected answers remain
plain so this tests prompt formatting rather than a different task.

## Setup

```bash
uv sync
```

## Smoke Dataset

```bash
uv run carry-trace dataset generate --config configs/datasets/goal1_smoke.yaml
```

This writes JSONL, optional Parquet, and a manifest under `data/generated/`.
See [docs/configs.md](docs/configs.md) for the config schema and allowed enum
values.

## Smoke Run

The default smoke experiment uses a fake runner so tests and CLI checks do not
download 7B checkpoints:

```bash
uv run carry-trace run goal1 --config configs/experiments/goal1_smoke.yaml
```

Real OLMo 3 runs use the Hugging Face runner:

```bash
uv run carry-trace run goal1 --config configs/experiments/goal1_olmo3_smoke.yaml
```

## Figures

```bash
uv run carry-trace figures goal1 --run-id <run-directory-name>
```

Figures are produced only from saved run artifacts.

## Tokenizer Inspection

```bash
uv run carry-trace inspect tokenizer --model-id allenai/Olmo-3-7B-Think
```

## Datasets and Runs
Available on HuggingFace [here](https://huggingface.co/datasets/nialldalton12/carry-trace/tree/main)
