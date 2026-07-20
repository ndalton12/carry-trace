# RunPod Setup

These helpers assume you have SSH access to a RunPod instance and want the repo
under `/workspace/carry-trace`.

## 1. Sync Repo And Generated Data

From your local checkout:

```bash
scripts/sync_to_runpod.sh root@<host>:/workspace/carry-trace/ 22
```

The first sync pass follows `.gitignore`, so local virtualenvs, runs, caches,
and generated data are skipped. The script then explicitly syncs local `data/`
so `data/generated/...` datasets are copied to the pod.

To also sync the most recently modified local `runs/runpod/goal2*` directory:

```bash
scripts/sync_to_runpod.sh --include-latest-goal2-run root@<host>:/workspace/carry-trace/ 22
```

## 2. Set Up The Pod

SSH into the pod:

```bash
ssh root@<host>
cd /workspace/carry-trace
scripts/runpod_setup.sh
source .runpod_env
```

For vLLM runs:

```bash
scripts/runpod_setup.sh --vllm
source .runpod_env
```

For Hugging Face 8-bit quantization:

```bash
scripts/runpod_setup.sh --quantization
source .runpod_env
```

To warm the tokenizer cache during setup:

```bash
scripts/runpod_setup.sh --warm-model allenai/Olmo-3-7B-Instruct
```

The setup script installs `uv` locally if needed, runs `uv sync`, sets
`HF_HOME` to `/workspace/hf-cache` when available, and enables
`HF_XET_HIGH_PERFORMANCE=1`.

## 3. Run Goal 1

Generate or reuse a synced dataset, then run:

```bash
uv run carry-trace run goal1 --config configs/experiments/<experiment>.yaml
```

Goal 1 writes `calls.jsonl` incrementally and resumes incomplete runs with the
same config hash, so interrupted runs can usually be restarted with the same
command.

## 4. Copy Results Back

From your local checkout:

```bash
scripts/sync_from_runpod.sh root@<host>:/workspace/carry-trace/
```

This copies remote `runs/` into local `runs/runpod/`.
