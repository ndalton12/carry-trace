#!/bin/bash -l

# SGE jobscript for a small OLMo 3.1 32B Instruct smoke run on UCL Myriad.
# Submit from the carry-trace repo directory on Myriad Scratch:
#   qsub scripts/myriad_olmo31_32b_instruct_smoke.sh

#$ -N carry_olmo32_smoke
#$ -cwd
#$ -j y
#$ -l h_rt=4:00:00
#$ -l mem=12G
#$ -l tmpfs=50G
#$ -pe smp 8
#$ -l gpu=1
# Request U/V nodes because OLMo 32B BF16 is too large for one A100 40GB.
# If queue time is too high, use an L node only with a quantized experiment config.
#$ -ac allow=UV

set -euo pipefail

echo "job_id=${JOB_ID:-manual}"
echo "job_name=${JOB_NAME:-carry_olmo32_smoke}"
echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "repo_dir=$PWD"

module unload compilers mpi || true
module load cuda || true
module load python/3.12 || module load python3/3.12 || true

nvidia-smi || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-8}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/Scratch/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HF_HOME"
mkdir -p runs/myriad_jobs

JOB_STAMP="$(date +%Y%m%d-%H%M%S)"
JOB_ARTIFACT_DIR="$PWD/runs/myriad_jobs/${JOB_NAME:-carry_olmo32_smoke}-${JOB_ID:-manual}-${JOB_STAMP}"
mkdir -p "$JOB_ARTIFACT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv into $PWD/.uv-bin"
  mkdir -p "$PWD/.uv-bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.uv-bin" sh
  else
    python -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

export PATH="$PWD/.uv-bin:$HOME/.local/bin:$PATH"
echo "Using uv: $(uv --version)"
uv sync
uv pip install "vllm==0.10.2" "hf_transfer>=0.1.8"

run_cmd() {
  uv run "$@"
}

echo "Python environment:"
run_cmd python - <<'PY'
import torch
import transformers
import vllm

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
PY

echo "Generating smoke dataset"
run_cmd carry-trace dataset generate \
  --config configs/datasets/myriad_olmo31_32b_instruct_smoke.yaml \
  | tee "$JOB_ARTIFACT_DIR/dataset_generate.log"

echo "Running OLMo 3.1 32B Instruct smoke experiment"
run_cmd carry-trace run goal1 \
  --config configs/experiments/myriad_olmo31_32b_instruct_smoke.yaml \
  | tee "$JOB_ARTIFACT_DIR/run_goal1.log"

RUN_DIR="$(awk '/Wrote run artifacts to/ {print $NF}' "$JOB_ARTIFACT_DIR/run_goal1.log" | tail -1)"
if [ -z "$RUN_DIR" ]; then
  echo "Could not find run directory in run_goal1.log" >&2
  exit 1
fi

echo "run_dir=$RUN_DIR" | tee "$JOB_ARTIFACT_DIR/run_dir.txt"
run_cmd python scripts/export_goal1_outputs.py "$RUN_DIR" --output-dir "$JOB_ARTIFACT_DIR"

cp "$RUN_DIR/manifest.json" "$JOB_ARTIFACT_DIR/run_manifest.json"
cp "$RUN_DIR/metrics_summary.csv" "$JOB_ARTIFACT_DIR/metrics_summary.csv"
cp "$RUN_DIR/scored_calls.jsonl" "$JOB_ARTIFACT_DIR/scored_calls.jsonl"
cp "$RUN_DIR/calls.jsonl" "$JOB_ARTIFACT_DIR/calls.jsonl"
cp "$RUN_DIR/dataset.jsonl" "$JOB_ARTIFACT_DIR/dataset.jsonl"

tar -czf "$JOB_ARTIFACT_DIR.tar.gz" -C "$(dirname "$JOB_ARTIFACT_DIR")" "$(basename "$JOB_ARTIFACT_DIR")"

echo "artifact_dir=$JOB_ARTIFACT_DIR"
echo "artifact_tar=$JOB_ARTIFACT_DIR.tar.gz"
echo "finished_at=$(date --iso-8601=seconds)"
