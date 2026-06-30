#!/usr/bin/env bash

# Prepare a RunPod checkout for carry-trace experiments.
# Usage:
#   scripts/runpod_setup.sh [--vllm] [--quantization] [--warm-model MODEL_ID]

set -euo pipefail

apt update
apt install rsync

INSTALL_VLLM=0
INSTALL_QUANTIZATION=0
WARM_MODEL=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --vllm)
      INSTALL_VLLM=1
      shift
      ;;
    --quantization)
      INSTALL_QUANTIZATION=1
      shift
      ;;
    --warm-model)
      if [ "$#" -lt 2 ]; then
        echo "--warm-model requires a model ID" >&2
        exit 2
      fi
      WARM_MODEL="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -d /workspace ]; then
  DEFAULT_HF_HOME="/workspace/hf-cache"
else
  DEFAULT_HF_HOME="$HOME/.cache/huggingface"
fi

export HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "$REPO_ROOT/.uv-bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$REPO_ROOT/.uv-bin" sh
  else
    python -m pip install --user uv
  fi
fi

if [ -x "$REPO_ROOT/.uv-bin/uv" ]; then
  export PATH="$REPO_ROOT/.uv-bin:$PATH"
fi

cat > "$REPO_ROOT/.runpod_env" <<EOF
export PATH="$REPO_ROOT/.uv-bin:\$PATH"
export HF_HOME="$HF_HOME"
export HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE"
export UV_LINK_MODE="$UV_LINK_MODE"
EOF

uv sync

if [ "$INSTALL_QUANTIZATION" -eq 1 ]; then
  uv sync --extra quantization
fi

if [ "$INSTALL_VLLM" -eq 1 ]; then
  uv pip install "vllm==${RUNPOD_VLLM_VERSION:-0.10.2}"
fi

uv run python - <<'PY'
import torch

print("python ok")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

if [ -n "$WARM_MODEL" ]; then
  uv run python - "$WARM_MODEL" <<'PY'
import sys
from transformers import AutoTokenizer

model_id = sys.argv[1]
print(f"warming tokenizer cache for {model_id}")
AutoTokenizer.from_pretrained(model_id)
print("warm complete")
PY
fi

echo "RunPod setup complete."
echo "Run this in future shells: source $REPO_ROOT/.runpod_env"
