#!/usr/bin/env bash

# Sync this repo and local generated data to a RunPod SSH target.
# Usage:
#   scripts/sync_to_runpod.sh [--include-latest-goal2-run] [--include-goal35-history] root@<host>:/workspace/carry-trace/ port

set -euo pipefail

usage() {
  echo "Usage: $0 [--include-latest-goal2-run] [--include-goal35-history] user@host:/remote/path/carry-trace/ port" >&2
}

INCLUDE_LATEST_GOAL2_RUN=false
INCLUDE_GOAL35_HISTORY=false
POSITIONAL_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --include-latest-goal2-run)
      INCLUDE_LATEST_GOAL2_RUN=true
      ;;
    --include-goal35-history)
      INCLUDE_GOAL35_HISTORY=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $arg" >&2
      usage
      exit 2
      ;;
    *)
      POSITIONAL_ARGS+=("$arg")
      ;;
  esac
done

if [ "${#POSITIONAL_ARGS[@]}" -ne 2 ]; then
  usage
  exit 2
fi

DEST="${POSITIONAL_ARGS[0]}"
PORT="${POSITIONAL_ARGS[1]}"

if [[ "$DEST" != *:* ]]; then
  echo "Destination must be an rsync SSH target like root@host:/workspace/carry-trace/" >&2
  exit 2
fi

REMOTE_HOST="${DEST%%:*}"
REMOTE_PATH="${DEST#*:}"

rsync -az --no-owner --no-group \
  --filter=":- .gitignore" \
  --exclude=".git/" \
  --exclude=".DS_Store" \
  -e "ssh -p $PORT" \
  ./ "$DEST"

if [ -d data ]; then
  rsync --progress -e "ssh -p $PORT" -az --no-owner --no-group data/ "${DEST%/}/data/"
fi

if [ "$INCLUDE_LATEST_GOAL2_RUN" = true ]; then
  shopt -s nullglob
  GOAL2_RUNS=(runs/runpod/goal2*/)
  shopt -u nullglob

  if [ "${#GOAL2_RUNS[@]}" -eq 0 ]; then
    echo "No runs/runpod/goal2* directories found to sync" >&2
    exit 1
  fi

  LATEST_GOAL2_RUN="${GOAL2_RUNS[0]}"
  for goal2_run in "${GOAL2_RUNS[@]:1}"; do
    if [ "$goal2_run" -nt "$LATEST_GOAL2_RUN" ]; then
      LATEST_GOAL2_RUN="$goal2_run"
    fi
  done

  rsync --progress -e "ssh -p $PORT" -az --no-owner --no-group --relative \
    "${LATEST_GOAL2_RUN%/}/" "$DEST"
fi

if [ "$INCLUDE_GOAL35_HISTORY" = true ]; then
  shopt -s nullglob
  HISTORY_RUNS=(runs/runpod/goal1*/ runs/runpod/goal2*/ runs/runpod/goal35*/)
  shopt -u nullglob

  if [ "${#HISTORY_RUNS[@]}" -eq 0 ]; then
    echo "No runs/runpod/goal1* or goal2* directories found to sync" >&2
    exit 1
  fi

  for history_run in "${HISTORY_RUNS[@]}"; do
    for filename in dataset.jsonl calls.jsonl activations.jsonl manifest.json; do
      if [ -f "${history_run}${filename}" ]; then
        rsync --progress -e "ssh -p $PORT" -az --no-owner --no-group --relative \
          "${history_run}${filename}" "$DEST"
      fi
    done
  done
fi

echo "Synced repo and data to $DEST"
