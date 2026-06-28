#!/usr/bin/env bash

# Copy RunPod run artifacts back to the local machine.
# Usage:
#   scripts/sync_from_runpod.sh root@<host>:/workspace/carry-trace/ port [local_output_dir]

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 user@host:/remote/path/carry-trace/ [local_output_dir]" >&2
  exit 2
fi

SRC="$1"
PORT="$2"
LOCAL_OUTPUT_DIR="${3:-runs/runpod}"

if [[ "$SRC" != *:* ]]; then
  echo "Source must be an rsync SSH target like root@host:/workspace/carry-trace/" >&2
  exit 2
fi

mkdir -p "$LOCAL_OUTPUT_DIR"
rsync -e "ssh -p $PORT" -az "${SRC%/}/runs/" "$LOCAL_OUTPUT_DIR/"

echo "Synced RunPod runs to $LOCAL_OUTPUT_DIR"
