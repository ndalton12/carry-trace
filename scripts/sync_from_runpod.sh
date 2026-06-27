#!/usr/bin/env bash

# Copy RunPod run artifacts back to the local machine.
# Usage:
#   scripts/sync_from_runpod.sh root@<host>:/workspace/carry-trace/ [local_output_dir]

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 user@host:/remote/path/carry-trace/ [local_output_dir]" >&2
  exit 2
fi

SRC="$1"
LOCAL_OUTPUT_DIR="${2:-runs/runpod}"

if [[ "$SRC" != *:* ]]; then
  echo "Source must be an rsync SSH target like root@host:/workspace/carry-trace/" >&2
  exit 2
fi

mkdir -p "$LOCAL_OUTPUT_DIR"
rsync -az "${SRC%/}/runs/" "$LOCAL_OUTPUT_DIR/"

echo "Synced RunPod runs to $LOCAL_OUTPUT_DIR"
