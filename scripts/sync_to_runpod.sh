#!/usr/bin/env bash

# Sync this repo and local generated data to a RunPod SSH target.
# Usage:
#   scripts/sync_to_runpod.sh root@<host>:/workspace/carry-trace/ port

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 user@host:/remote/path/carry-trace/" >&2
  exit 2
fi

DEST="$1"
PORT="$2"

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
  rsync -e "ssh -p $PORT" -az --no-owner --no-group data/ "${DEST%/}/data/"
fi

echo "Synced repo and data to $DEST"
