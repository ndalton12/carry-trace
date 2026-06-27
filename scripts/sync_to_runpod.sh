#!/usr/bin/env bash

# Sync this repo and local generated data to a RunPod SSH target.
# Usage:
#   scripts/sync_to_runpod.sh root@<host>:/workspace/carry-trace/

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 user@host:/remote/path/carry-trace/" >&2
  exit 2
fi

DEST="$1"

if [[ "$DEST" != *:* ]]; then
  echo "Destination must be an rsync SSH target like root@host:/workspace/carry-trace/" >&2
  exit 2
fi

REMOTE_HOST="${DEST%%:*}"
REMOTE_PATH="${DEST#*:}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'"

rsync -az --delete \
  --filter=":- .gitignore" \
  --exclude=".git/" \
  --exclude=".DS_Store" \
  ./ "$DEST"

if [ -d data ]; then
  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH/data'"
  rsync -az data/ "${DEST%/}/data/"
fi

echo "Synced repo and data to $DEST"
