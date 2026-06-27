#!/usr/bin/env bash

# Sync this repo to Myriad without gitignored local artifacts.
# Usage:
#   scripts/sync_to_myriad.sh uccaxxx@myriad.rc.ucl.ac.uk:/home/uccaxxx/Scratch/carry-trace/

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 user@myriad.rc.ucl.ac.uk:/remote/path/carry-trace/" >&2
  exit 2
fi

DEST="$1"

if [[ "$DEST" != *:* ]]; then
  echo "Destination must be an rsync SSH target like user@host:/remote/path/" >&2
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
  rsync -az --delete data/ "${DEST%/}/data/"
fi

echo "Synced repo to $DEST"
