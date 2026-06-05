#!/usr/bin/env bash
# Sync the local vllm working tree to a SEPARATE dev checkout on gpu-wb.
# Work artifact (untracked). Does NOT touch prod: it only rsyncs source into an
# isolated dev dir; nothing here stops/starts the prod qwen3.5-27b service or
# uses the GPUs. Running an actual PP=2 job (E3) still needs a maintenance window
# and is a separate, explicit step.
#
# Assumes a ONE-TIME setup on gpu-wb (see deploy-gpu-wb.md):
#   - an isolated dev checkout at $REMOTE_DIR with its own .venv built via
#     `VLLM_USE_PRECOMPILED=1 uv pip install -e .` (editable -> source rsync
#     takes effect with no rebuild for Python-only changes, which is all we make).
#
# Usage:
#   docs/superpowers/tools/sync-gpu-wb.sh            # dry-run by default (safe)
#   APPLY=1 docs/superpowers/tools/sync-gpu-wb.sh    # actually sync
#   APPLY=1 docs/superpowers/tools/sync-gpu-wb.sh -- <remote cmd...>  # sync then run
#
# Override via env: REMOTE_HOST, REMOTE_DIR.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gpu-wb}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/vllm-dev}"   # expanded on the remote
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Only ship source + tests. Never ship build trash, the local venv, caches, or
# the untracked work artifacts under docs/superpowers (kept local by convention).
RSYNC_ARGS=(
  -az --delete
  --filter=':- .gitignore'
  --exclude '.git/'
  --exclude '.venv/'
  --exclude 'docs/superpowers/'
  --exclude '**/__pycache__/'
  --exclude '*.so'
  --exclude 'build/'
  --exclude '.pytest_cache/'
)

DRY="--dry-run"
[[ "${APPLY:-0}" == "1" ]] && DRY=""

# Split optional remote command after a literal `--`.
REMOTE_CMD=()
seen_dashdash=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then seen_dashdash=1; continue; fi
  [[ $seen_dashdash == 1 ]] && REMOTE_CMD+=("$a")
done

echo ">> rsync ${DRY:-(APPLY)} $LOCAL_DIR/ -> $REMOTE_HOST:$REMOTE_DIR/"
# shellcheck disable=SC2029
rsync $DRY "${RSYNC_ARGS[@]}" "$LOCAL_DIR/" \
  "$REMOTE_HOST:$(ssh "$REMOTE_HOST" "echo $REMOTE_DIR")/"

if [[ ${#REMOTE_CMD[@]} -gt 0 && "${APPLY:-0}" == "1" ]]; then
  echo ">> remote: ${REMOTE_CMD[*]}"
  # shellcheck disable=SC2029
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ${REMOTE_CMD[*]}"
fi
