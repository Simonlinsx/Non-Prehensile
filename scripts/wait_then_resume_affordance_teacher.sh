#!/usr/bin/env bash
set -euo pipefail

# Wait for a declared checkpoint, then hand it to the existing teacher runner.
# This keeps multi-stage curricula reproducible without polling or relaunching
# them by hand. All task, manifest, seed, GPU, and iteration settings remain
# explicit environment variables consumed by train_affordance_teacher.sh.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_CHECKPOINT="${WAIT_CHECKPOINT:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-14400}"

if [[ -z "$WAIT_CHECKPOINT" ]]; then
  echo "Set WAIT_CHECKPOINT to the checkpoint that starts the next stage." >&2
  exit 2
fi
if (( POLL_SECONDS <= 0 || WAIT_TIMEOUT_SECONDS <= 0 )); then
  echo "POLL_SECONDS and WAIT_TIMEOUT_SECONDS must be positive." >&2
  exit 2
fi

wait_start_epoch="$(date +%s)"
while [[ ! -f "$WAIT_CHECKPOINT" ]]; do
  wait_now_epoch="$(date +%s)"
  if (( wait_now_epoch - wait_start_epoch >= WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for checkpoint: $WAIT_CHECKPOINT" >&2
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

export RESUME_CHECKPOINT="$WAIT_CHECKPOINT"
export WEIGHTS_ONLY="${WEIGHTS_ONLY:-0}"
exec bash "$REPO_ROOT/scripts/train_affordance_teacher.sh"
