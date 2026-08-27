#!/usr/bin/env bash
set -euo pipefail

# Resolve a timestamped RSL-RL run directory, then delegate checkpoint
# evaluation to the existing fixed-grid watcher. This wrapper changes no
# training or evaluation semantics; it only removes a manual timing step.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PARENT="${RUN_PARENT:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-14400}"

if [[ -z "$RUN_PARENT" || ! -d "$RUN_PARENT" ]]; then
  echo "Set RUN_PARENT to an existing RSL-RL experiment directory." >&2
  exit 2
fi
if [[ -z "$RUN_SUFFIX" ]]; then
  echo "Set RUN_SUFFIX to the exact run-name suffix to resolve." >&2
  exit 2
fi
if (( POLL_SECONDS <= 0 || WAIT_TIMEOUT_SECONDS <= 0 )); then
  echo "POLL_SECONDS and WAIT_TIMEOUT_SECONDS must be positive." >&2
  exit 2
fi

wait_start_epoch="$(date +%s)"
while true; do
  mapfile -t matching_runs < <(
    find "$RUN_PARENT" -mindepth 1 -maxdepth 1 -type d \
      -name "*_${RUN_SUFFIX}" -printf '%T@ %p\n' \
      | sort -n \
      | cut -d' ' -f2-
  )
  if (( ${#matching_runs[@]} > 0 )); then
    latest_index=$((${#matching_runs[@]} - 1))
    export CHECKPOINT_DIR="${matching_runs[$latest_index]}"
    echo "Resolved checkpoint directory: $CHECKPOINT_DIR"
    exec bash "$REPO_ROOT/scripts/watch_affordance_teacher_checkpoints.sh"
  fi

  wait_now_epoch="$(date +%s)"
  if (( wait_now_epoch - wait_start_epoch >= WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for run suffix '$RUN_SUFFIX' under $RUN_PARENT" >&2
    exit 3
  fi
  sleep "$POLL_SECONDS"
done
