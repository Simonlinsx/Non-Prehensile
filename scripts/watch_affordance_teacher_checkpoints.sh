#!/usr/bin/env bash
set -euo pipefail

# Wait for a fixed checkpoint grid and evaluate each checkpoint exactly once.
# This is intentionally a held-out evaluator, not a training-metric watcher.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINTS="${CHECKPOINTS:-}"
PROFILE="${PROFILE:-t0_dir90}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-7001}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EPISODES="${NUM_EPISODES:-128}"
RUN_PREFIX="${RUN_PREFIX:-teacher_checkpoint_watch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/teacher_eval}"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-7200}"

if [[ -z "$CHECKPOINT_DIR" || ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Set CHECKPOINT_DIR to an existing training run directory." >&2
  exit 2
fi
if [[ -z "$CHECKPOINTS" ]]; then
  echo "Set CHECKPOINTS to a whitespace-separated iteration list." >&2
  exit 2
fi
case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT"

for checkpoint_iteration in $CHECKPOINTS; do
  checkpoint_path="$CHECKPOINT_DIR/model_${checkpoint_iteration}.pt"
  output_dir="$OUTPUT_ROOT/${RUN_PREFIX}_model${checkpoint_iteration}_balanced${NUM_EPISODES}"
  summary_path="$output_dir/eval_summary.json"

  if [[ -f "$summary_path" ]]; then
    echo "Already evaluated: $summary_path"
    continue
  fi

  watch_start_epoch="$(date +%s)"
  while [[ ! -f "$checkpoint_path" ]]; do
    watch_now_epoch="$(date +%s)"
    if (( watch_now_epoch - watch_start_epoch >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for checkpoint: $checkpoint_path" >&2
      exit 3
    fi
    sleep "$POLL_SECONDS"
  done

  echo "Evaluating checkpoint: $checkpoint_path"
  OMNI_KIT_ACCEPT_EULA=YES \
  PROFILE="$PROFILE" \
  GPU_ID="$GPU_ID" \
  SEED="$SEED" \
  NUM_ENVS="$NUM_ENVS" \
  NUM_EPISODES="$NUM_EPISODES" \
  CHECKPOINT="$checkpoint_path" \
  RUN_LABEL="${RUN_PREFIX}_model${checkpoint_iteration}" \
  OUTPUT_DIR="$output_dir" \
    bash "$REPO_ROOT/scripts/evaluate_affordance_teacher.sh"
done
