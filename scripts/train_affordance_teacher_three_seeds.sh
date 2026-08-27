#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-17 23 41}"
RUN_SUFFIX_BASE="${RUN_SUFFIX_BASE:-t0_unified_progress_planarpush_fromscratch_v4}"

export TASK="${TASK:-Isaac-AffordanceTeacher-T0-UnifiedProgress-C1-Soft-Franka-v0}"
export DAPL_CLUTTER_MANIFEST="${DAPL_CLUTTER_MANIFEST:-$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_train1024_seed1701.jsonl}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
export LOGGER="${LOGGER:-wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"

for teacher_seed in $SEEDS; do
  echo "Starting sequential T0 teacher seed ${teacher_seed} on GPU ${GPU_ID:-0}."
  SEED="$teacher_seed" \
  RUN_SUFFIX="${RUN_SUFFIX_BASE}" \
    bash "$REPO_ROOT/scripts/train_affordance_teacher.sh"
done
