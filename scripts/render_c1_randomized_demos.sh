#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-7}"
SCENES="${SCENES:-0 7 17 35 64 78 82 103}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/logs/rsl_rl/franka_affordance_goalwrench_c1soft_seed17/2026-08-26_18-46-22_seed17_t0_goalwrench_c1soft_dir45_from_v26m350_short10_v30/model_359.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/teacher_demos/c1_accepted_randomized_v30_model359}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing accepted C1 checkpoint: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
for scene_index in $SCENES; do
  scene_label="scene$(printf '%03d' "$scene_index")"
  output_dir="$OUTPUT_ROOT/$scene_label/eval"
  video_dir="$OUTPUT_ROOT/$scene_label/video"
  mkdir -p "$output_dir" "$video_dir"
  set +e
  OMNI_KIT_ACCEPT_EULA=YES \
  GPU_ID="$GPU_ID" \
  PROFILE=c1_frozenv7_goalwrench_dir45 \
  CHECKPOINT="$CHECKPOINT" \
  SEED=17 \
  SCENE_INDEX="$scene_index" \
  NUM_EPISODES=1 \
  MAX_EPISODE_STEPS=300 \
  VIDEO=1 \
  VIDEO_NUM_EPISODES=1 \
  VIDEO_LENGTH=300 \
  VIDEO_FOLDER="$video_dir" \
  VIDEO_NAME_PREFIX="c1_v30_model359_${scene_label}" \
  RUN_LABEL="c1_v30_model359_${scene_label}" \
  OUTPUT_DIR="$output_dir" \
    bash "$REPO_ROOT/scripts/evaluate_affordance_teacher.sh" \
      > "$OUTPUT_ROOT/$scene_label/render.log" 2>&1
  eval_status=$?
  set -e

  # Isaac Sim can occasionally return a non-zero status while shutting down
  # after both evaluation and video encoding have completed.  Treat the
  # scene as complete only when both durable artifacts exist; otherwise fail
  # the batch and surface the original status for diagnosis.
  video_path="$video_dir/c1_v30_model359_${scene_label}-step-0.mp4"
  if [[ ! -s "$output_dir/eval_summary.json" || ! -s "$video_path" ]]; then
    echo "Failed to render $scene_label (exit=$eval_status); see $OUTPUT_ROOT/$scene_label/render.log" >&2
    exit "${eval_status:-1}"
  fi
  if (( eval_status != 0 )); then
    echo "Completed $scene_label despite Isaac shutdown exit=$eval_status"
  else
    echo "Completed $scene_label"
  fi
done
