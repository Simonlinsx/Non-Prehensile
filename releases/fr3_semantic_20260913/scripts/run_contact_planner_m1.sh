#!/usr/bin/env bash
set -euo pipefail

# Reproducible M1 entry point.  M1 uses oracle target geometry/affordance,
# deterministic contact sampling, Pinocchio IK, and closed-loop short pushes;
# it does not load an RL checkpoint.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-17}"
NUM_ENVS="${NUM_ENVS:-8}"
VIDEO="${VIDEO:-0}"
TASK="${TASK:-Isaac-AffordanceTeacher-Relation-C1-Franka-v0}"
EXECUTION_PROFILE="${EXECUTION_PROFILE:-adaptive}"
MAX_REPLANS="${MAX_REPLANS:-30}"
RUN_LABEL="${RUN_LABEL:-m1_oracle_c1_${EXECUTION_PROFILE}_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/contact_planner_m1}"
MANIFEST="${MANIFEST:-$REPO_ROOT/data/manifests/contact_planner_m3/hammer_c1_outward120_eval50_seed20260902_isaaclab.jsonl}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"

# Keep the accepted contact/servo timing fixed while testing push-distance
# efficiency.  The adaptive profile is the accepted faster A/B configuration;
# fixed5mm reproduces the conservative contact-geometry validation baseline;
# persistent keeps a legal contact and closes the pose loop every 3 steps.
case "$EXECUTION_PROFILE" in
  adaptive)
    profile_minimum_push_distance_m=0.008
    profile_maximum_push_distance_m=0.015
    profile_push_distance_samples=3
    profile_contact_execution_mode=macro
    ;;
  fixed5mm)
    profile_minimum_push_distance_m=0.005
    profile_maximum_push_distance_m=0.005
    profile_push_distance_samples=1
    profile_contact_execution_mode=macro
    ;;
  persistent)
    profile_minimum_push_distance_m=0.008
    profile_maximum_push_distance_m=0.015
    profile_push_distance_samples=3
    profile_contact_execution_mode=persistent
    ;;
  *)
    echo "Unknown EXECUTION_PROFILE: $EXECUTION_PROFILE (expected adaptive, fixed5mm, or persistent)" >&2
    exit 2
    ;;
esac
MINIMUM_PUSH_DISTANCE_M="${MINIMUM_PUSH_DISTANCE_M:-$profile_minimum_push_distance_m}"
MAXIMUM_PUSH_DISTANCE_M="${MAXIMUM_PUSH_DISTANCE_M:-$profile_maximum_push_distance_m}"
PUSH_DISTANCE_SAMPLES="${PUSH_DISTANCE_SAMPLES:-$profile_push_distance_samples}"
CONTACT_EXECUTION_MODE="${CONTACT_EXECUTION_MODE:-$profile_contact_execution_mode}"
PERSISTENT_MICRO_STEPS="${PERSISTENT_MICRO_STEPS:-3}"
PERSISTENT_MICRO_DISTANCE_M="${PERSISTENT_MICRO_DISTANCE_M:-0.002}"
PERSISTENT_MAX_CONTACT_TRAVEL_M="${PERSISTENT_MAX_CONTACT_TRAVEL_M:-0.060}"
PERSISTENT_COST_REGRESSION_TOLERANCE="${PERSISTENT_COST_REGRESSION_TOLERANCE:-0.25}"
PERSISTENT_MAX_MOMENT_ARM_ERROR_M="${PERSISTENT_MAX_MOMENT_ARM_ERROR_M:-0.010}"
PERSISTENT_MINIMUM_GOAL_AXIS_COSINE="${PERSISTENT_MINIMUM_GOAL_AXIS_COSINE:-0.2}"

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

for required in "$PYTHON_BIN" "$MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required M1 input: $required" >&2
    exit 2
  fi
done

if [[ "$VIDEO" == "1" ]]; then
  NUM_ENVS=1
fi

mkdir -p "$OUTPUT_ROOT"
command=(
  "$PYTHON_BIN"
  "$REPO_ROOT/scripts/run_contact_planner_m1.py"
  --headless
  --device cuda:0
  --task "$TASK"
  --num-envs "$NUM_ENVS"
  --seed "$SEED"
  --max-replans "$MAX_REPLANS"
  --minimum-push-distance-m "$MINIMUM_PUSH_DISTANCE_M"
  --maximum-push-distance-m "$MAXIMUM_PUSH_DISTANCE_M"
  --push-distance-samples "$PUSH_DISTANCE_SAMPLES"
  --contact-execution-mode "$CONTACT_EXECUTION_MODE"
  --persistent-micro-steps "$PERSISTENT_MICRO_STEPS"
  --persistent-micro-distance-m "$PERSISTENT_MICRO_DISTANCE_M"
  --persistent-max-contact-travel-m "$PERSISTENT_MAX_CONTACT_TRAVEL_M"
  --persistent-cost-regression-tolerance "$PERSISTENT_COST_REGRESSION_TOLERANCE"
  --persistent-max-moment-arm-error-m "$PERSISTENT_MAX_MOMENT_ARM_ERROR_M"
  --persistent-minimum-goal-axis-cosine "$PERSISTENT_MINIMUM_GOAL_AXIS_COSINE"
  --yaw-weight-m-per-rad 8
  --inside-yaw-weight-m-per-rad 8
  --predicted-yaw-guard-rad 0.075
  --output-candidates 32
  --push-direction-samples 13
  --push-direction-span-deg 90
  --hand-yaw-samples 9
  --hand-yaw-span-deg 120
  --physical-contact-force-threshold-n 0.02
  --output "$OUTPUT_ROOT/${RUN_LABEL}.json"
)
if [[ "$VIDEO" == "1" ]]; then
  command+=(
    --video
    --video-folder "$OUTPUT_ROOT/${RUN_LABEL}_video"
    --video-name-prefix "$RUN_LABEL"
  )
fi
# Explicit CLI arguments are appended last so a diagnostic can override any
# profile value without editing this reproducible entry point.
command+=("$@")

OMNI_KIT_ACCEPT_EULA=YES \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
DOMINO_ROOT="$DOMINO_DATA_ROOT" \
DOMINO_USD_ROOT="$DOMINO_CONVERTED_ROOT" \
DAPL_CLUTTER_MANIFEST="$MANIFEST" \
PYTHONPATH="$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
  "${command[@]}"

echo "M1 JSON: $OUTPUT_ROOT/${RUN_LABEL}.json"
if [[ "$VIDEO" == "1" ]]; then
  echo "M1 video directory: $OUTPUT_ROOT/${RUN_LABEL}_video"
fi
