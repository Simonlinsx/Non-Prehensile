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
RUN_LABEL="${RUN_LABEL:-m1_oracle_c1_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/contact_planner_m1}"
MANIFEST="${MANIFEST:-$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"

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
  --num-envs "$NUM_ENVS"
  --seed "$SEED"
  --output "$OUTPUT_ROOT/${RUN_LABEL}.json"
)
if [[ "$VIDEO" == "1" ]]; then
  command+=(
    --video
    --video-folder "$OUTPUT_ROOT/${RUN_LABEL}_video"
    --video-name-prefix "$RUN_LABEL"
  )
fi

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
