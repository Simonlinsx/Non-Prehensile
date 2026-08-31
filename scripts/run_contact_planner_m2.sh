#!/usr/bin/env bash
set -euo pipefail

# Reproducible M2 diagnostic.  M2 keeps the M1 oracle-affordance/IK safety
# layer and ranks explicit contact-direction-distance candidates with restored
# Isaac physics rollouts.  It is an experimental upper bound, not an accepted
# full-pose or sim-to-real result.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-17}"
NUM_ENVS="${NUM_ENVS:-1}"
RUN_LABEL="${RUN_LABEL:-m2_oracle_physics_mpc2_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/contact_planner_m2}"
MANIFEST="${MANIFEST:-$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"
ROLLOUT_CANDIDATES="${ROLLOUT_CANDIDATES:-8}"

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

for required in "$PYTHON_BIN" "$MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required M2 input: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"
OMNI_KIT_ACCEPT_EULA=YES \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
DOMINO_ROOT="$DOMINO_DATA_ROOT" \
DOMINO_USD_ROOT="$DOMINO_CONVERTED_ROOT" \
DAPL_CLUTTER_MANIFEST="$MANIFEST" \
PYTHONPATH="$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_contact_planner_m1.py" \
    --headless \
    --device cuda:0 \
    --num-envs "$NUM_ENVS" \
    --seed "$SEED" \
    --physics-rollout-candidates "$ROLLOUT_CANDIDATES" \
    --rollout-lookahead-steps 2 \
    --output-candidates 32 \
    --push-direction-samples 13 \
    --push-direction-span-deg 180 \
    --push-distance-samples 4 \
    --minimum-push-distance-m 0.003 \
    --maximum-push-distance-m 0.015 \
    --contact-penetration-m 0.002 \
    --output "$OUTPUT_ROOT/${RUN_LABEL}.json"

echo "M2 JSON: $OUTPUT_ROOT/${RUN_LABEL}.json"
