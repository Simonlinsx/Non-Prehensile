#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"
MANIFEST="${DAPL_CLUTTER_MANIFEST:-$REPO_ROOT/data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl}"
GPU_ID="${GPU_ID:-7}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEEDS="${SEEDS:-17 23 41}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
ISAAC_RESTART_DELAY="${ISAAC_RESTART_DELAY:-30}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data1/linsixu/tmp/isaaclab_nonprehensile_joint_pose}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-non-prehensile-affordance}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_SUFFIX="${RUN_SUFFIX:-contact_boundary_fixed_v7}"
TASK="${TASK:-Isaac-AffordanceHammer-Pose-Franka-v0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MIN_FREE_KB="${MIN_FREE_KB:-524288}"

mkdir -p \
  "$RUNTIME_ROOT/tmp" \
  "$RUNTIME_ROOT/cache" \
  "$RUNTIME_ROOT/optix" \
  "$RUNTIME_ROOT/wandb_cache" \
  "$RUNTIME_ROOT/wandb_data"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export OPTIX_CACHE_PATH="$RUNTIME_ROOT/optix"
export WANDB_CACHE_DIR="$RUNTIME_ROOT/wandb_cache"
export WANDB_DATA_DIR="$RUNTIME_ROOT/wandb_data"
export WANDB_MODE
# Scalar curves remain online, while checkpoints stay in the run directory.
# This avoids a failed artifact upload filling W&B's local transaction log.
export RSL_RL_WANDB_UPLOAD_CHECKPOINTS="${RSL_RL_WANDB_UPLOAD_CHECKPOINTS:-0}"

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires NVIDIA's Omniverse EULA acceptance." >&2
    echo "Review the license, then rerun with OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing 128-scene hammer manifest: $MANIFEST" >&2
  exit 1
fi

scene_count="$(wc -l < "$MANIFEST")"
if (( scene_count < 128 )); then
  echo "Joint-pose training requires at least 128 scenes; found $scene_count" >&2
  exit 1
fi

available_kb="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
if (( available_kb < MIN_FREE_KB )); then
  echo "Insufficient free disk: ${available_kb} KiB < ${MIN_FREE_KB} KiB" >&2
  exit 1
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2
  exit 1
fi

cd "$REPO_ROOT"
for seed in $SEEDS; do
  experiment="franka_affordance_joint_pose_seed${seed}"
  run_name="seed${seed}_${RUN_SUFFIX}"
  resume_args=()
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    resume_args=(--resume --checkpoint "$RESUME_CHECKPOINT")
  fi
  echo "Starting strict joint-pose training: task=$TASK seed=$seed envs=$NUM_ENVS scenes=$scene_count"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  DOMINO_ROOT="$DOMINO_DATA_ROOT" \
  DOMINO_USD_ROOT="$DOMINO_CONVERTED_ROOT" \
  DAPL_CLUTTER_ASSET_SOURCE=domino \
  DAPL_CLUTTER_MANIFEST="$MANIFEST" \
  DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0 \
  PYTHONPATH="$REPO_ROOT:$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" scripts/train.py \
      --task "$TASK" \
      --num_envs "$NUM_ENVS" \
      --seed "$seed" \
      --max_iterations "$MAX_ITERATIONS" \
      --experiment_name "$experiment" \
      --run_name "$run_name" \
      --logger "$LOGGER" \
      --log_project_name "$WANDB_PROJECT" \
      --headless \
      "${resume_args[@]}"

  sleep "$ISAAC_RESTART_DELAY"
done
