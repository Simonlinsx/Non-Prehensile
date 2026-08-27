#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"
MANIFEST="${DAPL_CLUTTER_MANIFEST:-$REPO_ROOT/data/manifests/domino_hammer_dapl_sparse_train1024_seed1701.jsonl}"
TASK="${TASK:-Isaac-AffordanceTeacher-T0-DAPL-C1-Soft-Franka-v0}"
GPU_ID="${GPU_ID:-6}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-17}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
RUN_SUFFIX="${RUN_SUFFIX:-t0_dapl_c1_soft_from_scratch_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka_affordance_teacher_seed${SEED}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WEIGHTS_ONLY="${WEIGHTS_ONLY:-0}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-non-prehensile-affordance}"
WANDB_MODE="${WANDB_MODE:-online}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data1/linsixu/tmp/isaaclab_nonprehensile_teacher}"
DAPL_LOCAL_FRANKA_USD_DIR="${DAPL_LOCAL_FRANKA_USD_DIR:-$RUNTIME_ROOT/franka_usd}"
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"

mkdir -p \
  "$RUNTIME_ROOT/tmp" \
  "$RUNTIME_ROOT/cache" \
  "$RUNTIME_ROOT/optix" \
  "$RUNTIME_ROOT/wandb_cache" \
  "$RUNTIME_ROOT/wandb_data" \
  "$DAPL_LOCAL_FRANKA_USD_DIR"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export OPTIX_CACHE_PATH="$RUNTIME_ROOT/optix"
export WANDB_CACHE_DIR="$RUNTIME_ROOT/wandb_cache"
export WANDB_DATA_DIR="$RUNTIME_ROOT/wandb_data"
export WANDB_MODE
export RSL_RL_WANDB_UPLOAD_CHECKPOINTS="${RSL_RL_WANDB_UPLOAD_CHECKPOINTS:-0}"
export DAPL_LOCAL_FRANKA_USD_DIR

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing teacher manifest: $MANIFEST" >&2
  exit 1
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2
  exit 1
fi
available_kb="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
if (( available_kb < MIN_FREE_KB )); then
  echo "Insufficient free disk: ${available_kb} KiB < ${MIN_FREE_KB} KiB" >&2
  exit 1
fi

resume_args=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  resume_args=(--resume --checkpoint "$RESUME_CHECKPOINT")
  if [[ "$WEIGHTS_ONLY" == "1" ]]; then
    resume_args+=(--weights_only)
  fi
elif [[ "$WEIGHTS_ONLY" == "1" ]]; then
  echo "WEIGHTS_ONLY=1 requires RESUME_CHECKPOINT." >&2
  exit 2
fi

save_interval_args=()
if [[ -n "$SAVE_INTERVAL" ]]; then
  if [[ ! "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAVE_INTERVAL must be a positive integer." >&2
    exit 2
  fi
  save_interval_args=(--save_interval "$SAVE_INTERVAL")
fi

experiment="$EXPERIMENT_NAME"
run_name="seed${SEED}_${RUN_SUFFIX}"
cd "$REPO_ROOT"
echo "Starting teacher: task=$TASK seed=$SEED envs=$NUM_ENVS iterations=$MAX_ITERATIONS"
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
    --seed "$SEED" \
    --max_iterations "$MAX_ITERATIONS" \
    "${save_interval_args[@]}" \
    --experiment_name "$experiment" \
    --run_name "$run_name" \
    --logger "$LOGGER" \
    --log_project_name "$WANDB_PROJECT" \
    --headless \
    "${resume_args[@]}"
