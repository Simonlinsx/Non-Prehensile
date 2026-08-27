#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEEDS="${SEEDS:-17 23 41}"
STAGE0_ITERS="${STAGE0_ITERS:-2000}"
STAGE1_ITERS="${STAGE1_ITERS:-2000}"
STAGE2_ITERS="${STAGE2_ITERS:-3000}"
STAGE3_ITERS="${STAGE3_ITERS:-5000}"
RESUME_SEED="${RESUME_SEED:-}"
RESUME_STAGE="${RESUME_STAGE:-0}"
ISAAC_RESTART_DELAY="${ISAAC_RESTART_DELAY:-30}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data1/linsixu/tmp/isaaclab_nonprehensile}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-isaaclab}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_SUFFIX="${RUN_SUFFIX:-strict_single_hammer_v2}"
START_STAGE="${START_STAGE:-0}"
END_STAGE="${END_STAGE:-3}"

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

if (( START_STAGE < 0 || START_STAGE > 3 || END_STAGE < START_STAGE || END_STAGE > 3 )); then
  echo "Require 0 <= START_STAGE <= END_STAGE <= 3" >&2
  exit 2
fi

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires NVIDIA's Omniverse EULA acceptance." >&2
    echo "Review the license, then rerun with OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

TASKS=(
  Isaac-AffordanceHammer-XY-Franka-v0
  Isaac-AffordanceHammer-Yaw-Franka-v0
  Isaac-AffordanceHammer-Avoid-Franka-v0
  Isaac-AffordanceHammer-Clutter-Franka-v0
)
MANIFESTS=(
  data/manifests/domino_hammer_stage0_xy_seed1701.jsonl
  data/manifests/domino_hammer_stage1_yaw_128_seed1702.jsonl
  data/manifests/domino_hammer_stage2_avoid_128_seed1703.jsonl
  data/manifests/domino_hammer_stage3_clutter_128_seed1704.jsonl
)
ITERATIONS=("$STAGE0_ITERS" "$STAGE1_ITERS" "$STAGE2_ITERS" "$STAGE3_ITERS")

cd "$REPO_ROOT"
for manifest in "${MANIFESTS[@]}"; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing curriculum manifest: $REPO_ROOT/$manifest" >&2
    exit 1
  fi
done

for seed in $SEEDS; do
  experiment="franka_affordance_curriculum_seed${seed}"
  resume_args=()
  stage_begin="$START_STAGE"
  if [[ -n "$RESUME_SEED" && "$seed" == "$RESUME_SEED" ]]; then
    stage_begin="$RESUME_STAGE"
  fi
  if (( stage_begin > 0 )); then
    previous_stage=$((stage_begin - 1))
    previous_name="seed${seed}_stage${previous_stage}_${RUN_SUFFIX}"
    previous_run="$(find "logs/rsl_rl/$experiment" -mindepth 1 -maxdepth 1 \
      -type d -name "*_${previous_name}" | sort | tail -n 1)"
    if [[ -z "$previous_run" ]]; then
      echo "Could not locate checkpoint run for $previous_name" >&2
      exit 1
    fi
    resume_args=(
      --resume
      --load_run "$(basename "$previous_run")"
      --checkpoint 'model_.*.pt'
    )
  fi
  for ((stage = stage_begin; stage <= END_STAGE; stage++)); do
    run_name="seed${seed}_stage${stage}_${RUN_SUFFIX}"
    echo "Starting seed=$seed stage=$stage task=${TASKS[$stage]} envs=$NUM_ENVS"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    DOMINO_ROOT="$DOMINO_DATA_ROOT" \
    DOMINO_USD_ROOT="$DOMINO_CONVERTED_ROOT" \
    DAPL_CLUTTER_ASSET_SOURCE=domino \
    DAPL_CLUTTER_MANIFEST="$REPO_ROOT/${MANIFESTS[$stage]}" \
    DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0 \
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" scripts/train.py \
        --task "${TASKS[$stage]}" \
        --num_envs "$NUM_ENVS" \
        --seed "$seed" \
        --max_iterations "${ITERATIONS[$stage]}" \
        --experiment_name "$experiment" \
        --run_name "$run_name" \
        --logger "$LOGGER" \
        --log_project_name "$WANDB_PROJECT" \
        --headless \
        "${resume_args[@]}"

    run_dir="$(find "logs/rsl_rl/$experiment" -mindepth 1 -maxdepth 1 \
      -type d -name "*_${run_name}" | sort | tail -n 1)"
    if [[ -z "$run_dir" ]]; then
      echo "Could not locate completed run for $run_name" >&2
      exit 1
    fi
    resume_args=(
      --resume
      --load_run "$(basename "$run_dir")"
      --checkpoint 'model_.*.pt'
    )
    sleep "$ISAAC_RESTART_DELAY"
  done
done
