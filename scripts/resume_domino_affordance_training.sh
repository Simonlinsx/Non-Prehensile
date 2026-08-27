#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
GPU_ID="${GPU_ID:-7}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-17}"
REFINE_ITERS="${REFINE_ITERS:-1000}"
LOAD_RUN="${LOAD_RUN:-2026-08-21_07-54-44_seed17_stage0}"
CURRICULUM_SEEDS="${CURRICULUM_SEEDS:-17 23 41}"
EXPERIMENT="franka_affordance_curriculum_seed${SEED}"
MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_stage0_xy_seed1701.jsonl"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data1/linsixu/tmp/isaaclab_nonprehensile}"

mkdir -p "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/optix"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export OPTIX_CACHE_PATH="$RUNTIME_ROOT/optix"

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires NVIDIA's Omniverse EULA acceptance." >&2
    exit 2
    ;;
esac

if [[ ! -f "$REPO_ROOT/logs/rsl_rl/$EXPERIMENT/$LOAD_RUN/model_1999.pt" ]]; then
  echo "Missing Stage 0 checkpoint: $EXPERIMENT/$LOAD_RUN/model_1999.pt" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Refining seed=$SEED stage=0 from $LOAD_RUN for $REFINE_ITERS iterations"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
DOMINO_ROOT="$DOMINO_DATA_ROOT" \
DOMINO_USD_ROOT="$DOMINO_CONVERTED_ROOT" \
DAPL_CLUTTER_ASSET_SOURCE=domino \
DAPL_CLUTTER_MANIFEST="$MANIFEST" \
DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0 \
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/train.py \
    --task Isaac-AffordanceHammer-XY-Franka-v0 \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max_iterations "$REFINE_ITERS" \
    --experiment_name "$EXPERIMENT" \
    --run_name "seed${SEED}_stage0" \
    --headless \
    --resume \
    --load_run "$LOAD_RUN" \
    --checkpoint 'model_.*.pt'

echo "Stage 0 refinement complete; continuing the four-stage, three-seed curriculum"
OMNI_KIT_ACCEPT_EULA=YES \
GPU_ID="$GPU_ID" \
NUM_ENVS="$NUM_ENVS" \
SEEDS="$CURRICULUM_SEEDS" \
RESUME_SEED="$SEED" \
RESUME_STAGE=1 \
  bash scripts/train_domino_affordance_curriculum.sh
