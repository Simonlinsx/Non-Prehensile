#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${PUSH_ANYTHING_ROOT:-/data1/linsixu/dairlib-push-anything}"
STAGE_PYTHON="${PUSH_ANYTHING_STAGE_PYTHON:-/data1/linsixu/miniconda3/envs/domino/bin/python}"
AUDIT_PYTHON="${PUSH_ANYTHING_AUDIT_PYTHON:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
RUN_NAME="${PUSH_ANYTHING_RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)_hammer_c1_seed17}"
OUTPUT_DIR="${PUSH_ANYTHING_OUTPUT_DIR:-$REPO_ROOT/outputs/contact_planner_m3/$RUN_NAME}"
BUILD="${PUSH_ANYTHING_BUILD:-1}"

for python_path in "$STAGE_PYTHON" "$AUDIT_PYTHON"; do
  if [[ ! -x "$python_path" ]]; then
    echo "ERROR: required Python is not executable: $python_path" >&2
    exit 2
  fi
done
if [[ "$BUILD" != "0" && "$BUILD" != "1" ]]; then
  echo "ERROR: PUSH_ANYTHING_BUILD must be 0 or 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
"$STAGE_PYTHON" scripts/stage_domino_hammer_push_anything.py \
  --upstream-root "$UPSTREAM_ROOT" \
  --generator-python "$STAGE_PYTHON" \
  --goal-distance 0.08 \
  --goal-yaw-deg 10 \
  --quaternion-weight 5 \
  --sampling-seed 17 \
  --semantic-guard-clearance 0.025 \
  --semantic-guard-stop-distance 0.055

if [[ "$BUILD" == "1" ]]; then
  PUSH_ANYTHING_ROOT="$UPSTREAM_ROOT" \
    bash scripts/build_push_anything_native.sh --build
fi

set +e
PUSH_ANYTHING_ROOT="$UPSTREAM_ROOT" \
PUSH_ANYTHING_RUN_NAME="$RUN_NAME" \
PUSH_ANYTHING_OUTPUT_DIR="$OUTPUT_DIR" \
PUSH_ANYTHING_TIMEOUT_S="${PUSH_ANYTHING_TIMEOUT_S:-180}" \
PUSH_ANYTHING_TCPQ_PORT="${PUSH_ANYTHING_TCPQ_PORT:-7700}" \
  bash scripts/run_push_anything_native_baseline.sh
geometry_status="$?"
set -e

if [[ ! -f "$OUTPUT_DIR/sampling_c3_debug.csv" ]]; then
  echo "ERROR: controller produced no trajectory CSV: $OUTPUT_DIR" >&2
  exit 1
fi

set +e
"$AUDIT_PYTHON" scripts/audit_push_anything_c1.py \
  --trajectory-csv "$OUTPUT_DIR/sampling_c3_debug.csv" \
  --semantic-dir data/push_anything_semantics/020_hammer_0 \
  --output-json "$OUTPUT_DIR/c1_semantic_audit.json" \
  --output-csv "$OUTPUT_DIR/c1_semantic_audit.csv"
semantic_status="$?"
set -e

set +e
"$AUDIT_PYTHON" scripts/verify_push_anything_c1_acceptance.py \
  --run-dir "$OUTPUT_DIR"
joint_status="$?"
set -e

if [[ "$geometry_status" -ne 0 || "$semantic_status" -ne 0 || "$joint_status" -ne 0 ]]; then
  echo "C1 joint acceptance failed; inspect $OUTPUT_DIR" >&2
  exit 1
fi
echo "C1 joint acceptance passed: $OUTPUT_DIR/joint_acceptance.json"
