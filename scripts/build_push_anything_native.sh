#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_COMMIT="9d988c835d6e99330397701487fce5ce4ceafa3c"
EXPECTED_C3_COMMIT="5c08cb2e14b1ab10e024cb46e8504970cffcd5ea"
UPSTREAM_ROOT="${PUSH_ANYTHING_ROOT:-/data1/linsixu/dairlib-push-anything}"
C3_ROOT="${PUSH_ANYTHING_C3_ROOT:-/data1/linsixu/c3-push-anything}"
OUTPUT_USER_ROOT="${PUSH_ANYTHING_BAZEL_ROOT:-/data1/linsixu/.cache/bazel-push-anything}"
BAZELISK_HOME="${PUSH_ANYTHING_BAZELISK_HOME:-/data1/linsixu/.cache/bazelisk}"
OPENBLAS_ROOT="${PUSH_ANYTHING_OPENBLAS_ROOT:-/data1/linsixu/miniconda3/envs/anydex-torch}"
C3_PATCH="$REPO_ROOT/third_party/push_anything/patches/0002-c3-no-gurobi-optional-m.patch"
MODE="${1:---check}"

usage() {
  cat <<'EOF'
Usage: build_push_anything_native.sh [--check|--build]

Environment:
  PUSH_ANYTHING_ROOT       Upstream checkout (default: /data1/linsixu/dairlib-push-anything)
  PUSH_ANYTHING_C3_ROOT    Pinned local C3 checkout
  PUSH_ANYTHING_BAZEL_ROOT Bazel cache/output root on a large disk
  PUSH_ANYTHING_OPENBLAS_ROOT Prefix containing lib/libopenblas.so

This builds the C3+ path natively and explicitly disables Gurobi/MIQP.
EOF
}

if [[ "$MODE" != "--check" && "$MODE" != "--build" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
  echo "ERROR: Push Anything checkout not found: $UPSTREAM_ROOT" >&2
  exit 3
fi

actual_commit="$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "ERROR: expected upstream $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 4
fi

if ! rg -q 'std::optional<std::vector<std::string>> sampling_meshes' \
  "$UPSTREAM_ROOT/examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h" || \
   ! rg -q 'controller_params_\.sampling_meshes' \
  "$UPSTREAM_ROOT/systems/controllers/sampling_based_c3_controller.cc" || \
   ! rg -q 'SemanticC1TrajectoryGuard' \
  "$UPSTREAM_ROOT/examples/sampling_c3/franka_osc_controller.cc" || \
   [[ ! -f "$UPSTREAM_ROOT/examples/sampling_c3/monitor_push_anything_baseline.py" ]]; then
  echo "ERROR: semantic C1 integration patch is not applied; run scripts/apply_push_anything_patches.sh first." >&2
  exit 5
fi

if [[ ! -d "$C3_ROOT/.git" ]]; then
  echo "ERROR: C3 checkout not found: $C3_ROOT" >&2
  exit 6
fi

actual_c3_commit="$(git -C "$C3_ROOT" rev-parse HEAD)"
if [[ "$actual_c3_commit" != "$EXPECTED_C3_COMMIT" ]]; then
  echo "ERROR: expected C3 $EXPECTED_C3_COMMIT, found $actual_c3_commit" >&2
  exit 7
fi

if ! git -C "$C3_ROOT" apply --reverse --check "$C3_PATCH" 2>/dev/null; then
  echo "ERROR: no-Gurobi C3 patch is not applied; run scripts/apply_c3_patches.sh first." >&2
  exit 8
fi

if command -v bazelisk >/dev/null 2>&1; then
  bazel_cmd="$(command -v bazelisk)"
elif [[ -x /data1/linsixu/.local/bin/bazelisk ]]; then
  bazel_cmd="/data1/linsixu/.local/bin/bazelisk"
elif command -v bazel >/dev/null 2>&1; then
  bazel_cmd="$(command -v bazel)"
else
  bazel_cmd=""
fi

missing=()
[[ -n "$bazel_cmd" ]] || missing+=("bazel/bazelisk")
if [[ -f "$OPENBLAS_ROOT/lib/libopenblas.so" ]]; then
  openblas_lib_dir="$OPENBLAS_ROOT/lib"
elif pkg-config --exists openblas 2>/dev/null; then
  openblas_lib_dir="$(pkg-config --variable=libdir openblas)"
else
  openblas_lib_dir=""
  missing+=("OpenBLAS shared library")
fi

available_kb="$(df -Pk /data1 | awk 'NR == 2 {print $4}')"
available_gb="$((available_kb / 1024 / 1024))"

echo "Push Anything: $UPSTREAM_ROOT"
echo "Upstream commit: $actual_commit"
echo "C3 commit: $actual_c3_commit"
echo "Bazel output root: $OUTPUT_USER_ROOT"
echo "Bazelisk cache: $BAZELISK_HOME"
echo "OpenBLAS library: ${openblas_lib_dir:-missing}"
echo "Free space on /data1: ${available_gb} GiB"
echo "Projection backend: C3+ (Gurobi/MIQP disabled)"

if (( available_gb < 30 )); then
  echo "WARNING: less than 30 GiB free; the pinned Drake source build may exhaust the disk." >&2
fi

if (( ${#missing[@]} > 0 )); then
  printf 'MISSING:' >&2
  printf ' %s' "${missing[@]}" >&2
  printf '\n' >&2
  echo "Run the one-time native prerequisites command documented in docs/CONTACT_PLANNER_M3_C3PLUS.md." >&2
  exit 6
fi

if [[ "$MODE" == "--check" ]]; then
  (
    cd "$UPSTREAM_ROOT"
    BAZELISK_HOME="$BAZELISK_HOME" "$bazel_cmd" \
      --output_user_root="$OUTPUT_USER_ROOT" --batch version
  )
  echo "Native C3+ build prerequisites passed."
  exit 0
fi

mkdir -p "$OUTPUT_USER_ROOT"
mkdir -p "$BAZELISK_HOME"
cd "$UPSTREAM_ROOT"

BAZELISK_HOME="$BAZELISK_HOME" "$bazel_cmd" \
  --output_user_root="$OUTPUT_USER_ROOT" \
  --batch \
  build \
  --override_module="c3=$C3_ROOT" \
  --define=WITH_GUROBI=OFF \
  --config=omp \
  --action_env="LD_LIBRARY_PATH=$openblas_lib_dir" \
  --linkopt="-L$openblas_lib_dir" \
  --linkopt="-Wl,-rpath,$openblas_lib_dir" \
  //examples/sampling_c3:franka_sim \
  //examples/sampling_c3:franka_osc_controller \
  //examples/sampling_c3:franka_sampling_c3_controller \
  //examples/sampling_c3:monitor_push_anything_baseline

echo "Native Push Anything C3+ targets built successfully."
