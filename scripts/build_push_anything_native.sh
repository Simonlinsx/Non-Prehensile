#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_COMMIT="9d988c835d6e99330397701487fce5ce4ceafa3c"
UPSTREAM_ROOT="${PUSH_ANYTHING_ROOT:-/data1/linsixu/dairlib-push-anything}"
OUTPUT_USER_ROOT="${PUSH_ANYTHING_BAZEL_ROOT:-/data1/linsixu/.cache/bazel-push-anything}"
INTEGRATION_PATCH="$REPO_ROOT/third_party/push_anything/patches/0001-safe-only-sampling-mesh.patch"
MODE="${1:---check}"

usage() {
  cat <<'EOF'
Usage: build_push_anything_native.sh [--check|--build]

Environment:
  PUSH_ANYTHING_ROOT       Upstream checkout (default: /data1/linsixu/dairlib-push-anything)
  PUSH_ANYTHING_BAZEL_ROOT Bazel cache/output root on a large disk

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

if ! git -C "$UPSTREAM_ROOT" apply --unidiff-zero --reverse --check \
  "$INTEGRATION_PATCH" 2>/dev/null; then
  echo "ERROR: semantic sampling patch is not applied; run scripts/apply_push_anything_patches.sh first." >&2
  exit 5
fi

if command -v bazelisk >/dev/null 2>&1; then
  bazel_cmd="$(command -v bazelisk)"
elif command -v bazel >/dev/null 2>&1; then
  bazel_cmd="$(command -v bazel)"
else
  bazel_cmd=""
fi

missing=()
[[ -n "$bazel_cmd" ]] || missing+=("bazel/bazelisk")
pkg-config --exists openblas 2>/dev/null || missing+=("libopenblas-dev")
pkg-config --exists lcm 2>/dev/null || missing+=("liblcm-dev")

available_kb="$(df -Pk /data1 | awk 'NR == 2 {print $4}')"
available_gb="$((available_kb / 1024 / 1024))"

echo "Push Anything: $UPSTREAM_ROOT"
echo "Upstream commit: $actual_commit"
echo "Bazel output root: $OUTPUT_USER_ROOT"
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
  "$bazel_cmd" --version
  echo "Native C3+ build prerequisites passed."
  exit 0
fi

mkdir -p "$OUTPUT_USER_ROOT"
cd "$UPSTREAM_ROOT"

"$bazel_cmd" \
  --output_user_root="$OUTPUT_USER_ROOT" \
  build \
  --define=WITH_GUROBI=OFF \
  --config=omp \
  //examples/sampling_c3:franka_sim \
  //examples/sampling_c3:franka_osc_controller \
  //examples/sampling_c3:franka_sampling_c3_controller

echo "Native Push Anything C3+ targets built successfully."
