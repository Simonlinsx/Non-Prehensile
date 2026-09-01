#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${PUSH_ANYTHING_ROOT:-/data1/linsixu/dairlib-push-anything}"
TIMEOUT_S="${PUSH_ANYTHING_TIMEOUT_S:-180}"
TCPQ_PORT="${PUSH_ANYTHING_TCPQ_PORT:-7700}"
LCM_URL="${PUSH_ANYTHING_LCM_URL:-tcpq://127.0.0.1:$TCPQ_PORT}"
RUN_NAME="${PUSH_ANYTHING_RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)_official_single_object}"
OUTPUT_DIR="${PUSH_ANYTHING_OUTPUT_DIR:-$REPO_ROOT/outputs/contact_planner_m3/$RUN_NAME}"
BIN_DIR="$UPSTREAM_ROOT/bazel-bin/examples/sampling_c3"

required_files=(
  "$BIN_DIR/franka_sim"
  "$BIN_DIR/franka_osc_controller"
  "$BIN_DIR/franka_sampling_c3_controller"
  "$BIN_DIR/monitor_push_anything_baseline"
  "$REPO_ROOT/scripts/lcm_tcpq_hub.py"
)
for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing build artifact: $path" >&2
    echo "Run scripts/build_push_anything_native.sh --build first." >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
child_pids=()
cleanup() {
  local pid
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

cd "$UPSTREAM_ROOT"
/usr/bin/python3 -u "$REPO_ROOT/scripts/lcm_tcpq_hub.py" --port "$TCPQ_PORT" \
  >"$OUTPUT_DIR/lcm_tcpq_hub.log" 2>&1 &
child_pids+=("$!")
sleep 0.2
if ! kill -0 "${child_pids[0]}" 2>/dev/null; then
  echo "ERROR: local LCM TCPQ hub failed to start; see $OUTPUT_DIR/lcm_tcpq_hub.log" >&2
  exit 3
fi

"$BIN_DIR/franka_sampling_c3_controller" --is_simulation=true --demo_name=anything \
  --lcm_url="$LCM_URL" \
  >"$OUTPUT_DIR/franka_sampling_c3_controller.log" 2>&1 &
child_pids+=("$!")
"$BIN_DIR/franka_osc_controller" --is_simulation=true --demo_name=anything \
  --lcm_url="$LCM_URL" \
  >"$OUTPUT_DIR/franka_osc_controller.log" 2>&1 &
child_pids+=("$!")
sleep 1

"$BIN_DIR/franka_sim" --demo_name=anything --lcm_url="$LCM_URL" \
  >"$OUTPUT_DIR/franka_sim.log" 2>&1 &
child_pids+=("$!")

set +e
"$BIN_DIR/monitor_push_anything_baseline" \
  --output_dir "$OUTPUT_DIR" \
  --timeout_s "$TIMEOUT_S" \
  --lcm_url "$LCM_URL" \
  >"$OUTPUT_DIR/monitor.log" 2>&1
monitor_status="$?"
set -e

for pid in "${child_pids[@]}"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "WARNING: child process $pid exited before monitoring completed." >&2
  fi
done

echo "M3 baseline output: $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/acceptance.json" ]]; then
  cat "$OUTPUT_DIR/acceptance.json"
else
  echo "ERROR: monitor exited without writing acceptance.json; see $OUTPUT_DIR/monitor.log" >&2
fi
exit "$monitor_status"
