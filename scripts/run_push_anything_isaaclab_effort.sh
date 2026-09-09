#!/usr/bin/env bash
set -euo pipefail

# Minimal reference bridge: released Push Anything C3 + released Franka OSC
# produce joint efforts; Isaac supplies only measured state and PhysX dynamics.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${PUSH_ANYTHING_ROOT:-/data1/linsixu/dairlib-push-anything-paper-baseline}"
PYTHON_BIN="${ISAACLAB_PYTHON:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
TCPQ_PORT="${PUSH_ANYTHING_TCPQ_PORT:-8071}"
RELAY_PORT="${PUSH_ANYTHING_RELAY_PORT:-8072}"
DEMO_NAME="${PUSH_ANYTHING_DEMO_NAME:-hammer_safe_physical}"
COMMAND_WAIT_MS="${PUSH_ANYTHING_COMMAND_WAIT_MS:-5}"
OBJECT_STATE_CHANNEL="${PUSH_ANYTHING_OBJECT_STATE_CHANNEL:-OBJECT_STATE_SIMULATION}"
FRANKA_INPUT_CHANNEL="${PUSH_ANYTHING_FRANKA_INPUT_CHANNEL:-FRANKA_INPUT}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MANIFEST.jsonl RESULT.json [Isaac executor arguments...]" >&2
  exit 2
fi

MANIFEST="$(realpath "$1")"
RESULT="$(realpath -m "$2")"
shift 2
OUTPUT_DIR="$(dirname "$RESULT")"
BIN_DIR="$UPSTREAM_ROOT/bazel-bin/examples/sampling_c3"
RUNFILES="$BIN_DIR/xbox_script.runfiles"
LCM_REPO="$RUNFILES/drake++drake_dep_repositories+lcm"
LCM_TYPES="$RUNFILES/_main/lcmtypes"
LCM_URL="tcpq://127.0.0.1:$TCPQ_PORT"
PROTOCOL="$REPO_ROOT/source/IsaacLab_nonPrehensile/dapl/contact_planner/c3_online_protocol.py"

required=(
  "$MANIFEST"
  "$PYTHON_BIN"
  "$BIN_DIR/franka_sampling_c3_controller"
  "$BIN_DIR/franka_osc_controller"
  "$LCM_TYPES"
  "$LCM_REPO/gen"
  "$PROTOCOL"
  "$REPO_ROOT/third_party/push_anything/online_bridge_relay.py"
  "$REPO_ROOT/scripts/run_c3_online_isaaclab.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing effort-bridge prerequisite: $path" >&2
    exit 3
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

/usr/bin/python3 -u "$REPO_ROOT/scripts/lcm_tcpq_hub.py" --port "$TCPQ_PORT" \
  >"$OUTPUT_DIR/hub_online.log" 2>&1 &
child_pids+=("$!")
sleep 0.2

cd "$UPSTREAM_ROOT"
"$BIN_DIR/franka_sampling_c3_controller" \
  --is_simulation=true --demo_name="$DEMO_NAME" --lcm_url="$LCM_URL" \
  >"$OUTPUT_DIR/controller_online.log" 2>&1 &
controller_pid="$!"
child_pids+=("$controller_pid")
"$BIN_DIR/franka_osc_controller" \
  --is_simulation=true --demo_name="$DEMO_NAME" --lcm_url="$LCM_URL" \
  >"$OUTPUT_DIR/osc_online.log" 2>&1 &
osc_pid="$!"
child_pids+=("$osc_pid")

# A dead native process otherwise looks like a stream of valid zero-effort
# watchdog packets and lets the gravity-enabled robot collapse for seconds.
sleep 0.2
if ! kill -0 "$controller_pid" 2>/dev/null; then
  echo "ERROR: C3 controller exited; see $OUTPUT_DIR/controller_online.log" >&2
  tail -20 "$OUTPUT_DIR/controller_online.log" >&2 || true
  exit 4
fi
if ! kill -0 "$osc_pid" 2>/dev/null; then
  echo "ERROR: OSC controller exited; see $OUTPUT_DIR/osc_online.log" >&2
  tail -20 "$OUTPUT_DIR/osc_online.log" >&2 || true
  exit 4
fi

env \
  "PYTHONPATH=$LCM_TYPES:$LCM_REPO/gen:$LCM_REPO${PYTHONPATH:+:$PYTHONPATH}" \
  /usr/bin/python3 -u "$REPO_ROOT/third_party/push_anything/online_bridge_relay.py" \
    --protocol-module "$PROTOCOL" \
    --listen-port "$RELAY_PORT" \
    --socket-timeout-s 60 \
    --lcm-url "$LCM_URL" \
    --command-mode effort \
    --object-state-channel "$OBJECT_STATE_CHANNEL" \
    --franka-input-channel "$FRANKA_INPUT_CHANNEL" \
    --command-wait-ms "$COMMAND_WAIT_MS" \
  >"$OUTPUT_DIR/relay_online.log" 2>&1 &
child_pids+=("$!")
relay_pid="$!"

for _attempt in $(seq 1 100); do
  if rg -q 'C3_ONLINE_RELAY_LISTENING' "$OUTPUT_DIR/relay_online.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$relay_pid" 2>/dev/null; then
    echo "ERROR: effort relay exited; see $OUTPUT_DIR/relay_online.log" >&2
    exit 4
  fi
  if ! kill -0 "$controller_pid" 2>/dev/null; then
    echo "ERROR: C3 controller exited while starting relay" >&2
    exit 4
  fi
  if ! kill -0 "$osc_pid" 2>/dev/null; then
    echo "ERROR: OSC controller exited while starting relay" >&2
    exit 4
  fi
  sleep 0.1
done
if ! rg -q 'C3_ONLINE_RELAY_LISTENING' "$OUTPUT_DIR/relay_online.log" 2>/dev/null; then
  echo "ERROR: timed out waiting for effort relay" >&2
  exit 5
fi

cd "$REPO_ROOT"
set +e
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH="$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" scripts/run_c3_online_isaaclab.py \
  --manifest "$MANIFEST" --output "$RESULT" --relay-port "$RELAY_PORT" "$@"
executor_status="$?"
set -e

wait "$relay_pid" 2>/dev/null || true
if [[ ! -f "$RESULT" ]]; then
  echo "ERROR: effort executor did not write $RESULT" >&2
  exit "${executor_status:-6}"
fi
exit "$executor_status"
