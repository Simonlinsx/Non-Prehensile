#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${PUSH_ANYTHING_ROOT:-}"
BINARY_ROOT="${PUSH_ANYTHING_BINARY_ROOT:-}"
PYTHON_BIN="${ISAACLAB_PYTHON:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
TCPQ_PORT="${PUSH_ANYTHING_TCPQ_PORT:-7796}"
RELAY_PORT="${PUSH_ANYTHING_RELAY_PORT:-7797}"
# The native Push Anything OSC evaluates a newly received trajectory at the
# current robot-state timestamp.  A full planner-period lookahead skips the
# contact-entry portion of short C3 segments and is especially harmful when a
# physical Franka cannot instantaneously track the new reference.
LOOKAHEAD_MS="${PUSH_ANYTHING_TASK_LOOKAHEAD_MS:-0}"
FRESH_TASK_TIMEOUT_MS="${PUSH_ANYTHING_FRESH_TASK_TIMEOUT_MS:-1000}"
FRESH_TIMESTAMP_TOLERANCE_US="${PUSH_ANYTHING_FRESH_TIMESTAMP_TOLERANCE_US:-100}"
RELAY_SOCKET_TIMEOUT_S="${PUSH_ANYTHING_RELAY_SOCKET_TIMEOUT_S:-60}"
STATE_MODE="${PUSH_ANYTHING_STATE_MODE:-single}"
SCENE_SPEC="${PUSH_ANYTHING_SCENE_SPEC:-}"
DEMO_NAME="${PUSH_ANYTHING_DEMO_NAME:-anything}"
PY_BINDINGS_ROOT="${PUSH_ANYTHING_PY_BINDINGS_ROOT:-/data1/linsixu/dairlib-push-anything-upstream-baseline}"
PY_RUNFILES="${PUSH_ANYTHING_PY_RUNFILES:-}"
OBJECT_STATE_CHANNEL="${PUSH_ANYTHING_OBJECT_STATE_CHANNEL:-}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EXECUTOR_MODE="${PUSH_ANYTHING_EXECUTOR_MODE:-task}"
EXECUTOR_SCRIPT="$REPO_ROOT/scripts/run_c3_online_isaaclab_task.py"
if [[ "$EXECUTOR_MODE" == "effort" ]]; then
  EXECUTOR_SCRIPT="$REPO_ROOT/scripts/run_c3_online_isaaclab.py"
elif [[ "$EXECUTOR_MODE" != "task" ]]; then
  echo "ERROR: PUSH_ANYTHING_EXECUTOR_MODE must be task or effort" >&2
  exit 2
fi
if [[ "$EXECUTOR_MODE" == "effort" && "$STATE_MODE" != "single" ]]; then
  echo "ERROR: native OSC diagnostic currently supports target-only packets" >&2
  exit 2
fi

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage: run_push_anything_isaaclab_online.sh MANIFEST.jsonl RESULT.json [Isaac executor arguments...]

The selected Push Anything checkout must already be staged for exactly the
same scene as MANIFEST.jsonl.  Use PUSH_ANYTHING_ROOT to select that checkout.
EOF
  exit 2
fi

MANIFEST="$(realpath "$1")"
RESULT="$(realpath -m "$2")"
shift 2
PROTOCOL="$REPO_ROOT/source/IsaacLab_nonPrehensile/dapl/contact_planner/c3_online_protocol.py"
OUTPUT_DIR="$(dirname "$RESULT")"
EFFECT_AUDIT_OUTPUT="${PUSH_ANYTHING_EFFECT_AUDIT_OUTPUT:-$OUTPUT_DIR/effect_audit.jsonl}"
EFFECT_AUDIT_SUMMARY="${PUSH_ANYTHING_EFFECT_AUDIT_SUMMARY:-$OUTPUT_DIR/effect_audit_summary.json}"
EFFECT_PREDICTION_STEP_S="${PUSH_ANYTHING_EFFECT_PREDICTION_STEP_S:-}"
EFFECT_PREDICTION_KNOT_INDEX="${PUSH_ANYTHING_EFFECT_PREDICTION_KNOT_INDEX:-1}"
LCM_URL="tcpq://127.0.0.1:$TCPQ_PORT"

if [[ "$STATE_MODE" != "single" && "$STATE_MODE" != "scene" ]]; then
  echo "ERROR: PUSH_ANYTHING_STATE_MODE must be single or scene" >&2
  exit 2
fi
if [[ "$STATE_MODE" == "scene" ]]; then
  if [[ -z "$SCENE_SPEC" ]]; then
    echo "ERROR: scene mode requires PUSH_ANYTHING_SCENE_SPEC" >&2
    exit 2
  fi
  SCENE_SPEC="$(realpath "$SCENE_SPEC")"
  if [[ -z "$UPSTREAM_ROOT" ]]; then
    UPSTREAM_ROOT="$("$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("runtime_root", ""))' \
      "$SCENE_SPEC")"
  fi
fi
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/data1/linsixu/dairlib-push-anything}"
UPSTREAM_ROOT="$(realpath "$UPSTREAM_ROOT")"
BINARY_ROOT="${BINARY_ROOT:-$UPSTREAM_ROOT}"
BINARY_ROOT="$(realpath "$BINARY_ROOT")"
export PUSH_ANYTHING_ROOT="$UPSTREAM_ROOT"
if [[ -z "$EFFECT_PREDICTION_STEP_S" ]]; then
  EFFECT_PREDICTION_STEP_S="$("$PYTHON_BIN" - "$UPSTREAM_ROOT" "$DEMO_NAME" "$EFFECT_PREDICTION_KNOT_INDEX" <<'PY'
import math
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
params = yaml.safe_load((root / "examples/sampling_c3" / sys.argv[2]
                         / "parameters/sampling_c3_controller_params.yaml").read_text())
options = yaml.safe_load((root / params["sampling_c3_options_file"]).read_text())
position_dt = float(options["planning_dt_position"])
pose_dt = float(options["planning_dt_pose"])
if position_dt <= 0 or not math.isclose(position_dt, pose_dt):
    raise ValueError("Effect audit needs a common physical knot duration; unequal mode durations require explicit auditing")
knot = int(sys.argv[3])
if knot <= 0:
    raise ValueError("Effect prediction knot must be positive")
print(position_dt * knot)
PY
)"
fi
BIN_DIR="$BINARY_ROOT/bazel-bin/examples/sampling_c3"
RELAY_BINARY="$BIN_DIR/online_bridge_relay"
RELAY_SOURCE="$REPO_ROOT/third_party/push_anything/online_bridge_relay.py"

# multiyaml_rewrite may give a staged object its own LCM state channel.  Read
# that channel from the exact runtime demo instead of silently publishing the
# legacy OBJECT_STATE_SIMULATION channel that some upstream demos use.
if [[ "$STATE_MODE" == "single" && -z "$OBJECT_STATE_CHANNEL" ]]; then
  controller_params="$UPSTREAM_ROOT/examples/sampling_c3/$DEMO_NAME/parameters/sampling_c3_controller_params.yaml"
  if [[ -f "$controller_params" ]]; then
    lcm_params_rel="$(sed -nE 's/^[[:space:]]*lcm_channels_simulation_file:[[:space:]]*([^#[:space:]]+).*$/\1/p' "$controller_params" | head -n 1)"
    if [[ -n "$lcm_params_rel" && -f "$UPSTREAM_ROOT/$lcm_params_rel" ]]; then
      OBJECT_STATE_CHANNEL="$(sed -nE 's/^[[:space:]]*object_state_channels:[[:space:]]*\[[[:space:]]*([^],[:space:]]+).*$/\1/p' "$UPSTREAM_ROOT/$lcm_params_rel" | head -n 1)"
    fi
  fi
  OBJECT_STATE_CHANNEL="${OBJECT_STATE_CHANNEL:-OBJECT_STATE_SIMULATION}"
fi

required_files=(
  "$MANIFEST"
  "$PYTHON_BIN"
  "$PROTOCOL"
  "$REPO_ROOT/scripts/lcm_tcpq_hub.py"
  "$EXECUTOR_SCRIPT"
  "$BIN_DIR/franka_sampling_c3_controller"
)
if [[ "$EXECUTOR_MODE" == "effort" ]]; then
  required_files+=("$BIN_DIR/franka_osc_controller")
fi
if [[ "$STATE_MODE" == "scene" ]]; then
  required_files+=("$SCENE_SPEC")
fi
for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing online C3+ prerequisite: $path" >&2
    exit 3
  fi
done

if [[ -x "$RELAY_BINARY" ]]; then
  relay_command=("$RELAY_BINARY")
else
  if [[ -n "$PY_RUNFILES" ]]; then
    relay_runfiles_candidates=("$PY_RUNFILES")
  else
    # Any Python target built from this workspace carries the generated LCM
    # bindings and Drake's native Python module.  Prefer relay/monitor targets,
    # but fall back to an already-built utility target instead of depending on
    # a second checkout whose Bazel outputs may have been cleaned.
    relay_runfiles_candidates=(
      "$BINARY_ROOT/bazel-bin/examples/sampling_c3/online_bridge_relay.runfiles"
      "$BINARY_ROOT/bazel-bin/examples/sampling_c3/monitor_push_anything_baseline.runfiles"
      "$BINARY_ROOT/bazel-bin/examples/sampling_c3/xbox_script.runfiles"
      "$UPSTREAM_ROOT/bazel-bin/examples/sampling_c3/online_bridge_relay.runfiles"
      "$UPSTREAM_ROOT/bazel-bin/examples/sampling_c3/monitor_push_anything_baseline.runfiles"
      "$UPSTREAM_ROOT/bazel-bin/examples/sampling_c3/xbox_script.runfiles"
      "$PY_BINDINGS_ROOT/bazel-bin/examples/sampling_c3/monitor_push_anything_baseline.runfiles"
      "$PY_BINDINGS_ROOT/bazel-bin/examples/sampling_c3/xbox_script.runfiles"
    )
  fi
  RELAY_RUNFILES=""
  for candidate in "${relay_runfiles_candidates[@]}"; do
    if [[ -d "$candidate/_main/lcmtypes" ]]; then
      RELAY_RUNFILES="$candidate"
      break
    fi
  done
  if [[ -z "$RELAY_RUNFILES" ]]; then
    echo "ERROR: no built Python target provides generated LCM bindings" >&2
    echo "Checked runfiles candidates:" >&2
    printf '  %s\n' "${relay_runfiles_candidates[@]}" >&2
    exit 3
  fi
  LCM_TYPES="$RELAY_RUNFILES/_main/lcmtypes"
  if [[ -d "$RELAY_RUNFILES/lcm+/lcm-python" ]]; then
    # Older external-repository layout used by the development checkout.
    LCM_PYTHON="$RELAY_RUNFILES/lcm+/lcm-python"
  else
    # Drake's current Bzlmod LCM target exposes a generated import shim under
    # ``gen``; it preloads the native _lcm extension with the correct RUNPATH.
    LCM_REPO="$RELAY_RUNFILES/drake++drake_dep_repositories+lcm"
    LCM_PYTHON="$LCM_REPO/gen:$LCM_REPO"
  fi
  for path in "$RELAY_SOURCE" "$LCM_TYPES"; do
    if [[ ! -e "$path" ]]; then
      echo "ERROR: missing fallback online-relay prerequisite: $path" >&2
      exit 3
    fi
  done
  IFS=: read -r -a lcm_python_paths <<<"$LCM_PYTHON"
  for path in "${lcm_python_paths[@]}"; do
    if [[ ! -d "$path" ]]; then
      echo "ERROR: missing fallback online-relay prerequisite: $path" >&2
      exit 3
    fi
  done
  relay_command=(
    env "PYTHONPATH=$LCM_TYPES:$RELAY_RUNFILES/c3+/lcmtypes:$LCM_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
    /usr/bin/python3 -u "$RELAY_SOURCE"
  )
fi

if [[ -n "${PUSH_ANYTHING_ROBOT_MODEL_MANIFEST:-}" ]]; then
  if [[ "$EXECUTOR_MODE" != "effort" ]]; then
    echo "ERROR: matched FR3 manifest currently requires the native OSC effort executor" >&2
    exit 2
  fi
  export PUSH_ANYTHING_ROBOT_MODEL_MANIFEST
  PUSH_ANYTHING_ROBOT_MODEL="$("$PYTHON_BIN" "$REPO_ROOT/scripts/fr3_robot_model_contract.py" "$PUSH_ANYTHING_ROBOT_MODEL_MANIFEST")"
  export PUSH_ANYTHING_ROBOT_MODEL
fi

mkdir -p "$OUTPUT_DIR"
echo "C3_ONLINE_RUNTIME_ROOT $UPSTREAM_ROOT"
echo "C3_ONLINE_BINARY_ROOT $BINARY_ROOT"
echo "C3_ONLINE_DEMO_NAME $DEMO_NAME"
echo "C3_ONLINE_OBJECT_STATE_CHANNEL $OBJECT_STATE_CHANNEL"
echo "C3_EFFECT_PREDICTION_STEP_S $EFFECT_PREDICTION_STEP_S"
osc_channel_args=()
if [[ -n "${PUSH_ANYTHING_PLANNER_PERIOD_MS:-}" ]]; then
  if [[ "$EXECUTOR_MODE" != "effort" ]]; then
    echo "ERROR: split planner/OSC clocks require effort execution" >&2
    exit 2
  fi
  OSC_STATE_CHANNEL="${PUSH_ANYTHING_OSC_STATE_CHANNEL:-FRANKA_STATE_OSC_SIMULATION}"
  osc_channel_args+=(--simulation_osc_state_channel="$OSC_STATE_CHANNEL")
fi
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
if ! kill -0 "${child_pids[0]}" 2>/dev/null; then
  echo "ERROR: LCM TCPQ hub failed; see $OUTPUT_DIR/hub_online.log" >&2
  exit 4
fi

cd "$UPSTREAM_ROOT"
"$BIN_DIR/franka_sampling_c3_controller" \
  --is_simulation=true --demo_name="$DEMO_NAME" --lcm_url="$LCM_URL" \
  >"$OUTPUT_DIR/controller_online.log" 2>&1 &
child_pids+=("$!")

if [[ "$EXECUTOR_MODE" == "effort" ]]; then
  "$BIN_DIR/franka_osc_controller" "${osc_channel_args[@]}" \
    --is_simulation=true --demo_name="$DEMO_NAME" --lcm_url="$LCM_URL" \
    >"$OUTPUT_DIR/osc_online.log" 2>&1 &
  child_pids+=("$!")
fi

relay_args=(
  --protocol-module "$PROTOCOL" \
  --listen-port "$RELAY_PORT" \
  --socket-timeout-s "$RELAY_SOCKET_TIMEOUT_S" \
  --lcm-url "$LCM_URL" \
  --command-mode "$EXECUTOR_MODE" \
  --object-state-channel "$OBJECT_STATE_CHANNEL" \
  --command-wait-ms 0 \
  --task-lookahead-ms "$LOOKAHEAD_MS" \
  --fresh-task-timeout-ms "$FRESH_TASK_TIMEOUT_MS" \
  --fresh-timestamp-tolerance-us "$FRESH_TIMESTAMP_TOLERANCE_US" \
  --effect-audit-output "$EFFECT_AUDIT_OUTPUT"
)
if [[ -n "${PUSH_ANYTHING_PLANNER_PERIOD_MS:-}" ]]; then
  relay_args+=(--franka-state-channel "$OSC_STATE_CHANNEL"
    --planner-state-channel FRANKA_STATE_SIMULATION --planner-period-ms "$PUSH_ANYTHING_PLANNER_PERIOD_MS")
fi
if [[ "$EXECUTOR_MODE" == "effort" ]]; then
  relay_args+=(--fresh-effort-timeout-ms "${PUSH_ANYTHING_FRESH_EFFORT_TIMEOUT_MS:-1000}")
  if [[ "${PUSH_ANYTHING_DIAGNOSTIC_OSC_HOLD:-0}" == "1" ]]; then
    relay_args+=(--diagnostic-osc-hold)
  fi
fi
if [[ -n "${PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S:-}" ]]; then
  relay_args+=(--diagnostic-reference-replay-start-s "$PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S"
               --diagnostic-reference-replay-duration-s "${PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_DURATION_S:-0.675}")
fi
if [[ "$STATE_MODE" == "scene" ]]; then
  relay_args+=(--state-mode scene --scene-spec "$SCENE_SPEC")
fi
"${relay_command[@]}" "${relay_args[@]}" \
  >"$OUTPUT_DIR/relay_online.log" 2>&1 &
child_pids+=("$!")
relay_pid="$!"

for _attempt in $(seq 1 100); do
  if rg -q 'C3_ONLINE_RELAY_LISTENING' "$OUTPUT_DIR/relay_online.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$relay_pid" 2>/dev/null; then
    echo "ERROR: online relay exited; see $OUTPUT_DIR/relay_online.log" >&2
    exit 5
  fi
  sleep 0.1
done
if ! rg -q 'C3_ONLINE_RELAY_LISTENING' "$OUTPUT_DIR/relay_online.log" 2>/dev/null; then
  echo "ERROR: timed out waiting for online relay" >&2
  exit 6
fi

cd "$REPO_ROOT"
if [[ -f "$UPSTREAM_ROOT/shared_contact_model.json" ]]; then
  export PUSH_ANYTHING_CONTACT_MODEL_MANIFEST="$UPSTREAM_ROOT/shared_contact_model.json"
else
  unset PUSH_ANYTHING_CONTACT_MODEL_MANIFEST
fi
executor_scene_args=()
if [[ "$STATE_MODE" == "scene" ]]; then
  executor_scene_args+=(--c3-scene-spec "$SCENE_SPEC")
fi
set +e
OMNI_KIT_ACCEPT_EULA=YES \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$EXECUTOR_SCRIPT" \
  --manifest "$MANIFEST" \
  --output "$RESULT" \
  --relay-port "$RELAY_PORT" \
  "${executor_scene_args[@]}" \
  "$@"
executor_status="$?"
set -e

# The relay flushes each audit event.  Waiting for its normal EOF here makes
# sure the final simulator state is present before producing the summary.
wait "$relay_pid" 2>/dev/null || true
if [[ "$EXECUTOR_MODE" == "task" && -s "$EFFECT_AUDIT_OUTPUT" ]]; then
  effect_audit_args=(
    "$EFFECT_AUDIT_OUTPUT" --output "$EFFECT_AUDIT_SUMMARY"
    --prediction-step-s "$EFFECT_PREDICTION_STEP_S"
    --prediction-knot-index "$EFFECT_PREDICTION_KNOT_INDEX"
  )
  if [[ -f "$RESULT" ]]; then
    effect_audit_args+=(--execution-result "$RESULT")
  fi
  if ! "$PYTHON_BIN" scripts/analyze_c3_effect_audit.py \
      "${effect_audit_args[@]}"; then
    echo "WARNING: failed to summarize C3 object-effect audit" >&2
  fi
fi

if [[ ! -f "$RESULT" ]]; then
  echo "ERROR: Isaac executor did not write $RESULT" >&2
  if (( executor_status != 0 )); then
    exit "$executor_status"
  fi
  exit 7
fi
if (( executor_status != 0 )); then
  exit "$executor_status"
fi
"$PYTHON_BIN" -c \
  'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["online_closed_loop_success"] else 2)' \
  "$RESULT"
