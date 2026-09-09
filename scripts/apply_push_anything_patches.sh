#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${1:-/data1/linsixu/dairlib-push-anything}"
PATCH_PROFILE="${2:-${PUSH_ANYTHING_PATCH_PROFILE:-extended}}"
EXPECTED_COMMIT="9d988c835d6e99330397701487fce5ce4ceafa3c"
SEMANTIC_PATCH="$REPO_ROOT/third_party/push_anything/patches/0001-safe-only-sampling-mesh.patch"
POSE_EFFECT_PATCH="$REPO_ROOT/third_party/push_anything/patches/0004-target-only-pose-effect-guard.patch"
ONLINE_BUILD_PATCH="$REPO_ROOT/third_party/push_anything/patches/0003-online-isaac-bridge-build.patch"
BEST_OBJECT_PLAN_PATCH="$REPO_ROOT/third_party/push_anything/patches/0005-fix-best-object-plan-channel.patch"
SIGNED_YAW_DEADBAND_PATCH="$REPO_ROOT/third_party/push_anything/patches/0006-signed-yaw-acceptance-deadband.patch"
CLOSED_GRIPPER_PATCH="$REPO_ROOT/third_party/push_anything/patches/0007-closed-gripper-convex-contact.patch"
SIMULATION_CLOCK_PATCH="$REPO_ROOT/third_party/push_anything/patches/0008-synchronous-simulation-plan-clock.patch"
RELAY_SOURCE="$REPO_ROOT/third_party/push_anything/online_bridge_relay.py"
RELAY_DESTINATION="$UPSTREAM_ROOT/examples/sampling_c3/online_bridge_relay.py"

if [[ ! -e "$UPSTREAM_ROOT/.git" ]]; then
  echo "Push Anything checkout not found: $UPSTREAM_ROOT" >&2
  exit 2
fi

ACTUAL_COMMIT="$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Expected upstream $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
  exit 3
fi
if [[ "$PATCH_PROFILE" != "canonical" && "$PATCH_PROFILE" != "extended" ]]; then
  echo "Patch profile must be canonical or extended, found: $PATCH_PROFILE" >&2
  exit 4
fi

if rg -q 'std::optional<std::vector<std::string>> sampling_meshes' \
    "$UPSTREAM_ROOT/examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h" && \
   rg -q 'neutral_yaw_contact_max_moment_arm' \
    "$UPSTREAM_ROOT/examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h" && \
   rg -q 'SemanticC1TrajectoryGuard' \
    "$UPSTREAM_ROOT/examples/sampling_c3/franka_osc_controller.cc" && \
   [[ -f "$UPSTREAM_ROOT/examples/sampling_c3/semantic_c1_trajectory_guard.cc" ]] && \
   [[ -f "$UPSTREAM_ROOT/examples/sampling_c3/monitor_push_anything_baseline.py" ]]; then
  echo "Semantic C1 patch already applied: $SEMANTIC_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --unidiff-zero --recount --check "$SEMANTIC_PATCH"
  git -C "$UPSTREAM_ROOT" apply --unidiff-zero --recount "$SEMANTIC_PATCH"
  echo "Applied: $SEMANTIC_PATCH"
fi

if rg -q 'use_simulation_time_for_plans' "$UPSTREAM_ROOT/systems/controllers/sampling_based_c3_controller.cc"; then
  git -C "$UPSTREAM_ROOT" apply --reverse --check "$SIMULATION_CLOCK_PATCH"
  echo "Simulation plan clock patch already applied: $SIMULATION_CLOCK_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --check "$SIMULATION_CLOCK_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$SIMULATION_CLOCK_PATCH"
  echo "Applied: $SIMULATION_CLOCK_PATCH"
fi

if [[ "$PATCH_PROFILE" == "extended" ]]; then
  if rg -q 'std::optional<std::vector<int>> sampleable_objects' \
      "$UPSTREAM_ROOT/examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h" && \
     rg -q 'pose_effect_max_horizon_rotation_error' \
      "$UPSTREAM_ROOT/examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h" && \
     rg -q 'TARGET_EFFECT selected=' \
      "$UPSTREAM_ROOT/systems/controllers/sampling_based_c3_controller.cc"; then
    echo "Target-only pose-effect patch already applied: $POSE_EFFECT_PATCH"
  else
    git -C "$UPSTREAM_ROOT" apply --check "$POSE_EFFECT_PATCH"
    git -C "$UPSTREAM_ROOT" apply "$POSE_EFFECT_PATCH"
    echo "Applied: $POSE_EFFECT_PATCH"
  fi
else
  echo "Canonical profile: skipped target pose-effect experiment patch"
fi

if rg -q 'name = "online_bridge_relay"' \
    "$UPSTREAM_ROOT/examples/sampling_c3/BUILD.bazel"; then
  echo "Online relay BUILD patch already applied: $ONLINE_BUILD_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --unidiff-zero --check "$ONLINE_BUILD_PATCH"
  git -C "$UPSTREAM_ROOT" apply --unidiff-zero "$ONLINE_BUILD_PATCH"
  echo "Applied: $ONLINE_BUILD_PATCH"
fi

if rg -q 'get_output_port_dynamically_feasible_best_plan_object' \
    "$UPSTREAM_ROOT/examples/sampling_c3/franka_sampling_c3_controller.cc"; then
  echo "Best object-plan channel patch already applied: $BEST_OBJECT_PLAN_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --check "$BEST_OBJECT_PLAN_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$BEST_OBJECT_PLAN_PATCH"
  echo "Applied: $BEST_OBJECT_PLAN_PATCH"
fi

if [[ -f "$UPSTREAM_ROOT/systems/controllers/closed_gripper_contact_model.h" ]]; then
  FR3_PARTS_PATCH="$REPO_ROOT/third_party/push_anything/patches/0025-fr3-multipart-finger-geometry.patch"
  if git -C "$UPSTREAM_ROOT" apply --reverse --check "$FR3_PARTS_PATCH" 2>/dev/null; then
    # 0007 adds the complete header, so its reverse check must see the
    # pre-0025 bytes. Verify that composition in a temporary copy; never
    # roll back the live checkout during an idempotency check.
    python3 "$REPO_ROOT/scripts/verify_push_anything_patch_composition.py" \
      "$UPSTREAM_ROOT" "$FR3_PARTS_PATCH" "$CLOSED_GRIPPER_PATCH"
  else
    git -C "$UPSTREAM_ROOT" apply --reverse --check "$CLOSED_GRIPPER_PATCH"
  fi
  echo "Closed-gripper geometry patch already applied: $CLOSED_GRIPPER_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --check "$CLOSED_GRIPPER_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$CLOSED_GRIPPER_PATCH"
  echo "Applied: $CLOSED_GRIPPER_PATCH (activated only by explicit mesh configuration)"
fi

if [[ "$PATCH_PROFILE" == "extended" ]]; then
  if rg -q 'neutral_yaw_contact_deadband' \
      "$UPSTREAM_ROOT/examples/sampling_c3/parameter_headers/sampling_c3_controller_params.h"; then
    echo "Signed-yaw acceptance deadband patch already applied: $SIGNED_YAW_DEADBAND_PATCH"
  else
    git -C "$UPSTREAM_ROOT" apply --check "$SIGNED_YAW_DEADBAND_PATCH"
    git -C "$UPSTREAM_ROOT" apply "$SIGNED_YAW_DEADBAND_PATCH"
    echo "Applied: $SIGNED_YAW_DEADBAND_PATCH"
  fi
else
  echo "Canonical profile: skipped signed-yaw acceptance experiment patch"
fi

ACTOR_WORKSPACE_PATCH="$REPO_ROOT/third_party/push_anything/patches/0009-actor-workspace-constraints.patch"
if rg -q 'enforce_actor_workspace_bounds' "$UPSTREAM_ROOT/systems/controllers/sampling_based_c3_controller.cc"; then
  git -C "$UPSTREAM_ROOT" apply --reverse --check "$ACTOR_WORKSPACE_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --check "$ACTOR_WORKSPACE_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$ACTOR_WORKSPACE_PATCH"
fi

SOLVER_BINDINGS_PATCH="$REPO_ROOT/third_party/push_anything/patches/0010-c3-solver-diagnostic-bindings.patch"
if rg -q '@c3//lcmtypes:lcmtypes_c3_py' "$UPSTREAM_ROOT/examples/sampling_c3/BUILD.bazel"; then
  git -C "$UPSTREAM_ROOT" apply --reverse --check "$SOLVER_BINDINGS_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --check "$SOLVER_BINDINGS_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$SOLVER_BINDINGS_PATCH"
fi

NONNEGATIVE_FORCE_PATCH="$REPO_ROOT/third_party/push_anything/patches/0011-nonnegative-contact-force-constraints.patch"
if rg -q 'enforce_nonnegative_contact_forces' "$UPSTREAM_ROOT/systems/controllers/sampling_based_c3_controller.cc"; then
  git -C "$UPSTREAM_ROOT" apply --reverse --check "$NONNEGATIVE_FORCE_PATCH"
else
  git -C "$UPSTREAM_ROOT" apply --check "$NONNEGATIVE_FORCE_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$NONNEGATIVE_FORCE_PATCH"
fi

SEMANTIC_QUATERNION_PATCH="$REPO_ROOT/third_party/push_anything/patches/0012-fix-semantic-guard-quaternion-order.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$SEMANTIC_QUATERNION_PATCH" 2>/dev/null; then
  echo "Semantic quaternion order patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$SEMANTIC_QUATERNION_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$SEMANTIC_QUATERNION_PATCH"
fi

FOH_PD_PATCH="$REPO_ROOT/third_party/push_anything/patches/0013-optional-foh-pd-rollout.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$FOH_PD_PATCH" 2>/dev/null; then
  echo "Optional FOH PD rollout patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$FOH_PD_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$FOH_PD_PATCH"
fi

PD_COMPARE_PATCH="$REPO_ROOT/third_party/push_anything/patches/0014-readonly-pd-reference-comparison.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$PD_COMPARE_PATCH" 2>/dev/null; then
  echo "Read-only PD reference comparison patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$PD_COMPARE_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$PD_COMPARE_PATCH"
fi

PD_CALIBRATION_PATCH="$REPO_ROOT/third_party/push_anything/patches/0015-readonly-pd-inertia-calibration.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$PD_CALIBRATION_PATCH" 2>/dev/null; then
  echo "Read-only PD inertia calibration patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$PD_CALIBRATION_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$PD_CALIBRATION_PATCH"
fi

RELINEARIZED_COST_PATCH="$REPO_ROOT/third_party/push_anything/patches/0016-optional-relinearized-pd-cost.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$RELINEARIZED_COST_PATCH" 2>/dev/null; then
  echo "Optional relinearized PD cost patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$RELINEARIZED_COST_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$RELINEARIZED_COST_PATCH"
fi

COARSE_MODEL_PATCH="$REPO_ROOT/third_party/push_anything/patches/0017-optional-osc-matched-coarse-dynamics.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$COARSE_MODEL_PATCH" 2>/dev/null; then
  echo "Optional OSC matched coarse dynamics patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$COARSE_MODEL_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$COARSE_MODEL_PATCH"
fi

FINGER_FLOOR_PATCH="$REPO_ROOT/third_party/push_anything/patches/0018-planner-finger-table-clearance.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$FINGER_FLOOR_PATCH" 2>/dev/null; then
  echo "Planner finger table clearance patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$FINGER_FLOOR_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$FINGER_FLOOR_PATCH"
fi

OFFLINE_PD_PATCH="$REPO_ROOT/third_party/push_anything/patches/0019-offline-pd-reference-replay.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$OFFLINE_PD_PATCH" 2>/dev/null; then
  echo "Offline PD reference replay patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$OFFLINE_PD_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$OFFLINE_PD_PATCH"
fi

FINITE_HYSTERESIS_PATCH="$REPO_ROOT/third_party/push_anything/patches/0020-finite-cost-hysteresis.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$FINITE_HYSTERESIS_PATCH" 2>/dev/null; then
  echo "Finite-cost hysteresis patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$FINITE_HYSTERESIS_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$FINITE_HYSTERESIS_PATCH"
fi

ALIGNED_CLEARANCE_PATCH="$REPO_ROOT/third_party/push_anything/patches/0021-align-closed-gripper-planning-clearance.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$ALIGNED_CLEARANCE_PATCH" 2>/dev/null; then
  echo "Closed-gripper planning clearance alignment patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$ALIGNED_CLEARANCE_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$ALIGNED_CLEARANCE_PATCH"
fi

PRECISE_CAPTURE_PATCH="$REPO_ROOT/third_party/push_anything/patches/0022-readonly-precise-pd-capture.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$PRECISE_CAPTURE_PATCH" 2>/dev/null; then
  echo "Read-only precise PD capture patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$PRECISE_CAPTURE_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$PRECISE_CAPTURE_PATCH"
fi

OSC_DYNAMICS_PATCH="$REPO_ROOT/third_party/push_anything/patches/0023-offline-native-osc-dynamics.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$REPO_ROOT/third_party/push_anything/patches/0026-separate-simulation-osc-state-channel.patch" 2>/dev/null; then
  python3 "$REPO_ROOT/scripts/verify_push_anything_patch_composition.py" "$UPSTREAM_ROOT" \
    "$REPO_ROOT/third_party/push_anything/patches/0026-separate-simulation-osc-state-channel.patch" "$OSC_DYNAMICS_PATCH"
  echo "Offline native OSC diagnostic and separate channel patches verified"
elif git -C "$UPSTREAM_ROOT" apply --reverse --check "$OSC_DYNAMICS_PATCH" 2>/dev/null; then
  echo "Offline native OSC dynamics diagnostic patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$OSC_DYNAMICS_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$OSC_DYNAMICS_PATCH"
fi

FR3_MODEL_PATCH="$REPO_ROOT/third_party/push_anything/patches/0024-optional-matched-fr3-robot-model.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$FR3_MODEL_PATCH" 2>/dev/null; then
  echo "Optional matched FR3 robot model patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$FR3_MODEL_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$FR3_MODEL_PATCH"
fi

FR3_PARTS_PATCH="$REPO_ROOT/third_party/push_anything/patches/0025-fr3-multipart-finger-geometry.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$FR3_PARTS_PATCH" 2>/dev/null; then
  echo "FR3 multipart finger geometry patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$FR3_PARTS_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$FR3_PARTS_PATCH"
fi

OSC_CHANNEL_PATCH="$REPO_ROOT/third_party/push_anything/patches/0026-separate-simulation-osc-state-channel.patch"
if git -C "$UPSTREAM_ROOT" apply --reverse --check "$OSC_CHANNEL_PATCH" 2>/dev/null; then
  echo "Separate simulation OSC state channel patch already applied"
else
  git -C "$UPSTREAM_ROOT" apply --check "$OSC_CHANNEL_PATCH"
  git -C "$UPSTREAM_ROOT" apply "$OSC_CHANNEL_PATCH"
fi

if cmp -s "$RELAY_SOURCE" "$RELAY_DESTINATION"; then
  echo "Relay source already synchronized: $RELAY_DESTINATION"
else
  install -m 0644 "$RELAY_SOURCE" "$RELAY_DESTINATION"
  echo "Synchronized relay source: $RELAY_DESTINATION"
fi
