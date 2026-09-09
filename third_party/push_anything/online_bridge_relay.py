#!/usr/bin/env python3
"""Relay a fixed-size TCP state/effort stream to Push Anything LCM.

The TCP client is the simulator or real-robot executor.  This process runs in
the Push Anything Bazel Python environment so it can use the upstream LCM
runtime and generated message bindings without installing either into Isaac.
"""

import argparse
import bisect
import copy
import math
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

import lcm

from c3 import lcmt_output
from dairlib import (
    lcmt_object_state,
    lcmt_radio_out,
    lcmt_robot_input,
    lcmt_robot_output,
    lcmt_timestamped_saved_traj,
)


FRANKA_POSITION_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
FRANKA_VELOCITY_NAMES = [f"panda_joint{index}dot" for index in range(1, 8)]
FRANKA_EFFORT_NAMES = [f"panda_motor{index}" for index in range(1, 8)]
DEFAULT_OBJECT_BODY_NAME = "DOMINO_020_hammer_safe"
DEFAULT_OBJECT_STATE_CHANNEL = (
    "OBJECT_DOMINO_020_hammer_safe_STATE_SIMULATION")


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--protocol-module", type=Path, required=True)
  parser.add_argument("--listen-host", default="127.0.0.1")
  parser.add_argument("--listen-port", type=int, default=7790)
  parser.add_argument("--accept-timeout-s", type=float, default=60.0)
  parser.add_argument(
      "--socket-timeout-s", type=float, default=60.0,
      help=(
          "maximum wall-clock idle time between simulator state packets; "
          "multi-object C3 plus rendering can legitimately exceed two seconds"))
  parser.add_argument("--command-wait-ms", type=int, default=2)
  parser.add_argument("--fresh-effort-timeout-ms", type=int, default=0,
      help="simulation handshake: wait for OSC torque at the measured-state timestamp")
  parser.add_argument("--diagnostic-osc-hold", action="store_true",
      help="effort-only integration check using upstream stationary teleop mode; not a pushing trial")
  parser.add_argument(
      "--command-mode", choices=("effort", "task"), default="effort")
  parser.add_argument(
      "--task-lookahead-ms", type=float, default=0.0,
      help=("receding-horizon task sample offset from the current measured "
            "state; zero matches the native Push Anything OSC clock"))
  parser.add_argument(
      "--fresh-task-timeout-ms", type=int, default=0,
      help="wait for a task trajectory planned from the just-published state")
  parser.add_argument(
      "--fresh-timestamp-tolerance-us", type=int, default=100,
      help="tolerance for floating-second to integer-microsecond truncation")
  parser.add_argument("--diagnostic-reference-replay-start-s", type=float,
      help="Planner clock seconds; Isaac timestamps include a 0.1 s origin offset")
  parser.add_argument("--diagnostic-reference-replay-duration-s", type=float, default=.675)
  parser.add_argument("--lcm-url", default="tcpq://127.0.0.1:7791")
  parser.add_argument("--franka-state-channel", default="FRANKA_STATE_SIMULATION")
  parser.add_argument("--planner-state-channel", help="Separate slower planner state channel; effort mode only")
  parser.add_argument("--planner-period-ms", type=float, default=50.0)
  parser.add_argument(
      "--state-mode", choices=("single", "scene"), default="single",
      help="v1 target-only packets or v2 target-plus-clutter packets")
  parser.add_argument(
      "--object-state-channel",
      default=DEFAULT_OBJECT_STATE_CHANNEL,
      help="legacy target channel used in --state-mode=single")
  parser.add_argument(
      "--object-body-names", nargs="+",
      help="target-first C3 base_names; required in scene mode")
  parser.add_argument(
      "--object-state-channels", nargs="+",
      help="target-first C3 LCM channels; required in scene mode")
  parser.add_argument(
      "--scene-spec", type=Path,
      help="shared target-first C3/Isaac object contract for scene mode")
  parser.add_argument("--franka-input-channel", default="FRANKA_INPUT_SIMULATION")
  parser.add_argument(
      "--tracking-channel", default="TRACKING_TRAJECTORY_ACTOR")
  parser.add_argument("--c3-mode-channel", default="IS_C3_MODE")
  parser.add_argument("--radio-channel", default="SAMPLING_C3_RADIO")
  parser.add_argument(
      "--object-plan-channel", default="DYNAMICALLY_FEASIBLE_CURR_PLAN")
  parser.add_argument(
      "--effect-audit-output", type=Path,
      help=(
          "optional JSONL trace containing the measured target state and "
          "C3+'s dynamically feasible object prediction; this is diagnostic "
          "only and never changes the returned command"))
  return parser.parse_args()


def load_protocol(path):
  source = path.expanduser().resolve()
  if not source.is_file():
    raise FileNotFoundError(f"protocol module does not exist: {source}")
  spec = importlib.util.spec_from_file_location("c3_online_protocol", source)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load protocol module: {source}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def load_scene_object_contract(path):
  source = path.expanduser().resolve()
  if not source.is_file():
    raise FileNotFoundError(f"scene spec does not exist: {source}")
  payload = json.loads(source.read_text(encoding="utf-8"))
  if payload.get("schema") != "nonprehensile.c3_online_scene.v1":
    raise ValueError("scene spec has an unsupported schema")
  objects = payload.get("objects")
  if not isinstance(objects, list) or not objects:
    raise ValueError("scene spec objects must be a non-empty list")
  if objects[0].get("role") != "target":
    raise ValueError("scene spec object zero must be the target")
  if any(item.get("role") != "clutter" for item in objects[1:]):
    raise ValueError("scene spec non-target objects must be clutter")
  try:
    body_names = [str(item["c3_body_name"]) for item in objects]
    state_channels = [str(item["state_channel"]) for item in objects]
  except (KeyError, TypeError) as exc:
    raise ValueError(
        "each scene object needs c3_body_name and state_channel") from exc
  if any(not value for value in (*body_names, *state_channels)):
    raise ValueError("scene object names and channels must be non-empty")
  return body_names, state_channels


def effort_is_current(message, state_utime_us, tolerance_us):
  """Reject both old and future torques, allowing only timestamp roundoff."""
  return (message is not None and
          abs(int(message.utime) - state_utime_us) <= tolerance_us)


def receive_exact(connection, size):
  chunks = bytearray()
  while len(chunks) < size:
    data = connection.recv(size - len(chunks))
    if not data:
      if chunks:
        raise ConnectionError("TCP stream ended inside a state packet")
      return None
    chunks.extend(data)
  return bytes(chunks)


def robot_state_message(state):
  message = lcmt_robot_output()
  message.utime = state.utime_us
  message.num_positions = 7
  message.num_velocities = 7
  message.num_efforts = 7
  message.position_names = list(FRANKA_POSITION_NAMES)
  message.position = list(state.joint_position_rad)
  message.velocity_names = list(FRANKA_VELOCITY_NAMES)
  message.velocity = list(state.joint_velocity_rad_s)
  message.effort_names = list(FRANKA_EFFORT_NAMES)
  message.effort = list(state.joint_effort_nm)
  message.imu_accel = [0.0, 0.0, 0.0]
  return message


def object_state_message(utime_us, rigid_body_state, body_name):
  position_names = [
      f"{body_name}_{suffix}"
      for suffix in ("qw", "qx", "qy", "qz", "x", "y", "z")]
  velocity_names = [
      f"{body_name}_{suffix}"
      for suffix in ("wx", "wy", "wz", "vx", "vy", "vz")]
  message = lcmt_object_state()
  message.utime = utime_us
  message.object_name = body_name
  message.num_positions = 7
  message.num_velocities = 6
  message.position_names = position_names
  message.position = [
      *rigid_body_state.quaternion_wxyz,
      *rigid_body_state.position_m,
  ]
  message.velocity_names = velocity_names
  message.velocity = [
      *rigid_body_state.angular_velocity_rad_s,
      *rigid_body_state.linear_velocity_m_s,
  ]
  return message


def legacy_object_state(protocol, state):
  return protocol.C3RigidBodyState(
      quaternion_wxyz=state.object_quaternion_wxyz,
      position_m=state.object_position_m,
      angular_velocity_rad_s=state.object_angular_velocity_rad_s,
      linear_velocity_m_s=state.object_linear_velocity_m_s)


def radio_state_message(protocol, state):
  message = lcmt_radio_out()
  message.radioReceiverSignalGood = True
  message.receiverMedullaSignalGood = True
  message.channel = [0.0] * 16
  if state.flags & protocol.C3StateFlags.FORCE_C3_MODE:
    # Push Anything already exposes force-C3 on radio channel 12.  The
    # executor requests it explicitly; merely observing legal safe contact is
    # behavior-neutral and remains available to diagnostics.
    message.channel[12] = 1.0
  return message


def trajectory_by_name(message, name):
  for trajectory in message.saved_traj.trajectories:
    if trajectory.trajectory_name == name and trajectory.num_points > 0:
      return trajectory
  return None


def target_object_trajectory(message, stem):
  """Return target-object trajectory from upstream's indexed C3 output.

  Push Anything names dynamically feasible object trajectories with an object
  suffix (``*_0`` for the target).  Accept the unsuffixed spelling as well so
  old audit logs remain readable.  This helper is diagnostic only and never
  changes the command relayed to Isaac or a real robot.
  """

  return (
      trajectory_by_name(message, f"{stem}_0")
      or trajectory_by_name(message, stem)
  )


def evaluate_trajectory(trajectory, query_time_s):
  times = [float(value) for value in trajectory.time_vec]
  points = trajectory.datapoints
  if len(times) == 1:
    left = right = 0
    alpha = 0.0
  elif query_time_s <= times[0]:
    # A freshly replanned trajectory starts exactly at the measured-state
    # timestamp.  Its right derivative is the command to execute during the
    # first servo interval; returning zero here makes a zero-latency online
    # executor reset to the first knot and stall at every replan.
    left = 0
    right = 1
    alpha = 0.0
  elif query_time_s >= times[-1]:
    left = right = len(times) - 1
    alpha = 0.0
  else:
    right = bisect.bisect_right(times, query_time_s)
    left = right - 1
    duration = times[right] - times[left]
    alpha = 0.0 if duration <= 0.0 else (
        query_time_s - times[left]) / duration
  value = tuple(
      float(row[left]) + alpha * (float(row[right]) - float(row[left]))
      for row in points)
  if right == left or times[right] <= times[left]:
    velocity = (0.0,) * min(3, len(points))
  else:
    inverse_duration = 1.0 / (times[right] - times[left])
    velocity = tuple(
        (float(points[axis][right]) - float(points[axis][left]))
        * inverse_duration
        for axis in range(min(3, len(points))))
  return value, velocity


class DiagnosticReferenceReplay:
  """One captured native C3 reference, for calibration excluded from acceptance."""

  def __init__(self, start_s, duration_s=.675, tolerance_us=100):
    if start_s is not None and (not math.isfinite(start_s) or start_s < 0):
      raise ValueError("Diagnostic replay start must be finite and nonnegative")
    if not math.isfinite(duration_s) or not 0 < duration_s <= .75:
      raise ValueError("Diagnostic replay duration must be in (0, .75] seconds")
    self.start_us = None if start_s is None else round(start_s * 1e6)
    self.duration_us = round(duration_s * 1e6)
    self.tolerance_us = tolerance_us
    self.captured = None
    self.end_us = None
    self.completed = False

  def select(self, stamp, message, mode, object_plan):
    native_c3 = bool(mode["message"] is not None and mode["value"] and
                    int(mode["message"].utime) + self.tolerance_us >= stamp)
    native = (message, native_c3, bool(mode["yaw_recovery"]), False, None)
    if self.start_us is None or self.completed:
      return native
    event = None
    if self.captured is not None and stamp >= self.end_us:
      self.completed = True
      return (*native[:4], {"event": "diagnostic_reference_replay_end",
                            "utime_us": stamp, "planned_end_utime_us": self.end_us})
    if self.captured is None:
      if (stamp < self.start_us or not native_c3 or message is None or object_plan is None
          or abs(int(message.utime) - stamp) > self.tolerance_us
          or abs(int(object_plan.utime) - int(message.utime)) > self.tolerance_us):
        return native
      position = trajectory_by_name(message, "end_effector_position_target")
      force = trajectory_by_name(message, "end_effector_force_target")
      if position is None or force is None:
        return native
      # The replay must fit in the published trajectory; no extrapolated inputs.
      for trajectory in (position, force):
        times = [float(t) for t in trajectory.time_vec]
        if (len(times) < 2 or not all(math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
          raise ValueError("Invalid diagnostic reference knot times")
        if abs(times[0] * 1e6 - stamp) > self.tolerance_us:
          return native
        if times[-1] * 1e6 + self.tolerance_us < stamp + self.duration_us:
          return native
      self.captured = (copy.deepcopy(message), True, native[2])
      self.end_us = stamp + self.duration_us
      event = {"event": "diagnostic_reference_replay_start", "utime_us": stamp,
               "plan_utime_us": int(message.utime), "end_utime_us": self.end_us,
               "duration_s": self.duration_us * 1e-6,
               "object_plan_utime_us": int(object_plan.utime),
               "object_trajectories": [{"name": t.trajectory_name,
                   "time_vec": list(t.time_vec), "datapoints": [list(row) for row in t.datapoints]}
                   for t in object_plan.saved_traj.trajectories]}
    return (*self.captured, True, event)


class PlannerClockGate:
  """Schedule planner publications using measured simulation microseconds."""
  def __init__(self, period_ms):
    if not math.isfinite(period_ms) or period_ms <= 0:
      raise ValueError("Planner period must be finite and positive")
    self.period_us = round(period_ms * 1000)
    if self.period_us < 1:
      raise ValueError("Planner period is below timestamp resolution")
    self.next_us = None
    self.last_us = None

  def due(self, stamp):
    if self.last_us is not None and stamp < self.last_us:
      raise ValueError("Simulation measurement clock moved backwards")
    self.last_us = stamp
    if self.next_us is None:
      self.next_us = stamp
    if stamp + 1 < self.next_us:
      return False
    while self.next_us <= stamp + 1:
      self.next_us += self.period_us
    return True


def main():
  args = parse_args()
  if not 1 <= args.listen_port <= 65535:
    raise ValueError("listen port is invalid")
  if args.accept_timeout_s <= 0.0 or args.socket_timeout_s <= 0.0:
    raise ValueError("socket timeouts must be positive")
  if args.command_wait_ms < 0:
    raise ValueError("command wait must be non-negative")
  if args.fresh_effort_timeout_ms < 0:
    raise ValueError("fresh effort timeout must be non-negative")
  if args.diagnostic_osc_hold and args.command_mode != "effort":
    raise ValueError("OSC hold diagnostic requires effort mode")
  if args.planner_state_channel and (args.command_mode != "effort" or args.planner_state_channel == args.franka_state_channel):
    raise ValueError("Separate planner clock requires distinct effort-mode state channels")
  planner_gate = PlannerClockGate(args.planner_period_ms) if args.planner_state_channel else None
  planner_publications = 0
  if args.task_lookahead_ms < 0.0:
    raise ValueError("task lookahead must be non-negative")
  if args.fresh_task_timeout_ms < 0:
    raise ValueError("fresh task timeout must be non-negative")
  if args.fresh_timestamp_tolerance_us < 0:
    raise ValueError("fresh timestamp tolerance must be non-negative")
  replay = DiagnosticReferenceReplay(args.diagnostic_reference_replay_start_s,
      args.diagnostic_reference_replay_duration_s, args.fresh_timestamp_tolerance_us)
  if args.diagnostic_reference_replay_start_s is not None and (
      args.command_mode != "task" or args.effect_audit_output is None or args.task_lookahead_ms != 0):
    raise ValueError("Diagnostic replay requires task mode, audit output, and zero lookahead")
  protocol = load_protocol(args.protocol_module)
  if args.state_mode == "single":
    if (
        args.object_body_names is not None
        or args.object_state_channels is not None
        or args.scene_spec is not None):
      raise ValueError(
          "scene object arguments require --state-mode=scene")
    object_body_names = [DEFAULT_OBJECT_BODY_NAME]
    object_state_channels = [args.object_state_channel]
    state_packet_size = protocol.STATE_PACKET_SIZE
    state_decoder = protocol.C3MeasuredState.unpack
  else:
    explicit_contract = (
        args.object_body_names is not None or args.object_state_channels is not None)
    if args.scene_spec is not None and explicit_contract:
      raise ValueError("use either --scene-spec or explicit scene object arguments")
    if args.scene_spec is not None:
      object_body_names, object_state_channels = load_scene_object_contract(
          args.scene_spec)
    else:
      if args.object_body_names is None or args.object_state_channels is None:
        raise ValueError(
            "scene mode requires --scene-spec or both explicit object lists")
      object_body_names = list(args.object_body_names)
      object_state_channels = list(args.object_state_channels)
    if len(object_body_names) != len(object_state_channels):
      raise ValueError("object body names and state channels must have equal length")
    if not 1 <= len(object_body_names) <= 1 + protocol.MAX_CLUTTER_OBJECTS:
      raise ValueError(
          "scene mode object count exceeds the fixed protocol capacity")
    if len(set(object_body_names)) != len(object_body_names):
      raise ValueError("object body names must be unique")
    if len(set(object_state_channels)) != len(object_state_channels):
      raise ValueError("object state channels must be unique")
    state_packet_size = protocol.SCENE_STATE_PACKET_SIZE
    state_decoder = protocol.C3MeasuredSceneState.unpack
  lcm_handle = lcm.LCM(args.lcm_url)
  latest_command = {"message": None, "count": 0}
  latest_tracking = {"message": None, "count": 0}
  latest_c3_mode = {
      "message": None,
      "value": False,
      "yaw_recovery": False,
      "count": 0,
  }
  latest_object_plan = {"message": None, "count": 0}
  effect_audit_stream = None
  last_audited_object_plan_count = 0

  if args.effect_audit_output is not None:
    effect_audit_path = args.effect_audit_output.expanduser().resolve()
    effect_audit_path.parent.mkdir(parents=True, exist_ok=True)
    effect_audit_stream = effect_audit_path.open("w", encoding="utf-8")
    effect_audit_stream.write(json.dumps({
        "event": "metadata",
        "schema": "nonprehensile.c3_object_effect_audit.v1",
        "command_mode": args.command_mode,
        "diagnostic_osc_hold": args.diagnostic_osc_hold,
        "object_plan_channel": args.object_plan_channel,
        "state_mode": args.state_mode,
        "target_body_name": object_body_names[0],
        "diagnostic_reference_replay_start_s": args.diagnostic_reference_replay_start_s,
        "diagnostic_reference_replay_duration_s": args.diagnostic_reference_replay_duration_s,
        "osc_state_channel": args.franka_state_channel,
        "planner_state_channel": args.planner_state_channel,
        "planner_period_ms": args.planner_period_ms if planner_gate else None,

    }, sort_keys=True) + "\n")
    effect_audit_stream.flush()

  def handle_command(_channel, payload):
    message = lcmt_robot_input.decode(payload)
    if message.num_efforts != 7 or len(message.efforts) != 7:
      return
    effort_by_name = dict(zip(message.effort_names, message.efforts))
    if not all(name in effort_by_name for name in FRANKA_EFFORT_NAMES):
      return
    latest_command["message"] = message
    latest_command["count"] += 1

  def handle_tracking(_channel, payload):
    message = lcmt_timestamped_saved_traj.decode(payload)
    position_trajectory = trajectory_by_name(
        message, "end_effector_position_target")
    if position_trajectory is None:
      return
    latest_tracking["message"] = message
    latest_tracking["count"] += 1
    if effect_audit_stream is not None:
      handle_planner_diagnostic(_channel, payload)
    if latest_tracking["count"] <= 3 or latest_tracking["count"] % 100 == 0:
      first = tuple(float(row[0]) for row in position_trajectory.datapoints[:3])
      last = tuple(float(row[-1]) for row in position_trajectory.datapoints[:3])
      print(
          "C3_ONLINE_RELAY_TRAJECTORY",
          f"count={latest_tracking['count']}",
          f"message_utime={message.utime}",
          f"time=({position_trajectory.time_vec[0]},"
          f"{position_trajectory.time_vec[-1]})",
          f"first={first}",
          f"last={last}",
          flush=True)

  def handle_c3_mode(_channel, payload):
    message = lcmt_timestamped_saved_traj.decode(payload)
    trajectory = trajectory_by_name(message, "is_c3_mode")
    if trajectory is None:
      return
    previous_mode = latest_c3_mode["value"]
    yaw_recovery_trajectory = trajectory_by_name(message, "is_yaw_recovery")
    latest_c3_mode["message"] = message
    latest_c3_mode["value"] = bool(float(trajectory.datapoints[0][0]) > 0.5)
    latest_c3_mode["yaw_recovery"] = bool(
        yaw_recovery_trajectory is not None
        and float(yaw_recovery_trajectory.datapoints[0][0]) > 0.5)
    latest_c3_mode["count"] += 1
    if latest_c3_mode["value"] != previous_mode:
      print(
          "C3_ONLINE_RELAY_MODE",
          f"message_utime={message.utime}",
          f"c3={int(latest_c3_mode['value'])}",
          f"yaw_recovery={int(latest_c3_mode['yaw_recovery'])}",
          flush=True)

  def handle_object_plan(_channel, payload):
    message = lcmt_timestamped_saved_traj.decode(payload)
    position_trajectory = target_object_trajectory(
        message, "object_position_target")
    orientation_trajectory = target_object_trajectory(
        message, "object_orientation_target")
    if position_trajectory is None or orientation_trajectory is None:
      return
    if (
        len(position_trajectory.datapoints) < 3
        or len(orientation_trajectory.datapoints) < 4):
      return
    latest_object_plan["message"] = message
    latest_object_plan["count"] += 1

  def handle_planner_diagnostic(channel, payload):
    message = lcmt_timestamped_saved_traj.decode(payload)
    effect_audit_stream.write(json.dumps({
        "event": "planner_diagnostic", "channel": channel,
        "utime_us": int(message.utime),
        "trajectories": [{"name": t.trajectory_name,
                          "times": list(t.time_vec),
                          "datapoints": [list(row) for row in t.datapoints]}
                         for t in message.saved_traj.trajectories],
    }, separators=(",", ":")) + "\n")

  def handle_solver_diagnostic(channel, payload):
    message = lcmt_output.decode(payload)
    solution, intermediates = message.solution, message.intermediates
    effect_audit_stream.write(json.dumps({
        "event": "solver_diagnostic", "channel": channel,
        "utime_us": int(message.utime),
        "x": [list(row) for row in solution.x_sol],
        "u": [list(row) for row in solution.u_sol],
        "lambda": [list(row) for row in solution.lambda_sol],
        "z": [list(row) for row in intermediates.z_sol],
        "delta": [list(row) for row in intermediates.delta_sol],
    }, separators=(",", ":")) + "\n")

  subscriptions = []
  if args.command_mode == "effort":
    subscriptions.append(
        lcm_handle.subscribe(args.franka_input_channel, handle_command))
    # Read-only diagnostics: the effort executor still returns the untouched
    # OSC command, but its relay log now exposes the task-space trajectory that
    # produced that effort.  This makes cross-engine tracking error observable
    # without adding fields to the deployment wire protocol.
    subscriptions.append(
        lcm_handle.subscribe(args.tracking_channel, handle_tracking))
  else:
    subscriptions.append(
        lcm_handle.subscribe(args.tracking_channel, handle_tracking))
  subscriptions.append(
      lcm_handle.subscribe(args.c3_mode_channel, handle_c3_mode))
  for subscription in subscriptions:
    subscription.set_queue_capacity(1)
  if effect_audit_stream is not None:
    object_plan_subscription = lcm_handle.subscribe(
        args.object_plan_channel, handle_object_plan)
    object_plan_subscription.set_queue_capacity(1)
    subscriptions.append(object_plan_subscription)
    # Observe raw optimizer output separately from the PD rollout used to
    # rank samples. Neither stream modifies commands or mode selection.
    for channel in ("C3_TRAJECTORY_OBJECT_CURR_PLAN", "C3_TRAJECTORY_OBJECT_BEST_PLAN",
                    "DYNAMICALLY_FEASIBLE_BEST_PLAN", "DYNAMICALLY_FEASIBLE_CURR_ACTOR_PLAN",
                    "SAMPLE_LOCATIONS", "SAMPLE_COSTS"):
      subscription = lcm_handle.subscribe(channel, handle_planner_diagnostic)
      subscription.set_queue_capacity(1)
      subscriptions.append(subscription)
    subscription = lcm_handle.subscribe("C3_DEBUG_CURR", handle_solver_diagnostic)
    subscription.set_queue_capacity(1)
    subscriptions.append(subscription)

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.listen_host, args.listen_port))
    server.listen(1)
    server.settimeout(args.accept_timeout_s)
    print(
        "C3_ONLINE_RELAY_LISTENING",
        f"tcp={args.listen_host}:{args.listen_port}",
        f"lcm={args.lcm_url}",
        f"state_mode={args.state_mode}",
        f"objects={len(object_body_names)}",
        f"command_mode={args.command_mode}",
        f"task_lookahead_ms={args.task_lookahead_ms}",
        f"fresh_task_timeout_ms={args.fresh_task_timeout_ms}",
        f"fresh_timestamp_tolerance_us={args.fresh_timestamp_tolerance_us}",
        flush=True)
    connection, address = server.accept()
    with connection:
      connection.settimeout(args.socket_timeout_s)
      print("C3_ONLINE_RELAY_CONNECTED", address, flush=True)
      state_count = 0
      returned_command_count = 0
      while True:
        packet = receive_exact(connection, state_packet_size)
        if packet is None:
          break
        state = state_decoder(packet)
        radio_message = radio_state_message(protocol, state)
        if args.diagnostic_osc_hold:
          radio_message.channel[14] = 1.0
        lcm_handle.publish(args.radio_channel, radio_message.encode())
        if args.state_mode == "single":
          rigid_body_states = [legacy_object_state(protocol, state)]
        else:
          rigid_body_states = [state.target, *state.clutter]
          if len(rigid_body_states) != len(object_body_names):
            raise ValueError(
                "scene packet active object count does not match staged C3 scene")
        for channel, body_name, body_state in zip(
            object_state_channels, object_body_names, rigid_body_states):
          lcm_handle.publish(
              channel,
              object_state_message(
                  state.utime_us, body_state, body_name).encode())
        robot_payload = robot_state_message(state).encode()
        if planner_gate is not None and planner_gate.due(int(state.utime_us)):
          lcm_handle.publish(args.planner_state_channel, robot_payload)
          planner_publications += 1
          # The first state initializes upstream's LCM loop. The second
          # planner state can produce the first plan. Later states must each
          # finish planning before advancing the simulated OSC/physics clock.
          if planner_publications > 1:
            deadline = time.monotonic() + args.fresh_task_timeout_ms * 1e-3
            while not (effort_is_current(latest_tracking["message"], state.utime_us, args.fresh_timestamp_tolerance_us)
                       and effort_is_current(latest_c3_mode["message"], state.utime_us, args.fresh_timestamp_tolerance_us)):
              remaining_ms = int((deadline - time.monotonic()) * 1000)
              if remaining_ms <= 0:
                raise TimeoutError(f"Planner did not return current plan at {state.utime_us} us")
              lcm_handle.handle_timeout(max(1, remaining_ms))
          if effect_audit_stream is not None:
            effect_audit_stream.write(json.dumps(dict(event="planner_state", utime_us=int(state.utime_us),
                publication=planner_publications, bootstrap=planner_publications == 1)) + "\n")
        lcm_handle.publish(args.franka_state_channel, robot_payload)
        state_count += 1

        latest = latest_command if args.command_mode == "effort" else latest_tracking
        if args.command_mode == "effort" and args.fresh_effort_timeout_ms > 0:
          deadline = time.monotonic() + args.fresh_effort_timeout_ms * 1.0e-3
          while not effort_is_current(latest["message"], state.utime_us,
                                      args.fresh_timestamp_tolerance_us):
            remaining_ms = int((deadline - time.monotonic()) * 1000.0)
            if remaining_ms <= 0:
              break
            lcm_handle.handle_timeout(max(1, remaining_ms))
        elif args.command_mode == "task" and args.fresh_task_timeout_ms > 0:
          deadline = time.monotonic() + args.fresh_task_timeout_ms * 1.0e-3
          while (
              latest["message"] is None
              or int(latest["message"].utime)
              + args.fresh_timestamp_tolerance_us < state.utime_us
              or latest_c3_mode["message"] is None
              or int(latest_c3_mode["message"].utime)
              + args.fresh_timestamp_tolerance_us < state.utime_us):
            remaining_ms = int((deadline - time.monotonic()) * 1000.0)
            if remaining_ms <= 0:
              break
            lcm_handle.handle_timeout(max(1, remaining_ms))
        else:
          lcm_handle.handle_timeout(args.command_wait_ms)
        # Multiple read-only subscriptions must not starve the effort stream.
        # Consume everything already queued without extending the per-state
        # wait budget; bounded draining also prevents a noisy publisher from
        # monopolizing the TCP control loop.
        for _ in range(max(1, 2 * len(subscriptions))):
          if not lcm_handle.handle_timeout(0):
            break
        if effect_audit_stream is not None:
          target_state = rigid_body_states[0]
          effect_audit_stream.write(json.dumps({
              "event": "measured_state",
              "sequence": int(state.sequence),
              "utime_us": int(state.utime_us),
              "target_position_m": list(target_state.position_m),
              "target_quaternion_wxyz": list(target_state.quaternion_wxyz),
          }, separators=(",", ":"), sort_keys=True) + "\n")
          if latest_object_plan["count"] != last_audited_object_plan_count:
            plan_message = latest_object_plan["message"]
            position_trajectory = target_object_trajectory(
                plan_message, "object_position_target")
            orientation_trajectory = target_object_trajectory(
                plan_message, "object_orientation_target")
            effect_audit_stream.write(json.dumps({
                "event": "object_plan",
                "relay_state_sequence": int(state.sequence),
                "relay_state_utime_us": int(state.utime_us),
                "plan_utime_us": int(plan_message.utime),
                "c3_mode": bool(latest_c3_mode["value"]),
                "yaw_recovery": bool(latest_c3_mode["yaw_recovery"]),
                "knot_indices": [
                    float(value) for value in position_trajectory.time_vec],
                "position_knots_m": [
                    [float(row[index]) for row in position_trajectory.datapoints[:3]]
                    for index in range(position_trajectory.num_points)],
                "quaternion_knots_wxyz": [
                    [float(row[index]) for row in orientation_trajectory.datapoints[:4]]
                    for index in range(orientation_trajectory.num_points)],
            }, separators=(",", ":"), sort_keys=True) + "\n")
            last_audited_object_plan_count = latest_object_plan["count"]
          effect_audit_stream.flush()
        message, command_c3_mode, command_yaw_recovery, replay_active, replay_event = replay.select(
            int(state.utime_us), latest["message"], latest_c3_mode, latest_object_plan["message"])
        if planner_gate is not None:
          command_c3_mode = bool(latest_c3_mode["value"])
          command_yaw_recovery = bool(latest_c3_mode["yaw_recovery"])
        if replay_event is not None:
          effect_audit_stream.write(json.dumps(replay_event, sort_keys=True) + "\n")
          effect_audit_stream.flush()
          print("C3_DIAGNOSTIC_REFERENCE_REPLAY", json.dumps(replay_event), flush=True)
        flags = protocol.C3CommandFlags(0)
        if message is None:
          flags |= protocol.C3CommandFlags.STALE_STATE
          command_utime = state.utime_us
          if args.command_mode == "effort":
            command = protocol.C3EffortCommand(
                sequence=state.sequence,
                utime_us=command_utime,
                flags=flags,
                joint_effort_nm=(0.0,) * 7)
          else:
            command = protocol.C3TaskCommand(
                sequence=state.sequence,
                utime_us=command_utime,
                flags=flags,
                position_m=(0.0,) * 3,
                velocity_m_s=(0.0,) * 3,
                feedforward_force_n=(0.0,) * 3)
        else:
          flags |= protocol.C3CommandFlags.READY
          if (
              not replay_active and int(message.utime) + args.fresh_timestamp_tolerance_us
              < state.utime_us):
            flags |= protocol.C3CommandFlags.STALE_STATE
          if (args.command_mode == "effort" and args.fresh_effort_timeout_ms > 0
              and not effort_is_current(message, state.utime_us, args.fresh_timestamp_tolerance_us)):
            flags |= protocol.C3CommandFlags.STALE_STATE
          command_utime = int(message.utime)
          if command_c3_mode:
            flags |= protocol.C3CommandFlags.C3_MODE
            if command_yaw_recovery:
              flags |= protocol.C3CommandFlags.YAW_RECOVERY
          returned_command_count += 1
          if args.command_mode == "effort":
            effort_by_name = dict(zip(message.effort_names, message.efforts))
            efforts = tuple(
                float(effort_by_name[name]) for name in FRANKA_EFFORT_NAMES)
            command = protocol.C3EffortCommand(
                sequence=state.sequence,
                utime_us=max(0, command_utime),
                flags=flags,
                joint_effort_nm=efforts)
          else:
            position_trajectory = trajectory_by_name(
                message, "end_effector_position_target")
            force_trajectory = trajectory_by_name(
                message, "end_effector_force_target")
            # Multi-object C3 solves can finish after the measurement timestamp
            # that triggered them.  Sampling before the returned trajectory's
            # own start time repeatedly returns its first knot and stalls the
            # online executor.  Advance a fixed horizon from whichever clock
            # is later while preserving the single-object behavior.
            query_time_s = max(
                state.utime_us * 1.0e-6,
                float(position_trajectory.time_vec[0]))
            query_time_s += args.task_lookahead_ms * 1.0e-3
            position, velocity = evaluate_trajectory(
                position_trajectory, query_time_s)
            if effect_audit_stream is not None:
              effect_audit_stream.write(json.dumps({
                  "event": "task_reference", "relay_state_utime_us": int(state.utime_us),
                  "plan_utime_us": int(message.utime),
                  "plan_start_time_s": float(position_trajectory.time_vec[0]),
                  "query_time_s": query_time_s, "c3_mode": bool(flags & protocol.C3CommandFlags.C3_MODE),
                  "command_flags": int(flags),
                  "diagnostic_reference_replay_active": replay_active,
                  "force_knots_n": ([[float(row[i]) for row in force_trajectory.datapoints[:3]]
                                     for i in range(force_trajectory.num_points)] if force_trajectory else []),
                  "knot_times_s": [float(t) for t in position_trajectory.time_vec],
                  "position_knots_m": [[float(row[i]) for row in position_trajectory.datapoints[:3]]
                                       for i in range(position_trajectory.num_points)],
              }, separators=(",", ":"), sort_keys=True) + "\n")
            if force_trajectory is None:
              force = (0.0,) * 3
            else:
              force_query_time_s = max(
                  state.utime_us * 1.0e-6,
                  float(force_trajectory.time_vec[0]))
              force_query_time_s += args.task_lookahead_ms * 1.0e-3
              force, _ = evaluate_trajectory(
                  force_trajectory, force_query_time_s)
            command = protocol.C3TaskCommand(
                sequence=state.sequence,
                # This sample is evaluated at the current measured-state
                # timestamp even when the underlying piecewise trajectory was
                # published earlier.
                utime_us=state.utime_us,
                flags=flags,
                position_m=tuple(position[:3]),
                velocity_m_s=tuple(velocity[:3]),
                feedforward_force_n=tuple(force[:3]))
        connection.sendall(command.pack())
        if state_count % 100 == 0:
          print(
              "C3_ONLINE_RELAY_PROGRESS",
              f"states={state_count}",
              f"commands={returned_command_count}",
              flush=True)
      print(
          "C3_ONLINE_RELAY_DONE",
          f"states={state_count}",
          f"commands={returned_command_count}",
          flush=True)
  if effect_audit_stream is not None:
    effect_audit_stream.close()


if __name__ == "__main__":
  main()
