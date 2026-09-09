#!/usr/bin/env python3
"""Run Push Anything C3+/OSC as an online torque controller for Isaac Lab.

Unlike ``run_push_anything_isaaclab_smoke.py``, this executor never loads a
recorded trajectory.  Every control step sends the measured Franka and target
state to the native C3 relay and applies the returned joint efforts.  The same
wire contract is intended for the later real-robot executor.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

from isaac_failure_evidence import execution_failure_evidence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AffordanceTeacher-C1-Franka-v0")
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--relay-host", default="127.0.0.1")
parser.add_argument("--relay-port", type=int, default=7795)
parser.add_argument("--socket-timeout-s", type=float, default=5.0)
parser.add_argument("--physics-dt-s", type=float, default=0.001)
parser.add_argument("--control-decimation", type=int, default=1)
parser.add_argument("--max-sim-time-s", type=float, default=30.0)
parser.add_argument("--dwell-time-s", type=float, default=0.5)
parser.add_argument("--watchdog-timeout-s", type=float, default=0.020)
parser.add_argument("--maximum-consecutive-watchdog-steps", type=int, default=20)
parser.add_argument(
    "--startup-timeout-s",
    type=float,
    default=2.0,
    help="fail-safe zero-effort handshake window before the first READY command",
)
parser.add_argument("--table-height-offset-m", type=float, default=0.029)
parser.add_argument("--contact-distance-m", type=float, default=0.010)
parser.add_argument("--robot-model-manifest", type=Path,
    default=os.environ.get("PUSH_ANYTHING_ROBOT_MODEL_MANIFEST"),
    help="Pinned matched FR3 model manifest; automatically selects closed stock gripper")
parser.add_argument("--stock-closed-gripper", action="store_true",
    help="Use the unchanged stock robot asset and closed fingers; diagnostic torque baseline")
parser.add_argument(
    "--pusher-collision-model",
    choices=("c3_tip_only", "stock_hand"),
    default="c3_tip_only",
    help=(
        "collision geometry used by Isaac: c3_tip_only matches the spherical "
        "contact body optimized by C3; stock_hand additionally retains the "
        "Panda hand and peg collisions for a later hardware-clearance audit"
    ),
)
parser.add_argument("--audit-stride", type=int, default=1)
parser.add_argument("--video", action="store_true", help="Record actual IsaacLab RGB rendering during physics execution")
parser.add_argument("--video-fps", type=int, default=25)
parser.add_argument("--goal-ghost-opacity", type=float, default=.35)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(1.35, 1.05, 1.00))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(.32, .20, .30))
parser.add_argument(
    "--trace-stride",
    type=int,
    default=100,
    help="store one compact state/command diagnostic every N control steps",
)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument(
    "--initial-joint-position-rad",
    type=float,
    nargs=7,
    default=(2.191, 1.1, -1.33, -2.22, 1.30, 2.02, 0.08),
)
parser.add_argument(
    "--support-quaternion-wxyz",
    type=float,
    nargs=4,
    default=(-0.4937799140314683, 0.5013379099086276,
             0.506216296031207, 0.4985847553024161),
)
parser.add_argument(
    "--effort-limits-nm",
    type=float,
    nargs=7,
    default=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
)
parser.add_argument(
    "--max-effort-rate-nm-s",
    type=float,
    default=0.0,
    help="optional executor-side slew limit; zero preserves upstream OSC output",
)
parser.add_argument(
    "--pace-realtime",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="sleep when simulation and relay run faster than wall clock",
)
parser.add_argument("--domino-root", type=Path, default=Path("/data1/linsixu/DOMINO"))
parser.add_argument(
    "--domino-usd-root",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "data/domino_usd",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

from fr3_robot_model_contract import load_contract
robot_model_contract = None
if args_cli.robot_model_manifest:
    robot_model_contract = load_contract(args_cli.robot_model_manifest)
    args_cli.stock_closed_gripper = True
    native_path = os.environ.get("PUSH_ANYTHING_ROBOT_MODEL")
    if not native_path or Path(native_path).resolve() != Path(robot_model_contract["native_urdf"]):
        raise ValueError("Native OSC must use the same FR3 manifest; set PUSH_ANYTHING_ROBOT_MODEL_MANIFEST in the launcher")
    limits = robot_model_contract["joint_limits"]
    for i, q in enumerate(args_cli.initial_joint_position_rad, 1):
        limit = limits[f"panda_joint{i}"]
        if not limit["lower"] <= q <= limit["upper"]:
            raise ValueError(f"FR3 initial joint {i} violates official limits")
    args_cli.effort_limits_nm = tuple(min(requested, limits[f"panda_joint{i}"]["effort"])
        for i, requested in enumerate(args_cli.effort_limits_nm, 1))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

from dapl.contact_planner.franka_push_tool_urdf import (
    PUSH_TOOL_TIP_FROM_HAND_M,
    PUSH_TOOL_TIP_RADIUS_M,
    build_push_anything_franka_urdf,
)

# Use the same physical 19.5 mm spherical pusher that C3 optimizes.  This must
# be selected before importing the task package because its robot asset config
# is resolved at import time.
isaacsim_spec = importlib.util.find_spec("isaacsim")
if isaacsim_spec is None or not isaacsim_spec.submodule_search_locations:
    raise RuntimeError("Cannot locate Isaac Sim's packaged Franka URDF")
isaacsim_root = Path(next(iter(isaacsim_spec.submodule_search_locations)))
physical_end_effector_urdf = None if args_cli.stock_closed_gripper else build_push_anything_franka_urdf(
    isaacsim_root
    / "exts/isaacsim.asset.importer.urdf/data/urdf/robots"
    / "franka_description/robots/panda_arm_hand.urdf",
    Path("/tmp/IsaacLab/nonprehensile_franka_push_anything_effort")
    / "panda_arm_hand_push_anything.urdf",
    sphere_only_collision=args_cli.pusher_collision_model == "c3_tip_only",
)
if robot_model_contract:
    physical_end_effector_urdf = Path(robot_model_contract["simulation_urdf"])
    os.environ["DAPL_LOCAL_FRANKA_URDF"] = str(physical_end_effector_urdf)
    os.environ["DAPL_LOCAL_FRANKA_USD_DIR"] = (
        "/tmp/IsaacLab/fr3_stock_" + robot_model_contract["simulation_urdf_sha256"][:16])
elif args_cli.stock_closed_gripper:
    os.environ.pop("DAPL_LOCAL_FRANKA_URDF", None)
    os.environ.pop("DAPL_LOCAL_FRANKA_USD_DIR", None)
else:
    os.environ["DAPL_LOCAL_FRANKA_URDF"] = str(physical_end_effector_urdf)
    os.environ["DAPL_LOCAL_FRANKA_USD_DIR"] = (
        "/tmp/IsaacLab/nonprehensile_franka_push_anything_effort/usd"
    )
TASK_TIP_FROM_HAND_M = 0.104279112 if args_cli.stock_closed_gripper else PUSH_TOOL_TIP_FROM_HAND_M
if robot_model_contract:
    TASK_TIP_FROM_HAND_M = robot_model_contract["task_tip_from_hand_m"]
TASK_TIP_RADIUS_M = 0.008 if args_cli.stock_closed_gripper else PUSH_TOOL_TIP_RADIUS_M

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from dapl.contact_planner.c3_online_protocol import (
    C3CommandFlags,
    C3EffortCommand,
    C3MeasuredState,
    C3StateFlags,
    COMMAND_PACKET_SIZE,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp


# Reflected motor inertia parsed by Drake from panda_arm.urdf:
# gear_ratio (100)^2 * rotor_inertia.  PhysX calls the same quantity armature.
UPSTREAM_ARMATURE_KG_M2 = (
    0.605721456,
    0.605721456,
    0.462474144,
    0.462474144,
    0.205544064,
    0.205544064,
    0.205544064,
)

if robot_model_contract:
    UPSTREAM_ARMATURE_KG_M2 = tuple(robot_model_contract["armature_kg_m2"])


def _never_terminate(env) -> torch.Tensor:
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("C3 relay closed inside a command packet")
        payload.extend(chunk)
    return bytes(payload)


def _quaternion_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(first, dim=-1)
    second = torch.nn.functional.normalize(second, dim=-1)
    chord = torch.minimum(
        torch.linalg.vector_norm(first - second, dim=-1),
        torch.linalg.vector_norm(first + second, dim=-1),
    )
    return 4.0 * torch.asin(torch.clamp(0.5 * chord, max=1.0))


def _pose_error_tensor(base) -> torch.Tensor:
    target = base.scene["target"]
    goal = getattr(base, "_c3_fixed_goal_pose_wxyz", None)
    if goal is None:
        goal = base.command_manager.get_command("target_object_pose")
    position = target.data.root_pos_w[:, :3] - base.scene.env_origins
    delta = goal[:, :3] - position
    return torch.stack(
        (
            torch.linalg.vector_norm(delta[0, :2]),
            torch.abs(delta[0, 2]),
            _quaternion_distance(target.data.root_quat_w, goal[:, 3:7])[0],
        )
    )


def _pose_errors(base) -> tuple[float, float, float]:
    values = _pose_error_tensor(base).detach().cpu().tolist()
    return tuple(float(value) for value in values)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _configure_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    env_cfg.seed = args_cli.seed
    env_cfg.disable_obs_noise = True
    env_cfg.enforce_joint_limits = False
    robot_position = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (
        float(robot_position[0]),
        float(robot_position[1]),
        float(robot_position[2]) + args_cli.table_height_offset_m,
    )
    for joint_index, joint_position in enumerate(
        args_cli.initial_joint_position_rad, start=1
    ):
        env_cfg.scene.robot.init_state.joint_pos[
            f"panda_joint{joint_index}"
        ] = float(joint_position)
    env_cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = (
        0.0 if args_cli.stock_closed_gripper else 0.04)
    env_cfg.actions.arm_action = JointEffortActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=1.0,
        clip={f"panda_joint{i}": (-limit, limit)
              for i, limit in enumerate(args_cli.effort_limits_nm, 1)},
    )
    env_cfg.sim.dt = args_cli.physics_dt_s
    env_cfg.decimation = args_cli.control_decimation
    env_cfg.sim.render_interval = args_cli.control_decimation
    if args_cli.video:
        if args_cli.video_fps <= 0:
            raise ValueError("Video FPS must be positive")
        steps_per_frame = 1.0 / (args_cli.video_fps * args_cli.physics_dt_s)
        if abs(steps_per_frame-round(steps_per_frame)) > 1e-8:
            raise ValueError("Video cadence must be an integer number of physics steps")
        env_cfg.sim.render_interval = round(steps_per_frame)
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = (1280, 720)
        env_cfg.sim.render.enable_translucency = True
    env_cfg.episode_length_s = args_cli.max_sim_time_s + 10.0

    # This executable consumes simulator state directly; it never consumes an
    # RL observation or reward.  Avoid rebuilding the 4096-value point-cloud
    # observation and twelve training rewards at 1 kHz.  The command manager,
    # physics, contact sensors, action manager, and acceptance checks remain
    # active, so this only removes unused bookkeeping from the control loop.
    for term_name in (
        "affordance_scene",
        "hand_state",
        "previous_action",
        "rel_goal",
        "target_twist",
    ):
        if hasattr(env_cfg.observations.policy, term_name):
            setattr(env_cfg.observations.policy, term_name, None)
    env_cfg.observations.critic = None
    for term_name in (
        "task_success",
        "safe_region_distance",
        "safe_region_progress",
        "first_safe_region_contact",
        "safe_contact_planar_progress",
        "safe_contact_height_progress",
        "safe_contact_rotation_progress",
        "near_goal_target_motion",
        "action_magnitude",
        "action_rate",
        "forbidden_region_contact",
        "forbidden_region_clearance",
    ):
        if hasattr(env_cfg.rewards, term_name):
            setattr(env_cfg.rewards, term_name, None)

    # The upstream simulation OSC includes model-based gravity compensation.
    # Isaac must therefore expose the same gravity rather than the high-PD
    # task's gravity-disabled arm dynamics.
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = False
    for name, actuator in env_cfg.scene.robot.actuators.items():
        if name != "panda_hand":
            actuator.stiffness = 0.0
            actuator.damping = 0.0
    env_cfg.scene.robot.actuators["panda_shoulder"].armature = {
        "panda_joint[1-2]": UPSTREAM_ARMATURE_KG_M2[0],
        "panda_joint[3-4]": UPSTREAM_ARMATURE_KG_M2[2],
    }
    env_cfg.scene.robot.actuators["panda_forearm"].armature = {
        "panda_joint[5-7]": UPSTREAM_ARMATURE_KG_M2[4],
    }

    if robot_model_contract:
        from dapl.contact_planner.fr3_model_runtime import spawn_fr3_with_rigid_finger_coupling
        env_cfg.scene.robot.spawn.func = spawn_fr3_with_rigid_finger_coupling
        hand_actuator = env_cfg.scene.robot.actuators["panda_hand"]
        hand_actuator.effort_limit_sim = {name: robot_model_contract["joint_limits"][name]["effort"] for name in ("panda_finger_joint1", "panda_finger_joint2")}
        hand_actuator.velocity_limit_sim = {name: robot_model_contract["joint_limits"][name]["velocity"] for name in ("panda_finger_joint1", "panda_finger_joint2")}
        for group, indices in (("panda_shoulder", range(1, 5)), ("panda_forearm", range(5, 8))):
            actuator = env_cfg.scene.robot.actuators[group]
            actuator.friction = 0.0
            actuator.damping = {f"panda_joint{i}": robot_model_contract["joint_dynamics"][f"panda_joint{i}"]["damping"] for i in indices}
            actuator.armature = {f"panda_joint{i}": UPSTREAM_ARMATURE_KG_M2[i-1] for i in indices}
            actuator.effort_limit_sim = {f"panda_joint{i}": args_cli.effort_limits_nm[i-1] for i in indices}
            actuator.velocity_limit_sim = {f"panda_joint{i}": robot_model_contract["joint_limits"][f"panda_joint{i}"]["velocity"] for i in indices}

    for term_name in (
        "time_out", "reached", "forbidden_region_contact", "object_dropped"
    ):
        term = getattr(env_cfg.terminations, term_name, None)
        if term is not None:
            term.func = _never_terminate
            term.params = {}
    return env_cfg


def main() -> int:
    if not args_cli.manifest.expanduser().is_file():
        raise FileNotFoundError(f"manifest does not exist: {args_cli.manifest}")
    if args_cli.physics_dt_s <= 0.0 or args_cli.control_decimation <= 0:
        raise ValueError("physics dt and control decimation must be positive")
    if args_cli.max_sim_time_s <= 0.0 or args_cli.dwell_time_s <= 0.0:
        raise ValueError("simulation and dwell times must be positive")
    if args_cli.watchdog_timeout_s <= 0.0 or args_cli.startup_timeout_s <= 0.0:
        raise ValueError("watchdog and startup timeouts must be positive")
    if (
        args_cli.maximum_consecutive_watchdog_steps <= 0
        or args_cli.audit_stride <= 0
        or args_cli.trace_stride <= 0
    ):
        raise ValueError("watchdog, audit, and trace step counts must be positive")
    if args_cli.max_effort_rate_nm_s < 0.0:
        raise ValueError("effort slew limit must be non-negative")

    manifest = args_cli.manifest.expanduser().resolve()
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    controller_metadata = {}
    if os.environ.get("PUSH_ANYTHING_ROOT"):
        import yaml
        runtime = Path(os.environ["PUSH_ANYTHING_ROOT"])
        demo = os.environ.get("PUSH_ANYTHING_DEMO_NAME", "anything")
        parameters = yaml.safe_load((runtime / "examples/sampling_c3" / demo /
            "parameters/sampling_c3_controller_params.yaml").read_text())
        native_goal = runtime / parameters["goal_params_file"]
        goal_config = yaml.safe_load(native_goal.read_text())
        if goal_config.get("goal_mode") != 2:
            raise ValueError("Native OSC simulation validation requires a fixed native goal")
        controller_metadata = {"native_goal_mode": 2,
            "native_goal_artifact": {"path": str(native_goal.resolve()),
                "sha256": hashlib.sha256(native_goal.read_bytes()).hexdigest()},
            "force_c3_on_contact": False, "native_osc": True,
            "planner_period_ms": os.environ.get("PUSH_ANYTHING_PLANNER_PERIOD_MS")}
    contact_model = None
    if robot_model_contract and os.environ.get("PUSH_ANYTHING_CONTACT_MODEL_MANIFEST"):
        contact_path = Path(os.environ["PUSH_ANYTHING_CONTACT_MODEL_MANIFEST"])
        contact_model = json.loads(contact_path.read_text())
        if (contact_model.get("asset_contract") or {}).get("robot_urdf_sha256") != robot_model_contract["simulation_urdf_sha256"]:
            raise ValueError("Planner contact geometry belongs to a different robot")
        controller_metadata["contact_model_sha256"] = hashlib.sha256(contact_path.read_bytes()).hexdigest()
    os.environ["DAPL_CLUTTER_MANIFEST"] = str(manifest)
    os.environ["DAPL_CLUTTER_ASSET_SOURCE"] = "domino"
    os.environ["DOMINO_ROOT"] = str(args_cli.domino_root.expanduser().resolve())
    os.environ["DOMINO_USD_ROOT"] = str(args_cli.domino_usd_root.expanduser().resolve())

    env = None
    video_recorder = None
    result: dict[str, object] = {
        "schema": "nonprehensile.c3_online_isaaclab.v1",
        "executor_mode": "measured_isaac_state_to_online_c3_osc_effort",
        "manifest": str(manifest),
        "relay": f"{args_cli.relay_host}:{args_cli.relay_port}",
        "robot_model_contract": robot_model_contract,
        "robot_model": "FR3" if robot_model_contract else "Panda",
        "controller_parameters": controller_metadata,
        "physical_end_effector": ("stock_franka_gripper_closed"
            if args_cli.stock_closed_gripper else "push_anything_spherical_pusher"),
        "deployment_ready": False,
        "deployment_blocker": "simulation_only_validation_pending_s2_acceptance",
    }
    try:
        env = gym.make(args_cli.task, cfg=_configure_env(), render_mode="rgb_array" if args_cli.video else None)
        env.reset()
        base = env.unwrapped
        robot = base.scene["robot"]
        target = base.scene["target"]
        robot_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        robot_cfg.resolve(base.scene)
        joint_ids = robot_cfg.joint_ids
        hand_id = robot_cfg.body_ids[0]
        finger_ids = [robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
        friction_application = None
        finger_limits_application = None
        if robot_model_contract:
            from dapl.contact_planner.fr3_model_runtime import synchronize_arm_friction, synchronize_finger_limits
            friction_application = synchronize_arm_friction(robot.root_physx_view, joint_ids)
            finger_limits_application = synchronize_finger_limits(robot, robot_model_contract)
        robot_target_sensor_name = getattr(
            base.cfg, "robot_target_sensor_name", None
        )
        hand_target_sensor_name = "target_hand_contacts"
        if not robot_target_sensor_name or any(
            base.scene.sensors.get(name) is None
            for name in (robot_target_sensor_name, hand_target_sensor_name)
        ):
            raise RuntimeError(
                "effort acceptance requires whole-arm and hand PhysX contact sensors"
            )

        initial_q = torch.tensor(
            args_cli.initial_joint_position_rad,
            device=base.device,
            dtype=robot.data.joint_pos.dtype,
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(
            initial_q, torch.zeros_like(initial_q), joint_ids=joint_ids
        )
        base.sim.forward()
        base.scene.update(0.0)

        def joint_state() -> list[float]:
            return robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()

        def tcp_position() -> list[float]:
            offset = torch.tensor(
                (0.0, 0.0, TASK_TIP_FROM_HAND_M),
                device=base.device,
                dtype=robot.data.body_pos_w.dtype,
            ).unsqueeze(0)
            position_w = (
                robot.data.body_pos_w[0:1, hand_id]
                + quat_apply(robot.data.body_quat_w[0:1, hand_id], offset)
            )
            return (position_w[0] - base.scene.env_origins[0]).detach().cpu().tolist()

        def filtered_contact_forces(sensor_name: str) -> dict[str, float]:
            sensor = base.scene.sensors[sensor_name]
            force_norm = torch.linalg.vector_norm(
                sensor.data.force_matrix_w[0], dim=-1
            )
            summary = {sensor_name: float(torch.max(force_norm))}
            filter_paths = tuple(sensor.cfg.filter_prim_paths_expr)
            if force_norm.ndim == 2 and force_norm.shape[1] == len(filter_paths):
                for index, path in enumerate(filter_paths):
                    body = str(path).rsplit("/", maxsplit=1)[-1]
                    summary[f"{sensor_name}:{body}"] = float(
                        torch.max(force_norm[:, index])
                    )
            return summary

        control_period_s = float(base.step_dt)
        max_steps = max(1, math.ceil(args_cli.max_sim_time_s / control_period_s))
        dwell_steps = max(1, math.ceil(args_cli.dwell_time_s / control_period_s))
        startup_steps = max(1, math.ceil(args_cli.startup_timeout_s / control_period_s))
        effort_limits = torch.tensor(
            args_cli.effort_limits_nm, device=base.device, dtype=initial_q.dtype
        ).unsqueeze(0)
        support_quaternion = torch.tensor(
            args_cli.support_quaternion_wxyz,
            device=base.device,
            dtype=target.data.root_quat_w.dtype,
        ).unsqueeze(0)
        support_quaternion = torch.nn.functional.normalize(support_quaternion, dim=-1)

        initial_target_pose = torch.cat(
            (target.data.root_pos_w[:, :3] - base.scene.env_origins,
             target.data.root_quat_w), dim=1
        )[0].detach().cpu().tolist()
        goal_pose = base.command_manager.get_command("target_object_pose")[0].detach().cpu().tolist()
        base._c3_fixed_goal_pose_wxyz = base.command_manager.get_command("target_object_pose").detach().clone()
        initial_robot_joint_position = joint_state()
        initial_tcp_position = tcp_position()
        initial_robot_dynamics = {
            "body_names": list(robot.body_names),
            "dof_friction_coefficients": robot.root_physx_view.get_dof_friction_coefficients()[0].detach().cpu().tolist(),
            "dof_friction_properties": robot.root_physx_view.get_dof_friction_properties()[0].detach().cpu().tolist(),
            "dof_stiffness": robot.root_physx_view.get_dof_stiffnesses()[0].detach().cpu().tolist(),
            "dof_velocity_limits": robot.root_physx_view.get_dof_max_velocities()[0].detach().cpu().tolist(),
            "dof_effort_limits": robot.root_physx_view.get_dof_max_forces()[0].detach().cpu().tolist(),
            "dof_position_limits": robot.root_physx_view.get_dof_limits()[0].detach().cpu().tolist(),
            "dof_damping": robot.root_physx_view.get_dof_dampings()[0].detach().cpu().tolist(),
            "dof_armatures_kg_m2": robot.root_physx_view.get_dof_armatures()[0].detach().cpu().tolist(),
            "joint_limits_rad": robot.data.joint_pos_limits[0].detach().cpu().tolist(),
            "base_position_world_m": robot.data.root_pos_w[0].detach().cpu().tolist(),
            "base_quaternion_wxyz": robot.data.root_quat_w[0].detach().cpu().tolist(),
            "hand_quaternion_wxyz": robot.data.body_quat_w[0, hand_id].detach().cpu().tolist(),
            "joint_names": list(robot.joint_names),
            "body_masses_kg": robot.root_physx_view.get_masses()[0].detach().cpu().tolist(),
            "body_com_pose_xyzw": robot.root_physx_view.get_coms()[0].detach().cpu().tolist(),
            "body_inertias_kg_m2": robot.root_physx_view.get_inertias()[0].detach().cpu().tolist(),
            "gravity_compensation_nm": robot.root_physx_view.get_gravity_compensation_forces()[0, joint_ids].detach().cpu().tolist(),
            "arm_mass_matrix": robot.root_physx_view.get_generalized_mass_matrices()[0][joint_ids][:, joint_ids].detach().cpu().tolist(),
        }
        if contact_model and contact_model.get("source_target_dynamics"):
            expected = contact_model["source_target_dynamics"]
            actual_mass = float(target.root_physx_view.get_masses()[0].reshape(-1)[0])
            actual_com = target.root_physx_view.get_coms()[0, :3]
            actual_inertia = target.root_physx_view.get_inertias()[0].reshape(3, 3).T
            if (abs(actual_mass - expected["mass_kg"]) > 1e-7
                or not torch.allclose(actual_com, actual_com.new_tensor(expected["com_position_body_m"]), atol=1e-6, rtol=0)
                or not torch.allclose(actual_inertia, actual_inertia.new_tensor(expected["inertia_body_about_com_kg_m2"]), atol=1e-9, rtol=0)):
                raise ValueError("Target dynamics differ from the planner contact model")

        counters = {"ready": 0, "fresh": 0, "stale": 0, "watchdog": 0,
                    "semantic_hold": 0, "planner_failure": 0}
        consecutive_watchdog = 0
        maximum_consecutive_watchdog = 0
        first_ready_step = None
        safe_contact_ever = False
        latest_legal_safe_contact = False
        forbidden_contact_ever = False
        first_safe_contact_step = None
        first_forbidden_contact_step = None
        minimum_safe_distance = math.inf
        minimum_forbidden_distance = math.inf
        minimum_planar = math.inf
        minimum_height = math.inf
        minimum_rotation = math.inf
        strict_dwell = 0
        maximum_strict_dwell = 0
        stopped_reason = "maximum_sim_time"
        last_effort = torch.zeros_like(initial_q)
        max_command_l2 = 0.0
        max_applied_l2 = 0.0
        max_joint_velocity = 0.0
        maximum_command_age_s = 0.0
        peak_robot_target_contact_forces: dict[str, float] = {}
        executed_steps = 0
        physics_step_in_progress = False
        contact_audited_steps = 0
        loop_started = time.monotonic()
        trace: list[dict[str, object]] = []
        finger_state_path = output.with_suffix(".finger_state.jsonl")
        finger_state_stream = finger_state_path.open("x", encoding="utf-8")
        finger_state_recorded_steps = 0
        maximum_finger_opening_m = [0.0, 0.0]
        maximum_finger_mismatch_m = 0.0
        phase_wall_time_s = {"state_pack": 0.0, "native_roundtrip": 0.0,
                             "physics_and_state_read": 0.0, "contact_and_trace": 0.0}
        if args_cli.video:
            from isaac_online_video import IsaacOnlineVideo
            video_recorder = IsaacOnlineVideo(base, output.with_suffix('.isaaclab.mp4'), args_cli.video_fps,
                                               args_cli.goal_ghost_opacity)
            video_recorder.capture(0, 0.0)
            video_step_stride = round(1.0 / (args_cli.video_fps * control_period_s))
            if video_step_stride < 1 or abs(video_step_stride * args_cli.video_fps * control_period_s - 1.0) > 1e-8:
                raise ValueError("Video cadence must align with control measurements")

        with socket.create_connection(
            (args_cli.relay_host, args_cli.relay_port),
            timeout=args_cli.socket_timeout_s,
        ) as client:
            client.settimeout(args_cli.socket_timeout_s)
            for sequence in range(max_steps):
                phase_started = time.monotonic()
                q = robot.data.joint_pos[0, joint_ids]
                dq = robot.data.joint_vel[0, joint_ids]
                measured_effort = robot.data.applied_torque[0, joint_ids]
                object_position = target.data.root_pos_w[0, :3] - base.scene.env_origins[0]
                object_position_c3 = object_position.clone()
                object_position_c3[2] -= args_cli.table_height_offset_m
                object_quaternion_c3 = quat_mul(
                    target.data.root_quat_w[0:1], quat_inv(support_quaternion)
                )[0]
                object_quaternion_c3 = torch.nn.functional.normalize(
                    object_quaternion_c3, dim=0
                )
                # Scalar conversion of CUDA tensors synchronizes the device.
                # One packed transfer keeps the 1 kHz contract unchanged while
                # avoiding roughly thirty serial synchronizations per step.
                measured_values = torch.cat(
                    (
                        q,
                        dq,
                        measured_effort,
                        object_quaternion_c3,
                        object_position_c3,
                        target.data.root_ang_vel_w[0],
                        target.data.root_lin_vel_w[0],
                    )
                ).detach().cpu().tolist()
                utime_us = 100_000 + round(sequence * control_period_s * 1.0e6)
                state = C3MeasuredState(
                    sequence=sequence,
                    utime_us=utime_us,
                    joint_position_rad=tuple(measured_values[0:7]),
                    joint_velocity_rad_s=tuple(measured_values[7:14]),
                    joint_effort_nm=tuple(measured_values[14:21]),
                    object_quaternion_wxyz=tuple(measured_values[21:25]),
                    object_position_m=tuple(measured_values[25:28]),
                    object_angular_velocity_rad_s=tuple(measured_values[28:31]),
                    object_linear_velocity_m_s=tuple(measured_values[31:34]),
                    flags=(
                        C3StateFlags.LEGAL_SAFE_CONTACT
                        if latest_legal_safe_contact
                        else C3StateFlags(0)
                    ),
                )
                phase_wall_time_s["state_pack"] += time.monotonic() - phase_started
                phase_started = time.monotonic()
                client.sendall(state.pack())
                command = C3EffortCommand.unpack(
                    _receive_exact(client, COMMAND_PACKET_SIZE)
                )
                if command.sequence != sequence:
                    raise RuntimeError(
                        f"relay sequence mismatch: {command.sequence} != {sequence}"
                    )
                phase_wall_time_s["native_roundtrip"] += time.monotonic() - phase_started
                phase_started = time.monotonic()

                ready = bool(command.flags & C3CommandFlags.READY)
                stale = bool(command.flags & C3CommandFlags.STALE_STATE)
                semantic_hold = bool(command.flags & C3CommandFlags.SEMANTIC_HOLD)
                planner_failure = bool(command.flags & C3CommandFlags.PLANNER_FAILURE)
                counters["ready"] += int(ready)
                counters["stale"] += int(stale)
                counters["fresh"] += int(ready and not stale)
                counters["semantic_hold"] += int(semantic_hold)
                counters["planner_failure"] += int(planner_failure)
                if ready and first_ready_step is None:
                    first_ready_step = sequence
                command_age_s = abs(utime_us - command.utime_us) * 1.0e-6
                maximum_command_age_s = max(maximum_command_age_s, command_age_s)
                watchdog = (
                    not ready
                    or planner_failure
                    or command_age_s > args_cli.watchdog_timeout_s
                )
                counters["watchdog"] += int(watchdog)
                consecutive_watchdog = consecutive_watchdog + 1 if watchdog else 0
                maximum_consecutive_watchdog = max(
                    maximum_consecutive_watchdog, consecutive_watchdog
                )

                if watchdog or semantic_hold:
                    effort = torch.zeros_like(initial_q)
                else:
                    effort = torch.tensor(
                        command.joint_effort_nm,
                        device=base.device,
                        dtype=initial_q.dtype,
                    ).unsqueeze(0)
                    effort = torch.clamp(effort, -effort_limits, effort_limits)
                    if args_cli.max_effort_rate_nm_s > 0.0:
                        maximum_delta = args_cli.max_effort_rate_nm_s * control_period_s
                        effort = torch.clamp(
                            effort,
                            last_effort - maximum_delta,
                            last_effort + maximum_delta,
                        )
                last_effort = effort
                max_command_l2 = max(
                    max_command_l2,
                    math.sqrt(sum(value * value for value in command.joint_effort_nm)),
                )
                physics_step_in_progress = True
                env.step(effort)
                physics_step_in_progress = False
                executed_steps += 1
                step_diagnostics = torch.cat(
                    (
                        torch.linalg.vector_norm(
                            robot.data.applied_torque[0, joint_ids]
                        ).reshape(1),
                        torch.max(
                            torch.abs(robot.data.joint_vel[0, joint_ids])
                        ).reshape(1),
                        _pose_error_tensor(base),
                        robot.data.joint_pos[0, finger_ids],
                        robot.data.joint_pos_target[0, finger_ids],
                    )
                ).detach().cpu().tolist()
                applied_l2, absolute_joint_velocity, planar, height, rotation = (
                    float(value) for value in step_diagnostics[:5]
                )
                measured_fingers = step_diagnostics[5:7]
                commanded_fingers = step_diagnostics[7:9]
                maximum_finger_opening_m = [max(old, abs(value)) for old, value in zip(maximum_finger_opening_m, measured_fingers)]
                maximum_finger_mismatch_m = max(maximum_finger_mismatch_m, abs(measured_fingers[0] - measured_fingers[1]))
                finger_state_stream.write(json.dumps({
                    "step": sequence,
                    "measurement_utime_us": 100000 + round((sequence + 1) * control_period_s * 1e6),
                    "position_m": measured_fingers, "target_m": commanded_fingers,
                }, separators=(",", ":")) + "\n")
                finger_state_recorded_steps += 1
                max_applied_l2 = max(
                    max_applied_l2,
                    applied_l2,
                )
                max_joint_velocity = max(
                    max_joint_velocity,
                    absolute_joint_velocity,
                )
                phase_wall_time_s["physics_and_state_read"] += time.monotonic() - phase_started
                phase_started = time.monotonic()

                strict = planar < 0.02 and height < 0.01 and rotation < 0.105
                # Keep every near-goal measured pose even when routine trace
                # rows are decimated, so the final dwell is independently
                # certifiable at the actual 1 kHz measurement frequency.
                if sequence % args_cli.trace_stride == 0 or strict:
                    trace.append({
                        "step": sequence,
                        "sim_time_s": (sequence + 1) * control_period_s,
                        "measurement_utime_us": 100_000 + round((sequence + 1) * control_period_s * 1e6),
                        "target_quaternion_wxyz": target.data.root_quat_w[0].detach().cpu().tolist(),
                        "planar_error_m": planar,
                        "height_error_m": height,
                        "rotation_error_rad": rotation,
                        "joint_position_rad": joint_state(),
                        "joint_velocity_rad_s": robot.data.joint_vel[
                            0, joint_ids
                        ].detach().cpu().tolist(),
                        "tcp_position_m": tcp_position(),
                        "hand_quaternion_wxyz": robot.data.body_quat_w[0, hand_id].detach().cpu().tolist(),
                        "finger_joint_position_m": robot.data.joint_pos[0, robot.find_joints("panda_finger_joint.*")[0]].detach().cpu().tolist(),
                        "target_position_m": (
                            target.data.root_pos_w[0, :3]
                            - base.scene.env_origins[0]
                        ).detach().cpu().tolist(),
                        "command_effort_nm": list(command.joint_effort_nm),
                        "applied_effort_nm": robot.data.applied_torque[
                            0, joint_ids
                        ].detach().cpu().tolist(),
                        "command_flags": int(command.flags),
                        "command_age_s": command_age_s,
                    })

                minimum_planar = min(minimum_planar, planar)
                minimum_height = min(minimum_height, height)
                minimum_rotation = min(minimum_rotation, rotation)
                strict_dwell = strict_dwell + 1 if strict else 0
                maximum_strict_dwell = max(maximum_strict_dwell, strict_dwell)

                if sequence % args_cli.audit_stride == 0:
                    contact = mdp.domino_affordance_contact_state(
                        base,
                        contact_distance_m=args_cli.contact_distance_m,
                        evaluate_protected=False,
                        pusher_tip_from_hand_m=None if robot_model_contract else TASK_TIP_FROM_HAND_M,
                        pusher_tip_radius_m=TASK_TIP_RADIUS_M,
                        physical_contact_force_threshold_n=0.5,
                        robot_target_sensor_name=robot_target_sensor_name,
                        hand_target_sensor_name=hand_target_sensor_name,
                    )
                    safe = bool(contact["legal_physical_safe_hand_contact"][0])
                    latest_legal_safe_contact = safe
                    forbidden = bool(contact["forbidden_robot_contact"][0])
                    # Preserve the safety decision even if force telemetry
                    # subsequently raises before the trace row is written.
                    safe_contact_ever = safe_contact_ever or safe
                    forbidden_contact_ever = forbidden_contact_ever or forbidden
                    contact_audited_steps += 1
                    if safe and first_safe_contact_step is None:
                        first_safe_contact_step = sequence
                    if forbidden and first_forbidden_contact_step is None:
                        first_forbidden_contact_step = sequence
                    contact_forces = {
                        **filtered_contact_forces(robot_target_sensor_name),
                        **filtered_contact_forces(hand_target_sensor_name),
                    }
                    if (safe or forbidden) and (not trace or trace[-1]["step"] != sequence):
                        trace.append({
                            "step": sequence,
                            "sim_time_s": (sequence + 1) * control_period_s,
                            "measurement_utime_us": 100_000 + round((sequence + 1) * control_period_s * 1e6),
                            "target_position_m": (target.data.root_pos_w[0, :3] - base.scene.env_origins[0]).detach().cpu().tolist(),
                            "target_quaternion_wxyz": target.data.root_quat_w[0].detach().cpu().tolist(),
                            "tcp_position_m": tcp_position(),
                            "joint_position_rad": joint_state(),
                            "hand_quaternion_wxyz": robot.data.body_quat_w[0, hand_id].detach().cpu().tolist(),
                            "finger_joint_position_m": robot.data.joint_pos[0, robot.find_joints("panda_finger_joint.*")[0]].detach().cpu().tolist(),
                            "contact_event_extra_sample": True,
                        })
                    if trace and trace[-1]["step"] == sequence:
                        trace[-1].update(legal_safe_robot_contact=safe,
                            forbidden_robot_contact=forbidden,
                            robot_target_contact_force_n_by_sensor=contact_forces)
                    for name, force_n in contact_forces.items():
                        peak_robot_target_contact_forces[name] = max(
                            peak_robot_target_contact_forces.get(name, 0.0), force_n
                        )
                    minimum_safe_distance = min(
                        minimum_safe_distance, float(contact["minimum_safe_distance"][0])
                    )
                    minimum_forbidden_distance = min(
                        minimum_forbidden_distance,
                        float(contact["minimum_robot_forbidden_distance"][0]),
                    )

                phase_wall_time_s["contact_and_trace"] += time.monotonic() - phase_started
                if video_recorder is not None and executed_steps % video_step_stride == 0:
                    video_recorder.capture(executed_steps, executed_steps * control_period_s)
                if forbidden_contact_ever:
                    stopped_reason = "oracle_c1_violation"
                    break
                startup_failed = first_ready_step is None and sequence + 1 >= startup_steps
                post_startup_watchdog = (
                    first_ready_step is not None
                    and consecutive_watchdog >= args_cli.maximum_consecutive_watchdog_steps
                )
                if startup_failed or post_startup_watchdog:
                    stopped_reason = "watchdog_timeout"
                    break
                if strict_dwell >= dwell_steps and safe_contact_ever:
                    stopped_reason = "strict_pose_dwell_success"
                    break
                if sequence % max(1, round(1.0 / control_period_s)) == 0:
                    finger_state_stream.flush()
                    print(
                        "C3_ONLINE_ISAAC_PROGRESS",
                        f"sim_time_s={sequence * control_period_s:.3f}",
                        f"planar_m={planar:.4f}",
                        f"rotation_rad={rotation:.4f}",
                        f"fresh={counters['fresh']}",
                        f"stale={counters['stale']}",
                        f"safe={int(safe_contact_ever)}",
                        flush=True,
                    )
                if args_cli.pace_realtime:
                    target_wall = loop_started + (sequence + 1) * control_period_s
                    remaining = target_wall - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)

        final_planar, final_height, final_rotation = _pose_errors(base)
        if video_recorder is not None:
            video_recorder.capture(executed_steps, executed_steps * control_period_s)
        final_target_pose = torch.cat(
            (target.data.root_pos_w[:, :3] - base.scene.env_origins,
             target.data.root_quat_w), dim=1
        )[0].detach().cpu().tolist()
        strict_geometry_final = (
            final_planar < 0.02 and final_height < 0.01 and final_rotation < 0.105
        )
        online_closed_loop_success = (
            stopped_reason == "strict_pose_dwell_success"
            and strict_geometry_final
            and safe_contact_ever
            and not forbidden_contact_ever
        )
        result.update({
            "control_period_s": control_period_s,
            "physics_dt_s": args_cli.physics_dt_s,
            "franka_base_height_m": args_cli.table_height_offset_m,
            "control_decimation": args_cli.control_decimation,
            "executed_steps": executed_steps,
            "executed_sim_time_s": executed_steps * control_period_s,
            "wall_time_s": time.monotonic() - loop_started,
            "phase_wall_time_s": phase_wall_time_s,
            "stopped_reason": stopped_reason,
            "initial_target_pose_wxyz": initial_target_pose,
            "goal_pose_wxyz": goal_pose,
            "final_target_pose_wxyz": final_target_pose,
            "initial_robot_joint_position_rad": initial_robot_joint_position,
            "initial_robot_dynamics": initial_robot_dynamics,
            "friction_backend_application": friction_application,
            "finger_limits_backend_application": finger_limits_application,
            "maximum_finger_opening_m": maximum_finger_opening_m,
            "maximum_finger_mismatch_m": maximum_finger_mismatch_m,
            "finger_state_artifact": {"path": str(finger_state_path), "recorded_steps": finger_state_recorded_steps},
            "robot_model_contract": robot_model_contract,
            "robot_model": "FR3" if robot_model_contract else "Panda",
            "controller_parameters": controller_metadata,
            "final_robot_joint_position_rad": joint_state(),
            "initial_tcp_position_m": initial_tcp_position,
            "final_tcp_position_m": tcp_position(),
            "final_planar_error_m": final_planar,
            "final_height_error_m": final_height,
            "final_rotation_error_rad": final_rotation,
            "minimum_planar_error_m": _finite_or_none(minimum_planar),
            "minimum_height_error_m": _finite_or_none(minimum_height),
            "minimum_rotation_error_rad": _finite_or_none(minimum_rotation),
            "strict_dwell_required_steps": dwell_steps,
            "strict_pose_thresholds": {"planar_m": .02, "height_m": .01,
                "rotation_rad": .105, "dwell_time_s": args_cli.dwell_time_s},
            "pose_error_reference": "frozen initial manifest goal, full normalized SO(3)",
            "startup_timeout_steps": startup_steps,
            "first_ready_step": first_ready_step,
            "maximum_strict_dwell_steps": maximum_strict_dwell,
            "strict_geometry_final": strict_geometry_final,
            "safe_robot_contact_ever": safe_contact_ever,
            "forbidden_robot_contact_ever": forbidden_contact_ever,
            "first_safe_contact_step": first_safe_contact_step,
            "first_forbidden_contact_step": first_forbidden_contact_step,
            "minimum_safe_distance_m": _finite_or_none(minimum_safe_distance),
            "minimum_robot_forbidden_distance_m": _finite_or_none(minimum_forbidden_distance),
            "peak_robot_target_contact_force_n_by_sensor": (
                peak_robot_target_contact_forces
            ),
            "physical_end_effector": ("stock_franka_gripper_closed"
                if args_cli.stock_closed_gripper else "push_anything_spherical_pusher"),
            "acceptance_eligible": False,
            "validation_scope": "native OSC torque integration diagnostic; formal source, pose-dwell and contact audit pending",
            "pusher_collision_model": ("stock_hand_closed" if args_cli.stock_closed_gripper else args_cli.pusher_collision_model),
            "joint_armature_kg_m2": list(UPSTREAM_ARMATURE_KG_M2),
            "physical_end_effector_urdf": str(physical_end_effector_urdf),
            "command_counts": counters,
            "maximum_consecutive_watchdog_steps": maximum_consecutive_watchdog,
            "maximum_command_age_s": maximum_command_age_s,
            "maximum_command_l2_nm": max_command_l2,
            "maximum_applied_effort_l2_nm": max_applied_l2,
            "maximum_absolute_joint_velocity_rad_s": max_joint_velocity,
            "trace_stride": args_cli.trace_stride,
            "audit_stride": args_cli.audit_stride,
            "contact_audited_steps": contact_audited_steps,
            "trace": trace,
            "online_closed_loop_success": online_closed_loop_success,
            "s2_single_scene_pass": online_closed_loop_success,
            "deployment_blocker": (
                "single_scene_simulation_pass_requires_randomized_s2_s3_s4"
                if online_closed_loop_success
                else "online_isaac_closed_loop_not_yet_successful"
            ),
        })
        print(
            "C3_ONLINE_ISAAC_RESULT",
            f"success={int(online_closed_loop_success)}",
            f"reason={stopped_reason}",
            f"planar_m={final_planar:.6f}",
            f"rotation_rad={final_rotation:.6f}",
            f"c1_pass={int(not forbidden_contact_ever)}",
            flush=True,
        )
        return 0 if online_closed_loop_success else 2
    except Exception as exc:
        # Use existing CPU records only: querying a failed simulator here can
        # mask the original error and discard the evidence a second time.
        result.update(execution_failure_evidence(locals(), vars(args_cli)))
        result.update({
            "online_closed_loop_success": False,
            "s2_single_scene_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(f"C3_ONLINE_ISAAC_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if video_recorder is not None:
            try:
                result['isaaclab_video'] = video_recorder.close()
            except Exception as video_error:
                result['isaaclab_video_error'] = f'{type(video_error).__name__}: {video_error}'
        if "finger_state_stream" in locals():
            finger_state_stream.close()
            if "finger_state_artifact" in result:
                result["finger_state_artifact"]["sha256"] = hashlib.sha256(finger_state_path.read_bytes()).hexdigest()
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"C3_ONLINE_ISAAC_OUTPUT {output}", flush=True)
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
