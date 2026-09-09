#!/usr/bin/env python3
"""Track live C3+ task-space commands in Isaac Lab with a local controller.

C3 receives the measured Franka/hammer state before every planner update and
returns a newly planned execution segment.  A faster local Isaac/robot servo
tracks that short segment between planner updates.  This is closed-loop
replanning, not playback of a previously recorded rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AffordanceTeacher-C1-Franka-v0")
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--c3-scene-spec",
    type=Path,
    help=(
        "shared target-first C3 object contract; enables v3 target-plus-clutter "
        "state publication and typed C1/C2/C3 auditing"
    ),
)
parser.add_argument("--relay-host", default="127.0.0.1")
parser.add_argument("--relay-port", type=int, default=7795)
parser.add_argument("--socket-timeout-s", type=float, default=5.0)
parser.add_argument("--physics-dt-s", type=float, default=0.0025)
parser.add_argument("--control-decimation", type=int, default=4)
parser.add_argument(
    "--planner-frequency-hz",
    type=float,
    default=20.0,
    help="fresh measured-state C3+ replans per second; servo remains at env step rate",
)
parser.add_argument("--max-sim-time-s", type=float, default=35.0)
parser.add_argument("--dwell-time-s", type=float, default=0.5)
parser.add_argument("--strict-position-threshold-m", type=float, default=0.020)
parser.add_argument("--strict-height-threshold-m", type=float, default=0.010)
parser.add_argument(
    "--strict-rotation-threshold-rad",
    type=float,
    default=0.105,
    help="terminal SO(3) error threshold (0.105 rad is approximately 6 degrees)",
)
parser.add_argument("--startup-timeout-s", type=float, default=2.0)
parser.add_argument("--maximum-consecutive-watchdog-steps", type=int, default=5)
parser.add_argument("--table-height-offset-m", type=float, default=0.029)
parser.add_argument("--contact-distance-m", type=float, default=0.012)
parser.add_argument("--protected-clearance-m", type=float, default=0.005)
parser.add_argument("--robot-obstacle-clearance-m", type=float, default=0.005)
parser.add_argument(
    "--hard-c2-termination",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "terminate on clutter--protected-target collision; disabled during the "
        "C1-hard/C2-soft/C3-soft curriculum phase"
    ),
)
parser.add_argument(
    "--hard-c3-termination",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "terminate on robot--clutter collision; disabled during the "
        "C1-hard/C2-soft/C3-soft curriculum phase"
    ),
)
parser.add_argument(
    "--c2-soft-activation-distance-m",
    type=float,
    default=0.050,
    help="clearance below which the continuous C2 audit penalty becomes nonzero",
)
parser.add_argument(
    "--c3-soft-activation-distance-m",
    type=float,
    default=0.050,
    help="clearance below which the continuous C3 audit penalty becomes nonzero",
)
parser.add_argument(
    "--physical-contact-force-threshold-n",
    type=float,
    default=0.1,
    help=(
        "measured pusher/contact force required to confirm contact; the "
        "default is calibrated for the 50 g DOMINO hammer (productive "
        "contacts are typically 0.10--0.22 N in PhysX)"
    ),
)
parser.add_argument(
    "--force-c3-on-legal-safe-contact",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "map measured physical safe contact to Push Anything's native "
        "force-C3 mode input"
    ),
)
parser.add_argument(
    "--force-c3-contact-max-duration-s",
    type=float,
    default=3.0,
    help=(
        "maximum duration of one measured-contact force-C3 window; the "
        "window also closes on measured planar regression"
    ),
)
parser.add_argument(
    "--require-contact-sensors",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "fail closed unless the independent PhysX C1 reporters are present; "
        "disable only for geometry-only debugging"
    ),
)
parser.add_argument("--audit-stride", type=int, default=1)
parser.add_argument("--trace-stride", type=int, default=10)
parser.add_argument("--ik-max-evaluations", type=int, default=30)
parser.add_argument("--ik-position-tolerance-m", type=float, default=0.003)
parser.add_argument("--max-joint-target-rate-rad-s", type=float, default=2.0)
parser.add_argument(
    "--max-task-target-speed-m-s",
    type=float,
    default=0.25,
    help=(
        "maximum Cartesian reference speed passed to the local OSC; this "
        "reference governor removes discontinuities between independently "
        "replanned C3 segments without changing the planner trajectory"
    ),
)
parser.add_argument("--task-height-floor-mode", choices=("fixed", "finger-geometry"), default="fixed")
parser.add_argument("--task-height-clearance-m", type=float, default=.002)
parser.add_argument("--semantic-c1-guard-mode", choices=("disabled", "native-equivalent"), default="disabled")
parser.add_argument(
    "--minimum-task-target-z-m",
    type=float,
    default=-0.005,
    help=(
        "minimum C3-frame spherical-pusher reference height; prevents relaxed "
        "contact predictions from commanding the physical tool through the table"
    ),
)
parser.add_argument(
    "--execution-rotation-guard-limit-rad",
    type=float,
    default=3.0,
    help=(
        "optional transient planar-yaw envelope for diagnostics; the default "
        "does not reuse the terminal tolerance as a path constraint"
    ),
)
parser.add_argument(
    "--execution-rotation-guard-lookahead-s",
    type=float,
    default=0.200,
    help="measured outward angular-velocity lookahead used by the local guard",
)
parser.add_argument(
    "--execution-rotation-recovery-outward-rate-tolerance-rad-s",
    type=float,
    default=0.005,
    help=(
        "when already outside the yaw envelope, permit a stationary/inward "
        "recovery contact unless measured outward yaw rate exceeds this value"
    ),
)
parser.add_argument(
    "--measured-yaw-recovery-min-progress-rad",
    type=float,
    default=0.002,
    help=(
        "minimum observed yaw-error reduction required to infer that an "
        "untagged physical contact is a corrective yaw action"
    ),
)
parser.add_argument(
    "--measured-c3-yaw-brake",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "release and retreat from a native C3 safe contact when measured "
        "object yaw is correcting into half of the terminal yaw band; this "
        "prevents contact inertia from carrying the object through the goal"
    ),
)
parser.add_argument(
    "--execution-rotation-guard-retreat-m",
    type=float,
    default=0.030,
    help="radial EE retreat commanded when the local rotation guard triggers",
)
parser.add_argument(
    "--semantic-yaw-brake-retreat-tolerance-m",
    type=float,
    default=0.002,
    help=(
        "task-space tolerance used to release a completed semantic yaw-brake "
        "retreat after measured physical contact has cleared"
    ),
)
parser.add_argument(
    "--local-controller",
    choices=("position_ik", "rmpflow_position", "cartesian_impedance"),
    default="position_ik",
    help=(
        "robot-specific Isaac servo: position_ik is the legacy baseline; "
        "rmpflow_position adds full-arm obstacle avoidance; "
        "cartesian_impedance also applies C3's feedforward wrench"
    ),
)
parser.add_argument(
    "--osc-track-orientation",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "hold the startup pusher orientation in the Isaac Cartesian "
        "impedance controller; C3 still plans only the spherical tip XYZ"
    ),
)
parser.add_argument("--osc-translation-stiffness", type=float, default=150.0)
parser.add_argument("--osc-rotation-stiffness", type=float, default=25.0)
parser.add_argument("--osc-reference-velocity-mode", choices=("trajectory", "governed"), default="trajectory")
parser.add_argument("--osc-control-point", choices=("hand", "tip"), default="hand")
parser.add_argument("--osc-damping-ratio", type=float, default=1.0)
parser.add_argument(
    "--osc-track-trajectory-velocity",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="use the C3 reference velocity in OSC damping (controlled bridge ablation)",
)
parser.add_argument(
    "--osc-inertial-dynamics-decoupling",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "map task-space accelerations through the measured operational-space "
        "inertia; enabled by default for stable Franka Cartesian control"
    ),
)
parser.add_argument("--osc-nullspace-stiffness", type=float, default=10.0)
parser.add_argument("--osc-nullspace-damping-ratio", type=float, default=1.0)
parser.add_argument(
    "--osc-nullspace-target",
    choices=("default", "push_anything_joint2", "rmpflow_semantic"),
    default="default",
    help=(
        "fixed startup posture, the upstream Push Anything panda_joint2-only "
        "redundancy objective, or a live RMPflow collision-aware posture "
        "reference projected through OSC's null space"
    ),
)
parser.add_argument("--rmpflow-proxy-padding-m", type=float, default=0.010)
parser.add_argument(
    "--rmpflow-max-joint-reference-error-rad", type=float, default=0.35
)
parser.add_argument(
    "--rmpflow-safe-approach-switch-m",
    type=float,
    default=0.060,
    help=(
        "distance at which the full-target avoidance proxy is replaced by "
        "the protected-region proxy to permit safe-handle contact"
    ),
)
parser.add_argument(
    "--max-feedforward-force-n",
    type=float,
    default=5.0,
    help="norm cap on the open-loop C3 feedforward force applied by Isaac OSC",
)
parser.add_argument(
    "--c3-force-action-sign",
    type=float,
    choices=(-1.0, 1.0),
    default=-1.0,
    help=(
        "map C3's optimized external contact force to the robot-applied "
        "wrench. Dairlib solves M*dv+c=B*u+J^T*lambda, so its actuator "
        "counterforce is -J^T*lambda; Isaac OSC directly adds J^T*wrench "
        "and therefore requires the default -1 mapping"
    ),
)
parser.add_argument(
    "--position-feedforward-compliance-m-per-n",
    "--rmpflow-feedforward-compliance-m-per-n",
    dest="position_feedforward_compliance_m_per_n",
    type=float,
    default=0.005,
    help=(
        "virtual XY compliance used to convert C3 feedforward force into an "
        "equilibrium-point offset for either position-controlled backend; "
        "zero disables the conversion. The old --rmpflow-* spelling remains "
        "as a compatibility alias"
    ),
)
parser.add_argument(
    "--position-feedforward-max-offset-m",
    "--rmpflow-feedforward-max-offset-m",
    dest="position_feedforward_max_offset_m",
    type=float,
    default=0.010,
    help="norm cap on the C3 force-derived position-control XY target offset",
)
parser.add_argument(
    "--semantic-contact-push-speed-m-s",
    type=float,
    default=0.0,
    help=(
        "after a measured legal safe contact, maintain bounded object-relative "
        "compression along C3's planned contact velocity at this speed; zero "
        "keeps the unmodified C3 task trajectory"
    ),
)
parser.add_argument(
    "--semantic-contact-activation-planar-threshold-m",
    type=float,
    default=None,
    help=(
        "enable the bounded legal-contact servo only after measured XY goal "
        "error is at or below this terminal threshold; omitted preserves "
        "the original all-distances behavior"
    ),
)
parser.add_argument(
    "--semantic-contact-axis-source",
    choices=("planned", "goal", "goal_wrench"),
    default="planned",
    help=(
        "direction for the bounded legal-contact servo: planned preserves "
        "C3's instantaneous EE velocity; goal uses the measured target-to-goal "
        "XY direction; goal_wrench additionally uses the measured safe contact "
        "point, object COM, yaw error, and yaw rate to regulate planar torque"
    ),
)
parser.add_argument(
    "--semantic-contact-yaw-moment-gain-m-per-rad",
    type=float,
    default=0.04,
    help=(
        "goal_wrench proportional gain from current-minus-goal yaw error to "
        "the desired contact-force moment arm"
    ),
)
parser.add_argument(
    "--semantic-contact-yaw-rate-moment-gain-m-s-per-rad",
    type=float,
    default=0.02,
    help=(
        "goal_wrench damping gain from measured object yaw rate to the desired "
        "contact-force moment arm"
    ),
)
parser.add_argument(
    "--semantic-contact-max-axis-deviation-rad",
    type=float,
    default=1.22,
    help=(
        "maximum goal_wrench push-direction deviation from the target-to-goal "
        "axis (1.22 rad is approximately 70 degrees)"
    ),
)
parser.add_argument(
    "--semantic-contact-planar-regression-tolerance-m",
    type=float,
    default=0.001,
    help=(
        "stop a bounded contact pulse and replan when measured XY goal error "
        "regresses this far beyond the best value seen in the pulse"
    ),
)
parser.add_argument(
    "--force-c3-yaw-regression-tolerance-rad",
    type=float,
    default=0.005,
    help=(
        "release a latched native C3 contact when measured absolute yaw "
        "error grows this far beyond the best value observed in the contact; "
        "this uses deployable pose feedback rather than predicted contact "
        "effect"
    ),
)
parser.add_argument(
    "--contact-acquisition-speed-m-s",
    type=float,
    default=0.03,
    help=(
        "when C3 reaches the measured safe surface but no physical contact is "
        "reported, close the remaining model/registration gap toward the nearest "
        "safe point at this speed; zero disables acquisition"
    ),
)
parser.add_argument(
    "--contact-acquisition-trigger-m",
    type=float,
    default=0.018,
    help="safe-surface distance below which bounded contact acquisition may start",
)
parser.add_argument(
    "--contact-acquisition-max-travel-m",
    type=float,
    default=0.018,
    help="maximum planar safe-point approach applied during one acquisition",
)
parser.add_argument(
    "--contact-acquisition-forbidden-margin-m",
    type=float,
    default=0.003,
    help=(
        "minimum amount by which the safe surface must be closer than the "
        "forbidden surface before contact acquisition is admitted"
    ),
)
parser.add_argument(
    "--semantic-contact-max-duration-s",
    type=float,
    default=1.0,
    help=(
        "maximum duration of one measured-contact servo pulse before control "
        "returns to C3 for a fresh contact plan"
    ),
)
parser.add_argument(
    "--semantic-contact-compression-m",
    type=float,
    default=0.002,
    help="radial compression retained by the object-relative contact servo",
)
parser.add_argument(
    "--semantic-contact-max-additional-compression-m",
    type=float,
    default=0.015,
    help="maximum extra axial advance beyond the first-contact command",
)
parser.add_argument(
    "--semantic-contact-yaw-gain-m-per-rad",
    type=float,
    default=0.0,
    help="lateral safe-contact shift per radian of signed target yaw error",
)
parser.add_argument(
    "--semantic-contact-max-lateral-offset-m",
    type=float,
    default=0.020,
    help="absolute yaw-correction contact offset within the safe handle",
)
parser.add_argument(
    "--drake-tip-from-hand-m",
    type=float,
    default=None,
    help=(
        "panda_hand-frame distance to C3's contact proxy; defaults to the "
        "closed-fingertip proxy (104.279 mm), or 126.5 mm for the explicit "
        "attached-pusher ablation"
    ),
)
parser.add_argument(
    "--use-push-anything-end-effector",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "replace the stock gripper collisions with Push Anything's 19.5 mm "
        "spherical flange-mounted pusher. Disable this for the deployable "
        "closed-gripper contact mode"
    ),
)
parser.add_argument(
    "--pusher-collision-model",
    choices=("physical", "sphere_only"),
    default="physical",
    help=(
        "physical uses the hand, peg, and tip collisions; sphere_only is an "
        "explicit C3 point-actor diagnostic"
    ),
)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--video", action="store_true")
parser.add_argument(
    "--video-folder",
    type=Path,
    default=Path("outputs/contact_planner_m3/online_isaac_videos"),
)
parser.add_argument("--video-name-prefix", default="c3_online_task")
parser.add_argument("--video-fps", type=float, default=20.0)
parser.add_argument("--goal-ghost-opacity", type=float, default=0.68)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(1.05, 0.75, 0.72))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.40, 0.20, 0.04))
parser.add_argument(
    "--initial-joint-position-rad", type=float, nargs=7,
    default=(2.191, 1.1, -1.33, -2.22, 1.30, 2.02, 0.08),
)
parser.add_argument(
    "--support-quaternion-wxyz", type=float, nargs=4,
    default=(-0.4937799140314683, 0.5013379099086276,
             0.506216296031207, 0.4985847553024161),
)
parser.add_argument("--domino-root", type=Path, default=Path("/data1/linsixu/DOMINO"))
parser.add_argument(
    "--domino-usd-root", type=Path,
    default=Path(__file__).resolve().parents[1] / "data/domino_usd",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--osc-dynamics-audit", action="store_true")
args_cli = parser.parse_args()
if args_cli.drake_tip_from_hand_m is None:
    args_cli.drake_tip_from_hand_m = (
        0.1265
        if args_cli.use_push_anything_end_effector
        else 0.10427911200523377
    )
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.controllers.rmp_flow import RmpFlow
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)

physical_end_effector_urdf: Path | None = None
if args_cli.use_push_anything_end_effector:
    from dapl.contact_planner.franka_push_tool_urdf import (
        PUSH_TOOL_TIP_FROM_HAND_M,
        build_push_anything_franka_urdf,
    )

    if not math.isclose(
        args_cli.drake_tip_from_hand_m,
        PUSH_TOOL_TIP_FROM_HAND_M,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "--drake-tip-from-hand-m must remain 0.1265 when the physical "
            "Push Anything end effector is enabled"
        )
    isaacsim_spec = importlib.util.find_spec("isaacsim")
    if isaacsim_spec is None or not isaacsim_spec.submodule_search_locations:
        raise RuntimeError("Cannot locate Isaac Sim's packaged Franka URDF")
    isaacsim_root = Path(next(iter(isaacsim_spec.submodule_search_locations)))
    stock_franka_urdf = (
        isaacsim_root
        / "exts"
        / "isaacsim.asset.importer.urdf"
        / "data"
        / "urdf"
        / "robots"
        / "franka_description"
        / "robots"
        / "panda_arm_hand.urdf"
    )
    generated_tool_root = (
        Path("/tmp/IsaacLab/nonprehensile_franka_push_anything")
        / args_cli.pusher_collision_model
    )
    physical_end_effector_urdf = build_push_anything_franka_urdf(
        stock_franka_urdf,
        generated_tool_root / "panda_arm_hand_push_anything.urdf",
        sphere_only_collision=(args_cli.pusher_collision_model == "sphere_only"),
    )
    os.environ["DAPL_LOCAL_FRANKA_URDF"] = str(physical_end_effector_urdf)
    os.environ["DAPL_LOCAL_FRANKA_USD_DIR"] = str(generated_tool_root / "usd")

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from dapl.contact_planner.c3_online_protocol import (
    C3CommandFlags,
    C3MeasuredSceneState,
    C3MeasuredState,
    C3RigidBodyState,
    C3StateFlags,
    C3TaskCommand,
    MAX_CLUTTER_OBJECTS,
    TASK_COMMAND_PACKET_SIZE,
)
from dapl.contact_planner.contact_servo import goal_wrench_contact_axis
from dapl.contact_planner.franka_tcp_ik import FrankaTcpIK
from dapl.contact_planner.isaac_bridge import (
    execution_rotation_guard_requires_retreat,
    measured_contact_is_yaw_recovery,
    measured_yaw_brake_requested,
    measured_yaw_regression_brake_requested,
)
from dapl.contact_planner.isaac_visualization import (
    M1MarkerUpdateWrapper,
    create_m1_video_markers,
)
from dapl.contact_planner.streaming_video import StreamingRgbVideoWriter
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp


FRANKA_URDF = str(
    Path(isaaclab.__file__).resolve().parent
    / "controllers/config/data/lula_franka_gen.urdf"
)


class _RmpFlowVisualCuboid(VisualCuboid):
    """Present CUDA-backed IsaacLab proxy prims through Lula's NumPy API."""

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def get_world_pose(self):
        position, quaternion = super().get_world_pose()
        return self._numpy(position), self._numpy(quaternion)

    def get_local_scale(self):
        return self._numpy(super().get_local_scale())

    def get_size(self):
        return self._numpy(super().get_size())


def _never_terminate(env) -> torch.Tensor:
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("C3 relay closed inside a task-command packet")
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


def _load_scene_spec(path: Path | None) -> list[dict[str, object]] | None:
    if path is None:
        return None
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"C3 scene spec does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "nonprehensile.c3_online_scene.v1":
        raise ValueError("C3 scene spec has an unsupported schema")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("C3 scene spec objects must be a non-empty list")
    if len(objects) > 1 + MAX_CLUTTER_OBJECTS:
        raise ValueError("C3 scene spec exceeds protocol clutter capacity")
    if not isinstance(objects[0], dict) or objects[0].get("role") != "target":
        raise ValueError("C3 scene spec object zero must be the target")
    clutter_indices = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError("C3 scene spec object entries must be mappings")
        expected_role = "target" if index == 0 else "clutter"
        if item.get("role") != expected_role:
            raise ValueError(f"C3 scene spec object {index} must be {expected_role}")
        for field in (
            "asset_id",
            "c3_body_name",
            "state_channel",
            "support_quaternion_wxyz",
        ):
            if field not in item:
                raise ValueError(f"C3 scene spec object {index} is missing {field}")
        quaternion = item["support_quaternion_wxyz"]
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            raise ValueError("support_quaternion_wxyz must contain four values")
        quaternion = [float(value) for value in quaternion]
        if not all(math.isfinite(value) for value in quaternion):
            raise ValueError("support quaternion must be finite")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3):
            raise ValueError("support quaternion must be normalized")
        item = dict(item)
        item["support_quaternion_wxyz"] = quaternion
        if expected_role == "clutter":
            obstacle_index = item.get("isaac_obstacle_index")
            if not isinstance(obstacle_index, int) or obstacle_index < 0:
                raise ValueError(
                    "clutter objects need a non-negative isaac_obstacle_index")
            clutter_indices.append(obstacle_index)
        objects[index] = item
    if clutter_indices != list(range(len(clutter_indices))):
        raise ValueError(
            "clutter isaac_obstacle_index values must be contiguous and target-first")
    return objects


def _validate_scene_spec_manifest(
    manifest: Path, scene_objects: list[dict[str, object]] | None
) -> None:
    if scene_objects is None:
        return
    expected_asset_ids = [str(item["asset_id"]) for item in scene_objects]
    scene_count = 0
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            scene_count += 1
            payload = json.loads(line)
            objects = payload.get("objects")
            if not isinstance(objects, list) or len(objects) < len(scene_objects):
                raise ValueError(
                    f"manifest line {line_number} has too few objects for C3 scene spec")
            actual_asset_ids = [
                str(item.get("asset_id"))
                for item in objects[: len(scene_objects)]
            ]
            if actual_asset_ids != expected_asset_ids:
                raise ValueError(
                    f"manifest line {line_number} object order does not match C3 scene spec")
    if scene_count == 0:
        raise ValueError("manifest contains no scenes")


def _pose_errors(base) -> tuple[float, float, float]:
    target = base.scene["target"]
    goal = base.command_manager.get_command("target_object_pose")
    position = target.data.root_pos_w[:, :3] - base.scene.env_origins
    delta = goal[:, :3] - position
    return (
        float(torch.linalg.vector_norm(delta[0, :2])),
        float(torch.abs(delta[0, 2])),
        float(_quaternion_distance(target.data.root_quat_w, goal[:, 3:7])[0]),
    )


def _signed_planar_yaw_error(
    current_quaternion_wxyz: torch.Tensor,
    goal_quaternion_wxyz: torch.Tensor,
) -> float:
    """Return current-minus-goal yaw in ``[-pi, pi]``."""

    relative = quat_mul(
        current_quaternion_wxyz.unsqueeze(0),
        quat_inv(goal_quaternion_wxyz.unsqueeze(0)),
    )[0]
    w, x, y, z = relative
    return float(torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    ))


def _configure_env():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True
    )
    env_cfg.use_torch_compile = False
    env_cfg.seed = args_cli.seed
    env_cfg.disable_obs_noise = True
    env_cfg.enforce_joint_limits = False
    # Push Anything welds the Franka base 29 mm above its ground plane
    # (kFrankaToGroundOffset = [0, 0, -0.029]).  The Isaac manifest bridge
    # already shifts object poses by this amount; shift the robot as well so
    # C3 task-space Z values do not drive the physical spherical tip through
    # Isaac's z=0 support surface.
    robot_initial_position = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (
        float(robot_initial_position[0]),
        float(robot_initial_position[1]),
        float(robot_initial_position[2]) + args_cli.table_height_offset_m,
    )
    # The action term resolves its fixed null-space target while the scene is
    # constructed.  Make the commanded startup posture the articulation's
    # default as well as writing it after reset, so OSC does not silently pull
    # toward an unrelated configuration.
    for joint_index, joint_position in enumerate(
        args_cli.initial_joint_position_rad, start=1
    ):
        env_cfg.scene.robot.init_state.joint_pos[
            f"panda_joint{joint_index}"
        ] = float(joint_position)
    if args_cli.use_push_anything_end_effector:
        # The custom tool is flange-mounted, as in Push Anything.  Keep the
        # now visual-only fingers open so the peg and spherical tip are visible.
        env_cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.04
    else:
        # The deployable default pushes with the unmodified, fully closed
        # Franka fingertips.  Set this explicitly instead of inheriting a task
        # configuration whose stock gripper default may be open.
        env_cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.0
    if args_cli.local_controller in ("position_ik", "rmpflow_position"):
        env_cfg.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
            clip={
                "panda_joint1": (-2.8973, 2.8973),
                "panda_joint2": (-1.7628, 1.7628),
                "panda_joint3": (-2.8973, 2.8973),
                "panda_joint4": (-3.0718, -0.0698),
                "panda_joint5": (-2.8973, 2.8973),
                "panda_joint6": (-0.0175, 3.7525),
                "panda_joint7": (-2.8973, 2.8973),
            },
        )
    else:
        # C3 optimizes a spherical task point rather than Franka joint
        # torques.  Track that same point with an Isaac-native OSC and add
        # C3's feedforward contact force in the robot-base frame.  Keeping the
        # robot-specific inverse dynamics on the Isaac side is also the
        # interface that can later be replaced by a real Franka impedance
        # controller without replaying Drake torques.
        env_cfg.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        env_cfg.scene.robot.actuators["panda_shoulder"].damping = 0.0
        env_cfg.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        env_cfg.scene.robot.actuators["panda_forearm"].damping = 0.0
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        env_cfg.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            # The task-point action below corrects IsaacLab 2.2's offset
            # Jacobian. The hand mode retains the earlier analytic mapping
            # for explicitly recorded comparisons.
            body_offset=(OperationalSpaceControllerActionCfg.OffsetCfg(
                pos=(0.0, 0.0, args_cli.drake_tip_from_hand_m))
                if args_cli.osc_control_point == "tip" else None),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs", "wrench_abs"],
                # Push Anything controls a spherical point actor and publishes
                # no orientation reference.  Constraining a fixed 6-D pose
                # over-specifies that interface and can make the operational
                # mass inverse singular during low table contacts.
                motion_control_axes_task=(
                    (1, 1, 1, 1, 1, 1)
                    if args_cli.osc_track_orientation
                    else (1, 1, 1, 0, 0, 0)
                ),
                contact_wrench_control_axes_task=(1, 1, 1, 0, 0, 0),
                inertial_dynamics_decoupling=(
                    args_cli.osc_inertial_dynamics_decoupling
                ),
                # C3 controls only the spherical tip translation.  Decouple
                # translation from the unconstrained rotational block so the
                # operational-space inverse remains well conditioned near the
                # table.  This matches IsaacLab's supported Franka OSC setup.
                partial_inertial_dynamics_decoupling=(
                    args_cli.osc_inertial_dynamics_decoupling
                ),
                gravity_compensation=False,
                impedance_mode="fixed",
                motion_stiffness_task=(
                    args_cli.osc_translation_stiffness,
                    args_cli.osc_translation_stiffness,
                    args_cli.osc_translation_stiffness,
                    args_cli.osc_rotation_stiffness,
                    args_cli.osc_rotation_stiffness,
                    args_cli.osc_rotation_stiffness,
                ),
                motion_damping_ratio_task=(args_cli.osc_damping_ratio,) * 6,
                contact_wrench_stiffness_task=None,
                nullspace_control="position",
                nullspace_stiffness=args_cli.osc_nullspace_stiffness,
                nullspace_damping_ratio=(
                    args_cli.osc_nullspace_damping_ratio
                ),
            ),
            position_scale=1.0,
            orientation_scale=1.0,
            wrench_scale=1.0,
            nullspace_joint_pos_target="default",
        )
    if args_cli.osc_track_trajectory_velocity:
        if args_cli.local_controller != "cartesian_impedance":
            raise ValueError("trajectory velocity tracking requires cartesian_impedance")
        from dapl.contact_planner.isaac_trajectory_action import (
            TrajectoryOperationalSpaceControllerAction,
        )
        env_cfg.actions.arm_action.class_type = TrajectoryOperationalSpaceControllerAction
    if args_cli.osc_control_point == "tip":
        if args_cli.local_controller != "cartesian_impedance":
            raise ValueError("tip control point requires Cartesian impedance")
        from dapl.contact_planner.isaac_trajectory_action import TaskPointOperationalSpaceControllerAction
        env_cfg.actions.arm_action.class_type = TaskPointOperationalSpaceControllerAction
    env_cfg.sim.dt = args_cli.physics_dt_s
    env_cfg.decimation = args_cli.control_decimation
    if args_cli.video:
        video_step_stride = max(
            1,
            round(
                1.0
                / (
                    args_cli.physics_dt_s
                    * args_cli.control_decimation
                    * args_cli.video_fps
                )
            ),
        )
        env_cfg.sim.render_interval = (
            args_cli.control_decimation * video_step_stride
        )
    else:
        env_cfg.sim.render_interval = args_cli.control_decimation
    env_cfg.episode_length_s = args_cli.max_sim_time_s + 10.0
    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
        env_cfg.viewer.origin_type = "world"
        ground_material = getattr(
            getattr(getattr(env_cfg.scene, "ground", None), "spawn", None),
            "visual_material",
            None,
        )
        if ground_material is not None:
            ground_material.diffuse_color = (0.08, 0.20, 0.32)
            ground_material.roughness = 0.85
        light_spawn = getattr(getattr(env_cfg.scene, "light", None), "spawn", None)
        if light_spawn is not None and hasattr(light_spawn, "intensity"):
            light_spawn.intensity = 1200.0
        env_cfg.sim.render.enable_translucency = True
    for term_name in (
        "time_out",
        "reached",
        "forbidden_region_contact",
        "protected_region_collision",
        "robot_obstacle_collision",
        "object_dropped",
    ):
        term = getattr(env_cfg.terminations, term_name, None)
        if term is not None:
            term.func = _never_terminate
            term.params = {}
    return env_cfg


def main() -> int:
    for value, label in (
        (args_cli.physics_dt_s, "physics dt"),
        (args_cli.max_sim_time_s, "maximum simulation time"),
        (args_cli.dwell_time_s, "dwell time"),
        (args_cli.strict_position_threshold_m, "strict position threshold"),
        (args_cli.strict_height_threshold_m, "strict height threshold"),
        (args_cli.strict_rotation_threshold_rad, "strict rotation threshold"),
        (args_cli.startup_timeout_s, "startup timeout"),
        (args_cli.max_joint_target_rate_rad_s, "joint target rate"),
        (args_cli.max_task_target_speed_m_s, "task target speed limit"),
        (args_cli.task_height_clearance_m, "task height clearance"),
        (args_cli.planner_frequency_hz, "planner frequency"),
        (args_cli.video_fps, "video fps"),
        (args_cli.osc_translation_stiffness, "OSC translation stiffness"),
        (args_cli.osc_rotation_stiffness, "OSC rotation stiffness"),
        (args_cli.osc_damping_ratio, "OSC damping ratio"),
        (args_cli.osc_nullspace_stiffness, "OSC null-space stiffness"),
        (
            args_cli.osc_nullspace_damping_ratio,
            "OSC null-space damping ratio",
        ),
        (args_cli.max_feedforward_force_n, "feedforward-force cap"),
        (args_cli.rmpflow_safe_approach_switch_m, "RMPflow approach switch"),
        (
            args_cli.rmpflow_max_joint_reference_error_rad,
            "RMPflow maximum joint-reference error",
        ),
        (
            args_cli.execution_rotation_guard_limit_rad,
            "execution rotation-guard limit",
        ),
        (
            args_cli.execution_rotation_guard_lookahead_s,
            "execution rotation-guard lookahead",
        ),
        (
            args_cli.execution_rotation_guard_retreat_m,
            "execution rotation-guard retreat",
        ),
    ):
        if value <= 0.0:
            raise ValueError(f"{label} must be positive")
    if not math.isfinite(args_cli.minimum_task_target_z_m):
        raise ValueError("minimum task target Z must be finite")
    if min(
        args_cli.control_decimation,
        args_cli.maximum_consecutive_watchdog_steps,
        args_cli.audit_stride,
        args_cli.trace_stride,
        args_cli.ik_max_evaluations,
    ) <= 0:
        raise ValueError("all step/count parameters must be positive")
    if args_cli.protected_clearance_m < 0.0:
        raise ValueError("protected clearance must be non-negative")
    if args_cli.robot_obstacle_clearance_m < 0.0:
        raise ValueError("robot-obstacle clearance must be non-negative")
    if (
        args_cli.c2_soft_activation_distance_m
        <= args_cli.protected_clearance_m
    ):
        raise ValueError(
            "C2 soft activation distance must exceed protected clearance"
        )
    if (
        args_cli.c3_soft_activation_distance_m
        <= args_cli.robot_obstacle_clearance_m
    ):
        raise ValueError(
            "C3 soft activation distance must exceed robot-obstacle clearance"
        )
    if args_cli.physical_contact_force_threshold_n < 0.0:
        raise ValueError("physical contact-force threshold must be non-negative")
    if args_cli.rmpflow_proxy_padding_m < 0.0:
        raise ValueError("RMPflow proxy padding must be non-negative")
    if args_cli.position_feedforward_compliance_m_per_n < 0.0:
        raise ValueError("position feedforward compliance must be non-negative")
    if args_cli.position_feedforward_max_offset_m < 0.0:
        raise ValueError("position feedforward offset cap must be non-negative")
    if args_cli.execution_rotation_recovery_outward_rate_tolerance_rad_s < 0.0:
        raise ValueError(
            "execution rotation-recovery outward-rate tolerance must be "
            "non-negative"
        )
    if args_cli.measured_yaw_recovery_min_progress_rad < 0.0:
        raise ValueError(
            "measured yaw-recovery minimum progress must be non-negative"
        )
    if args_cli.semantic_contact_push_speed_m_s < 0.0:
        raise ValueError("semantic contact push speed must be non-negative")
    if (
        args_cli.semantic_contact_activation_planar_threshold_m is not None
        and args_cli.semantic_contact_activation_planar_threshold_m <= 0.0
    ):
        raise ValueError(
            "semantic contact activation planar threshold must be positive"
        )
    if args_cli.semantic_contact_planar_regression_tolerance_m < 0.0:
        raise ValueError(
            "semantic contact planar regression tolerance must be non-negative"
        )
    if args_cli.force_c3_yaw_regression_tolerance_rad < 0.0:
        raise ValueError(
            "force-C3 yaw regression tolerance must be non-negative"
        )
    if (
        args_cli.force_c3_on_legal_safe_contact
        and args_cli.force_c3_contact_max_duration_s <= 0.0
    ):
        raise ValueError("force-C3 contact maximum duration must be positive")
    if args_cli.contact_acquisition_speed_m_s < 0.0:
        raise ValueError("contact acquisition speed must be non-negative")
    if args_cli.contact_acquisition_trigger_m <= 0.0:
        raise ValueError("contact acquisition trigger must be positive")
    if args_cli.contact_acquisition_max_travel_m < 0.0:
        raise ValueError("contact acquisition maximum travel must be non-negative")
    if args_cli.contact_acquisition_forbidden_margin_m < 0.0:
        raise ValueError("contact acquisition forbidden margin must be non-negative")
    if (
        args_cli.semantic_contact_push_speed_m_s > 0.0
        and args_cli.semantic_contact_max_duration_s <= 0.0
    ):
        raise ValueError("semantic contact maximum duration must be positive")
    if args_cli.semantic_contact_compression_m < 0.0:
        raise ValueError("semantic contact compression must be non-negative")
    if args_cli.semantic_contact_max_additional_compression_m < 0.0:
        raise ValueError(
            "semantic contact maximum additional compression must be non-negative"
        )
    if args_cli.semantic_contact_yaw_gain_m_per_rad < 0.0:
        raise ValueError("semantic contact yaw gain must be non-negative")
    if args_cli.semantic_contact_max_lateral_offset_m < 0.0:
        raise ValueError("semantic contact lateral offset must be non-negative")
    if args_cli.semantic_contact_yaw_moment_gain_m_per_rad < 0.0:
        raise ValueError("semantic contact yaw moment gain must be non-negative")
    if args_cli.semantic_contact_yaw_rate_moment_gain_m_s_per_rad < 0.0:
        raise ValueError(
            "semantic contact yaw-rate moment gain must be non-negative"
        )
    if not 0.0 <= args_cli.semantic_contact_max_axis_deviation_rad <= math.pi:
        raise ValueError(
            "semantic contact maximum axis deviation must lie in [0, pi]"
        )
    semantic_contact_servo_enabled = (
        args_cli.semantic_contact_push_speed_m_s > 0.0
        or args_cli.contact_acquisition_speed_m_s > 0.0
    )
    if (
        semantic_contact_servo_enabled
        and args_cli.local_controller == "cartesian_impedance"
    ):
        raise ValueError(
            "semantic contact acquisition/push requires a position-controlled "
            "backend"
        )
    if (
        semantic_contact_servo_enabled
        and args_cli.local_controller == "position_ik"
        and args_cli.c3_scene_spec is not None
    ):
        raise ValueError(
            "position_ik semantic contact servo is target-only; C1+C2+C3 scene "
            "execution requires rmpflow_position for full-arm clutter avoidance"
        )
    if (
        args_cli.osc_nullspace_target == "rmpflow_semantic"
        and args_cli.local_controller != "cartesian_impedance"
    ):
        raise ValueError(
            "rmpflow_semantic null-space targets require cartesian_impedance"
        )

    manifest = args_cli.manifest.expanduser().resolve()
    output = args_cli.output.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DAPL_CLUTTER_MANIFEST"] = str(manifest)
    os.environ["DAPL_CLUTTER_ASSET_SOURCE"] = "domino"
    os.environ["DOMINO_ROOT"] = str(args_cli.domino_root.expanduser().resolve())
    os.environ["DOMINO_USD_ROOT"] = str(args_cli.domino_usd_root.expanduser().resolve())
    scene_objects = _load_scene_spec(args_cli.c3_scene_spec)
    _validate_scene_spec_manifest(manifest, scene_objects)
    multi_object_mode = scene_objects is not None
    contact_model_path = os.environ.get("PUSH_ANYTHING_CONTACT_MODEL_MANIFEST")
    contact_model = json.loads(Path(contact_model_path).read_text()) if contact_model_path else None
    native_pd_rollout_interpolation = None
    native_osc_matched_coarse_model = False
    native_relinearized_pd_cost = False
    native_planning_horizon = None
    native_planner_finger_table_clearance = None
    native_goal_mode = None
    native_goal_artifact = None
    runtime_root = os.environ.get("PUSH_ANYTHING_ROOT")
    if runtime_root:
        import yaml
        runtime = Path(runtime_root)
        demo = os.environ.get("PUSH_ANYTHING_DEMO_NAME", "anything")
        controller_params = yaml.safe_load((runtime / "examples/sampling_c3" / demo /
            "parameters/sampling_c3_controller_params.yaml").read_text())
        native_pd_rollout_interpolation = "foh" if controller_params.get("use_foh_pd_rollout", False) else "zoh"
        native_osc_matched_coarse_model = bool(controller_params.get("use_osc_matched_coarse_model", False))
        native_planner_finger_table_clearance = controller_params.get("planner_finger_table_clearance")
        native_relinearized_pd_cost = bool(controller_params.get("use_relinearized_pd_cost", False))
        native_planning_horizon = int(yaml.safe_load((runtime / controller_params["sampling_c3_options_file"]).read_text())["N"])
        native_goal_path = runtime / controller_params["goal_params_file"]
        native_goal_bytes = native_goal_path.read_bytes()
        native_goal_mode = yaml.safe_load(native_goal_bytes)["goal_mode"]
        native_goal_artifact = {"path": str(native_goal_path.resolve()),
            "sha256": hashlib.sha256(native_goal_bytes).hexdigest()}
    semantic_trajectory_guard = None
    semantic_guard_active = False
    semantic_guard_distance = None
    semantic_guard_activation_count = 0
    if args_cli.semantic_c1_guard_mode == "native-equivalent":
        from dapl.contact_planner.semantic_trajectory_guard import SemanticTrajectoryGuard
        runtime_root = os.environ.get("PUSH_ANYTHING_ROOT")
        if not runtime_root:
            raise ValueError("Native-equivalent semantic guard requires the staged runtime")
        semantic_trajectory_guard = SemanticTrajectoryGuard.from_runtime(
            runtime_root, os.environ.get("PUSH_ANYTHING_DEMO_NAME", "anything"))
    finger_reference_vertices = []
    if args_cli.task_height_floor_mode == "finger-geometry":
        if not contact_model:
            raise ValueError("Finger geometry height floor requires shared contact geometry")
        for relative in contact_model["files"]:
            if Path(relative).name in ("panda_leftfinger.obj", "panda_rightfinger.obj"):
                for line in (Path(contact_model_path).parent / relative).read_text().splitlines():
                    if line.startswith("v "):
                        finger_reference_vertices.append(tuple(map(float, line.split()[1:4])))
        if not finger_reference_vertices:
            raise ValueError("Missing closed finger reference vertices")
    if contact_model:
        if args_cli.use_push_anything_end_effector or not args_cli.osc_track_orientation:
            raise ValueError("shared finger contact model requires the closed gripper with orientation tracking")
        if not math.isclose(contact_model["reference_offset_m"], args_cli.drake_tip_from_hand_m, abs_tol=1e-9):
            raise ValueError("shared finger model task reference differs from the executor")
        asset_contract = contact_model.get("asset_contract")
        if asset_contract:
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                target_spec = json.loads(line)["objects"][0]
                if target_spec["asset_id"] != asset_contract["target_asset_id"] or target_spec["scale"] != asset_contract["target_scale"]:
                    raise ValueError("shared collision export does not match the target asset/scale")
            target_usd = args_cli.domino_usd_root / "020_hammer/base0.usd"
            if hashlib.sha256(target_usd.read_bytes()).hexdigest() != asset_contract["target_usd_sha256"]:
                raise ValueError("Isaac target asset changed; re-export the shared PhysX contact model")
    if (
        args_cli.osc_nullspace_target == "rmpflow_semantic"
        or args_cli.local_controller == "rmpflow_position"
    ) and not multi_object_mode:
        raise ValueError("semantic RMPflow requires --c3-scene-spec")

    env = None
    video_writer = None
    result: dict[str, object] = {
        "schema": (
            "nonprehensile.c3_online_isaaclab_task.v2"
            if multi_object_mode
            else "nonprehensile.c3_online_isaaclab_task.v1"
        ),
        "executor_mode": (
            "measured_state_online_c3_task_trajectory_local_cartesian_impedance"
            if args_cli.local_controller == "cartesian_impedance"
            else (
                "measured_state_online_c3_task_trajectory_local_rmpflow"
                if args_cli.local_controller == "rmpflow_position"
                else "measured_state_online_c3_task_trajectory_local_joint_servo"
            )
        ),
        "local_controller": args_cli.local_controller,
        "physical_end_effector": (
            f"push_anything_spherical_pusher_{args_cli.pusher_collision_model}"
            if args_cli.use_push_anything_end_effector
            else "stock_franka_gripper_closed"
        ),
        "diagnostic_reference_replay": bool(os.environ.get("PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S")),
        "diagnostic_pd_contact_relinearization": os.environ.get("PUSH_ANYTHING_DIAGNOSTIC_PD_RELINEARIZE") == "1",
        "diagnostic_pd_inertia_calibration": os.environ.get("PUSH_ANYTHING_DIAGNOSTIC_PD_CALIBRATION") == "1",
        "diagnostic_compare_pd_references": os.environ.get("PUSH_ANYTHING_DIAGNOSTIC_PD_COMPARE") == "1",
        "diagnostic_reference_replay_start_s": os.environ.get("PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S") or None,
        "physical_end_effector_urdf": (
            str(physical_end_effector_urdf)
            if physical_end_effector_urdf is not None else None
        ),
        "push_anything_runtime_root": os.environ.get("PUSH_ANYTHING_ROOT"),
        "controller_parameters": {
            "osc_translation_stiffness": args_cli.osc_translation_stiffness,
            "osc_rotation_stiffness": args_cli.osc_rotation_stiffness,
            "osc_control_point": args_cli.osc_control_point,
            "osc_damping_ratio": args_cli.osc_damping_ratio,
            "osc_track_trajectory_velocity": args_cli.osc_track_trajectory_velocity,
            "osc_reference_velocity_mode": args_cli.osc_reference_velocity_mode,
            "osc_dynamics_audit": args_cli.osc_dynamics_audit,
            "osc_nullspace_stiffness": args_cli.osc_nullspace_stiffness,
            "osc_nullspace_damping_ratio": (
                args_cli.osc_nullspace_damping_ratio
            ),
            "osc_nullspace_target": args_cli.osc_nullspace_target,
            "osc_track_orientation": args_cli.osc_track_orientation,
            "rmpflow_proxy_padding_m": args_cli.rmpflow_proxy_padding_m,
            "rmpflow_safe_approach_switch_m": (
                args_cli.rmpflow_safe_approach_switch_m
            ),
            "rmpflow_max_joint_reference_error_rad": (
                args_cli.rmpflow_max_joint_reference_error_rad
            ),
            "execution_rotation_guard_limit_rad": (
                args_cli.execution_rotation_guard_limit_rad
            ),
            "execution_rotation_guard_lookahead_s": (
                args_cli.execution_rotation_guard_lookahead_s
            ),
            "execution_rotation_recovery_outward_rate_tolerance_rad_s": (
                args_cli.execution_rotation_recovery_outward_rate_tolerance_rad_s
            ),
            "measured_yaw_recovery_min_progress_rad": (
                args_cli.measured_yaw_recovery_min_progress_rad
            ),
            "measured_c3_yaw_brake": args_cli.measured_c3_yaw_brake,
            "execution_rotation_guard_retreat_m": (
                args_cli.execution_rotation_guard_retreat_m
            ),
            "osc_motion_axes": (
                "xyz_rpy_hold"
                if args_cli.osc_track_orientation else "xyz_only"
            ),
            "osc_control_frame": (
                "c3_task_point_consistent_pose_twist_jacobian_wrench"
                if args_cli.osc_control_point == "tip" else
                "panda_hand_analytic_attached_sphere_center"
                if args_cli.use_push_anything_end_effector
                else "panda_hand_analytic_closed_gripper_proxy_center"
            ),
            "osc_inertial_dynamics_decoupling": (
                args_cli.osc_inertial_dynamics_decoupling
            ),
            "max_task_target_speed_m_s": args_cli.max_task_target_speed_m_s,
            "minimum_task_target_z_m": args_cli.minimum_task_target_z_m,
            "task_height_floor_mode": args_cli.task_height_floor_mode,
            "task_height_clearance_m": args_cli.task_height_clearance_m,
            "semantic_c1_guard_mode": args_cli.semantic_c1_guard_mode,
            "native_pd_rollout_interpolation": native_pd_rollout_interpolation,
            "native_osc_matched_coarse_model": native_osc_matched_coarse_model,
            "native_relinearized_pd_cost": native_relinearized_pd_cost,
            "native_planning_horizon": native_planning_horizon,
            "native_planner_finger_table_clearance": native_planner_finger_table_clearance,
            "native_goal_mode": native_goal_mode,
            "native_goal_artifact": native_goal_artifact,
            "semantic_c1_guard_stop_distance_m": (
                semantic_trajectory_guard.stop_distance_m
                if semantic_trajectory_guard is not None else None
            ),
            "max_feedforward_force_n": args_cli.max_feedforward_force_n,
            "physical_contact_force_threshold_n": (
                args_cli.physical_contact_force_threshold_n
            ),
            "c3_force_action_sign": args_cli.c3_force_action_sign,
            "position_feedforward_compliance_m_per_n": (
                args_cli.position_feedforward_compliance_m_per_n
            ),
            "position_feedforward_max_offset_m": (
                args_cli.position_feedforward_max_offset_m
            ),
            "force_c3_on_legal_safe_contact": (
                args_cli.force_c3_on_legal_safe_contact
            ),
            "force_c3_contact_max_duration_s": (
                args_cli.force_c3_contact_max_duration_s
            ),
            "semantic_contact_push_speed_m_s": (
                args_cli.semantic_contact_push_speed_m_s
            ),
            "semantic_contact_activation_planar_threshold_m": (
                args_cli.semantic_contact_activation_planar_threshold_m
            ),
            "semantic_contact_axis_source": (
                args_cli.semantic_contact_axis_source
            ),
            "semantic_contact_yaw_moment_gain_m_per_rad": (
                args_cli.semantic_contact_yaw_moment_gain_m_per_rad
            ),
            "semantic_contact_yaw_rate_moment_gain_m_s_per_rad": (
                args_cli.semantic_contact_yaw_rate_moment_gain_m_s_per_rad
            ),
            "semantic_contact_max_axis_deviation_rad": (
                args_cli.semantic_contact_max_axis_deviation_rad
            ),
            "semantic_contact_planar_regression_tolerance_m": (
                args_cli.semantic_contact_planar_regression_tolerance_m
            ),
            "force_c3_yaw_regression_tolerance_rad": (
                args_cli.force_c3_yaw_regression_tolerance_rad
            ),
            "contact_acquisition_speed_m_s": (
                args_cli.contact_acquisition_speed_m_s
            ),
            "contact_acquisition_trigger_m": (
                args_cli.contact_acquisition_trigger_m
            ),
            "contact_acquisition_max_travel_m": (
                args_cli.contact_acquisition_max_travel_m
            ),
            "contact_acquisition_forbidden_margin_m": (
                args_cli.contact_acquisition_forbidden_margin_m
            ),
            "semantic_contact_max_duration_s": (
                args_cli.semantic_contact_max_duration_s
            ),
            "semantic_contact_compression_m": (
                args_cli.semantic_contact_compression_m
            ),
            "semantic_contact_max_additional_compression_m": (
                args_cli.semantic_contact_max_additional_compression_m
            ),
            "semantic_contact_yaw_gain_m_per_rad": (
                args_cli.semantic_contact_yaw_gain_m_per_rad
            ),
            "semantic_contact_max_lateral_offset_m": (
                args_cli.semantic_contact_max_lateral_offset_m
            ),
        },
        "task": args_cli.task,
        "manifest": str(manifest),
        "planner_contact_model": contact_model,
        "relay": f"{args_cli.relay_host}:{args_cli.relay_port}",
        "state_mode": "scene" if multi_object_mode else "single",
        "constraint_schedule": {
            "c1_robot_target": "hard_termination",
            "c2_clutter_protected_target": (
                "hard_termination"
                if args_cli.hard_c2_termination else "soft_penalty"
            ),
            "c3_robot_clutter": (
                "hard_termination"
                if args_cli.hard_c3_termination else "soft_penalty"
            ),
            "c2_soft_activation_distance_m": (
                args_cli.c2_soft_activation_distance_m
            ),
            "c3_soft_activation_distance_m": (
                args_cli.c3_soft_activation_distance_m
            ),
        },
        "franka_base_height_m": args_cli.table_height_offset_m,
        "strict_pose_thresholds": {
            "planar_m": args_cli.strict_position_threshold_m,
            "height_m": args_cli.strict_height_threshold_m,
            "rotation_rad": args_cli.strict_rotation_threshold_rad,
            "dwell_time_s": args_cli.dwell_time_s,
        },
        "c3_scene_spec": (
            str(args_cli.c3_scene_spec.expanduser().resolve())
            if multi_object_mode
            else None
        ),
        "deployment_ready": False,
        "deployment_blocker": "simulation_only_validation_pending_s2_acceptance",
    }
    try:
        env = gym.make(
            args_cli.task,
            cfg=_configure_env(),
            render_mode="rgb_array" if args_cli.video else None,
        )
        env.reset()
        base = env.unwrapped
        if contact_model and contact_model.get("asset_contract"):
            robot_source = Path(base.cfg.scene.robot.spawn.asset_path)
            if hashlib.sha256(robot_source.read_bytes()).hexdigest() != contact_model["asset_contract"]["robot_urdf_sha256"]:
                raise ValueError("Franka asset changed; re-export the shared PhysX contact model")
        if args_cli.video:
            markers = create_m1_video_markers(
                env, goal_ghost_opacity=args_cli.goal_ghost_opacity
            )
            env = M1MarkerUpdateWrapper(env, markers)
            video_writer = StreamingRgbVideoWriter(
                args_cli.video_folder / f"{args_cli.video_name_prefix}.mp4",
                fps=args_cli.video_fps,
            )
        robot = base.scene["robot"]
        target = base.scene["target"]
        obstacles = base.scene["obstacles"] if multi_object_mode else None
        robot_target_sensor_name = getattr(
            base.cfg, "robot_target_sensor_name", None
        )
        hand_target_sensor_name = "target_hand_contacts"
        required_contact_sensors = tuple(
            name for name in (
                robot_target_sensor_name,
                hand_target_sensor_name,
            ) if name
        )
        missing_contact_sensors = tuple(
            name for name in required_contact_sensors
            if base.scene.sensors.get(name) is None
        )
        if args_cli.require_contact_sensors and (
            not robot_target_sensor_name
            or missing_contact_sensors
        ):
            raise RuntimeError(
                "strict online C1 acceptance requires the independent PhysX "
                "whole-arm and hand target-contact reporters; missing="
                f"{missing_contact_sensors or ('robot_target_sensor_name',)}. "
                "Use a teacher task such as "
                "Isaac-AffordanceTeacher-C1-Franka-v0."
            )
        clutter_count = 0 if scene_objects is None else len(scene_objects) - 1
        if multi_object_mode:
            active_obstacle_count = int(
                getattr(
                    base,
                    "_clutter_active_obstacle_count",
                    getattr(base.cfg, "active_obstacle_count", obstacles.num_objects),
                )
            )
            if clutter_count != active_obstacle_count:
                raise ValueError(
                    "C3 scene spec clutter count does not match active Isaac obstacles: "
                    f"{clutter_count} != {active_obstacle_count}"
                )
            if clutter_count > obstacles.num_objects:
                raise ValueError("C3 scene spec references unavailable Isaac obstacles")

        def audited_contact_state() -> dict[str, torch.Tensor]:
            """Use the same filtered PhysX reporters as the teacher task."""

            return mdp.domino_affordance_contact_state(
                base,
                contact_distance_m=args_cli.contact_distance_m,
                protected_clearance_m=args_cli.protected_clearance_m,
                robot_obstacle_clearance_m=(
                    args_cli.robot_obstacle_clearance_m
                ),
                evaluate_protected=clutter_count > 0,
                evaluate_robot_obstacle=clutter_count > 0,
                require_physical_protected_contact=clutter_count > 0,
                physical_contact_force_threshold_n=(
                    args_cli.physical_contact_force_threshold_n
                ),
                pusher_tip_from_hand_m=(
                    args_cli.drake_tip_from_hand_m
                    if args_cli.use_push_anything_end_effector else None
                ),
                pusher_tip_radius_m=0.0195,
                robot_target_sensor_name=robot_target_sensor_name,
                hand_target_sensor_name=hand_target_sensor_name,
                robot_obstacle_sensor_name=(
                    getattr(base.cfg, "robot_obstacle_sensor_name", None)
                    if multi_object_mode else None
                ),
                target_obstacle_sensor_name=(
                    getattr(base.cfg, "target_obstacle_sensor_name", None)
                    if multi_object_mode else None
                ),
            )

        def filtered_contact_force_summary(
            sensor_names: str | tuple[str, ...] | None,
        ) -> dict[str, float]:
            """Return reporter and filter-pair forces for collision diagnosis."""

            if not sensor_names:
                return {}
            names = (sensor_names,) if isinstance(sensor_names, str) else sensor_names
            summary: dict[str, float] = {}
            for name in names:
                sensor = base.scene.sensors.get(name)
                if sensor is None or sensor.data.force_matrix_w is None:
                    continue
                force_norm = torch.linalg.vector_norm(
                    sensor.data.force_matrix_w[0], dim=-1
                )
                summary[name] = float(torch.max(force_norm))
                filter_paths = tuple(
                    getattr(sensor.cfg, "filter_prim_paths_expr", ())
                )
                if force_norm.ndim == 2 and force_norm.shape[1] == len(filter_paths):
                    for filter_index, filter_path in enumerate(filter_paths):
                        filter_name = str(filter_path).rsplit("/", maxsplit=1)[-1]
                        summary[f"{name}:{filter_name}"] = float(
                            torch.max(force_norm[:, filter_index])
                        )
            return summary

        robot_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        robot_cfg.resolve(base.scene)
        joint_ids = robot_cfg.joint_ids
        hand_id = robot_cfg.body_ids[0]
        finger_joint_ids, _ = robot.find_joints("panda_finger_joint.*")
        contact_body_ids, contact_body_names = robot.find_bodies(
            ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
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

        def tcp_position_tensor() -> torch.Tensor:
            return (
                base.scene["ee_frame"].data.target_pos_w[0, 0]
                - base.scene.env_origins[0]
            )

        def tcp_position_list() -> list[float]:
            return tcp_position_tensor().detach().cpu().tolist()

        control_period_s = float(base.step_dt)
        video_stride = max(
            1, round(1.0 / (control_period_s * args_cli.video_fps))
        )
        planner_stride = max(
            1, round(1.0 / (control_period_s * args_cli.planner_frequency_hz))
        )
        planner_period_s = planner_stride * control_period_s
        max_steps = max(1, math.ceil(args_cli.max_sim_time_s / control_period_s))
        dwell_steps = max(1, math.ceil(args_cli.dwell_time_s / control_period_s))
        startup_steps = max(1, math.ceil(args_cli.startup_timeout_s / control_period_s))
        support_values = (
            [item["support_quaternion_wxyz"] for item in scene_objects]
            if multi_object_mode
            else [args_cli.support_quaternion_wxyz]
        )
        support_quaternions = torch.tensor(
            support_values,
            device=base.device,
            dtype=target.data.root_quat_w.dtype,
        )
        support_quaternions = torch.nn.functional.normalize(
            support_quaternions, dim=-1
        )

        initial_target_pose = torch.cat(
            (target.data.root_pos_w[:, :3] - base.scene.env_origins,
             target.data.root_quat_w), dim=1
        )[0].detach().cpu().tolist()
        # The DOMINO link frame is not generally located at the rigid body's
        # center of mass.  Planar contact torque must therefore be computed
        # about the PhysX COM rather than about ``root_pos_w``.  Record both
        # the body-frame offset and world COM so planner/model alignment can be
        # audited without privileged contact-force information.  The same
        # offset can be estimated from an RGB-D mesh on the real robot.
        initial_target_com_pose_b = (
            target.data.body_com_pose_b[0, 0].detach().cpu().tolist()
        )
        initial_target_com_position = (
            target.data.root_com_pos_w[0] - base.scene.env_origins[0]
        ).detach().cpu().tolist()
        initial_target_mass = float(target.data.default_mass[0, 0])
        initial_target_inertia = (
            target.data.default_inertia[0].detach().cpu().reshape(3, 3).tolist()
        )
        if contact_model and contact_model.get("source_target_dynamics"):
            expected = contact_model["source_target_dynamics"]
            mass_error = abs(initial_target_mass - expected["mass_kg"])
            com_error = float(np.linalg.norm(np.asarray(initial_target_com_pose_b[:3])
                                            - expected["com_position_body_m"]))
            inertia_error = float(np.max(np.abs(np.asarray(initial_target_inertia)
                                - expected["inertia_body_about_com_kg_m2"])))
            result["planner_dynamics_validation"] = {
                "mass_error_kg": mass_error, "com_error_m": com_error,
                "maximum_inertia_entry_error_kg_m2": inertia_error,
                "passed": mass_error < 1e-7 and com_error < 1e-6 and inertia_error < 1e-9,
            }
            if not result["planner_dynamics_validation"]["passed"]:
                raise ValueError("Isaac target dynamics changed; re-export the shared contact model")
        goal_pose = base.command_manager.get_command(
            "target_object_pose"
        )[0].detach().cpu().tolist()
        initial_tcp_position = tcp_position_list()
        ik = FrankaTcpIK(
            FRANKA_URDF,
            hand_to_tcp_m=args_cli.drake_tip_from_hand_m,
            max_evaluations=args_cli.ik_max_evaluations,
        )
        soft_limits = robot.data.soft_joint_pos_limits[
            0, joint_ids
        ].detach().cpu().numpy()
        ik.lower = np.maximum(ik.lower, soft_limits[:, 0] + 1.0e-4)
        ik.upper = np.minimum(ik.upper, soft_limits[:, 1] - 1.0e-4)
        desired_q = initial_q.clone()

        def model_planner_tip_position_list() -> list[float]:
            current_q = robot.data.joint_pos[
                0, joint_ids
            ].detach().cpu().numpy()
            return ik.forward(current_q).translation.tolist()

        def planner_tip_position_tensor() -> torch.Tensor:
            offset_hand = torch.tensor(
                (0.0, 0.0, args_cli.drake_tip_from_hand_m),
                device=base.device,
                dtype=robot.data.body_pos_w.dtype,
            ).unsqueeze(0)
            tip_position_w = (
                robot.data.body_pos_w[0:1, hand_id]
                + quat_apply(robot.data.body_quat_w[0:1, hand_id], offset_hand)
            )
            return quat_apply_inverse(
                robot.data.root_quat_w[0:1],
                tip_position_w - robot.data.root_pos_w[0:1],
            )[0]

        def planner_tip_position_list() -> list[float]:
            return planner_tip_position_tensor().detach().cpu().tolist()

        initial_planner_tip_position = planner_tip_position_list()
        initial_model_planner_tip_position = model_planner_tip_position_list()
        initial_cross_model_tip_error = float(np.linalg.norm(
            np.asarray(initial_planner_tip_position)
            - np.asarray(initial_model_planner_tip_position)
        ))
        root_quaternion_w = robot.data.root_quat_w[0:1]
        hand_quaternion_w = robot.data.body_quat_w[0:1, hand_id]
        initial_tip_quaternion_b = quat_mul(
            quat_inv(root_quaternion_w), hand_quaternion_w
        )[0]
        initial_tip_quaternion_b = torch.nn.functional.normalize(
            initial_tip_quaternion_b, dim=0
        )
        last_desired_position = np.asarray(
            initial_planner_tip_position, dtype=float
        )
        governed_task_target = last_desired_position.copy()
        applied_feedforward_force = np.zeros(3, dtype=float)
        rmpflow = None
        rmpflow_proxies: list[dict[str, object]] = []
        rmpflow_joint_target = None
        rmpflow_full_target_enabled = None
        rmpflow_target_proxy_switch_step = None
        osc_action_term = None

        if (
            args_cli.osc_nullspace_target == "rmpflow_semantic"
            or args_cli.local_controller == "rmpflow_position"
        ):
            # C3 models the robot as one spherical task point.  Build a
            # complementary, robot-specific RMPflow world model so its joint
            # posture reference accounts for every Franka collision sphere.
            semantic_state = audited_contact_state()

            def local_box(
                points_e: torch.Tensor,
                root_position_e: torch.Tensor,
                root_quaternion: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                quaternion = root_quaternion.unsqueeze(0).expand(
                    points_e.shape[0], -1
                )
                local_points = quat_apply_inverse(
                    quaternion, points_e - root_position_e.unsqueeze(0)
                )
                lower = torch.amin(local_points, dim=0)
                upper = torch.amax(local_points, dim=0)
                center = 0.5 * (lower + upper)
                extent = torch.clamp(
                    upper - lower + 2.0 * args_cli.rmpflow_proxy_padding_m,
                    min=0.010,
                )
                return center, extent

            origin_w = base.scene.env_origins[0]
            target_points_e = semantic_state["target_points"][0]
            target_position_e = target.data.root_pos_w[0] - origin_w
            target_center_l, target_extent = local_box(
                target_points_e,
                target_position_e,
                target.data.root_quat_w[0],
            )
            target_proxy = _RmpFlowVisualCuboid(
                prim_path="/World/C3RmpFlowProxies/Target",
                name="c3_rmpflow_target_proxy",
                position=np.zeros(3),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                scale=target_extent.detach().cpu().numpy(),
                size=1.0,
                visible=False,
            )
            rmpflow_proxies.append({
                "proxy": target_proxy,
                "role": "target_full",
                "index": 0,
                "center_local": target_center_l,
                "extent_m": target_extent.detach().cpu().tolist(),
            })

            protected_mask = semantic_state["protected_mask"][0]
            protected_points_e = target_points_e[protected_mask]
            if protected_points_e.shape[0] < 4:
                raise RuntimeError(
                    "semantic RMPflow requires at least four protected points"
                )
            protected_center_l, protected_extent = local_box(
                protected_points_e,
                target_position_e,
                target.data.root_quat_w[0],
            )
            protected_proxy = _RmpFlowVisualCuboid(
                prim_path="/World/C3RmpFlowProxies/TargetProtected",
                name="c3_rmpflow_target_protected_proxy",
                position=np.zeros(3),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                scale=protected_extent.detach().cpu().numpy(),
                size=1.0,
                visible=False,
            )
            rmpflow_proxies.append({
                "proxy": protected_proxy,
                "role": "target_protected",
                "index": 0,
                "center_local": protected_center_l,
                "extent_m": protected_extent.detach().cpu().tolist(),
            })

            if clutter_count:
                obstacle_points_e = semantic_state["obstacle_points"][0].reshape(
                    clutter_count, -1, 3
                )
                for obstacle_index in range(clutter_count):
                    obstacle_position_e = (
                        obstacles.data.object_pos_w[0, obstacle_index] - origin_w
                    )
                    center_l, extent = local_box(
                        obstacle_points_e[obstacle_index],
                        obstacle_position_e,
                        obstacles.data.object_quat_w[0, obstacle_index],
                    )
                    proxy = _RmpFlowVisualCuboid(
                        prim_path=(
                            "/World/C3RmpFlowProxies/Clutter_"
                            f"{obstacle_index:02d}"
                        ),
                        name=f"c3_rmpflow_clutter_proxy_{obstacle_index:02d}",
                        position=np.zeros(3),
                        orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                        scale=extent.detach().cpu().numpy(),
                        size=1.0,
                        visible=False,
                    )
                    rmpflow_proxies.append({
                        "proxy": proxy,
                        "role": "clutter",
                        "index": obstacle_index,
                        "center_local": center_l,
                        "extent_m": extent.detach().cpu().tolist(),
                    })

            rmpflow_config = load_supported_motion_policy_config(
                "Franka", "RMPflow"
            )
            rmpflow = RmpFlow(**rmpflow_config)
            # Roll the Lula state forward to provide an actual posture
            # reference, rather than a one-servo-tick target that is nearly
            # identical to the measured joints.  The reference is bounded
            # below before it reaches either local controller.
            rmpflow.set_ignore_state_updates(True)
            rmpflow.set_robot_base_pose(
                robot.data.root_pos_w[0].detach().cpu().numpy(),
                robot.data.root_quat_w[0].detach().cpu().numpy(),
            )
            for item in rmpflow_proxies:
                if not rmpflow.add_obstacle(item["proxy"], static=False):
                    raise RuntimeError(
                        f"RMPflow rejected {item['role']} collision proxy"
                    )
            rmpflow_full_target_enabled = True
            rmpflow_target_proxy_switch_step = None
            if not rmpflow.disable_obstacle(protected_proxy):
                raise RuntimeError(
                    "RMPflow could not disable the protected approach proxy"
                )
            if args_cli.local_controller == "cartesian_impedance":
                osc_action_term = base.action_manager.get_term("arm_action")
                if getattr(
                    osc_action_term, "_nullspace_joint_pos_target", None
                ) is None:
                    raise RuntimeError(
                        "OSC action term has no null-space target buffer"
                    )

            def update_rmpflow_reference(
                desired_tip_position_b: np.ndarray,
            ) -> None:
                nonlocal rmpflow_joint_target
                nonlocal rmpflow_full_target_enabled
                nonlocal rmpflow_target_proxy_switch_step
                if (
                    rmpflow_full_target_enabled
                    and current_minimum_safe_distance
                    <= args_cli.rmpflow_safe_approach_switch_m
                ):
                    if not rmpflow.disable_obstacle(target_proxy):
                        raise RuntimeError(
                            "RMPflow could not disable the full target proxy"
                        )
                    if not rmpflow.enable_obstacle(protected_proxy):
                        raise RuntimeError(
                            "RMPflow could not enable the protected target proxy"
                        )
                    rmpflow_full_target_enabled = False
                    rmpflow_target_proxy_switch_step = sequence
                for item in rmpflow_proxies:
                    if str(item["role"]).startswith("target_"):
                        position_w = target.data.root_pos_w[0]
                        quaternion_w = target.data.root_quat_w[0]
                    else:
                        obstacle_index = int(item["index"])
                        position_w = obstacles.data.object_pos_w[
                            0, obstacle_index
                        ]
                        quaternion_w = obstacles.data.object_quat_w[
                            0, obstacle_index
                        ]
                    center_w = position_w + quat_apply(
                        quaternion_w.unsqueeze(0),
                        item["center_local"].unsqueeze(0),
                    )[0]
                    item["proxy"].set_world_pose(
                        position=center_w.detach().cpu().numpy(),
                        orientation=quaternion_w.detach().cpu().numpy(),
                    )
                rmpflow.update_world()

                desired_tip_b = torch.tensor(
                    desired_tip_position_b,
                    device=base.device,
                    dtype=robot.data.root_pos_w.dtype,
                ).unsqueeze(0)
                desired_tip_w = robot.data.root_pos_w[0:1] + quat_apply(
                    robot.data.root_quat_w[0:1], desired_tip_b
                )
                # Isaac 5.0's supported Franka RMPflow configuration controls
                # ``right_gripper``, 100 mm above panda_hand.  Push Anything's
                # spherical tip is 126.5 mm above panda_hand.
                lula_to_c3_tip = torch.tensor(
                    (0.0, 0.0, args_cli.drake_tip_from_hand_m - 0.100),
                    device=base.device,
                    dtype=robot.data.body_pos_w.dtype,
                ).unsqueeze(0)
                desired_lula_ee_w = desired_tip_w - quat_apply(
                    robot.data.body_quat_w[0:1, hand_id], lula_to_c3_tip
                )
                rmpflow.set_end_effector_target(
                    target_position=(
                        desired_lula_ee_w[0].detach().cpu().numpy()
                    ),
                    target_orientation=None,
                )
                q_target, _ = rmpflow.compute_joint_targets(
                    robot.data.joint_pos[0, joint_ids].detach().cpu().numpy(),
                    robot.data.joint_vel[0, joint_ids].detach().cpu().numpy(),
                    np.empty((0,), dtype=float),
                    np.empty((0,), dtype=float),
                    control_period_s,
                )
                measured_q = robot.data.joint_pos[
                    0, joint_ids
                ].detach().cpu().numpy()
                maximum_reference_error = (
                    args_cli.rmpflow_max_joint_reference_error_rad
                )
                q_target = np.clip(
                    q_target,
                    measured_q - maximum_reference_error,
                    measured_q + maximum_reference_error,
                )
                q_target = np.clip(q_target, soft_limits[:, 0], soft_limits[:, 1])
                if not np.all(np.isfinite(q_target)):
                    raise RuntimeError("RMPflow produced a non-finite joint target")
                rmpflow_joint_target = torch.tensor(
                    q_target, device=base.device, dtype=initial_q.dtype
                ).unsqueeze(0)
                if osc_action_term is not None:
                    osc_action_term._nullspace_joint_pos_target.copy_(
                        rmpflow_joint_target
                    )

        if (
            args_cli.local_controller == "cartesian_impedance"
            and args_cli.osc_nullspace_target == "push_anything_joint2"
        ):
            osc_action_term = base.action_manager.get_term("arm_action")
            if getattr(
                osc_action_term, "_nullspace_joint_pos_target", None
            ) is None:
                raise RuntimeError("OSC action term has no null-space target buffer")

        def osc_action(
            desired_position: np.ndarray, force_n: np.ndarray
        ) -> torch.Tensor:
            # Hand mode retains the historical analytic position conversion.
            # Tip mode supplies the C3 reference directly to the action term,
            # which shifts pose, Jacobian, twist and wrench consistently.
            current_hand_quaternion_b = quat_mul(
                quat_inv(robot.data.root_quat_w[0:1]),
                robot.data.body_quat_w[0:1, hand_id],
            )[0]
            current_hand_quaternion_b = torch.nn.functional.normalize(
                current_hand_quaternion_b, dim=0
            )
            tip_offset_hand = torch.tensor(
                (0.0, 0.0, args_cli.drake_tip_from_hand_m),
                device=base.device,
                dtype=initial_q.dtype,
            )
            tip_offset_b = quat_apply(
                current_hand_quaternion_b.unsqueeze(0),
                tip_offset_hand.unsqueeze(0),
            )[0]
            desired_hand_position = torch.as_tensor(
                desired_position,
                device=base.device,
                dtype=initial_q.dtype,
            ) - tip_offset_b
            if args_cli.osc_control_point == "tip":
                desired_hand_position = torch.as_tensor(
                    desired_position, device=base.device, dtype=initial_q.dtype)
            pose = torch.cat(
                (
                    desired_hand_position,
                    (
                        initial_tip_quaternion_b
                        if args_cli.osc_track_orientation
                        else current_hand_quaternion_b
                    ),
                )
            )
            wrench = torch.cat(
                (
                    torch.as_tensor(
                        force_n, device=base.device, dtype=initial_q.dtype
                    ),
                    torch.zeros(3, device=base.device, dtype=initial_q.dtype),
                )
            )
            return torch.cat((pose, wrench)).unsqueeze(0)

        def planner_rigid_body_state(
            root_position_w: torch.Tensor,
            root_quaternion_w: torch.Tensor,
            angular_velocity_w: torch.Tensor,
            linear_velocity_w: torch.Tensor,
            support_quaternion: torch.Tensor,
        ) -> C3RigidBodyState:
            position_c3 = root_position_w - base.scene.env_origins[0]
            position_c3 = position_c3.clone()
            position_c3[2] -= args_cli.table_height_offset_m
            quaternion_c3 = quat_mul(
                root_quaternion_w.unsqueeze(0),
                quat_inv(support_quaternion.unsqueeze(0)),
            )[0]
            quaternion_c3 = torch.nn.functional.normalize(quaternion_c3, dim=0)
            return C3RigidBodyState(
                quaternion_wxyz=tuple(float(value) for value in quaternion_c3),
                position_m=tuple(float(value) for value in position_c3),
                angular_velocity_rad_s=tuple(
                    float(value) for value in angular_velocity_w
                ),
                linear_velocity_m_s=tuple(float(value) for value in linear_velocity_w),
            )

        counters = {
            "planner_cycles": 0,
            "ready": 0,
            "fresh": 0,
            "stale": 0,
            "watchdog": 0,
        }
        first_ready_step = None
        consecutive_watchdog = 0
        maximum_consecutive_watchdog = 0
        safe_contact_ever = False
        forbidden_contact_ever = False
        protected_obstacle_collision_ever = False
        robot_obstacle_collision_ever = False
        c2_soft_penalty_sum = 0.0
        c3_soft_penalty_sum = 0.0
        maximum_c2_soft_penalty = 0.0
        maximum_c3_soft_penalty = 0.0
        c2_collision_audit_count = 0
        c3_collision_audit_count = 0
        safety_audit_count = 0
        first_safe_contact_step = None
        first_forbidden_contact_step = None
        minimum_safe_distance = math.inf
        minimum_forbidden_distance = math.inf
        minimum_protected_obstacle_clearance = math.inf
        minimum_robot_obstacle_clearance = math.inf
        minimum_hand_obstacle_clearance = math.inf
        minimum_arm_obstacle_clearance = math.inf
        minimum_arm_obstacle_centerline_distance = math.inf
        minimum_planar = math.inf
        minimum_height = math.inf
        minimum_rotation = math.inf
        strict_dwell = 0
        maximum_strict_dwell = 0
        maximum_ik_position_error = 0.0
        maximum_task_tracking_error = 0.0
        current_task_tracking_error = 0.0
        maximum_feedforward_force = 0.0
        maximum_applied_feedforward_force = 0.0
        maximum_position_feedforward_offset = 0.0
        maximum_abs_joint_effort = 0.0
        task_target_governor_activation_count = 0
        task_target_z_clamp_count = 0
        maximum_raw_task_target_step_m = 0.0
        maximum_task_target_governor_correction_m = 0.0
        position_feedforward_offset = np.zeros(3, dtype=float)
        current_semantic_safe_proximity = False
        current_safe_contact = False
        current_legal_safe_contact = False
        contact_audit_measurement_utime_us = None
        current_forbidden_contact = False
        current_protected_obstacle_collision = False
        current_robot_obstacle_collision = False
        current_c2_soft_penalty = 0.0
        current_c3_soft_penalty = 0.0
        current_minimum_safe_distance = math.inf
        current_minimum_forbidden_distance = math.inf
        current_protected_obstacle_clearance = math.inf
        current_robot_obstacle_clearance = math.inf
        current_hand_obstacle_clearance = math.inf
        current_arm_obstacle_clearance = math.inf
        current_arm_obstacle_centerline_distance = math.inf
        current_robot_target_contact_forces: dict[str, float] = {}
        peak_robot_target_contact_forces: dict[str, float] = {}
        current_robot_obstacle_contact_forces: dict[str, float] = {}
        peak_robot_obstacle_contact_forces: dict[str, float] = {}
        current_closest_safe_point_xy: np.ndarray | None = None
        contact_acquisition_active = False
        contact_acquisition_armed = True
        contact_acquisition_start_tip: np.ndarray | None = None
        contact_acquisition_start_signed_yaw_error: float | None = None
        contact_acquisition_start_is_yaw_recovery: bool | None = None
        contact_acquisition_start_action_axis_xy: np.ndarray | None = None
        contact_acquisition_axis_xy: np.ndarray | None = None
        contact_acquisition_target: np.ndarray | None = None
        contact_acquisition_travel_m = 0.0
        contact_acquisition_activation_count = 0
        contact_acquisition_deactivation_count = 0
        first_contact_acquisition_step: int | None = None
        maximum_contact_acquisition_travel_m = 0.0
        semantic_contact_push_active = False
        semantic_contact_push_armed = True
        semantic_contact_push_target: np.ndarray | None = None
        semantic_contact_push_axis_xy: np.ndarray | None = None
        semantic_contact_requested_moment_arm_m: float | None = None
        semantic_contact_achieved_moment_arm_m: float | None = None
        semantic_contact_relative_offset_b_m: np.ndarray | None = None
        semantic_contact_start_signed_yaw_error: float | None = None
        semantic_contact_is_yaw_recovery_pulse = False
        semantic_contact_from_committed_acquisition = False
        semantic_contact_best_planar_error_m: float | None = None
        semantic_contact_compression_offset_m = 0.0
        semantic_contact_push_start_step: int | None = None
        semantic_contact_approach_standoff_m: float | None = None
        semantic_contact_standoff_m: float | None = None
        semantic_contact_minimum_standoff_m: float | None = None
        semantic_contact_axial_target_m: float | None = None
        semantic_contact_push_activation_step = None
        semantic_contact_standoff_latch_step = None
        semantic_contact_push_deactivation_count = 0
        measured_yaw_recovery_inference_count = 0
        semantic_yaw_brake_active = False
        semantic_yaw_brake_target: np.ndarray | None = None
        semantic_yaw_brake_activation_count = 0
        first_semantic_yaw_brake_step: int | None = None
        maximum_semantic_contact_lateral_offset_m = 0.0
        force_c3_contact_active = False
        force_c3_contact_armed = True
        force_c3_contact_start_step: int | None = None
        force_c3_contact_best_planar_error_m: float | None = None
        force_c3_contact_best_abs_yaw_error_rad: float | None = None
        force_c3_contact_start_signed_yaw_error: float | None = None
        force_c3_contact_is_yaw_recovery_pulse = False
        force_c3_contact_activation_count = 0
        force_c3_contact_deactivation_count = 0
        measured_c3_yaw_brake_activation_count = 0
        force_c3_yaw_regression_brake_activation_count = 0
        first_measured_c3_yaw_brake_step: int | None = None
        first_force_c3_yaw_regression_brake_step: int | None = None
        force_c3_window_retreat_active = False
        force_c3_window_retreat_activation_count = 0
        first_force_c3_window_retreat_step: int | None = None
        last_audit_step = None
        stopped_reason = "maximum_sim_time"
        executed_steps = 0
        trace: list[dict[str, object]] = []
        last_command_position_isaac = None
        active_command = None
        active_command_step = None
        active_command_is_c3_mode = False
        active_command_is_yaw_recovery = False
        execution_rotation_guard_active = False
        execution_rotation_guard_retreat_target: np.ndarray | None = None
        execution_rotation_guard_activation_count = 0
        first_execution_rotation_guard_step = None
        maximum_execution_rotation_risk = 0.0
        loop_started = time.monotonic()

        with socket.create_connection(
            (args_cli.relay_host, args_cli.relay_port),
            timeout=args_cli.socket_timeout_s,
        ) as client:
            client.settimeout(args_cli.socket_timeout_s)
            for sequence in range(max_steps):
                executed_steps = sequence + 1
                q = robot.data.joint_pos[0, joint_ids]
                dq = robot.data.joint_vel[0, joint_ids]
                measured_effort = robot.data.applied_torque[0, joint_ids]
                planning_cycle = sequence % planner_stride == 0
                if planning_cycle:
                    counters["planner_cycles"] += 1
                    utime_us = 100_000 + round(
                        sequence * control_period_s * 1.0e6
                    )
                    state_flags = C3StateFlags(0)
                    if current_legal_safe_contact:
                        state_flags |= C3StateFlags.LEGAL_SAFE_CONTACT
                    if force_c3_contact_active:
                        state_flags |= C3StateFlags.FORCE_C3_MODE
                    target_state = planner_rigid_body_state(
                        target.data.root_pos_w[0, :3],
                        target.data.root_quat_w[0],
                        target.data.root_link_vel_w[0, 3:6],
                        target.data.root_link_vel_w[0, 0:3],
                        support_quaternions[0],
                    )
                    state_kwargs = {
                        "sequence": sequence,
                        "utime_us": utime_us,
                        "joint_position_rad": tuple(float(value) for value in q),
                        "joint_velocity_rad_s": tuple(float(value) for value in dq),
                        "joint_effort_nm": tuple(
                            float(value) for value in measured_effort
                        ),
                        "flags": state_flags,
                    }
                    if multi_object_mode:
                        clutter_states = tuple(
                            planner_rigid_body_state(
                                obstacles.data.object_pos_w[0, obstacle_index],
                                obstacles.data.object_quat_w[0, obstacle_index],
                                obstacles.data.body_link_vel_w[
                                    0, obstacle_index, 3:6
                                ],
                                obstacles.data.body_link_vel_w[
                                    0, obstacle_index, 0:3
                                ],
                                support_quaternions[obstacle_index + 1],
                            )
                            for obstacle_index in range(clutter_count)
                        )
                        state = C3MeasuredSceneState(
                            **state_kwargs,
                            target=target_state,
                            clutter=clutter_states,
                        )
                    else:
                        state = C3MeasuredState(
                            **state_kwargs,
                            object_quaternion_wxyz=target_state.quaternion_wxyz,
                            object_position_m=target_state.position_m,
                            object_angular_velocity_rad_s=(
                                target_state.angular_velocity_rad_s
                            ),
                            object_linear_velocity_m_s=target_state.linear_velocity_m_s,
                        )
                    client.sendall(state.pack())
                    command = C3TaskCommand.unpack(
                        _receive_exact(client, TASK_COMMAND_PACKET_SIZE)
                    )
                    if command.sequence != sequence:
                        raise RuntimeError(
                            f"relay sequence mismatch: {command.sequence} != {sequence}"
                        )
                    ready = bool(command.flags & C3CommandFlags.READY)
                    stale = bool(command.flags & C3CommandFlags.STALE_STATE)
                    semantic_hold = bool(
                        command.flags & C3CommandFlags.SEMANTIC_HOLD
                    )
                    command_is_c3_mode = bool(
                        command.flags & C3CommandFlags.C3_MODE
                    )
                    command_is_yaw_recovery = bool(
                        command.flags & C3CommandFlags.YAW_RECOVERY
                    )
                    if command_is_yaw_recovery and not command_is_c3_mode:
                        raise RuntimeError(
                            "relay returned YAW_RECOVERY outside C3_MODE"
                        )
                    counters["ready"] += int(ready)
                    counters["fresh"] += int(ready and not stale)
                    counters["stale"] += int(stale)
                    if ready and not stale and first_ready_step is None:
                        first_ready_step = sequence
                    watchdog = (
                        not ready
                        or stale
                        or bool(command.flags & C3CommandFlags.PLANNER_FAILURE)
                    )
                    counters["watchdog"] += int(watchdog)
                    consecutive_watchdog = (
                        consecutive_watchdog + 1 if watchdog else 0
                    )
                    maximum_consecutive_watchdog = max(
                        maximum_consecutive_watchdog, consecutive_watchdog
                    )
                    if ready and not stale and not semantic_hold:
                        active_command = command
                        active_command_step = sequence
                        active_command_is_c3_mode = command_is_c3_mode
                        active_command_is_yaw_recovery = (
                            command_is_yaw_recovery
                        )
                        maximum_feedforward_force = max(
                            maximum_feedforward_force,
                            float(np.linalg.norm(command.feedforward_force_n)),
                        )
                    else:
                        # Fail closed rather than extrapolating an old segment.
                        active_command = None
                        active_command_step = None
                        active_command_is_c3_mode = False
                        active_command_is_yaw_recovery = False

                goal_command = base.command_manager.get_command(
                    "target_object_pose"
                )[0]
                signed_yaw_error = _signed_planar_yaw_error(
                    target.data.root_quat_w[0], goal_command[3:7]
                )
                target_yaw_rate = float(target.data.root_ang_vel_w[0, 2])
                measured_c3_yaw_projected_error = signed_yaw_error
                measured_c3_yaw_brake_requested_now = False
                force_c3_yaw_regression_brake_requested_now = False
                force_c3_contact_window_expired = bool(
                    force_c3_contact_active
                    and force_c3_contact_start_step is not None
                    and (sequence - force_c3_contact_start_step)
                    * control_period_s
                    >= args_cli.force_c3_contact_max_duration_s
                )
                if abs(signed_yaw_error) > 1.0e-6:
                    outward_yaw_rate = max(
                        0.0,
                        math.copysign(1.0, signed_yaw_error) * target_yaw_rate,
                    )
                else:
                    outward_yaw_rate = abs(target_yaw_rate)
                execution_rotation_risk = (
                    abs(signed_yaw_error)
                    + args_cli.execution_rotation_guard_lookahead_s
                    * outward_yaw_rate
                )
                # Outside the envelope, only an explicitly planner-labelled
                # recovery trajectory may return to contact.  Measured outward
                # yaw still revokes that authorization at servo rate.
                rotation_guard_violation = (
                    execution_rotation_guard_requires_retreat(
                        signed_error_rad=signed_yaw_error,
                        yaw_rate_rad_s=target_yaw_rate,
                        limit_rad=args_cli.execution_rotation_guard_limit_rad,
                        lookahead_s=(
                            args_cli.execution_rotation_guard_lookahead_s
                        ),
                        recovery_outward_rate_tolerance_rad_s=(
                            args_cli
                            .execution_rotation_recovery_outward_rate_tolerance_rad_s
                        ),
                        planner_marks_yaw_recovery=(
                            active_command_is_yaw_recovery
                        ),
                    )
                )
                maximum_execution_rotation_risk = max(
                    maximum_execution_rotation_risk, execution_rotation_risk
                )
                close_to_safe_surface = (
                    current_legal_safe_contact
                    or current_minimum_safe_distance
                    <= args_cli.contact_distance_m
                )
                rotation_guard_retreat_reached = bool(
                    execution_rotation_guard_active
                    and execution_rotation_guard_retreat_target is not None
                    and np.linalg.norm(
                        np.asarray(planner_tip_position_list(), dtype=float)
                        - execution_rotation_guard_retreat_target
                    )
                    <= args_cli.semantic_yaw_brake_retreat_tolerance_m
                )
                if (
                    rotation_guard_retreat_reached
                    and not current_legal_safe_contact
                    and current_minimum_safe_distance
                    > args_cli.contact_distance_m
                ):
                    execution_rotation_guard_active = False
                    execution_rotation_guard_retreat_target = None
                    # The command received at the start of this servo step was
                    # planned before retreat completion.  Drop it so the next
                    # contact action is based on the cleared measured state.
                    active_command = None
                    active_command_step = None
                    active_command_is_c3_mode = False
                    active_command_is_yaw_recovery = False
                    applied_feedforward_force = np.zeros(3, dtype=float)
                if (
                    not execution_rotation_guard_active
                    and close_to_safe_surface
                    and rotation_guard_violation
                ):
                    execution_rotation_guard_active = True
                    execution_rotation_guard_activation_count += 1
                    if first_execution_rotation_guard_step is None:
                        first_execution_rotation_guard_step = sequence
                    tip_position = np.asarray(
                        planner_tip_position_list(), dtype=float
                    )
                    target_com_position = (
                        target.data.root_com_pos_w[0, :3]
                        - base.scene.env_origins[0]
                    ).detach().cpu().numpy()
                    retreat_reference_xy = target_com_position[:2]
                    retreat_direction = (
                        tip_position[:2] - retreat_reference_xy
                    )
                    retreat_norm = float(np.linalg.norm(retreat_direction))
                    if retreat_norm <= 1.0e-9:
                        retreat_direction = np.asarray((1.0, 0.0))
                    else:
                        retreat_direction /= retreat_norm
                    execution_rotation_guard_retreat_target = tip_position.copy()
                    execution_rotation_guard_retreat_target[:2] += (
                        args_cli.execution_rotation_guard_retreat_m
                        * retreat_direction
                    )
                if execution_rotation_guard_active:
                    # Drop the current C3 segment immediately and retreat from
                    # contact.  The next measured-state replan may still choose
                    # a corrective contact once the EE has cleared the object.
                    active_command = None
                    active_command_step = None
                    active_command_is_c3_mode = False
                    active_command_is_yaw_recovery = False
                    applied_feedforward_force = np.zeros(3, dtype=float)
                    assert execution_rotation_guard_retreat_target is not None
                    last_desired_position = (
                        execution_rotation_guard_retreat_target.copy()
                    )
                    last_command_position_isaac = last_desired_position.copy()

                # A native C3 contact can be dynamically correct yet remain
                # active long enough to rotate through the requested yaw.  Use
                # measured pose progress to identify such a corrective pulse,
                # then release before it crosses zero.  This is the same
                # observable feedback available from a real RGB-D tracker and
                # does not alter C3's contact optimizer or add a waypoint.
                if (
                    args_cli.measured_c3_yaw_brake
                    and force_c3_contact_active
                    and force_c3_contact_start_signed_yaw_error is not None
                    and force_c3_contact_best_abs_yaw_error_rad is not None
                    and not execution_rotation_guard_active
                ):
                    force_c3_contact_best_abs_yaw_error_rad = min(
                        force_c3_contact_best_abs_yaw_error_rad,
                        abs(signed_yaw_error),
                    )
                    force_c3_yaw_regression_brake_requested_now = (
                        measured_yaw_regression_brake_requested(
                            current_signed_error_rad=signed_yaw_error,
                            best_abs_error_rad=(
                                force_c3_contact_best_abs_yaw_error_rad
                            ),
                            regression_tolerance_rad=(
                                args_cli.force_c3_yaw_regression_tolerance_rad
                            ),
                        )
                    )
                    if (
                        not force_c3_contact_is_yaw_recovery_pulse
                        and measured_contact_is_yaw_recovery(
                            start_signed_error_rad=(
                                force_c3_contact_start_signed_yaw_error
                            ),
                            current_signed_error_rad=signed_yaw_error,
                            current_yaw_rate_rad_s=target_yaw_rate,
                            activation_threshold_rad=max(
                                args_cli.measured_yaw_recovery_min_progress_rad,
                                1.0e-6,
                            ),
                            minimum_progress_rad=(
                                args_cli.measured_yaw_recovery_min_progress_rad
                            ),
                            minimum_rate_rad_s=(
                                args_cli
                                .execution_rotation_recovery_outward_rate_tolerance_rad_s
                            ),
                        )
                    ):
                        force_c3_contact_is_yaw_recovery_pulse = True
                        measured_yaw_recovery_inference_count += 1
                    measured_c3_yaw_projected_error = (
                        signed_yaw_error
                        + args_cli.execution_rotation_guard_lookahead_s
                        * target_yaw_rate
                    )
                    recovery_brake_requested = (
                        measured_yaw_brake_requested(
                            start_signed_error_rad=(
                                force_c3_contact_start_signed_yaw_error
                            ),
                            current_signed_error_rad=signed_yaw_error,
                            current_yaw_rate_rad_s=target_yaw_rate,
                            recovery_latched=(
                                force_c3_contact_is_yaw_recovery_pulse
                            ),
                            # Native C3 is allowed to use the whole terminal
                            # yaw band for translation.  A corrective contact
                            # is released at the predicted zero crossing,
                            # rather than prematurely at half the band.
                            exit_threshold_rad=0.0,
                            lookahead_s=(
                                args_cli.execution_rotation_guard_lookahead_s
                            ),
                        )
                    )
                    measured_c3_yaw_brake_requested_now = (
                        recovery_brake_requested
                        or force_c3_yaw_regression_brake_requested_now
                    )
                if (
                    measured_c3_yaw_brake_requested_now
                    or force_c3_contact_window_expired
                ) and not execution_rotation_guard_active:
                    current_tip = np.asarray(
                        planner_tip_position_list(), dtype=float
                    )
                    target_com_position_e = (
                        target.data.root_com_pos_w[0, :3]
                        - base.scene.env_origins[0]
                    ).detach().cpu().numpy()
                    # Retreat radially away from the tracked object COM.  A
                    # nearest sampled surface point is reliable for approach,
                    # but its planar difference becomes ill-conditioned at a
                    # top/bottom contact and can accidentally point inward.
                    retreat_reference_xy = target_com_position_e[:2]
                    retreat_direction = current_tip[:2] - retreat_reference_xy
                    retreat_norm = float(np.linalg.norm(retreat_direction))
                    if retreat_norm <= 1.0e-9:
                        retreat_direction = np.asarray((1.0, 0.0))
                    else:
                        retreat_direction /= retreat_norm
                    semantic_yaw_brake_target = current_tip.copy()
                    semantic_yaw_brake_target[:2] += (
                        args_cli.execution_rotation_guard_retreat_m
                        * retreat_direction
                    )
                    semantic_yaw_brake_active = True
                    force_c3_window_retreat_active = (
                        force_c3_contact_window_expired
                    )
                    if measured_c3_yaw_brake_requested_now:
                        semantic_yaw_brake_activation_count += 1
                        measured_c3_yaw_brake_activation_count += 1
                        if first_semantic_yaw_brake_step is None:
                            first_semantic_yaw_brake_step = sequence
                        if first_measured_c3_yaw_brake_step is None:
                            first_measured_c3_yaw_brake_step = sequence
                        if force_c3_yaw_regression_brake_requested_now:
                            force_c3_yaw_regression_brake_activation_count += 1
                            if first_force_c3_yaw_regression_brake_step is None:
                                first_force_c3_yaw_regression_brake_step = sequence
                    if force_c3_contact_window_expired:
                        force_c3_window_retreat_activation_count += 1
                        if first_force_c3_window_retreat_step is None:
                            first_force_c3_window_retreat_step = sequence
                    force_c3_contact_active = False
                    force_c3_contact_armed = False
                    force_c3_contact_start_step = None
                    force_c3_contact_best_planar_error_m = None
                    force_c3_contact_best_abs_yaw_error_rad = None
                    force_c3_contact_start_signed_yaw_error = None
                    force_c3_contact_is_yaw_recovery_pulse = False
                    force_c3_contact_deactivation_count += 1

                semantic_yaw_brake_retreat_reached = bool(
                    semantic_yaw_brake_active
                    and semantic_yaw_brake_target is not None
                    and np.linalg.norm(
                        np.asarray(planner_tip_position_list(), dtype=float)
                        - semantic_yaw_brake_target
                    )
                    <= args_cli.semantic_yaw_brake_retreat_tolerance_m
                )
                if (
                    semantic_yaw_brake_retreat_reached
                    and not current_legal_safe_contact
                    and current_minimum_safe_distance
                    > args_cli.contact_distance_m
                ):
                    semantic_yaw_brake_active = False
                    semantic_yaw_brake_target = None
                    force_c3_window_retreat_active = False
                    # Do not execute a C3 segment computed before the retreat
                    # completed.  The next planner tick receives the new EE and
                    # object state and may deliberately select a fresh contact.
                    active_command = None
                    active_command_step = None
                    active_command_is_c3_mode = False
                    active_command_is_yaw_recovery = False
                    applied_feedforward_force = np.zeros(3, dtype=float)
                if semantic_yaw_brake_active:
                    # A corrective yaw pulse or bounded force-C3 contact window
                    # requested release. Ignore subsequent contact commands
                    # until the pusher has physically cleared the target. This
                    # gives the measured-state planner a clean state from which
                    # to select the next contact.
                    active_command = None
                    active_command_step = None
                    active_command_is_c3_mode = False
                    active_command_is_yaw_recovery = False
                    applied_feedforward_force = np.zeros(3, dtype=float)
                    assert semantic_yaw_brake_target is not None
                    last_desired_position = semantic_yaw_brake_target.copy()
                    last_command_position_isaac = last_desired_position.copy()

                if active_command is not None:
                    # C3 evaluates this point using the staged contact model.
                    # Between 20 Hz replans, its local velocity feeds the 100 Hz
                    # robot-specific servo; no recorded/global path is replayed.
                    command_age_s = (
                        sequence - active_command_step
                    ) * control_period_s
                    desired_position = (
                        np.asarray(active_command.position_m, dtype=float)
                        + command_age_s
                        * np.asarray(active_command.velocity_m_s, dtype=float)
                    )
                    # Native Drake OSC consumes C3's published force as an
                    # external wrench. Dairlib's inverse-dynamics equality is
                    # M*dv+c=B*u+J^T*lambda, whereas Isaac OSC directly adds
                    # J^T*wrench to the robot torque.  The deployable wrench
                    # must therefore be the actuator counterforce, -lambda.
                    requested_force = (
                        args_cli.c3_force_action_sign
                        * np.asarray(
                            active_command.feedforward_force_n, dtype=float
                        )
                    )
                    force_norm = float(np.linalg.norm(requested_force))
                    if force_norm > args_cli.max_feedforward_force_n:
                        requested_force *= (
                            args_cli.max_feedforward_force_n / force_norm
                        )
                    applied_feedforward_force = (
                        requested_force
                        if args_cli.local_controller == "cartesian_impedance"
                        else np.zeros(3, dtype=float)
                    )
                    maximum_applied_feedforward_force = max(
                        maximum_applied_feedforward_force,
                        float(np.linalg.norm(applied_feedforward_force)),
                    )
                    position_feedforward_offset.fill(0.0)
                    if (
                        args_cli.local_controller in (
                            "position_ik", "rmpflow_position"
                        )
                        and args_cli.position_feedforward_compliance_m_per_n
                        > 0.0
                    ):
                        # Push Anything optimizes an impedance/contact-force
                        # trajectory, while the deployable position backends
                        # accept an equilibrium position. Convert only the
                        # planned planar force into the equivalent bounded
                        # compliance offset. RMPflow/IK still own the robot
                        # kinematics and the high-rate safety guards below retain
                        # priority over this nominal target.
                        force_offset_xy = (
                            args_cli.position_feedforward_compliance_m_per_n
                            * requested_force[:2]
                        )
                        force_offset_norm = float(np.linalg.norm(force_offset_xy))
                        if (
                            force_offset_norm
                            > args_cli.position_feedforward_max_offset_m
                            and force_offset_norm > 1.0e-12
                        ):
                            force_offset_xy *= (
                                args_cli.position_feedforward_max_offset_m
                                / force_offset_norm
                            )
                        desired_position[:2] += force_offset_xy
                        position_feedforward_offset[:2] = force_offset_xy
                        maximum_position_feedforward_offset = max(
                            maximum_position_feedforward_offset,
                            float(np.linalg.norm(force_offset_xy)),
                        )
                    last_desired_position = desired_position
                    last_command_position_isaac = desired_position
                else:
                    # Hold the last safe pose during startup/dropout, but
                    # remove all feedforward contact force immediately.
                    applied_feedforward_force = np.zeros(3, dtype=float)
                    position_feedforward_offset.fill(0.0)
                    if args_cli.local_controller in (
                        "position_ik", "rmpflow_position"
                    ):
                        desired_q = desired_q.clone()

                contact_acquisition_commit_latched = False
                if args_cli.contact_acquisition_speed_m_s > 0.0:
                    planner_contact_authorization_available = (
                        active_command is not None
                        and active_command_is_c3_mode
                    )
                    contact_acquisition_commit_latched = (
                        contact_acquisition_active
                        and contact_acquisition_start_is_yaw_recovery is False
                        and contact_acquisition_start_action_axis_xy is not None
                        and not planner_contact_authorization_available
                    )
                    c3_planar_action_ready = (
                        contact_acquisition_commit_latched
                        or (
                            planner_contact_authorization_available
                            and (
                                contact_acquisition_start_action_axis_xy
                                is not None
                                or float(np.linalg.norm(np.asarray(
                                    active_command.velocity_m_s[:2], dtype=float
                                ))) > 1.0e-4
                            )
                        )
                    )
                    acquisition_unsafe = (
                        current_forbidden_contact
                        or current_protected_obstacle_collision
                        or current_robot_obstacle_collision
                        or execution_rotation_guard_active
                        or (
                            not planner_contact_authorization_available
                            and not contact_acquisition_commit_latched
                        )
                    )
                    # ``contact_acquisition_target`` can advance faster than
                    # the joint-position servo actually tracks it.  Exhausting
                    # the safety budget from commanded motion made the pusher
                    # lift with a repeatable 3--5 mm physical gap still open.
                    # Keep the command itself capped below, but release only
                    # after the measured tip has consumed the bounded travel
                    # (or a legal physical contact triggers the normal handoff).
                    measured_acquisition_travel_m = 0.0
                    if (
                        contact_acquisition_active
                        and contact_acquisition_start_tip is not None
                    ):
                        measured_acquisition_tip = np.asarray(
                            planner_tip_position_list(), dtype=float
                        )
                        measured_acquisition_travel_m = float(np.linalg.norm(
                            measured_acquisition_tip[:2]
                            - contact_acquisition_start_tip[:2]
                        ))
                    acquisition_exhausted = (
                        contact_acquisition_active
                        and measured_acquisition_travel_m
                        >= max(
                            0.0,
                            args_cli.contact_acquisition_max_travel_m - 5.0e-4,
                        )
                    )
                    if (
                        contact_acquisition_active
                        and (
                            acquisition_unsafe
                            or acquisition_exhausted
                            or (
                                current_legal_safe_contact
                                and c3_planar_action_ready
                            )
                        )
                    ):
                        handoff_to_measured_c3_contact = (
                            current_legal_safe_contact
                            and c3_planar_action_ready
                        )
                        contact_acquisition_active = False
                        contact_acquisition_start_tip = None
                        contact_acquisition_axis_xy = None
                        contact_acquisition_target = None
                        contact_acquisition_travel_m = 0.0
                        contact_acquisition_deactivation_count += 1
                        if handoff_to_measured_c3_contact:
                            # A legal handoff is not a failed acquisition.
                            # Keep reacquisition armed so ordinary contact-
                            # sensor chatter can close a residual 3--8 mm gap
                            # without first retreating 1.5 trigger distances.
                            contact_acquisition_armed = True
                        else:
                            contact_acquisition_start_signed_yaw_error = None
                            contact_acquisition_start_is_yaw_recovery = None
                            contact_acquisition_start_action_axis_xy = None
                    if (
                        not contact_acquisition_active
                        and not contact_acquisition_armed
                        and current_minimum_safe_distance
                        > 1.5 * args_cli.contact_acquisition_trigger_m
                    ):
                        # A failed/cancelled attempt must clear the semantic
                        # surface before another acquisition can be latched.
                        contact_acquisition_armed = True
                    safe_is_unambiguously_closest = (
                        math.isfinite(current_minimum_safe_distance)
                        and math.isfinite(current_minimum_forbidden_distance)
                        and current_minimum_forbidden_distance
                        >= current_minimum_safe_distance
                        + args_cli.contact_acquisition_forbidden_margin_m
                    )
                    # The acquisition servo stops at the measured safe
                    # surface; it cannot consume more travel than the current
                    # safe-surface distance.  Reserving the configured 30 mm
                    # maximum unconditionally rejects otherwise safe yaw
                    # contacts that are already 10--20 mm from the surface.
                    # This remains fail-closed under the worst-case direction:
                    # even after the whole required acquisition displacement,
                    # robot--clutter clearance must exceed the hard threshold.
                    required_acquisition_travel_m = min(
                        args_cli.contact_acquisition_max_travel_m,
                        max(0.0, current_minimum_safe_distance),
                    )
                    robot_clutter_margin_available = (
                        not math.isfinite(current_robot_obstacle_clearance)
                        or current_robot_obstacle_clearance
                        >= args_cli.robot_obstacle_clearance_m
                        + required_acquisition_travel_m
                    )
                    if (
                        not contact_acquisition_active
                        and contact_acquisition_armed
                        and not acquisition_unsafe
                        and not current_legal_safe_contact
                        and current_closest_safe_point_xy is not None
                        and current_minimum_safe_distance
                        <= args_cli.contact_acquisition_trigger_m
                        and safe_is_unambiguously_closest
                        and robot_clutter_margin_available
                    ):
                        current_tip = np.asarray(
                            planner_tip_position_list(), dtype=float
                        )
                        safe_delta_xy = (
                            current_closest_safe_point_xy - current_tip[:2]
                        )
                        safe_delta_norm = float(np.linalg.norm(safe_delta_xy))
                        if safe_delta_norm > 1.0e-6:
                            contact_acquisition_active = True
                            contact_acquisition_armed = False
                            contact_acquisition_start_tip = current_tip.copy()
                            contact_acquisition_start_signed_yaw_error = (
                                signed_yaw_error
                            )
                            contact_acquisition_start_is_yaw_recovery = (
                                active_command_is_yaw_recovery
                            )
                            acquisition_action_xy = np.asarray(
                                active_command.velocity_m_s[:2], dtype=float
                            )
                            acquisition_action_norm = float(
                                np.linalg.norm(acquisition_action_xy)
                            )
                            # A short C3 horizon can end before the relay's
                            # lookahead sample.  In that case the returned
                            # terminal-knot velocity is exactly zero.  Do not
                            # turn it into a NaN "valid" action axis: that
                            # falsely latches a translation commitment after
                            # the planner leaves C3 mode.
                            contact_acquisition_start_action_axis_xy = (
                                acquisition_action_xy / acquisition_action_norm
                                if acquisition_action_norm > 1.0e-6
                                else None
                            )
                            contact_acquisition_axis_xy = (
                                safe_delta_xy / safe_delta_norm
                            )
                            contact_acquisition_target = current_tip.copy()
                            contact_acquisition_travel_m = 0.0
                            contact_acquisition_activation_count += 1
                            if first_contact_acquisition_step is None:
                                first_contact_acquisition_step = sequence
                    if contact_acquisition_active:
                        assert contact_acquisition_start_tip is not None
                        assert contact_acquisition_axis_xy is not None
                        assert contact_acquisition_target is not None
                        # Once a bounded translation acquisition is committed,
                        # the high-level planner may switch back to reposition
                        # because its geometric model already considers the
                        # contact reached.  Preserve the last contact-approach Z
                        # rather than following that new lift command.
                        nominal_desired_z = (
                            float(last_desired_position[2])
                            if planner_contact_authorization_available
                            else float(contact_acquisition_target[2])
                        )
                        if contact_acquisition_commit_latched:
                            applied_feedforward_force = np.zeros(3, dtype=float)
                        current_tip = np.asarray(
                            planner_tip_position_list(), dtype=float
                        )
                        if current_closest_safe_point_xy is not None:
                            current_safe_delta_xy = (
                                current_closest_safe_point_xy - current_tip[:2]
                            )
                            current_safe_delta_norm = float(
                                np.linalg.norm(current_safe_delta_xy)
                            )
                            if current_safe_delta_norm > 1.0e-6:
                                # Follow the measured distance field rather
                                # than a point latched at acquisition start.
                                # On a curved/discrete safe surface, the old
                                # fixed direction can become tangential before
                                # physical contact is established.
                                contact_acquisition_axis_xy = (
                                    current_safe_delta_xy
                                    / current_safe_delta_norm
                                )
                        commanded_acquisition_travel_m = float(np.linalg.norm(
                            contact_acquisition_target[:2]
                            - contact_acquisition_start_tip[:2]
                        ))
                        acquisition_step_m = max(
                            0.0,
                            min(
                                args_cli.contact_acquisition_speed_m_s
                                * control_period_s,
                                args_cli.contact_acquisition_max_travel_m
                                - commanded_acquisition_travel_m,
                            ),
                        )
                        contact_acquisition_travel_m = min(
                            args_cli.contact_acquisition_max_travel_m,
                            measured_acquisition_travel_m,
                        )
                        maximum_contact_acquisition_travel_m = max(
                            maximum_contact_acquisition_travel_m,
                            contact_acquisition_travel_m,
                        )
                        contact_acquisition_target[:2] += (
                            acquisition_step_m
                            * contact_acquisition_axis_xy
                        )
                        # Acquisition corrects only planar registration error.
                        # Preserve the planner's vertical approach; freezing Z
                        # at the trigger pose can sweep over a low-profile tool
                        # without ever loading the physical contact sensor.
                        contact_acquisition_target[2] = nominal_desired_z
                        last_desired_position = contact_acquisition_target.copy()
                        last_command_position_isaac = last_desired_position.copy()

                semantic_yaw_projected_error = signed_yaw_error
                semantic_yaw_recovery_predictive_brake = False
                semantic_yaw_recovery_reversing = False
                if args_cli.semantic_contact_push_speed_m_s > 0.0:
                    current_planar_error, _, current_rotation_error = (
                        _pose_errors(base)
                    )
                    # Match the planner's yaw-recovery hysteresis: entering
                    # the strict band is not enough margin for the next push.
                    # Continue a corrective pulse to half the terminal yaw
                    # threshold, then predictively brake before overshoot.
                    yaw_recovery_exit_threshold = (
                        0.5 * args_cli.strict_rotation_threshold_rad
                    )
                    if (
                        semantic_contact_push_active
                        and args_cli.semantic_contact_axis_source
                        != "goal_wrench"
                        and not semantic_contact_is_yaw_recovery_pulse
                        and semantic_contact_start_signed_yaw_error is not None
                        and measured_contact_is_yaw_recovery(
                            start_signed_error_rad=(
                                semantic_contact_start_signed_yaw_error
                            ),
                            current_signed_error_rad=signed_yaw_error,
                            current_yaw_rate_rad_s=target_yaw_rate,
                            activation_threshold_rad=yaw_recovery_exit_threshold,
                            minimum_progress_rad=(
                                args_cli.measured_yaw_recovery_min_progress_rad
                            ),
                            minimum_rate_rad_s=(
                                args_cli.execution_rotation_recovery_outward_rate_tolerance_rad_s
                            ),
                        )
                    ):
                        # Native C3 builds predating the semantic mode flag can
                        # still reveal the action's role through measured target
                        # motion.  Latch that evidence for the rest of this
                        # bounded pulse so the existing predictive brake stops
                        # before the yaw error crosses zero.
                        semantic_contact_is_yaw_recovery_pulse = True
                        measured_yaw_recovery_inference_count += 1
                    semantic_yaw_recovery_pulse = (
                        semantic_contact_push_active
                        and semantic_contact_is_yaw_recovery_pulse
                    )
                    semantic_yaw_recovery_measured_complete = (
                        semantic_yaw_recovery_pulse
                        and (
                            abs(signed_yaw_error)
                            <= yaw_recovery_exit_threshold
                            or semantic_contact_start_signed_yaw_error
                            * signed_yaw_error
                            <= 0.0
                        )
                    )
                    # The object keeps rotating briefly after the pusher starts
                    # retreating.  Braking only after the measured error enters
                    # the half-threshold band therefore reacts one servo cycle
                    # too late and can carry a corrective pulse through zero.
                    # Predict that crossing from the measured object yaw rate;
                    # this state is observable from RGB-D tracking on hardware
                    # and does not depend on simulator-only contact impulses.
                    semantic_yaw_motion_is_corrective = (
                        signed_yaw_error * target_yaw_rate < 0.0
                    )
                    semantic_yaw_recovery_reversing = bool(
                        semantic_yaw_recovery_pulse
                        and abs(target_yaw_rate)
                        > args_cli.execution_rotation_recovery_outward_rate_tolerance_rad_s
                        and signed_yaw_error * target_yaw_rate > 0.0
                    )
                    semantic_yaw_projected_error = (
                        signed_yaw_error
                        + args_cli.execution_rotation_guard_lookahead_s
                        * target_yaw_rate
                    )
                    semantic_yaw_recovery_predictive_brake = (
                        semantic_yaw_recovery_pulse
                        and semantic_yaw_motion_is_corrective
                        and (
                            abs(semantic_yaw_projected_error)
                            <= yaw_recovery_exit_threshold
                            or signed_yaw_error
                            * semantic_yaw_projected_error
                            <= 0.0
                        )
                    )
                    semantic_yaw_recovery_complete = (
                        semantic_yaw_recovery_measured_complete
                        or semantic_yaw_recovery_predictive_brake
                    )
                    contact_pulse_timed_out = (
                        semantic_contact_push_active
                        and semantic_contact_push_start_step is not None
                        and (sequence - semantic_contact_push_start_step)
                        * control_period_s
                        >= args_cli.semantic_contact_max_duration_s
                    )
                    contact_pulse_reached_position = (
                        semantic_contact_push_active
                        and current_planar_error
                        <= args_cli.strict_position_threshold_m
                    )
                    contact_pulse_regressed = (
                        semantic_contact_push_active
                        and semantic_contact_best_planar_error_m is not None
                        and current_planar_error
                        > semantic_contact_best_planar_error_m
                        + args_cli.semantic_contact_planar_regression_tolerance_m
                    )
                    if (
                        semantic_contact_push_active
                        and (
                            current_minimum_safe_distance
                            > 2.0 * args_cli.contact_distance_m
                            or current_forbidden_contact
                            or current_protected_obstacle_collision
                            or current_robot_obstacle_collision
                            or execution_rotation_guard_active
                            # A directly tracked C3 contact must stop when the
                            # planner revokes C3 mode.  A contact acquired from
                            # the same C3 command may finish its bounded local
                            # pulse: the high-rate C1/C2/C3 and yaw guards below
                            # still revoke it immediately on measured risk.
                            or (
                                not active_command_is_c3_mode
                                and not semantic_contact_from_committed_acquisition
                            )
                            or current_rotation_error
                            >= args_cli.execution_rotation_guard_limit_rad
                            or semantic_yaw_recovery_complete
                            or semantic_yaw_recovery_reversing
                            or contact_pulse_timed_out
                            or contact_pulse_reached_position
                            or contact_pulse_regressed
                        )
                    ):
                        goal_wrench_release_requested = bool(
                            args_cli.semantic_contact_axis_source
                            == "goal_wrench"
                            and current_minimum_safe_distance
                            <= 2.0 * args_cli.contact_distance_m
                        )
                        semantic_yaw_brake_requested = bool(
                            (
                                goal_wrench_release_requested
                                or (
                                    semantic_yaw_recovery_pulse
                                    and (
                                        semantic_yaw_recovery_complete
                                        or semantic_yaw_recovery_reversing
                                        or contact_pulse_timed_out
                                    )
                                )
                            )
                            and not execution_rotation_guard_active
                            and not current_forbidden_contact
                            and not current_protected_obstacle_collision
                            and not current_robot_obstacle_collision
                        )
                        semantic_contact_push_active = False
                        semantic_contact_push_target = None
                        semantic_contact_push_axis_xy = None
                        semantic_contact_requested_moment_arm_m = None
                        semantic_contact_achieved_moment_arm_m = None
                        semantic_contact_relative_offset_b_m = None
                        semantic_contact_start_signed_yaw_error = None
                        semantic_contact_is_yaw_recovery_pulse = False
                        semantic_contact_from_committed_acquisition = False
                        semantic_contact_best_planar_error_m = None
                        semantic_contact_compression_offset_m = 0.0
                        semantic_contact_push_start_step = None
                        semantic_contact_approach_standoff_m = None
                        semantic_contact_standoff_m = None
                        semantic_contact_minimum_standoff_m = None
                        semantic_contact_axial_target_m = None
                        semantic_contact_push_deactivation_count += 1
                        if semantic_yaw_brake_requested:
                            current_tip = np.asarray(
                                planner_tip_position_list(), dtype=float
                            )
                            target_com_position_e = (
                                target.data.root_com_pos_w[0, :3]
                                - base.scene.env_origins[0]
                            ).detach().cpu().numpy()
                            retreat_reference_xy = target_com_position_e[:2]
                            retreat_direction = (
                                current_tip[:2] - retreat_reference_xy
                            )
                            retreat_norm = float(np.linalg.norm(retreat_direction))
                            if retreat_norm <= 1.0e-9:
                                retreat_direction = np.asarray((1.0, 0.0))
                            else:
                                retreat_direction /= retreat_norm
                            semantic_yaw_brake_target = current_tip.copy()
                            semantic_yaw_brake_target[:2] += (
                                args_cli.execution_rotation_guard_retreat_m
                                * retreat_direction
                            )
                            semantic_yaw_brake_active = True
                            force_c3_window_retreat_active = False
                            semantic_yaw_brake_activation_count += 1
                            if first_semantic_yaw_brake_step is None:
                                first_semantic_yaw_brake_step = sequence
                            active_command = None
                            active_command_step = None
                            active_command_is_c3_mode = False
                            active_command_is_yaw_recovery = False
                            applied_feedforward_force = np.zeros(3, dtype=float)
                            last_desired_position = (
                                semantic_yaw_brake_target.copy()
                            )
                            last_command_position_isaac = (
                                last_desired_position.copy()
                            )
                    if (
                        not semantic_contact_push_active
                        and not semantic_contact_push_armed
                        and not current_legal_safe_contact
                        and current_minimum_safe_distance
                        > args_cli.contact_distance_m
                    ):
                        # Require a genuine contact break before another pulse;
                        # otherwise a timed-out pulse would restart forever and
                        # starve the receding-horizon planner.
                        semantic_contact_push_armed = True
                    if (
                        not semantic_contact_push_active
                        and semantic_contact_push_armed
                        and current_legal_safe_contact
                        and (
                            (
                                active_command is not None
                                and active_command_is_c3_mode
                            )
                            or (
                                contact_acquisition_start_action_axis_xy
                                is not None
                                and contact_acquisition_start_is_yaw_recovery
                                is False
                            )
                        )
                        and not current_forbidden_contact
                        and not current_protected_obstacle_collision
                        and not current_robot_obstacle_collision
                        and not execution_rotation_guard_active
                        and current_planar_error
                        > args_cli.strict_position_threshold_m
                        and (
                            args_cli
                            .semantic_contact_activation_planar_threshold_m
                            is None
                            or current_planar_error
                            <= args_cli
                            .semantic_contact_activation_planar_threshold_m
                        )
                        and current_rotation_error
                        < args_cli.execution_rotation_guard_limit_rad
                    ):
                        target_position_e = (
                            target.data.root_pos_w[0, :3]
                            - base.scene.env_origins[0]
                        ).detach().cpu().numpy()
                        planned_velocity_xy = (
                            np.asarray(
                                active_command.velocity_m_s[:2], dtype=float
                            )
                            if active_command is not None
                            and active_command_is_c3_mode
                            else np.zeros(2, dtype=float)
                        )
                        planned_speed = float(np.linalg.norm(planned_velocity_xy))
                        goal_position_e = base.command_manager.get_command(
                            "target_object_pose"
                        )[0, :3].detach().cpu().numpy()
                        goal_axis_xy = goal_position_e[:2] - target_position_e[:2]
                        goal_axis_norm = float(np.linalg.norm(goal_axis_xy))
                        use_goal_axis = bool(
                            args_cli.semantic_contact_axis_source
                            in ("goal", "goal_wrench")
                            and goal_axis_norm > 1.0e-6
                        )
                        contact_is_yaw_recovery = (
                            contact_acquisition_start_is_yaw_recovery
                            if contact_acquisition_start_is_yaw_recovery
                            is not None
                            else active_command_is_yaw_recovery
                        )
                        use_latched_translation_axis = (
                            contact_acquisition_start_action_axis_xy is not None
                            and not contact_is_yaw_recovery
                        )
                        if (
                            use_latched_translation_axis
                            or planned_speed > 1.0e-4
                            or use_goal_axis
                        ):
                            planned_axis = (
                                goal_axis_xy / goal_axis_norm
                                if use_goal_axis
                                else (
                                    contact_acquisition_start_action_axis_xy.copy()
                                    if use_latched_translation_axis
                                    else planned_velocity_xy / planned_speed
                                )
                            )
                            requested_moment_arm_m = None
                            achieved_moment_arm_m = None
                            if (
                                args_cli.semantic_contact_axis_source
                                == "goal_wrench"
                                and current_closest_safe_point_xy is not None
                                and goal_axis_norm > 1.0e-6
                            ):
                                target_com_position_e = (
                                    target.data.root_com_pos_w[0, :2]
                                    - base.scene.env_origins[0, :2]
                                ).detach().cpu().numpy()
                                (
                                    planned_axis,
                                    requested_moment_arm_m,
                                    achieved_moment_arm_m,
                                ) = goal_wrench_contact_axis(
                                    goal_delta_xy_m=goal_axis_xy,
                                    contact_point_xy_m=(
                                        current_closest_safe_point_xy
                                    ),
                                    object_com_xy_m=target_com_position_e,
                                    signed_yaw_error_rad=signed_yaw_error,
                                    object_yaw_rate_rad_s=target_yaw_rate,
                                    yaw_moment_gain_m_per_rad=(
                                        args_cli
                                        .semantic_contact_yaw_moment_gain_m_per_rad
                                    ),
                                    yaw_rate_moment_gain_m_s_per_rad=(
                                        args_cli
                                        .semantic_contact_yaw_rate_moment_gain_m_s_per_rad
                                    ),
                                    max_axis_deviation_rad=(
                                        args_cli
                                        .semantic_contact_max_axis_deviation_rad
                                    ),
                                )
                            # Continue only a command explicitly tagged as C3
                            # contact mode.  Repositioning trajectories can
                            # have the same planar velocity shape and must not
                            # be mistaken for a push.  C3's target-effect filter
                            # already decides whether translation or yaw motion
                            # is useful; the local servo only preserves contact.
                            current_tip = np.asarray(
                                planner_tip_position_list(), dtype=float
                            )
                            semantic_contact_push_active = True
                            semantic_contact_push_armed = False
                            semantic_contact_push_target = current_tip.copy()
                            semantic_contact_push_axis_xy = planned_axis
                            semantic_contact_requested_moment_arm_m = (
                                requested_moment_arm_m
                            )
                            semantic_contact_achieved_moment_arm_m = (
                                achieved_moment_arm_m
                            )
                            relative_offset_w = torch.tensor(
                                current_tip - target_position_e,
                                device=base.device,
                                dtype=target.data.root_pos_w.dtype,
                            ).unsqueeze(0)
                            semantic_contact_relative_offset_b_m = (
                                quat_apply_inverse(
                                    target.data.root_quat_w[0:1],
                                    relative_offset_w,
                                )[0].detach().cpu().numpy()
                            )
                            semantic_contact_start_signed_yaw_error = (
                                contact_acquisition_start_signed_yaw_error
                                if contact_acquisition_start_signed_yaw_error
                                is not None
                                else signed_yaw_error
                            )
                            semantic_contact_is_yaw_recovery_pulse = (
                                contact_is_yaw_recovery
                                and args_cli.semantic_contact_axis_source
                                != "goal_wrench"
                            )
                            semantic_contact_from_committed_acquisition = (
                                use_latched_translation_axis
                            )
                            semantic_contact_best_planar_error_m = (
                                current_planar_error
                            )
                            contact_acquisition_start_signed_yaw_error = None
                            contact_acquisition_start_is_yaw_recovery = None
                            contact_acquisition_start_action_axis_xy = None
                            semantic_contact_compression_offset_m = 0.0
                            semantic_contact_push_start_step = sequence
                            semantic_contact_approach_standoff_m = None
                            semantic_contact_standoff_m = max(
                                0.0,
                                float(np.dot(
                                    target_position_e[:2] - current_tip[:2],
                                    planned_axis,
                                )),
                            )
                            semantic_contact_minimum_standoff_m = max(
                                0.0,
                                semantic_contact_standoff_m
                                - args_cli.semantic_contact_max_additional_compression_m,
                            )
                            semantic_contact_axial_target_m = float(
                                np.dot(current_tip[:2], planned_axis)
                            )
                            semantic_contact_standoff_latch_step = sequence
                            if semantic_contact_push_activation_step is None:
                                semantic_contact_push_activation_step = sequence
                    if semantic_contact_push_active:
                        assert semantic_contact_push_target is not None
                        assert semantic_contact_push_axis_xy is not None
                        assert semantic_contact_relative_offset_b_m is not None
                        target_position_e = (
                            target.data.root_pos_w[0, :3]
                            - base.scene.env_origins[0]
                        ).detach().cpu().numpy()
                        goal_command = base.command_manager.get_command(
                            "target_object_pose"
                        )[0]
                        goal_position_e = goal_command[:3].detach().cpu().numpy()
                        goal_delta_xy = goal_position_e[:2] - target_position_e[:2]
                        goal_distance_xy = float(np.linalg.norm(goal_delta_xy))
                        if (
                            args_cli.semantic_contact_axis_source
                            == "goal_wrench"
                            and current_closest_safe_point_xy is not None
                            and goal_distance_xy > 1.0e-6
                        ):
                            target_com_position_e = (
                                target.data.root_com_pos_w[0, :2]
                                - base.scene.env_origins[0, :2]
                            ).detach().cpu().numpy()
                            (
                                semantic_contact_push_axis_xy,
                                semantic_contact_requested_moment_arm_m,
                                semantic_contact_achieved_moment_arm_m,
                            ) = goal_wrench_contact_axis(
                                goal_delta_xy_m=goal_delta_xy,
                                contact_point_xy_m=current_closest_safe_point_xy,
                                object_com_xy_m=target_com_position_e,
                                signed_yaw_error_rad=signed_yaw_error,
                                object_yaw_rate_rad_s=target_yaw_rate,
                                yaw_moment_gain_m_per_rad=(
                                    args_cli
                                    .semantic_contact_yaw_moment_gain_m_per_rad
                                ),
                                yaw_rate_moment_gain_m_s_per_rad=(
                                    args_cli
                                    .semantic_contact_yaw_rate_moment_gain_m_s_per_rad
                                ),
                                max_axis_deviation_rad=(
                                    args_cli
                                    .semantic_contact_max_axis_deviation_rad
                                ),
                            )
                        semantic_contact_best_planar_error_m = min(
                            semantic_contact_best_planar_error_m
                            if semantic_contact_best_planar_error_m is not None
                            else goal_distance_xy,
                            goal_distance_xy,
                        )
                        if goal_distance_xy >= 0.015:
                            semantic_contact_compression_offset_m = min(
                                args_cli.semantic_contact_max_additional_compression_m,
                                semantic_contact_compression_offset_m
                                + args_cli.semantic_contact_push_speed_m_s
                                * control_period_s,
                            )
                        relative_offset_b = torch.tensor(
                            semantic_contact_relative_offset_b_m,
                            device=base.device,
                            dtype=target.data.root_pos_w.dtype,
                        ).unsqueeze(0)
                        relative_offset_w = quat_apply(
                            target.data.root_quat_w[0:1], relative_offset_b
                        )[0].detach().cpu().numpy()
                        # Shift the maintained safe-handle contact laterally so
                        # the push moment opposes the measured current-minus-goal
                        # yaw error.  For push axis u and left normal n, choosing
                        # r_yaw = error * n gives cross(r_yaw, u) = -error.
                        # The shift is bounded inside the configured safe-handle
                        # allowance and is zero when yaw is already correct.
                        lateral_axis_xy = np.asarray(
                            (
                                -semantic_contact_push_axis_xy[1],
                                semantic_contact_push_axis_xy[0],
                            ),
                            dtype=float,
                        )
                        lateral_offset_m = (
                            0.0
                            if args_cli.semantic_contact_axis_source
                            == "goal_wrench"
                            else float(np.clip(
                                args_cli.semantic_contact_yaw_gain_m_per_rad
                                * signed_yaw_error,
                                -args_cli.semantic_contact_max_lateral_offset_m,
                                args_cli.semantic_contact_max_lateral_offset_m,
                            ))
                        )
                        maximum_semantic_contact_lateral_offset_m = max(
                            maximum_semantic_contact_lateral_offset_m,
                            abs(lateral_offset_m),
                        )
                        semantic_contact_push_target[:2] = (
                            target_position_e[:2]
                            + relative_offset_w[:2]
                            + semantic_contact_compression_offset_m
                            * semantic_contact_push_axis_xy
                            + lateral_offset_m * lateral_axis_xy
                        )
                        last_desired_position = semantic_contact_push_target.copy()
                        last_command_position_isaac = last_desired_position.copy()

                # Reject relaxed below-table targets for every physical servo.
                # The model may temporarily use penetration in its compliant
                # contact rollout, but neither Isaac nor a real robot can track
                # such a reference without stalling against the support plane.
                raw_task_target = np.asarray(
                    last_desired_position, dtype=float
                ).copy()
                candidate_task_target = raw_task_target.copy()
                reference_velocity = np.zeros(3, dtype=float)
                if active_command is not None:
                    reference_velocity = np.asarray(
                        active_command.velocity_m_s, dtype=float
                    ).copy()
                # The online bridge replaces native franka_osc_controller, so
                # its high-rate SemanticC1TrajectoryGuard must run here too.
                # Use fresh measured object poses at every servo step, rather
                # than the finite-iteration QP's predicted object trajectory.
                if semantic_trajectory_guard is not None:
                    guard_object_states = [planner_rigid_body_state(
                        target.data.root_pos_w[0], target.data.root_quat_w[0],
                        target.data.root_link_vel_w[0, 3:6],
                        target.data.root_link_vel_w[0, 0:3], support_quaternions[0])]
                    if multi_object_mode:
                        guard_object_states.extend(planner_rigid_body_state(
                            obstacles.data.object_pos_w[0, index],
                            obstacles.data.object_quat_w[0, index],
                            obstacles.data.body_link_vel_w[0, index, 3:6],
                            obstacles.data.body_link_vel_w[0, index, 0:3],
                            support_quaternions[index + 1]) for index in range(clutter_count))
                    candidate_task_target, semantic_guard_active, semantic_guard_distance = (
                        semantic_trajectory_guard.apply(
                            planner_tip_position_list(), candidate_task_target,
                            guard_object_states))
                    if semantic_guard_active:
                        applied_feedforward_force = np.zeros(3, dtype=float)
                        reference_velocity.fill(0.0)
                        semantic_guard_activation_count += 1
                reference_floor = args_cli.minimum_task_target_z_m
                if args_cli.task_height_floor_mode == "finger-geometry":
                    from dapl.contact_planner.isaac_bridge import minimum_reference_height
                    hand_q_b = quat_mul(quat_inv(robot.data.root_quat_w[0:1]),
                                        robot.data.body_quat_w[0:1, hand_id])[0].tolist()
                    reference_floor = minimum_reference_height(
                        finger_reference_vertices, hand_q_b, -args_cli.table_height_offset_m,
                        clearance_m=args_cli.task_height_clearance_m)
                if candidate_task_target[2] < reference_floor:
                    candidate_task_target[2] = reference_floor
                    reference_velocity[2] = max(0.0, reference_velocity[2])
                    task_target_z_clamp_count += 1

                if args_cli.local_controller == "cartesian_impedance":
                    # Each C3 solve is dynamically smooth internally, but a
                    # newly selected contact can make the next receding-horizon
                    # segment discontinuous with the segment currently being
                    # tracked.  Drake's native OSC and a real Franka controller
                    # both impose actuator/reference limits.  Apply the same
                    # robot-side contract here.
                    maximum_raw_task_target_step_m = max(
                        maximum_raw_task_target_step_m,
                        float(np.linalg.norm(
                            raw_task_target - governed_task_target
                        )),
                    )
                    target_delta = (
                        candidate_task_target - governed_task_target
                    )
                    target_delta_norm = float(np.linalg.norm(target_delta))
                    maximum_target_step = (
                        args_cli.max_task_target_speed_m_s * control_period_s
                    )
                    if target_delta_norm > maximum_target_step:
                        candidate_task_target = (
                            governed_task_target
                            + maximum_target_step
                            * target_delta
                            / target_delta_norm
                        )
                    governor_correction = float(np.linalg.norm(
                        raw_task_target - candidate_task_target
                    ))
                    maximum_task_target_governor_correction_m = max(
                        maximum_task_target_governor_correction_m,
                        governor_correction,
                    )
                    if governor_correction > 1.0e-9:
                        task_target_governor_activation_count += 1
                        # A new MPC segment restarts at measured position. Its
                        # reconciliation jump is not the trajectory derivative:
                        # differentiating it can reverse an inward C3 command.
                        # Retain the local polynomial derivative by default;
                        # the historical behavior remains an explicit ablation.
                        if args_cli.osc_reference_velocity_mode == "governed":
                            reference_velocity = (
                                candidate_task_target - governed_task_target
                            ) / control_period_s
                    governed_task_target = candidate_task_target
                else:
                    governed_task_target = candidate_task_target
                last_desired_position = governed_task_target.copy()
                last_command_position_isaac = governed_task_target.copy()
                if semantic_guard_active:
                    # The optional governed-velocity ablation must not
                    # reintroduce motion damping feedforward after a hold.
                    reference_velocity.fill(0.0)

                # Resolve IK only after semantic and rotation guards have had
                # their chance to replace the nominal C3 target.  Solving at
                # command-receive time leaves a position-controlled robot
                # tracking the stale contact pose while the safety state says
                # it is retreating.
                if args_cli.local_controller == "position_ik":
                    solved, position_error = ik.solve_translation(
                        seed=desired_q[0].detach().cpu().numpy(),
                        regularization_reference=q.detach().cpu().numpy(),
                        desired_translation=last_desired_position,
                    )
                    maximum_ik_position_error = max(
                        maximum_ik_position_error, position_error
                    )
                    if position_error > args_cli.ik_position_tolerance_m:
                        stopped_reason = "ik_tolerance_failure"
                        break
                    solved_tensor = torch.tensor(
                        solved, device=base.device, dtype=initial_q.dtype
                    ).unsqueeze(0)
                    maximum_delta = (
                        args_cli.max_joint_target_rate_rad_s * control_period_s
                    )
                    desired_q = torch.clamp(
                        solved_tensor,
                        desired_q - maximum_delta,
                        desired_q + maximum_delta,
                    )

                if rmpflow is not None:
                    update_rmpflow_reference(last_desired_position)
                    if args_cli.local_controller == "rmpflow_position":
                        desired_q = rmpflow_joint_target.clone()
                elif (
                    args_cli.local_controller == "cartesian_impedance"
                    and args_cli.osc_nullspace_target == "push_anything_joint2"
                ):
                    # Match Push Anything's Franka OSC redundancy contract:
                    # only panda_joint2 has a position objective (1.1 rad).
                    # Updating the other targets to their measured positions
                    # avoids introducing the seven-joint startup-posture
                    # objective that changed the arm path in the Isaac bridge.
                    joint2_only_target = robot.data.joint_pos[
                        0:1, joint_ids
                    ].clone()
                    joint2_only_target[:, 1] = 1.1
                    osc_action_term._nullspace_joint_pos_target.copy_(
                        joint2_only_target
                    )
                if args_cli.local_controller == "cartesian_impedance":
                    if args_cli.osc_track_trajectory_velocity:
                        if active_command is None:
                            reference_velocity.fill(0.0)
                        reference_speed = float(np.linalg.norm(reference_velocity))
                        if reference_speed > args_cli.max_task_target_speed_m_s:
                            reference_velocity *= (
                                args_cli.max_task_target_speed_m_s / reference_speed
                            )
                        velocity_b = torch.zeros(
                            (1, 6), device=base.device, dtype=initial_q.dtype
                        )
                        velocity_b[0, :3] = torch.as_tensor(
                            reference_velocity, device=base.device, dtype=initial_q.dtype
                        )
                        # The position reference controls panda_hand at
                        # p_tip - R_hand * offset. Differentiate that measured
                        # frame conversion too; angular pose target stays fixed.
                        omega_b = quat_apply(
                            quat_inv(robot.data.root_quat_w[0:1]),
                            robot.data.body_ang_vel_w[0:1, hand_id],
                        )
                        hand_quat_b = quat_mul(
                            quat_inv(robot.data.root_quat_w[0:1]),
                            robot.data.body_quat_w[0:1, hand_id],
                        )
                        offset_b = quat_apply(
                            hand_quat_b,
                            torch.tensor(
                                [[0.0, 0.0, args_cli.drake_tip_from_hand_m]],
                                device=base.device, dtype=initial_q.dtype,
                            ),
                        )
                        if active_command is not None and args_cli.osc_control_point == "hand":
                            velocity_b[:, :3] -= torch.linalg.cross(omega_b, offset_b)
                        base.action_manager.get_term("arm_action").set_desired_velocity(velocity_b)
                    local_action = osc_action(
                        last_desired_position, applied_feedforward_force
                    )
                else:
                    local_action = desired_q
                env.step(local_action)
                maximum_abs_joint_effort = max(
                    maximum_abs_joint_effort,
                    float(torch.max(torch.abs(
                        robot.data.applied_torque[0, joint_ids]
                    ))),
                )
                if (
                    video_writer is not None
                    and (sequence + 1) % video_stride == 0
                ):
                    video_writer.write(env.render())
                if last_command_position_isaac is not None:
                    current_task_tracking_error = float(np.linalg.norm(
                        np.asarray(planner_tip_position_list())
                        - last_command_position_isaac
                    ))
                    maximum_task_tracking_error = max(
                        maximum_task_tracking_error,
                        current_task_tracking_error,
                    )

                planar, height, rotation = _pose_errors(base)
                minimum_planar = min(minimum_planar, planar)
                minimum_height = min(minimum_height, height)
                minimum_rotation = min(minimum_rotation, rotation)
                strict = (
                    planar < args_cli.strict_position_threshold_m
                    and height < args_cli.strict_height_threshold_m
                    and rotation < args_cli.strict_rotation_threshold_rad
                )
                strict_dwell = strict_dwell + 1 if strict else 0
                maximum_strict_dwell = max(maximum_strict_dwell, strict_dwell)

                if sequence % args_cli.audit_stride == 0:
                    contact_audit_measurement_utime_us = 100_000 + round(
                        (sequence + 1) * control_period_s * 1.0e6
                    )
                    contact = audited_contact_state()
                    current_semantic_safe_proximity = bool(
                        contact["safe_robot_contact"][0]
                    )
                    hand_sensor_available = bool(
                        contact["hand_target_sensor_available"][0]
                    )
                    safe = (
                        bool(contact["legal_physical_safe_hand_contact"][0])
                        if hand_sensor_available
                        else bool(contact["legal_safe_robot_contact"][0])
                    )
                    current_safe_contact = safe
                    current_legal_safe_contact = safe
                    forbidden = bool(contact["forbidden_robot_contact"][0])
                    current_forbidden_contact = forbidden
                    protected_obstacle_collision = bool(
                        contact["protected_obstacle_collision"][0]
                    )
                    current_protected_obstacle_collision = (
                        protected_obstacle_collision
                    )
                    robot_obstacle_collision = bool(
                        contact["robot_obstacle_collision"][0]
                    )
                    current_robot_obstacle_collision = robot_obstacle_collision
                    safe_contact_ever = safe_contact_ever or safe
                    forbidden_contact_ever = forbidden_contact_ever or forbidden
                    protected_obstacle_collision_ever = (
                        protected_obstacle_collision_ever
                        or protected_obstacle_collision
                    )
                    robot_obstacle_collision_ever = (
                        robot_obstacle_collision_ever or robot_obstacle_collision
                    )
                    if safe and first_safe_contact_step is None:
                        first_safe_contact_step = sequence
                    if forbidden and first_forbidden_contact_step is None:
                        first_forbidden_contact_step = sequence
                    current_minimum_safe_distance = float(
                        contact["minimum_safe_distance"][0]
                    )
                    current_minimum_forbidden_distance = float(
                        contact["minimum_robot_forbidden_distance"][0]
                    )
                    current_protected_obstacle_clearance = float(
                        contact["protected_clearance"][0]
                    )
                    current_robot_obstacle_clearance = float(
                        contact["robot_obstacle_clearance"][0]
                    )
                    current_hand_obstacle_clearance = float(
                        contact["hand_obstacle_clearance"][0]
                    )
                    current_arm_obstacle_clearance = float(
                        contact["arm_obstacle_clearance"][0]
                    )
                    current_arm_obstacle_centerline_distance = float(
                        contact["arm_obstacle_centerline_distance"][0]
                    )
                    if math.isfinite(current_protected_obstacle_clearance):
                        current_c2_soft_penalty = min(1.0, max(
                            0.0,
                            (
                                args_cli.c2_soft_activation_distance_m
                                - current_protected_obstacle_clearance
                            ) / (
                                args_cli.c2_soft_activation_distance_m
                                - args_cli.protected_clearance_m
                            ),
                        ))
                    else:
                        current_c2_soft_penalty = 0.0
                    if math.isfinite(current_robot_obstacle_clearance):
                        current_c3_soft_penalty = min(1.0, max(
                            0.0,
                            (
                                args_cli.c3_soft_activation_distance_m
                                - current_robot_obstacle_clearance
                            ) / (
                                args_cli.c3_soft_activation_distance_m
                                - args_cli.robot_obstacle_clearance_m
                            ),
                        ))
                    else:
                        current_c3_soft_penalty = 0.0
                    safety_audit_count += 1
                    c2_soft_penalty_sum += current_c2_soft_penalty
                    c3_soft_penalty_sum += current_c3_soft_penalty
                    maximum_c2_soft_penalty = max(
                        maximum_c2_soft_penalty, current_c2_soft_penalty
                    )
                    maximum_c3_soft_penalty = max(
                        maximum_c3_soft_penalty, current_c3_soft_penalty
                    )
                    c2_collision_audit_count += int(
                        protected_obstacle_collision
                    )
                    c3_collision_audit_count += int(robot_obstacle_collision)
                    safe_points_e = contact["target_points"][0][
                        contact["safe_mask"][0]
                    ]
                    if safe_points_e.shape[0] > 0:
                        tip_e = planner_tip_position_tensor()
                        closest_safe_index = torch.argmin(
                            torch.linalg.vector_norm(
                                safe_points_e - tip_e.unsqueeze(0), dim=1
                            )
                        )
                        current_closest_safe_point_xy = (
                            safe_points_e[closest_safe_index, :2]
                            .detach().cpu().numpy()
                        )
                    else:
                        current_closest_safe_point_xy = None
                    current_robot_target_contact_forces = (
                        filtered_contact_force_summary((
                            robot_target_sensor_name,
                            hand_target_sensor_name,
                        ))
                    )
                    for sensor_name, force_n in (
                        current_robot_target_contact_forces.items()
                    ):
                        peak_robot_target_contact_forces[sensor_name] = max(
                            peak_robot_target_contact_forces.get(sensor_name, 0.0),
                            force_n,
                        )
                    current_robot_obstacle_contact_forces = (
                        filtered_contact_force_summary(
                            getattr(base.cfg, "robot_obstacle_sensor_name", None)
                        )
                        if multi_object_mode else {}
                    )
                    for sensor_name, force_n in (
                        current_robot_obstacle_contact_forces.items()
                    ):
                        peak_robot_obstacle_contact_forces[sensor_name] = max(
                            peak_robot_obstacle_contact_forces.get(sensor_name, 0.0),
                            force_n,
                        )
                    last_audit_step = sequence
                    minimum_safe_distance = min(
                        minimum_safe_distance, current_minimum_safe_distance
                    )
                    minimum_forbidden_distance = min(
                        minimum_forbidden_distance,
                        current_minimum_forbidden_distance,
                    )
                    minimum_protected_obstacle_clearance = min(
                        minimum_protected_obstacle_clearance,
                        current_protected_obstacle_clearance,
                    )
                    minimum_robot_obstacle_clearance = min(
                        minimum_robot_obstacle_clearance,
                        current_robot_obstacle_clearance,
                    )
                    minimum_hand_obstacle_clearance = min(
                        minimum_hand_obstacle_clearance,
                        current_hand_obstacle_clearance,
                    )
                    minimum_arm_obstacle_clearance = min(
                        minimum_arm_obstacle_clearance,
                        current_arm_obstacle_clearance,
                    )
                    minimum_arm_obstacle_centerline_distance = min(
                        minimum_arm_obstacle_centerline_distance,
                        current_arm_obstacle_centerline_distance,
                    )

                if args_cli.force_c3_on_legal_safe_contact:
                    force_c3_contact_unsafe = (
                        current_forbidden_contact
                        or current_protected_obstacle_collision
                        or current_robot_obstacle_collision
                        or execution_rotation_guard_active
                    )
                    if force_c3_contact_unsafe:
                        if force_c3_contact_active:
                            force_c3_contact_deactivation_count += 1
                        force_c3_contact_active = False
                        force_c3_contact_armed = False
                        force_c3_contact_start_step = None
                        force_c3_contact_best_planar_error_m = None
                        force_c3_contact_best_abs_yaw_error_rad = None
                        force_c3_contact_start_signed_yaw_error = None
                        force_c3_contact_is_yaw_recovery_pulse = False
                    elif force_c3_contact_active:
                        assert force_c3_contact_start_step is not None
                        assert force_c3_contact_best_planar_error_m is not None
                        force_c3_contact_best_planar_error_m = min(
                            force_c3_contact_best_planar_error_m, planar
                        )
                        force_c3_contact_timed_out = (
                            sequence - force_c3_contact_start_step
                        ) * control_period_s >= (
                            args_cli.force_c3_contact_max_duration_s
                        )
                        force_c3_contact_regressed = planar > (
                            force_c3_contact_best_planar_error_m
                            + args_cli.semantic_contact_planar_regression_tolerance_m
                        )
                        if (
                            force_c3_contact_timed_out
                            or force_c3_contact_regressed
                        ):
                            force_c3_contact_active = False
                            force_c3_contact_armed = False
                            force_c3_contact_start_step = None
                            force_c3_contact_best_planar_error_m = None
                            force_c3_contact_best_abs_yaw_error_rad = None
                            force_c3_contact_start_signed_yaw_error = None
                            force_c3_contact_is_yaw_recovery_pulse = False
                            force_c3_contact_deactivation_count += 1
                    elif current_legal_safe_contact and force_c3_contact_armed:
                        # Contact sensors commonly chatter for one or more
                        # high-rate servo steps at first touch.  Latch the
                        # authorization until its bounded timeout/regression
                        # condition instead of clearing it before the next
                        # lower-rate planner tick can observe FORCE_C3_MODE.
                        force_c3_contact_active = True
                        force_c3_contact_armed = False
                        force_c3_contact_start_step = sequence
                        force_c3_contact_best_planar_error_m = planar
                        force_c3_contact_best_abs_yaw_error_rad = abs(
                            signed_yaw_error
                        )
                        force_c3_contact_start_signed_yaw_error = (
                            signed_yaw_error
                        )
                        force_c3_contact_is_yaw_recovery_pulse = False
                        force_c3_contact_activation_count += 1
                    elif not current_legal_safe_contact:
                        force_c3_contact_armed = True
                        force_c3_contact_start_signed_yaw_error = None
                        force_c3_contact_is_yaw_recovery_pulse = False

                local_control_phase = (
                    "rotation_guard_retreat"
                    if execution_rotation_guard_active
                    else (
                        (
                            "force_c3_window_retreat"
                            if force_c3_window_retreat_active
                            else "semantic_yaw_brake_retreat"
                        )
                        if semantic_yaw_brake_active
                        else (
                            "safe_contact_acquisition"
                            if contact_acquisition_active
                            else (
                                "semantic_goal_push"
                                if semantic_contact_push_active
                                else "c3_task_tracking"
                            )
                        )
                    )
                )
                if semantic_guard_active:
                    local_control_phase = "semantic_c1_guard"
                # Retain every physical-contact and strict-pose measurement.
                # A successful dwell must be independently reconstructable
                # even after the fingers have left the target.
                if (sequence % args_cli.trace_stride == 0
                        or strict or current_legal_safe_contact or forbidden_contact_ever):
                    trace_command = active_command
                    trace.append({
                        "step": sequence,
                        "sim_time_s": sequence * control_period_s,
                        "task_reference_floor_c3_m": reference_floor,
                        # Measurements below are taken after env.step. Match
                        # the relay's 100 ms timestamp origin explicitly.
                        "measurement_utime_us": 100_000 + round(
                            (sequence + 1) * control_period_s * 1.0e6
                        ),
                        "tcp_position_m": tcp_position_list(),
                        "planner_tip_position_m": planner_tip_position_list(),
                        "semantic_c1_guard_active": semantic_guard_active,
                        "semantic_c1_guard_distance_m": semantic_guard_distance,
                        "semantic_c1_guard_activation_count": semantic_guard_activation_count,
                        "robot_root_pose_w": robot.data.root_state_w[0, :7].detach().cpu().tolist(),
                        "env_origin_w": base.scene.env_origins[0].detach().cpu().tolist(),
                        "finger_joint_position_m": robot.data.joint_pos[0, finger_joint_ids].detach().cpu().tolist(),
                        "contact_body_poses_w": {
                            name: robot.data.body_state_w[0, body_id, :7].detach().cpu().tolist()
                            for name, body_id in zip(contact_body_names, contact_body_ids)
                        },
                        "contact_audit_measurement_utime_us": contact_audit_measurement_utime_us,
                        "target_position_m": (
                            target.data.root_pos_w[0, :3]
                            - base.scene.env_origins[0]
                        ).detach().cpu().tolist(),
                        "target_com_position_m": (
                            target.data.root_com_pos_w[0]
                            - base.scene.env_origins[0]
                        ).detach().cpu().tolist(),
                        "target_quaternion_wxyz": (
                            target.data.root_quat_w[0].detach().cpu().tolist()
                        ),
                        "target_angular_velocity_rad_s": (
                            target.data.root_ang_vel_w[0].detach().cpu().tolist()
                        ),
                        "execution_rotation_risk_rad": execution_rotation_risk,
                        "execution_rotation_guard_active": (
                            execution_rotation_guard_active
                        ),
                        "clutter_positions_m": (
                            (
                                obstacles.data.object_pos_w[0, :clutter_count]
                                - base.scene.env_origins[0]
                            ).detach().cpu().tolist()
                            if multi_object_mode else []
                        ),
                        "clutter_quaternions_wxyz": (
                            obstacles.data.object_quat_w[0, :clutter_count]
                            .detach().cpu().tolist()
                            if multi_object_mode else []
                        ),
                        "task_position_c3_m": (
                            list(trace_command.position_m)
                            if trace_command is not None else None
                        ),
                        "osc_dynamics_audit": (
                            base.action_manager.get_term("arm_action").dynamics_audit_snapshot()
                            if args_cli.osc_dynamics_audit else None
                        ),
                        "raw_task_target_c3_m": raw_task_target.tolist(),
                        "governed_task_target_c3_m": (
                            governed_task_target.tolist()
                        ),
                        "task_velocity_m_s": (
                            list(trace_command.velocity_m_s)
                            if trace_command is not None else None
                        ),
                        "osc_reference_tip_velocity_m_s": (
                            reference_velocity.tolist()
                            if args_cli.osc_track_trajectory_velocity else None
                        ),
                        "feedforward_force_n": (
                            list(trace_command.feedforward_force_n)
                            if trace_command is not None else None
                        ),
                        "applied_feedforward_force_n": (
                            applied_feedforward_force.tolist()
                        ),
                        "position_feedforward_offset_m": (
                            position_feedforward_offset.tolist()
                        ),
                        "desired_joint_position_rad": (
                            desired_q[0].detach().cpu().tolist()
                            if args_cli.local_controller != "cartesian_impedance"
                            else None
                        ),
                        "rmpflow_nullspace_joint_target_rad": (
                            rmpflow_joint_target[0].detach().cpu().tolist()
                            if rmpflow_joint_target is not None else None
                        ),
                        "rmpflow_target_proxy_phase": (
                            None if rmpflow_full_target_enabled is None
                            else (
                                "full_target"
                                if rmpflow_full_target_enabled
                                else "protected_only"
                            )
                        ),
                        "local_control_phase": local_control_phase,
                        "contact_acquisition_active": (
                            contact_acquisition_active
                        ),
                        "contact_acquisition_commit_latched": (
                            contact_acquisition_commit_latched
                        ),
                        "contact_acquisition_travel_m": (
                            contact_acquisition_travel_m
                        ),
                        "contact_acquisition_action_axis_xy": (
                            contact_acquisition_start_action_axis_xy.tolist()
                            if contact_acquisition_start_action_axis_xy
                            is not None else None
                        ),
                        "force_c3_contact_active": force_c3_contact_active,
                        "force_c3_window_retreat_active": (
                            force_c3_window_retreat_active
                        ),
                        "force_c3_contact_window_expired": (
                            force_c3_contact_window_expired
                        ),
                        "force_c3_contact_yaw_recovery": (
                            force_c3_contact_is_yaw_recovery_pulse
                        ),
                        "measured_c3_yaw_brake_requested": (
                            measured_c3_yaw_brake_requested_now
                        ),
                        "force_c3_yaw_regression_brake_requested": (
                            force_c3_yaw_regression_brake_requested_now
                        ),
                        "force_c3_contact_best_abs_yaw_error_rad": (
                            force_c3_contact_best_abs_yaw_error_rad
                        ),
                        "measured_c3_yaw_projected_error_rad": (
                            measured_c3_yaw_projected_error
                        ),
                        "semantic_contact_action_axis_xy": (
                            semantic_contact_push_axis_xy.tolist()
                            if semantic_contact_push_axis_xy is not None
                            else None
                        ),
                        "semantic_contact_requested_moment_arm_m": (
                            semantic_contact_requested_moment_arm_m
                        ),
                        "semantic_contact_achieved_moment_arm_m": (
                            semantic_contact_achieved_moment_arm_m
                        ),
                        "semantic_yaw_brake_active": semantic_yaw_brake_active,
                        "semantic_yaw_brake_target_m": (
                            semantic_yaw_brake_target.tolist()
                            if semantic_yaw_brake_target is not None
                            else None
                        ),
                        "signed_yaw_error_rad": signed_yaw_error,
                        "projected_signed_yaw_error_rad": (
                            semantic_yaw_projected_error
                        ),
                        "semantic_yaw_predictive_brake": (
                            semantic_yaw_recovery_predictive_brake
                        ),
                        "semantic_yaw_recovery_reversing": (
                            semantic_yaw_recovery_reversing
                        ),
                        "closest_safe_point_xy_m": (
                            current_closest_safe_point_xy.tolist()
                            if current_closest_safe_point_xy is not None
                            else None
                        ),
                        "local_action": local_action[0].detach().cpu().tolist(),
                        "measured_joint_position_rad": robot.data.joint_pos[
                            0, joint_ids
                        ].detach().cpu().tolist(),
                        "task_tracking_error_m": current_task_tracking_error,
                        "audit_step": last_audit_step,
                        "safe_robot_contact": current_safe_contact,
                        "semantic_safe_robot_proximity": (
                            current_semantic_safe_proximity
                        ),
                        "legal_safe_robot_contact": current_legal_safe_contact,
                        "forbidden_robot_contact": current_forbidden_contact,
                        "protected_obstacle_collision": (
                            current_protected_obstacle_collision
                        ),
                        "robot_obstacle_collision": (
                            current_robot_obstacle_collision
                        ),
                        "c2_soft_penalty": current_c2_soft_penalty,
                        "c3_soft_penalty": current_c3_soft_penalty,
                        "minimum_safe_distance_m": (
                            current_minimum_safe_distance
                            if math.isfinite(current_minimum_safe_distance)
                            else None
                        ),
                        "minimum_robot_forbidden_distance_m": (
                            current_minimum_forbidden_distance
                            if math.isfinite(current_minimum_forbidden_distance)
                            else None
                        ),
                        "protected_obstacle_clearance_m": (
                            current_protected_obstacle_clearance
                            if math.isfinite(current_protected_obstacle_clearance)
                            else None
                        ),
                        "robot_obstacle_clearance_m": (
                            current_robot_obstacle_clearance
                            if math.isfinite(current_robot_obstacle_clearance)
                            else None
                        ),
                        "hand_obstacle_clearance_m": (
                            current_hand_obstacle_clearance
                            if math.isfinite(current_hand_obstacle_clearance)
                            else None
                        ),
                        "arm_obstacle_clearance_m": (
                            current_arm_obstacle_clearance
                            if math.isfinite(current_arm_obstacle_clearance)
                            else None
                        ),
                        "arm_obstacle_centerline_distance_m": (
                            current_arm_obstacle_centerline_distance
                            if math.isfinite(
                                current_arm_obstacle_centerline_distance
                            ) else None
                        ),
                        "robot_obstacle_contact_force_n_by_sensor": (
                            current_robot_obstacle_contact_forces
                        ),
                        "robot_target_contact_force_n_by_sensor": (
                            current_robot_target_contact_forces
                        ),
                        "flags": (
                            int(trace_command.flags)
                            if trace_command is not None else 0
                        ),
                        "c3_mode": active_command_is_c3_mode,
                            "yaw_recovery_mode": active_command_is_yaw_recovery,
                            "semantic_contact_yaw_recovery": (
                                semantic_contact_is_yaw_recovery_pulse
                            ),
                    })

                if forbidden_contact_ever:
                    stopped_reason = "oracle_c1_violation"
                    break
                if (
                    args_cli.hard_c2_termination
                    and protected_obstacle_collision_ever
                ):
                    stopped_reason = "oracle_c2_violation"
                    break
                if (
                    args_cli.hard_c3_termination
                    and robot_obstacle_collision_ever
                ):
                    stopped_reason = "oracle_c3_violation"
                    break
                startup_failed = first_ready_step is None and sequence + 1 >= startup_steps
                post_startup_watchdog = (
                    first_ready_step is not None
                    and consecutive_watchdog
                    >= args_cli.maximum_consecutive_watchdog_steps
                )
                if startup_failed or post_startup_watchdog:
                    stopped_reason = "watchdog_timeout"
                    break
                if strict_dwell >= dwell_steps and safe_contact_ever:
                    stopped_reason = "strict_pose_dwell_success"
                    break
                if sequence % max(1, round(1.0 / control_period_s)) == 0:
                    print(
                        "C3_ONLINE_TASK_PROGRESS",
                        f"sim_time_s={sequence * control_period_s:.2f}",
                        f"planar_m={planar:.4f}",
                        f"rotation_rad={rotation:.4f}",
                        f"tracking_m={current_task_tracking_error:.4f}",
                        f"tracking_max_m={maximum_task_tracking_error:.4f}",
                        f"safe={int(safe_contact_ever)}",
                        f"contact_now={int(current_legal_safe_contact)}",
                        f"c3_mode={int(active_command_is_c3_mode)}",
                        f"yaw_recovery={int(active_command_is_yaw_recovery)}",
                        f"phase={local_control_phase}",
                        f"tip_z_m={planner_tip_position_list()[2]:.4f}",
                        (
                            f"safe_distance_m={current_minimum_safe_distance:.4f}"
                            if math.isfinite(current_minimum_safe_distance)
                            else "safe_distance_m=inf"
                        ),
                        flush=True,
                    )

        final_planar, final_height, final_rotation = _pose_errors(base)
        final_target_pose = torch.cat(
            (target.data.root_pos_w[:, :3] - base.scene.env_origins,
             target.data.root_quat_w), dim=1
        )[0].detach().cpu().tolist()
        strict_geometry_final = (
            final_planar < args_cli.strict_position_threshold_m
            and final_height < args_cli.strict_height_threshold_m
            and final_rotation < args_cli.strict_rotation_threshold_rad
        )
        pose_c1_success = (
            stopped_reason == "strict_pose_dwell_success"
            and strict_geometry_final
            and safe_contact_ever
            and not forbidden_contact_ever
        )
        full_safe_success = (
            pose_c1_success
            and not protected_obstacle_collision_ever
            and not robot_obstacle_collision_ever
        )
        active_schedule_success = (
            pose_c1_success
            and (
                not args_cli.hard_c2_termination
                or not protected_obstacle_collision_ever
            )
            and (
                not args_cli.hard_c3_termination
                or not robot_obstacle_collision_ever
            )
        )
        result.update({
            "control_period_s": control_period_s,
            "planner_period_s": planner_period_s,
            "planner_frequency_hz": 1.0 / planner_period_s,
            "executed_steps": executed_steps,
            "semantic_c1_guard_activation_count": semantic_guard_activation_count,
            "executed_sim_time_s": executed_steps * control_period_s,
            "wall_time_s": time.monotonic() - loop_started,
            "stopped_reason": stopped_reason,
            "initial_target_pose_wxyz": initial_target_pose,
            "initial_target_com_pose_b_xyzw": initial_target_com_pose_b,
            "initial_target_com_position_m": initial_target_com_position,
            "initial_target_mass_kg": initial_target_mass,
            "initial_target_inertia_kg_m2": initial_target_inertia,
            "goal_pose_wxyz": goal_pose,
            "final_target_pose_wxyz": final_target_pose,
            "initial_tcp_position_m": initial_tcp_position,
            "final_tcp_position_m": tcp_position_list(),
            "initial_planner_tip_position_m": initial_planner_tip_position,
            "final_planner_tip_position_m": planner_tip_position_list(),
            "initial_model_planner_tip_position_m": (
                initial_model_planner_tip_position
            ),
            "initial_cross_model_tip_error_m": initial_cross_model_tip_error,
            "final_planar_error_m": final_planar,
            "final_height_error_m": final_height,
            "final_rotation_error_rad": final_rotation,
            "minimum_planar_error_m": minimum_planar,
            "minimum_height_error_m": minimum_height,
            "minimum_rotation_error_rad": minimum_rotation,
            "strict_dwell_required_steps": dwell_steps,
            "maximum_strict_dwell_steps": maximum_strict_dwell,
            "strict_geometry_final": strict_geometry_final,
            "safe_robot_contact_ever": safe_contact_ever,
            "safe_contact_acceptance": (
                "physical_hand_contact_and_semantic_safe_region"
            ),
            "forbidden_robot_contact_ever": forbidden_contact_ever,
            "protected_obstacle_collision_ever": (
                protected_obstacle_collision_ever
            ),
            "robot_obstacle_collision_ever": robot_obstacle_collision_ever,
            "safety_audit_count": safety_audit_count,
            "c2_collision_audit_fraction": (
                c2_collision_audit_count / safety_audit_count
                if safety_audit_count else 0.0
            ),
            "c3_collision_audit_fraction": (
                c3_collision_audit_count / safety_audit_count
                if safety_audit_count else 0.0
            ),
            "mean_c2_soft_penalty": (
                c2_soft_penalty_sum / safety_audit_count
                if safety_audit_count else 0.0
            ),
            "mean_c3_soft_penalty": (
                c3_soft_penalty_sum / safety_audit_count
                if safety_audit_count else 0.0
            ),
            "maximum_c2_soft_penalty": maximum_c2_soft_penalty,
            "maximum_c3_soft_penalty": maximum_c3_soft_penalty,
            "first_safe_contact_step": first_safe_contact_step,
            "first_forbidden_contact_step": first_forbidden_contact_step,
            "minimum_safe_distance_m": (
                minimum_safe_distance if math.isfinite(minimum_safe_distance) else None
            ),
            "minimum_robot_forbidden_distance_m": (
                minimum_forbidden_distance
                if math.isfinite(minimum_forbidden_distance) else None
            ),
            "minimum_protected_obstacle_clearance_m": (
                minimum_protected_obstacle_clearance
                if math.isfinite(minimum_protected_obstacle_clearance)
                else None
            ),
            "minimum_robot_obstacle_clearance_m": (
                minimum_robot_obstacle_clearance
                if math.isfinite(minimum_robot_obstacle_clearance)
                else None
            ),
            "minimum_hand_obstacle_clearance_m": (
                minimum_hand_obstacle_clearance
                if math.isfinite(minimum_hand_obstacle_clearance)
                else None
            ),
            "minimum_arm_obstacle_clearance_m": (
                minimum_arm_obstacle_clearance
                if math.isfinite(minimum_arm_obstacle_clearance)
                else None
            ),
            "minimum_arm_obstacle_centerline_distance_m": (
                minimum_arm_obstacle_centerline_distance
                if math.isfinite(minimum_arm_obstacle_centerline_distance)
                else None
            ),
            "peak_robot_obstacle_contact_force_n_by_sensor": (
                peak_robot_obstacle_contact_forces
            ),
            "peak_robot_target_contact_force_n_by_sensor": (
                peak_robot_target_contact_forces
            ),
            "contact_acquisition_activation_count": (
                contact_acquisition_activation_count
            ),
            "contact_acquisition_deactivation_count": (
                contact_acquisition_deactivation_count
            ),
            "first_contact_acquisition_step": first_contact_acquisition_step,
            "first_contact_acquisition_time_s": (
                first_contact_acquisition_step * control_period_s
                if first_contact_acquisition_step is not None else None
            ),
            "maximum_contact_acquisition_travel_m": (
                maximum_contact_acquisition_travel_m
            ),
            "semantic_contact_push_activation_step": (
                semantic_contact_push_activation_step
            ),
            "semantic_contact_push_activation_time_s": (
                semantic_contact_push_activation_step * control_period_s
                if semantic_contact_push_activation_step is not None else None
            ),
            "semantic_contact_push_deactivation_count": (
                semantic_contact_push_deactivation_count
            ),
            "force_c3_contact_activation_count": (
                force_c3_contact_activation_count
            ),
            "force_c3_contact_deactivation_count": (
                force_c3_contact_deactivation_count
            ),
            "measured_c3_yaw_brake_activation_count": (
                measured_c3_yaw_brake_activation_count
            ),
            "force_c3_yaw_regression_brake_activation_count": (
                force_c3_yaw_regression_brake_activation_count
            ),
            "first_force_c3_yaw_regression_brake_step": (
                first_force_c3_yaw_regression_brake_step
            ),
            "first_force_c3_yaw_regression_brake_time_s": (
                first_force_c3_yaw_regression_brake_step * control_period_s
                if first_force_c3_yaw_regression_brake_step is not None
                else None
            ),
            "first_measured_c3_yaw_brake_step": (
                first_measured_c3_yaw_brake_step
            ),
            "first_measured_c3_yaw_brake_time_s": (
                first_measured_c3_yaw_brake_step * control_period_s
                if first_measured_c3_yaw_brake_step is not None else None
            ),
            "force_c3_window_retreat_activation_count": (
                force_c3_window_retreat_activation_count
            ),
            "first_force_c3_window_retreat_step": (
                first_force_c3_window_retreat_step
            ),
            "first_force_c3_window_retreat_time_s": (
                first_force_c3_window_retreat_step * control_period_s
                if first_force_c3_window_retreat_step is not None else None
            ),
            "measured_yaw_recovery_inference_count": (
                measured_yaw_recovery_inference_count
            ),
            "semantic_yaw_brake_activation_count": (
                semantic_yaw_brake_activation_count
            ),
            "first_semantic_yaw_brake_step": first_semantic_yaw_brake_step,
            "first_semantic_yaw_brake_time_s": (
                first_semantic_yaw_brake_step * control_period_s
                if first_semantic_yaw_brake_step is not None else None
            ),
            "semantic_contact_standoff_latch_step": (
                semantic_contact_standoff_latch_step
            ),
            "semantic_contact_standoff_m": semantic_contact_standoff_m,
            "semantic_contact_minimum_standoff_m": (
                semantic_contact_minimum_standoff_m
            ),
            "maximum_semantic_contact_lateral_offset_m": (
                maximum_semantic_contact_lateral_offset_m
            ),
            "command_counts": counters,
            "first_ready_step": first_ready_step,
            "maximum_consecutive_watchdog_steps": maximum_consecutive_watchdog,
            "maximum_ik_position_error_m": maximum_ik_position_error,
            "maximum_task_tracking_error_m": maximum_task_tracking_error,
            "maximum_feedforward_force_n": maximum_feedforward_force,
            "maximum_applied_feedforward_force_n": (
                maximum_applied_feedforward_force
            ),
            "maximum_position_feedforward_offset_m": (
                maximum_position_feedforward_offset
            ),
            "maximum_abs_joint_effort_nm": maximum_abs_joint_effort,
            "task_target_governor_activation_count": (
                task_target_governor_activation_count
            ),
            "task_target_z_clamp_count": task_target_z_clamp_count,
            "maximum_raw_task_target_step_m": (
                maximum_raw_task_target_step_m
            ),
            "maximum_task_target_governor_correction_m": (
                maximum_task_target_governor_correction_m
            ),
            "execution_rotation_guard_activation_count": (
                execution_rotation_guard_activation_count
            ),
            "first_execution_rotation_guard_step": (
                first_execution_rotation_guard_step
            ),
            "first_execution_rotation_guard_time_s": (
                first_execution_rotation_guard_step * control_period_s
                if first_execution_rotation_guard_step is not None else None
            ),
            "maximum_execution_rotation_risk_rad": (
                maximum_execution_rotation_risk
            ),
            "rmpflow_collision_proxies": [
                {
                    "role": item["role"],
                    "index": item["index"],
                    "extent_m": item["extent_m"],
                }
                for item in rmpflow_proxies
            ],
            "rmpflow_internal_state_rollout": rmpflow is not None,
            "rmpflow_target_proxy_switch_step": (
                rmpflow_target_proxy_switch_step
            ),
            "rmpflow_target_proxy_switch_time_s": (
                rmpflow_target_proxy_switch_step * control_period_s
                if rmpflow_target_proxy_switch_step is not None else None
            ),
            "trace_stride": args_cli.trace_stride,
            "trace_sampling_policy": "stride_plus_all_physical_contact_steps",
            "audit_stride": args_cli.audit_stride,
            "drake_tip_from_hand_m": args_cli.drake_tip_from_hand_m,
            "video_capture_stride": video_stride if args_cli.video else None,
            "trace": trace,
            "pose_c1_success": pose_c1_success,
            "full_c1_c2_c3_success": full_safe_success,
            "online_closed_loop_success": active_schedule_success,
            "s2_single_scene_pass": active_schedule_success,
            "deployment_blocker": (
                "single_scene_full_safe_pass_requires_randomized_s2_s3_s4"
                if full_safe_success
                else (
                    "c1_hard_stage_pass_but_c2_or_c3_soft_violation_remains"
                    if active_schedule_success
                    else "online_isaac_task_closed_loop_not_yet_successful"
                )
            ),
        })
        print(
            "C3_ONLINE_TASK_RESULT",
            f"success={int(active_schedule_success)}",
            f"full_safe_success={int(full_safe_success)}",
            f"reason={stopped_reason}",
            f"planar_m={final_planar:.6f}",
            f"rotation_rad={final_rotation:.6f}",
            f"c1_pass={int(not forbidden_contact_ever)}",
            f"c2_pass={int(not protected_obstacle_collision_ever)}",
            f"c3_pass={int(not robot_obstacle_collision_ever)}",
            flush=True,
        )
        return 0 if active_schedule_success else 2
    except Exception as exc:
        result.update({
            "online_closed_loop_success": False,
            "s2_single_scene_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        print(f"C3_ONLINE_TASK_ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if video_writer is not None:
            try:
                result["video"] = video_writer.close()
            except Exception as exc:
                result["video_error"] = f"{type(exc).__name__}: {exc}"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"C3_ONLINE_TASK_OUTPUT {output}", flush=True)
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
