#!/usr/bin/env python3
"""Run the M1 oracle-affordance contact planner in closed loop.

M1 deliberately uses no learned policy and no RGB-D predictor.  Isaac Lab
supplies the target pose, goal pose, metric target point cloud, and oracle
``safe``/``protected`` scores.  The planner selects C1-legal handle contacts,
rejects kinematically unreachable or semantically unsafe joint paths, executes
one short push, observes the new object pose, and replans.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-AffordanceTeacher-FrozenV7-GoalWrench-C1-Franka-v0",
)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--max-replans", type=int, default=12)
parser.add_argument("--approach-steps", type=int, default=30)
parser.add_argument("--contact-steps", type=int, default=12)
parser.add_argument("--endpoint-hold-steps", type=int, default=25)
parser.add_argument("--servo-gain", type=float, default=3.0)
parser.add_argument("--joint-action-scale-rad", type=float, default=0.12)
parser.add_argument("--push-steps", type=int, default=18)
parser.add_argument("--retreat-steps", type=int, default=12)
parser.add_argument("--inter-push-settle-steps", type=int, default=3)
parser.add_argument("--final-hold-steps", type=int, default=5)
parser.add_argument("--dwell-steps", type=int, default=5)
parser.add_argument("--ik-max-evaluations", type=int, default=500)
parser.add_argument("--ik-position-tolerance-m", type=float, default=0.0015)
parser.add_argument("--ik-rotation-tolerance-rad", type=float, default=0.015)
parser.add_argument("--path-samples", type=int, default=11)
parser.add_argument("--output-candidates", type=int, default=16)
parser.add_argument("--push-direction-samples", type=int, default=7)
parser.add_argument("--push-direction-span-deg", type=float, default=60.0)
parser.add_argument("--hand-yaw-samples", type=int, default=5)
parser.add_argument("--hand-yaw-span-deg", type=float, default=60.0)
parser.add_argument("--contact-distance-m", type=float, default=0.010)
parser.add_argument("--gate-contact-distance-m", type=float, default=0.010)
parser.add_argument("--forbidden-clearance-m", type=float, default=0.020)
parser.add_argument("--approach-clearance-m", type=float, default=0.015)
parser.add_argument("--support-clearance-m", type=float, default=0.002)
parser.add_argument("--precontact-standoff-m", type=float, default=0.050)
parser.add_argument("--contact-penetration-m", type=float, default=0.0)
parser.add_argument("--minimum-push-distance-m", type=float, default=0.008)
parser.add_argument("--maximum-push-distance-m", type=float, default=0.015)
parser.add_argument("--push-distance-samples", type=int, default=1)
parser.add_argument("--translation-efficiency", type=float, default=0.35)
parser.add_argument("--rotation-efficiency", type=float, default=0.40)
parser.add_argument("--adaptive-dynamics-alpha", type=float, default=0.0)
parser.add_argument("--yaw-weight-m-per-rad", type=float, default=2.00)
parser.add_argument("--inside-yaw-weight-m-per-rad", type=float, default=0.35)
parser.add_argument("--predicted-yaw-guard-rad", type=float, default=0.09)
parser.add_argument("--yaw-guard-penalty-m-per-rad", type=float, default=10.0)
parser.add_argument(
    "--physics-rollout-candidates",
    type=int,
    default=0,
    help=(
        "Evaluate this many executable contacts through restored Isaac "
        "rollouts before each real push; zero preserves the M1 analytic ranker."
    ),
)
parser.add_argument("--rollout-predicate-violation-weight", type=float, default=1.0)
parser.add_argument("--rollout-mean-ratio-weight", type=float, default=0.25)
parser.add_argument("--rollout-minimum-cost-improvement", type=float, default=0.02)
parser.add_argument(
    "--rollout-lookahead-steps", type=int, choices=(1, 2), default=1
)
parser.add_argument("--rollout-lookahead-intermediate-weight", type=float, default=0.25)
parser.add_argument("--rollout-restore-position-tolerance-m", type=float, default=1e-5)
parser.add_argument("--rollout-restore-rotation-tolerance-rad", type=float, default=1e-4)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record the single-environment run with goal and semantic overlays.",
)
parser.add_argument(
    "--video-length",
    type=int,
    default=0,
    help="Recorded simulator steps; zero selects the full configured M1 horizon.",
)
parser.add_argument(
    "--video-folder",
    type=Path,
    default=Path("outputs/contact_planner_m1/videos"),
)
parser.add_argument("--video-name-prefix", default="contact_planner_m1")
parser.add_argument("--goal-ghost-opacity", type=float, default=0.68)
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=(1.15, 0.70, 0.82),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--camera-lookat",
    type=float,
    nargs=3,
    default=(0.48, 0.00, 0.06),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--video-ground-color",
    type=float,
    nargs=3,
    default=(0.08, 0.20, 0.32),
    metavar=("R", "G", "B"),
)
parser.add_argument("--video-dome-light-intensity", type=float, default=1200.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab
import numpy as np
import pinocchio as pin
import torch
from scipy.optimize import least_squares

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from dapl.contact_planner import (
    OracleContactPlannerConfig,
    OraclePlanningScene,
    OracleSafeContactPlanner,
    PhysicsRolloutScoringConfig,
    joint_threshold_cost,
    rank_physics_rollout_pairs,
    rank_physics_rollouts,
)
from dapl.contact_planner.isaac_visualization import (
    M1MarkerUpdateWrapper,
    create_m1_video_markers,
    show_selected_plan,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
    get_end_effector_pointcloud_in_env_frame,
    get_object_pointcloud_in_env_frame,
)


FRANKA_URDF = str(
    Path(isaaclab.__file__).resolve().parent
    / "controllers/config/data/lula_franka_gen.urdf"
)
PANDA_HAND_TO_TCP_M = 0.1034


@dataclass
class ExecutableCandidatePlan:
    """A semantic/IK-valid macro-action awaiting analytic or physics ranking."""

    candidate_rank: int
    q_precontact: np.ndarray
    q_contact: np.ndarray
    q_push: np.ndarray
    q_retreat: np.ndarray
    bridge: tuple[pin.SE3, np.ndarray, np.ndarray]
    raw_yaw_response: float
    push_direction_xy: np.ndarray
    push_distance_m: float
    diagnostic: dict[str, object]


def _never_terminate(env) -> torch.Tensor:
    """Keep task predicates observable without allowing automatic resets."""

    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


class FrankaEndpointIK:
    """Pinocchio TCP IK calibrated to the live Isaac Lab frame.

    The Lula URDF exposes ``panda_hand``, whereas Isaac Lab's first
    ``ee_frame`` target is the task TCP 0.1034 m along the hand's local Z
    axis.  The offset must be composed in forward kinematics so it rotates
    with the wrist.  Treating it as a one-time world translation produces an
    orientation-dependent position error of up to roughly 10 cm.
    """

    def __init__(self, max_evaluations: int) -> None:
        if not Path(FRANKA_URDF).is_file():
            raise FileNotFoundError(f"Franka IK URDF is missing: {FRANKA_URDF}")
        self.model = pin.buildModelFromUrdf(FRANKA_URDF)
        self.frame_id = self.model.getFrameId("panda_hand")
        self.hand_to_tcp = pin.SE3(
            np.eye(3), np.array((0.0, 0.0, PANDA_HAND_TO_TCP_M))
        )
        self.lower = self.model.lowerPositionLimit[:7] + 1.0e-4
        self.upper = self.model.upperPositionLimit[:7] - 1.0e-4
        self.max_evaluations = max_evaluations

    def forward(self, q_arm: np.ndarray) -> pin.SE3:
        data = self.model.createData()
        q_full = np.concatenate((q_arm, np.array((0.04, 0.04))))
        pin.forwardKinematics(self.model, data, q_full)
        pin.updateFramePlacements(self.model, data)
        return data.oMf[self.frame_id] * self.hand_to_tcp

    def bridge(
        self,
        q_reference: np.ndarray,
        tcp_position_env: np.ndarray,
        tcp_rotation_env: np.ndarray,
    ) -> tuple[pin.SE3, np.ndarray, np.ndarray]:
        pin_reference = self.forward(q_reference)
        env_from_pin_rotation = tcp_rotation_env @ pin_reference.rotation.T
        return pin_reference, env_from_pin_rotation, tcp_position_env

    def solve(
        self,
        *,
        seed: np.ndarray,
        regularization_reference: np.ndarray,
        bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
        desired_tcp_env: np.ndarray,
        desired_rotation_env: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        pin_reference, env_from_pin_rotation, tcp_reference_env = bridge
        desired_translation_pin = pin_reference.translation + (
            env_from_pin_rotation.T @ (desired_tcp_env - tcp_reference_env)
        )
        desired_rotation_pin = env_from_pin_rotation.T @ desired_rotation_env

        def residual(q_arm: np.ndarray) -> np.ndarray:
            pose = self.forward(q_arm)
            return np.concatenate(
                (
                    20.0 * (pose.translation - desired_translation_pin),
                    pin.log3(pose.rotation.T @ desired_rotation_pin),
                    5.0e-4 * (q_arm - regularization_reference),
                )
            )

        result = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            max_nfev=self.max_evaluations,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        solved = self.forward(result.x)
        position_error = float(
            np.linalg.norm(solved.translation - desired_translation_pin)
        )
        rotation_error = float(
            np.linalg.norm(pin.log3(solved.rotation.T @ desired_rotation_pin))
        )
        return result.x, position_error, rotation_error

    def tcp_pose_in_env(
        self,
        q_arm: np.ndarray,
        bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        pin_reference, env_from_pin_rotation, tcp_reference_env = bridge
        pose = self.forward(q_arm)
        position = tcp_reference_env + env_from_pin_rotation @ (
            pose.translation - pin_reference.translation
        )
        rotation = env_from_pin_rotation @ pose.rotation
        return position, rotation


def _minimum_point_distance(source: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return math.inf
    return float(
        np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2).min()
    )


def _joint_segment_semantic_clearance(
    ik: FrankaEndpointIK,
    *,
    q_start: np.ndarray,
    q_end: np.ndarray,
    bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
    hand_points_local: np.ndarray,
    target_points: np.ndarray,
    target_translation: np.ndarray,
    target_center: np.ndarray,
    target_yaw_delta: float,
    samples: int,
) -> float:
    minimum = math.inf
    for alpha in np.linspace(0.0, 1.0, samples):
        q_arm = (1.0 - alpha) * q_start + alpha * q_end
        tcp, rotation = ik.tcp_pose_in_env(q_arm, bridge)
        posed_hand = hand_points_local @ rotation.T + tcp
        angle = alpha * target_yaw_delta
        cosine, sine = math.cos(angle), math.sin(angle)
        yaw_rotation = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        translated_target = (
            (target_points - target_center) @ yaw_rotation.T
            + target_center
            + alpha * target_translation
        )
        minimum = min(
            minimum, _minimum_point_distance(posed_hand, translated_target)
        )
    return minimum


def _joint_segment_support_clearance(
    ik: FrankaEndpointIK,
    *,
    q_start: np.ndarray,
    q_end: np.ndarray,
    bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
    hand_points_local: np.ndarray,
    support_height: float,
    samples: int,
) -> float:
    """Return minimum hand height above the planar support along a joint path."""

    minimum = math.inf
    for alpha in np.linspace(0.0, 1.0, samples):
        q_arm = (1.0 - alpha) * q_start + alpha * q_end
        tcp, rotation = ik.tcp_pose_in_env(q_arm, bridge)
        posed_hand = hand_points_local @ rotation.T + tcp
        minimum = min(minimum, float(posed_hand[:, 2].min() - support_height))
    return minimum


def _preflight_candidate(
    ik: FrankaEndpointIK,
    *,
    q_current: np.ndarray,
    q_precontact: np.ndarray,
    q_contact: np.ndarray,
    q_push: np.ndarray,
    q_retreat: np.ndarray,
    bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
    hand_points_local: np.ndarray,
    target_points: np.ndarray,
    safe_points: np.ndarray,
    forbidden_points: np.ndarray,
    target_center: np.ndarray,
    push_translation: np.ndarray,
    push_yaw_delta: float,
    cfg: OracleContactPlannerConfig,
    samples: int,
) -> tuple[bool, dict[str, float]]:
    support_height = float(target_points[:, 2].min())
    approach_clearance = _joint_segment_semantic_clearance(
        ik,
        q_start=q_current,
        q_end=q_precontact,
        bridge=bridge,
        hand_points_local=hand_points_local,
        target_points=forbidden_points,
        target_translation=np.zeros(3),
        target_center=target_center,
        target_yaw_delta=0.0,
        samples=samples,
    )
    contact_forbidden_clearance = _joint_segment_semantic_clearance(
        ik,
        q_start=q_precontact,
        q_end=q_contact,
        bridge=bridge,
        hand_points_local=hand_points_local,
        target_points=forbidden_points,
        target_translation=np.zeros(3),
        target_center=target_center,
        target_yaw_delta=0.0,
        samples=samples,
    )
    push_forbidden_clearance = _joint_segment_semantic_clearance(
        ik,
        q_start=q_contact,
        q_end=q_push,
        bridge=bridge,
        hand_points_local=hand_points_local,
        target_points=forbidden_points,
        target_translation=cfg.translation_efficiency * push_translation,
        target_center=target_center,
        target_yaw_delta=push_yaw_delta,
        samples=samples,
    )
    approach_support_clearance = _joint_segment_support_clearance(
        ik,
        q_start=q_current,
        q_end=q_precontact,
        bridge=bridge,
        hand_points_local=hand_points_local,
        support_height=support_height,
        samples=samples,
    )
    contact_support_clearance = _joint_segment_support_clearance(
        ik,
        q_start=q_precontact,
        q_end=q_contact,
        bridge=bridge,
        hand_points_local=hand_points_local,
        support_height=support_height,
        samples=samples,
    )
    push_support_clearance = _joint_segment_support_clearance(
        ik,
        q_start=q_contact,
        q_end=q_push,
        bridge=bridge,
        hand_points_local=hand_points_local,
        support_height=support_height,
        samples=samples,
    )
    cosine, sine = math.cos(push_yaw_delta), math.sin(push_yaw_delta)
    final_yaw_rotation = np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    predicted_target_translation = cfg.translation_efficiency * push_translation
    final_forbidden_points = (
        (forbidden_points - target_center) @ final_yaw_rotation.T
        + target_center
        + predicted_target_translation
    )
    retreat_forbidden_clearance = _joint_segment_semantic_clearance(
        ik,
        q_start=q_push,
        q_end=q_retreat,
        bridge=bridge,
        hand_points_local=hand_points_local,
        target_points=final_forbidden_points,
        target_translation=np.zeros(3),
        target_center=target_center + predicted_target_translation,
        target_yaw_delta=0.0,
        samples=samples,
    )
    retreat_support_clearance = _joint_segment_support_clearance(
        ik,
        q_start=q_push,
        q_end=q_retreat,
        bridge=bridge,
        hand_points_local=hand_points_local,
        support_height=support_height,
        samples=samples,
    )
    contact_tcp, contact_rotation = ik.tcp_pose_in_env(q_contact, bridge)
    contact_hand = hand_points_local @ contact_rotation.T + contact_tcp
    final_safe_distance = _minimum_point_distance(contact_hand, safe_points)
    values = {
        "joint_approach_clearance_m": approach_clearance,
        "joint_contact_forbidden_clearance_m": contact_forbidden_clearance,
        "joint_push_forbidden_clearance_m": push_forbidden_clearance,
        "joint_contact_safe_distance_m": final_safe_distance,
        "joint_approach_support_clearance_m": approach_support_clearance,
        "joint_contact_support_clearance_m": contact_support_clearance,
        "joint_push_support_clearance_m": push_support_clearance,
        "joint_retreat_forbidden_clearance_m": retreat_forbidden_clearance,
        "joint_retreat_support_clearance_m": retreat_support_clearance,
    }
    valid = bool(
        approach_clearance > cfg.approach_clearance_m
        and contact_forbidden_clearance > cfg.forbidden_clearance_m
        and push_forbidden_clearance > cfg.forbidden_clearance_m
        and final_safe_distance <= cfg.contact_distance_m
        and approach_support_clearance > cfg.support_clearance_m
        and contact_support_clearance > cfg.support_clearance_m
        and push_support_clearance > cfg.support_clearance_m
        and retreat_forbidden_clearance > cfg.forbidden_clearance_m
        and retreat_support_clearance > cfg.support_clearance_m
    )
    return valid, values


def _quaternion_distance(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Sign-invariant quaternion geodesic without ``acos(1-eps)`` noise."""

    first = torch.nn.functional.normalize(first, dim=-1)
    second = torch.nn.functional.normalize(second, dim=-1)
    chord = torch.minimum(
        torch.linalg.vector_norm(first - second, dim=-1),
        torch.linalg.vector_norm(first + second, dim=-1),
    )
    return 2.0 * torch.asin(torch.clamp(0.5 * chord, max=1.0))


def _quaternion_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Hamilton product for Isaac Lab ``wxyz`` quaternions."""

    w1, x1, y1, z1 = first.unbind(dim=-1)
    w2, x2, y2, z2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _pose_errors(base) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = base.scene["target"]
    goal = base.command_manager.get_command("target_object_pose")
    position = target.data.root_pos_w[:, :3] - base.scene.env_origins
    delta = goal[:, :3] - position
    planar = torch.linalg.vector_norm(delta[:, :2], dim=1)
    height = torch.abs(delta[:, 2])
    rotation = _quaternion_distance(target.data.root_quat_w, goal[:, 3:7])
    return planar, height, rotation


def _finite_float_or_none(value: torch.Tensor) -> float | None:
    """Convert a scalar diagnostic to strict-JSON-compatible form."""

    result = float(value.item())
    return result if math.isfinite(result) else None


def _signed_yaw_error(base) -> torch.Tensor:
    target_rotation = matrix_from_quat(base.scene["target"].data.root_quat_w)
    goal_quaternion = base.command_manager.get_command("target_object_pose")[:, 3:7]
    goal_rotation = matrix_from_quat(goal_quaternion)
    relative = goal_rotation @ target_rotation.transpose(1, 2)
    return torch.atan2(relative[:, 1, 0], relative[:, 0, 0])


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.max_replans <= 0:
        raise ValueError("num-envs and max-replans must be positive")
    phase_steps = (
        args_cli.approach_steps,
        args_cli.contact_steps,
        args_cli.push_steps,
        args_cli.retreat_steps,
    )
    if min(phase_steps) <= 0:
        raise ValueError("all motion phases must contain at least one step")
    if args_cli.path_samples < 2 or args_cli.dwell_steps <= 0:
        raise ValueError("path-samples must be >=2 and dwell-steps positive")
    if args_cli.servo_gain < 1.0:
        raise ValueError("servo-gain must be at least 1.0")
    if args_cli.joint_action_scale_rad <= 0.0:
        raise ValueError("joint-action-scale-rad must be positive")
    if args_cli.gate_contact_distance_m < args_cli.contact_distance_m:
        raise ValueError(
            "gate-contact-distance-m must be at least contact-distance-m"
        )
    if not 0.0 <= args_cli.adaptive_dynamics_alpha <= 1.0:
        raise ValueError("adaptive-dynamics-alpha must be in [0, 1]")
    if args_cli.video and args_cli.num_envs != 1:
        raise ValueError("--video requires --num-envs 1")
    if args_cli.video_length < 0:
        raise ValueError("--video-length must be non-negative")
    if args_cli.physics_rollout_candidates < 0:
        raise ValueError("--physics-rollout-candidates must be non-negative")
    if args_cli.physics_rollout_candidates > args_cli.output_candidates:
        raise ValueError(
            "--physics-rollout-candidates cannot exceed --output-candidates"
        )
    if args_cli.physics_rollout_candidates > 0 and args_cli.video:
        raise ValueError(
            "physics rollout and video cannot share one environment; run the "
            "quantitative rollout evaluator first and replay selected actions later"
        )
    if (
        args_cli.rollout_restore_position_tolerance_m <= 0.0
        or args_cli.rollout_restore_rotation_tolerance_rad <= 0.0
    ):
        raise ValueError("rollout restore tolerances must be positive")
    if args_cli.rollout_lookahead_intermediate_weight < 0.0:
        raise ValueError("rollout lookahead intermediate weight must be non-negative")
    if (
        args_cli.inside_yaw_weight_m_per_rad < 0.0
        or args_cli.predicted_yaw_guard_rad <= 0.0
        or args_cli.yaw_guard_penalty_m_per_rad < 0.0
    ):
        raise ValueError("yaw guard weights must be non-negative and radius positive")

    planner_cfg = OracleContactPlannerConfig(
        contact_distance_m=args_cli.contact_distance_m,
        forbidden_clearance_m=args_cli.forbidden_clearance_m,
        approach_clearance_m=args_cli.approach_clearance_m,
        support_clearance_m=args_cli.support_clearance_m,
        precontact_standoff_m=args_cli.precontact_standoff_m,
        contact_penetration_m=args_cli.contact_penetration_m,
        minimum_push_distance_m=args_cli.minimum_push_distance_m,
        maximum_push_distance_m=args_cli.maximum_push_distance_m,
        push_distance_samples=args_cli.push_distance_samples,
        translation_efficiency=args_cli.translation_efficiency,
        rotation_efficiency=args_cli.rotation_efficiency,
        yaw_weight_m_per_rad=args_cli.yaw_weight_m_per_rad,
        path_samples=args_cli.path_samples,
        output_candidates=args_cli.output_candidates,
        push_direction_samples=args_cli.push_direction_samples,
        push_direction_span_deg=args_cli.push_direction_span_deg,
        hand_yaw_samples=args_cli.hand_yaw_samples,
        hand_yaw_span_deg=args_cli.hand_yaw_span_deg,
    )
    planner = OracleSafeContactPlanner(planner_cfg)
    rollout_scoring_cfg = PhysicsRolloutScoringConfig(
        predicate_violation_weight=(
            args_cli.rollout_predicate_violation_weight
        ),
        mean_ratio_weight=args_cli.rollout_mean_ratio_weight,
        minimum_cost_improvement=args_cli.rollout_minimum_cost_improvement,
    )
    ik = FrankaEndpointIK(args_cli.ik_max_evaluations)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    env_cfg.seed = args_cli.seed
    env_cfg.disable_obs_noise = True
    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
        env_cfg.viewer.origin_type = "world"
        # These are presentation-only changes.  The darker support and gentler
        # dome light prevent the white-on-white render seen in early demos.
        ground_cfg = getattr(env_cfg.scene, "ground", None)
        ground_spawn = getattr(ground_cfg, "spawn", None)
        ground_material = getattr(ground_spawn, "visual_material", None)
        if ground_material is not None:
            ground_material.diffuse_color = tuple(args_cli.video_ground_color)
            ground_material.roughness = 0.85
        light_cfg = getattr(env_cfg.scene, "light", None)
        light_spawn = getattr(light_cfg, "spawn", None)
        if light_spawn is not None and hasattr(light_spawn, "intensity"):
            light_spawn.intensity = args_cli.video_dome_light_intensity
        env_cfg.sim.render.enable_translucency = True
    # The RL task limits each relative joint target to 0.03 rad.  That small
    # target offset is appropriate for policy exploration but can saturate the
    # position actuator under gravity before an IK endpoint is reached.  M1's
    # deterministic joint servo uses a larger bounded offset while retaining
    # the same robot, actuator gains, and velocity limits.
    env_cfg.actions.arm_action.scale = args_cli.joint_action_scale_rad
    # M1 owns episode accounting.  Automatic success/C1 resets would replace
    # the object before we can attribute the exact planner trajectory.
    for term_name in (
        "time_out",
        "reached",
        "forbidden_region_contact",
        "object_dropped",
    ):
        term = getattr(env_cfg.terminations, term_name, None)
        if term is not None:
            term.func = _never_terminate
            term.params = {}
    env_cfg.episode_length_s = 120.0

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    video_markers = None
    if args_cli.video:
        video_markers = create_m1_video_markers(
            env, goal_ghost_opacity=args_cli.goal_ghost_opacity
        )
        env = M1MarkerUpdateWrapper(env, video_markers)
        configured_horizon = (
            args_cli.settle_steps
            + args_cli.max_replans
            * (
                args_cli.approach_steps
                + args_cli.endpoint_hold_steps
                + args_cli.contact_steps
                + args_cli.endpoint_hold_steps
                + args_cli.push_steps
                + args_cli.retreat_steps
                + args_cli.inter_push_settle_steps
            )
            + args_cli.final_hold_steps
        )
        video_length = args_cli.video_length or configured_horizon
        video_folder = args_cli.video_folder.expanduser().resolve()
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=args_cli.video_name_prefix,
            disable_logger=True,
        )
        print(
            "M1_VIDEO",
            f"folder={video_folder}",
            f"name_prefix={args_cli.video_name_prefix}",
            f"length={video_length}",
            flush=True,
        )
    try:
        env.reset()
        base = env.unwrapped
        robot = base.scene["robot"]
        robot_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        robot_cfg.resolve(base.scene)
        joint_ids = robot_cfg.joint_ids
        action_scale = float(base.cfg.actions.arm_action.scale)
        # Match the optimizer bounds to the actual action term.  The latter
        # clips targets to Isaac Lab's soft limits plus its configured safety
        # margin; using only the URDF hard limits can therefore yield an IK
        # endpoint that the controller can approach but never attain.
        action_limit_margin = float(
            getattr(base.cfg.actions.arm_action, "joint_limit_margin", 0.0)
        )
        soft_limits = (
            robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
        )
        ik.lower = np.maximum(
            ik.lower, soft_limits[:, 0] + action_limit_margin + 1.0e-4
        )
        ik.upper = np.minimum(
            ik.upper, soft_limits[:, 1] - action_limit_margin - 1.0e-4
        )
        if np.any(ik.lower >= ik.upper):
            raise RuntimeError("invalid Franka IK bounds after action-limit alignment")
        zero_action = torch.zeros((base.num_envs, 7), device=base.device)
        for _ in range(args_cli.settle_steps):
            env.step(zero_action)

        initial_target_position = (
            base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        ).clone()
        initial_goal = base.command_manager.get_command("target_object_pose").clone()
        initial_delta = initial_goal[:, :2] - initial_target_position[:, :2]
        initial_direction_deg = torch.rad2deg(
            torch.atan2(initial_delta[:, 1], initial_delta[:, 0])
        )
        initial_yaw_error = _signed_yaw_error(base).clone()

        safe_contact_ever = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        forbidden_contact_ever = torch.zeros_like(safe_contact_ever)
        forbidden_hand_contact_ever = torch.zeros_like(safe_contact_ever)
        arm_target_contact_ever = torch.zeros_like(safe_contact_ever)
        strict_pose_ever = torch.zeros_like(safe_contact_ever)
        freeze_active = torch.zeros_like(safe_contact_ever)
        frozen_joint_position = torch.zeros(
            (base.num_envs, 7), device=base.device, dtype=robot.data.joint_pos.dtype
        )
        success = torch.zeros_like(safe_contact_ever)
        planner_candidate_ever = torch.zeros_like(safe_contact_ever)
        ik_plan_ever = torch.zeros_like(safe_contact_ever)
        exhausted = torch.zeros_like(safe_contact_ever)
        dwell = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
        selected_pushes = torch.zeros_like(dwell)
        planned_contact_attempts = torch.zeros_like(dwell)
        adaptive_translation_gain = torch.full(
            (base.num_envs,),
            planner_cfg.translation_efficiency,
            device=base.device,
        )
        adaptive_rotation_gain = torch.full(
            (base.num_envs,),
            planner_cfg.rotation_efficiency,
            device=base.device,
        )
        minimum_planar_error = torch.full(
            (base.num_envs,), torch.inf, device=base.device
        )
        minimum_height_error = torch.full_like(minimum_planar_error, torch.inf)
        minimum_rotation_error = torch.full_like(minimum_planar_error, torch.inf)
        minimum_planar_while_rotation_valid = torch.full_like(
            minimum_planar_error, torch.inf
        )
        minimum_rotation_while_planar_valid = torch.full_like(
            minimum_planar_error, torch.inf
        )
        minimum_safe_distance = torch.full_like(minimum_planar_error, torch.inf)
        rollout_candidate_evaluations = torch.zeros_like(dwell)
        rollout_legal_evaluations = torch.zeros_like(dwell)
        rollout_selected_actions = torch.zeros_like(dwell)
        rollout_transition_position_error_sum = torch.zeros_like(
            minimum_planar_error
        )
        rollout_transition_rotation_error_sum = torch.zeros_like(
            minimum_planar_error
        )
        rollout_transition_comparisons = torch.zeros_like(dwell)
        plan_diagnostics: list[list[dict[str, object]]] = [
            [] for _ in range(base.num_envs)
        ]
        global_step = 0

        def update_metrics() -> None:
            nonlocal global_step
            planar, height, rotation = _pose_errors(base)
            contact = mdp.domino_affordance_contact_state(
                base,
                contact_distance_m=args_cli.contact_distance_m,
                evaluate_protected=False,
            )
            safe_now = contact["safe_robot_contact"]
            forbidden_now = contact["forbidden_robot_contact"]
            safe_contact_ever.logical_or_(safe_now)
            forbidden_contact_ever.logical_or_(forbidden_now)
            forbidden_hand_contact_ever.logical_or_(
                contact["forbidden_hand_contact"]
            )
            arm_target_contact_ever.logical_or_(
                contact["arm_target_physical_contact"]
            )
            strict_now = (planar < 0.02) & (height < 0.01) & (rotation < 0.10)
            strict_pose_ever.logical_or_(strict_now)
            newly_strict = strict_now & ~freeze_active
            frozen_joint_position[newly_strict] = robot.data.joint_pos[
                newly_strict
            ][:, joint_ids]
            freeze_active.logical_or_(strict_now)
            dwell.copy_(torch.where(strict_now, dwell + 1, torch.zeros_like(dwell)))
            success.logical_or_(
                (dwell >= args_cli.dwell_steps)
                & safe_contact_ever
                & ~forbidden_contact_ever
            )
            minimum_planar_error.copy_(torch.minimum(minimum_planar_error, planar))
            minimum_height_error.copy_(torch.minimum(minimum_height_error, height))
            minimum_rotation_error.copy_(torch.minimum(minimum_rotation_error, rotation))
            minimum_planar_while_rotation_valid.copy_(
                torch.minimum(
                    minimum_planar_while_rotation_valid,
                    torch.where(
                        rotation < 0.10,
                        planar,
                        torch.full_like(planar, torch.inf),
                    ),
                )
            )
            minimum_rotation_while_planar_valid.copy_(
                torch.minimum(
                    minimum_rotation_while_planar_valid,
                    torch.where(
                        planar < 0.02,
                        rotation,
                        torch.full_like(rotation, torch.inf),
                    ),
                )
            )
            minimum_safe_distance.copy_(
                torch.minimum(minimum_safe_distance, contact["minimum_safe_distance"])
            )
            global_step += 1

        def execute_phase(
            q_start: torch.Tensor,
            q_end: torch.Tensor,
            steps: int,
            plan_enabled: torch.Tensor,
            endpoint_hold_steps: int = 0,
        ) -> None:
            for phase_step in range(steps):
                alpha = (phase_step + 1) / steps
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                desired = q_start + alpha * (q_end - q_start)
                current = robot.data.joint_pos[:, joint_ids]
                moving = (
                    plan_enabled
                    & ~success
                    & ~forbidden_contact_ever
                    & ~freeze_active
                )
                desired = torch.where(moving[:, None], desired, current)
                desired = torch.where(
                    freeze_active[:, None], frozen_joint_position, desired
                )
                action = torch.clamp(
                    args_cli.servo_gain * (desired - current) / action_scale,
                    min=-1.0,
                    max=1.0,
                )
                env.step(action)
                update_metrics()
            for _ in range(endpoint_hold_steps):
                current = robot.data.joint_pos[:, joint_ids]
                moving = (
                    plan_enabled
                    & ~success
                    & ~forbidden_contact_ever
                    & ~freeze_active
                )
                desired = torch.where(moving[:, None], q_end, current)
                desired = torch.where(
                    freeze_active[:, None], frozen_joint_position, desired
                )
                action = torch.clamp(
                    args_cli.servo_gain * (desired - current) / action_scale,
                    min=-1.0,
                    max=1.0,
                )
                env.step(action)
                update_metrics()

        def execute_until_safe_contact(
            q_start: torch.Tensor,
            q_end: torch.Tensor,
            steps: int,
            plan_enabled: torch.Tensor,
            maximum_hold_steps: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Close slowly and latch the first legal C1 contact per scene."""

            contact_latched = torch.zeros_like(plan_enabled)
            q_at_contact = robot.data.joint_pos[:, joint_ids].clone()
            safe_distance_at_gate = torch.full(
                (base.num_envs,), torch.inf, device=base.device
            )
            forbidden_distance_at_gate = torch.full_like(
                safe_distance_at_gate, torch.inf
            )
            total_steps = steps + maximum_hold_steps
            for contact_step in range(total_steps):
                if contact_step < steps:
                    alpha = (contact_step + 1) / steps
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    desired_endpoint = q_start + alpha * (q_end - q_start)
                else:
                    desired_endpoint = q_end
                current = robot.data.joint_pos[:, joint_ids]
                moving = (
                    plan_enabled
                    & ~contact_latched
                    & ~success
                    & ~forbidden_contact_ever
                    & ~freeze_active
                )
                desired = torch.where(moving[:, None], desired_endpoint, current)
                desired = torch.where(
                    freeze_active[:, None], frozen_joint_position, desired
                )
                action = torch.clamp(
                    args_cli.servo_gain * (desired - current) / action_scale,
                    min=-1.0,
                    max=1.0,
                )
                env.step(action)
                update_metrics()
                state = mdp.domino_affordance_contact_state(
                    base,
                    contact_distance_m=args_cli.gate_contact_distance_m,
                    evaluate_protected=False,
                )
                legal_now = (
                    plan_enabled
                    & state["legal_safe_robot_contact"]
                    & ~forbidden_contact_ever
                )
                newly_latched = legal_now & ~contact_latched
                if bool(newly_latched.any()):
                    q_at_contact[newly_latched] = robot.data.joint_pos[
                        newly_latched
                    ][:, joint_ids]
                    safe_distance_at_gate[newly_latched] = state[
                        "minimum_safe_distance"
                    ][newly_latched]
                    forbidden_distance_at_gate[newly_latched] = state[
                        "minimum_robot_forbidden_distance"
                    ][newly_latched]
                    contact_latched.logical_or_(newly_latched)
                finished = (
                    ~plan_enabled
                    | contact_latched
                    | forbidden_contact_ever
                    | success
                    | freeze_active
                )
                if bool(finished.all()):
                    break

            final_state = mdp.domino_affordance_contact_state(
                base,
                contact_distance_m=args_cli.gate_contact_distance_m,
                evaluate_protected=False,
            )
            missed = plan_enabled & ~contact_latched
            safe_distance_at_gate[missed] = final_state["minimum_safe_distance"][
                missed
            ]
            forbidden_distance_at_gate[missed] = final_state[
                "minimum_robot_forbidden_distance"
            ][missed]
            q_at_contact[missed] = robot.data.joint_pos[missed][:, joint_ids]
            return (
                contact_latched,
                q_at_contact,
                safe_distance_at_gate,
                forbidden_distance_at_gate,
            )

        def restore_rollout_snapshot(
            snapshot: dict,
            expected_joint_position: torch.Tensor,
            expected_target_pose: torch.Tensor,
        ) -> tuple[float, float, float]:
            """Restore simulator state and verify that the reset is lossless."""

            base.scene.reset()
            base.scene.reset_to(snapshot, is_relative=False)
            base.sim.forward()
            base.scene.update(0.0)
            # ``InteractiveScene`` snapshots do not include manager-owned
            # controller buffers.  The latched relative action term must be
            # synchronized with the restored joints before another rollout.
            base.action_manager.reset()
            restored_joint_position = robot.data.joint_pos[:, joint_ids]
            restored_target_pose = base.scene["target"].data.root_pose_w
            joint_error = float(
                torch.max(
                    torch.abs(restored_joint_position - expected_joint_position)
                ).item()
            )
            position_error = float(
                torch.max(
                    torch.linalg.vector_norm(
                        restored_target_pose[:, :3] - expected_target_pose[:, :3],
                        dim=1,
                    )
                ).item()
            )
            rotation_error = float(
                torch.max(
                    _quaternion_distance(
                        restored_target_pose[:, 3:7],
                        expected_target_pose[:, 3:7],
                    )
                ).item()
            )
            if (
                joint_error > args_cli.rollout_restore_position_tolerance_m
                or position_error > args_cli.rollout_restore_position_tolerance_m
                or rotation_error
                > args_cli.rollout_restore_rotation_tolerance_rad
            ):
                raise RuntimeError(
                    "physics rollout restore exceeded tolerance: "
                    f"joint={joint_error:.3e} rad, "
                    f"position={position_error:.3e} m, "
                    f"rotation={rotation_error:.3e} rad"
                )
            return joint_error, position_error, rotation_error

        def rollout_contact_state() -> dict[str, torch.Tensor]:
            return mdp.domino_affordance_contact_state(
                base,
                contact_distance_m=args_cli.contact_distance_m,
                evaluate_protected=False,
            )

        def execute_rollout_phase(
            q_start: torch.Tensor,
            q_end: torch.Tensor,
            steps: int,
            enabled: torch.Tensor,
            c1_violation: torch.Tensor,
            *,
            endpoint_hold_steps: int = 0,
        ) -> None:
            """Execute a shadow phase without touching official M1 metrics."""

            for phase_step in range(steps + endpoint_hold_steps):
                if phase_step < steps:
                    alpha = (phase_step + 1) / steps
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    desired_endpoint = q_start + alpha * (q_end - q_start)
                else:
                    desired_endpoint = q_end
                current = robot.data.joint_pos[:, joint_ids]
                moving = enabled & ~c1_violation
                desired = torch.where(moving[:, None], desired_endpoint, current)
                action = torch.clamp(
                    args_cli.servo_gain * (desired - current) / action_scale,
                    min=-1.0,
                    max=1.0,
                )
                env.step(action)
                state = rollout_contact_state()
                c1_violation.logical_or_(
                    enabled & state["forbidden_robot_contact"]
                )

        def execute_rollout_until_contact(
            q_start: torch.Tensor,
            q_end: torch.Tensor,
            enabled: torch.Tensor,
            c1_violation: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Latch the first strict legal contact during a shadow rollout."""

            contact_latched = torch.zeros_like(enabled)
            q_at_contact = robot.data.joint_pos[:, joint_ids].clone()
            total_steps = args_cli.contact_steps + args_cli.endpoint_hold_steps
            for contact_step in range(total_steps):
                if contact_step < args_cli.contact_steps:
                    alpha = (contact_step + 1) / args_cli.contact_steps
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    desired_endpoint = q_start + alpha * (q_end - q_start)
                else:
                    desired_endpoint = q_end
                current = robot.data.joint_pos[:, joint_ids]
                moving = enabled & ~contact_latched & ~c1_violation
                desired = torch.where(moving[:, None], desired_endpoint, current)
                action = torch.clamp(
                    args_cli.servo_gain * (desired - current) / action_scale,
                    min=-1.0,
                    max=1.0,
                )
                env.step(action)
                state = rollout_contact_state()
                c1_violation.logical_or_(
                    enabled & state["forbidden_robot_contact"]
                )
                legal_now = (
                    enabled
                    & state["legal_safe_robot_contact"]
                    & ~c1_violation
                )
                newly_latched = legal_now & ~contact_latched
                if bool(newly_latched.any()):
                    q_at_contact[newly_latched] = robot.data.joint_pos[
                        newly_latched
                    ][:, joint_ids]
                    contact_latched.logical_or_(newly_latched)
                if bool(
                    (
                        ~enabled
                        | contact_latched
                        | c1_violation
                    ).all()
                ):
                    break
            return contact_latched, q_at_contact

        def execute_physics_rollout(
            *,
            q_start: torch.Tensor,
            q_precontact: torch.Tensor,
            q_contact: torch.Tensor,
            q_push: torch.Tensor,
            q_retreat: torch.Tensor,
            enabled: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """Run one candidate macro-action and measure its true outcome."""

            c1_violation = torch.zeros_like(enabled)
            execute_rollout_phase(
                q_start,
                q_precontact,
                args_cli.approach_steps,
                enabled,
                c1_violation,
                endpoint_hold_steps=args_cli.endpoint_hold_steps,
            )
            contact_ready, q_contact_actual = execute_rollout_until_contact(
                q_precontact,
                q_contact,
                enabled,
                c1_violation,
            )
            execute_rollout_phase(
                q_contact_actual,
                q_push,
                args_cli.push_steps,
                contact_ready,
                c1_violation,
            )
            retreat_start = torch.where(
                contact_ready[:, None], q_push, q_contact_actual
            )
            retreat_target = torch.where(
                contact_ready[:, None], q_retreat, q_precontact
            )
            execute_rollout_phase(
                retreat_start,
                retreat_target,
                args_cli.retreat_steps,
                enabled,
                c1_violation,
            )
            for _ in range(args_cli.inter_push_settle_steps):
                env.step(zero_action)
                state = rollout_contact_state()
                c1_violation.logical_or_(
                    enabled & state["forbidden_robot_contact"]
                )
            planar, height, rotation = _pose_errors(base)
            target_pose = torch.cat(
                (
                    base.scene["target"].data.root_pos_w[:, :3]
                    - base.scene.env_origins,
                    base.scene["target"].data.root_quat_w,
                ),
                dim=1,
            ).clone()
            return {
                "contact_ready": contact_ready,
                "c1_violation": c1_violation,
                "planar_error": planar.clone(),
                "height_error": height.clone(),
                "rotation_error": rotation.clone(),
                "target_pose": target_pose,
            }

        for replan_index in range(args_cli.max_replans):
            active = ~success & ~forbidden_contact_ever & ~exhausted
            if not bool(active.any()):
                break

            target_cfg = SceneEntityCfg("target")
            target_points = get_object_pointcloud_in_env_frame(
                base, target_cfg
            ).reshape(base.num_envs, -1, 3)
            affordance = mdp.domino_target_affordance(base, target_cfg).reshape(
                base.num_envs, -1, 2
            )
            hand_points = get_end_effector_pointcloud_in_env_frame(base)
            tcp_position = (
                base.scene["ee_frame"].data.target_pos_w[:, 0]
                - base.scene.env_origins
            )
            tcp_quaternion = base.scene["ee_frame"].data.target_quat_w[:, 0]
            local_hand = quat_apply_inverse(
                tcp_quaternion[:, None, :]
                .expand(-1, hand_points.shape[1], -1)
                .reshape(-1, 4),
                (hand_points - tcp_position[:, None, :]).reshape(-1, 3),
            ).reshape_as(hand_points)
            target_position = (
                base.scene["target"].data.root_pos_w[:, :3]
                - base.scene.env_origins
            )
            goal = base.command_manager.get_command("target_object_pose")
            yaw_error = _signed_yaw_error(base)
            candidates = planner.plan(
                OraclePlanningScene(
                    target_points=target_points,
                    safe_scores=affordance[..., 0],
                    protected_scores=affordance[..., 1],
                    target_position=target_position,
                    goal_position=goal[:, :3],
                    tcp_position=tcp_position,
                    hand_points_local=local_hand,
                    tcp_rotation=matrix_from_quat(tcp_quaternion),
                    yaw_error=yaw_error,
                )
            )
            planner_candidate_ever.logical_or_(candidates.any_valid & active)

            current_q = robot.data.joint_pos[:, joint_ids].clone()
            q_pre = current_q.clone()
            q_contact = current_q.clone()
            q_push = current_q.clone()
            q_retreat = current_q.clone()
            plan_enabled = torch.zeros_like(active)
            selected_rank = torch.full_like(dwell, -1)
            selected_raw_yaw_response = torch.zeros(
                base.num_envs, device=base.device
            )
            selected_push_direction = torch.zeros(
                (base.num_envs, 2), device=base.device
            )
            selected_push_distance = torch.zeros(
                base.num_envs, device=base.device
            )
            bridges: list[tuple[pin.SE3, np.ndarray, np.ndarray] | None] = [
                None for _ in range(base.num_envs)
            ]
            executable_options: list[list[ExecutableCandidatePlan]] = [
                [] for _ in range(base.num_envs)
            ]
            selected_rollout_target_pose = torch.full(
                (base.num_envs, 7),
                torch.nan,
                device=base.device,
                dtype=target_position.dtype,
            )
            selected_rollout_score = torch.full(
                (base.num_envs,), torch.inf, device=base.device
            )
            option_limit = max(args_cli.physics_rollout_candidates, 1)

            for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                q_current_np = current_q[env_id].detach().cpu().numpy()
                tcp_np = tcp_position[env_id].detach().cpu().numpy()
                tcp_rotation_np = (
                    matrix_from_quat(tcp_quaternion[env_id])
                    .detach()
                    .cpu()
                    .numpy()
                )
                bridge = ik.bridge(q_current_np, tcp_np, tcp_rotation_np)
                safe_mask = (
                    (affordance[env_id, :, 0] >= planner_cfg.safe_threshold)
                    & (
                        affordance[env_id, :, 1]
                        < planner_cfg.protected_threshold
                    )
                )
                points_np = target_points[env_id].detach().cpu().numpy()
                safe_np = target_points[env_id, safe_mask].detach().cpu().numpy()
                forbidden_np = (
                    target_points[env_id, ~safe_mask].detach().cpu().numpy()
                )
                target_center_np = target_position[env_id].detach().cpu().numpy()
                centered_xy = points_np[:, :2] - target_center_np[:2]
                planar_gyration_sq = max(
                    float(np.mean(np.sum(np.square(centered_xy), axis=1))),
                    1.0e-4,
                )
                hand_local_np = local_hand[env_id].detach().cpu().numpy()
                candidate_attempts = 0
                adaptive_scores: dict[int, float] = {}
                for candidate_rank in range(candidates.valid.shape[1]):
                    if not bool(candidates.valid[env_id, candidate_rank]):
                        continue
                    direction_xy = (
                        candidates.push_direction[env_id, candidate_rank, :2]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    distance = float(
                        candidates.push_distance[env_id, candidate_rank].item()
                    )
                    raw_yaw_response = float(
                        distance
                        * candidates.contact_moment_arm[
                            env_id, candidate_rank
                        ].item()
                        / planar_gyration_sq
                    )
                    predicted_translation = (
                        float(adaptive_translation_gain[env_id].item())
                        * distance
                        * direction_xy
                    )
                    predicted_yaw = (
                        float(adaptive_rotation_gain[env_id].item())
                        * raw_yaw_response
                    )
                    predicted_yaw_error = abs(
                        float(yaw_error[env_id].item()) - predicted_yaw
                    )
                    yaw_is_inside = (
                        abs(float(yaw_error[env_id].item())) < 0.10
                    )
                    yaw_cost_weight = (
                        args_cli.inside_yaw_weight_m_per_rad
                        if yaw_is_inside
                        else planner_cfg.yaw_weight_m_per_rad
                    )
                    yaw_guard_cost = (
                        args_cli.yaw_guard_penalty_m_per_rad
                        * max(
                            0.0,
                            predicted_yaw_error
                            - args_cli.predicted_yaw_guard_rad,
                        )
                        if yaw_is_inside
                        else 0.0
                    )
                    residual_xy = (
                        goal[env_id, :2] - target_position[env_id, :2]
                    ).detach().cpu().numpy() - predicted_translation
                    adaptive_scores[candidate_rank] = float(
                        np.linalg.norm(residual_xy)
                        + yaw_cost_weight * predicted_yaw_error
                        + yaw_guard_cost
                        + 0.05
                        * float(candidates.score[env_id, candidate_rank].item())
                    )
                analytic_rank_order = sorted(
                    adaptive_scores, key=adaptive_scores.get
                )
                if args_cli.physics_rollout_candidates > 0:
                    moment_representatives: list[int] = []
                    for moment_bin in (-1, 0, 1):
                        members: list[int] = []
                        for rank in analytic_rank_order:
                            moment = float(
                                candidates.contact_moment_arm[
                                    env_id, rank
                                ].item()
                            )
                            candidate_bin = (
                                -1
                                if moment
                                < -planner_cfg.moment_arm_neutral_band_m
                                else (
                                    1
                                    if moment
                                    > planner_cfg.moment_arm_neutral_band_m
                                    else 0
                                )
                            )
                            if candidate_bin == moment_bin:
                                members.append(rank)
                        if members:
                            moment_representatives.append(
                                min(members, key=adaptive_scores.get)
                            )
                    moment_representatives.sort(key=adaptive_scores.get)
                    distance_representatives: list[int] = []
                    seen_distances: set[float] = set()
                    for rank in analytic_rank_order:
                        distance_key = round(
                            float(candidates.push_distance[env_id, rank].item()),
                            6,
                        )
                        if distance_key in seen_distances:
                            continue
                        seen_distances.add(distance_key)
                        distance_representatives.append(rank)
                    representatives = list(
                        dict.fromkeys(
                            moment_representatives + distance_representatives
                        )
                    )
                    representatives.sort(key=adaptive_scores.get)
                    representative_set = set(representatives)
                    rank_order = representatives + [
                        rank
                        for rank in analytic_rank_order
                        if rank not in representative_set
                    ]
                else:
                    rank_order = analytic_rank_order

                for rank in rank_order:
                    if not bool(candidates.valid[env_id, rank]):
                        continue
                    candidate_attempts += 1
                    rotation_np = (
                        candidates.hand_rotation[env_id, rank]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    endpoints = (
                        candidates.precontact_tcp[env_id, rank],
                        candidates.contact_tcp[env_id, rank],
                        candidates.push_tcp[env_id, rank],
                        candidates.push_tcp[env_id, rank]
                        - planner_cfg.precontact_standoff_m
                        * candidates.approach_direction[env_id, rank],
                    )
                    solved: list[np.ndarray] = []
                    errors: list[tuple[float, float]] = []
                    seed = q_current_np
                    ik_valid = True
                    for endpoint in endpoints:
                        q_solution, position_error, rotation_error = ik.solve(
                            seed=seed,
                            regularization_reference=q_current_np,
                            bridge=bridge,
                            desired_tcp_env=endpoint.detach().cpu().numpy(),
                            desired_rotation_env=rotation_np,
                        )
                        solved.append(q_solution)
                        errors.append((position_error, rotation_error))
                        seed = q_solution
                        if (
                            position_error > args_cli.ik_position_tolerance_m
                            or rotation_error > args_cli.ik_rotation_tolerance_rad
                        ):
                            ik_valid = False
                            break
                    diagnostic: dict[str, object] = {
                        "replan": replan_index,
                        "rank": rank,
                        "planner_score": float(
                            candidates.score[env_id, rank].item()
                        ),
                        "yaw_error_before_rad": float(yaw_error[env_id].item()),
                        "contact_moment_arm_m": float(
                            candidates.contact_moment_arm[env_id, rank].item()
                        ),
                        "push_direction": (
                            candidates.push_direction[env_id, rank]
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                        "approach_direction": (
                            candidates.approach_direction[env_id, rank]
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                        "hand_yaw_offset_rad": float(
                            candidates.hand_yaw_offset[env_id, rank].item()
                        ),
                        "push_distance_m": float(
                            candidates.push_distance[env_id, rank].item()
                        ),
                        "predicted_yaw_error_rad": float(
                            candidates.predicted_yaw_error[env_id, rank].item()
                        ),
                        "ik_errors": errors,
                        "ik_valid": ik_valid,
                    }
                    if not ik_valid:
                        plan_diagnostics[env_id].append(diagnostic)
                        continue

                    push_translation = (
                        candidates.push_tcp[env_id, rank]
                        - candidates.contact_tcp[env_id, rank]
                    ).detach().cpu().numpy()
                    push_yaw_delta = float(
                        planner_cfg.rotation_efficiency
                        * candidates.push_distance[env_id, rank].item()
                        * candidates.contact_moment_arm[env_id, rank].item()
                        / planar_gyration_sq
                    )
                    diagnostic["predicted_yaw_delta_rad"] = push_yaw_delta
                    diagnostic["adaptive_candidate_score"] = adaptive_scores[rank]
                    diagnostic["adaptive_translation_gain"] = float(
                        adaptive_translation_gain[env_id].item()
                    )
                    diagnostic["adaptive_rotation_gain"] = float(
                        adaptive_rotation_gain[env_id].item()
                    )
                    preflight_valid, preflight_values = _preflight_candidate(
                        ik,
                        q_current=q_current_np,
                        q_precontact=solved[0],
                        q_contact=solved[1],
                        q_push=solved[2],
                        q_retreat=solved[3],
                        bridge=bridge,
                        hand_points_local=hand_local_np,
                        target_points=points_np,
                        safe_points=safe_np,
                        forbidden_points=forbidden_np,
                        target_center=target_center_np,
                        push_translation=push_translation,
                        push_yaw_delta=push_yaw_delta,
                        cfg=planner_cfg,
                        samples=args_cli.path_samples,
                    )
                    diagnostic.update(preflight_values)
                    diagnostic["preflight_valid"] = preflight_valid
                    plan_diagnostics[env_id].append(diagnostic)
                    if not preflight_valid:
                        continue

                    diagnostic["executable_option_index"] = len(
                        executable_options[env_id]
                    )
                    executable_options[env_id].append(
                        ExecutableCandidatePlan(
                            candidate_rank=rank,
                            q_precontact=solved[0],
                            q_contact=solved[1],
                            q_push=solved[2],
                            q_retreat=solved[3],
                            bridge=bridge,
                            raw_yaw_response=float(
                                candidates.push_distance[env_id, rank].item()
                                * candidates.contact_moment_arm[
                                    env_id, rank
                                ].item()
                                / planar_gyration_sq
                            ),
                            push_direction_xy=(
                                candidates.push_direction[env_id, rank, :2]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            push_distance_m=float(
                                candidates.push_distance[env_id, rank].item()
                            ),
                            diagnostic=diagnostic,
                        )
                    )
                    if len(executable_options[env_id]) >= option_limit:
                        break

                has_executable = bool(executable_options[env_id])
                ik_plan_ever[env_id] |= has_executable
                if not has_executable:
                    exhausted[env_id] = True
                    plan_diagnostics[env_id].append(
                        {
                            "replan": replan_index,
                            "planner_valid_candidates": int(
                                candidates.valid[env_id].sum().item()
                            ),
                            "candidate_attempts": candidate_attempts,
                            "failure": "no_ik_and_semantic_path_valid_candidate",
                        }
                    )

            def select_executable_option(
                env_id: int, option: ExecutableCandidatePlan
            ) -> None:
                q_pre[env_id] = torch.as_tensor(
                    option.q_precontact,
                    device=base.device,
                    dtype=current_q.dtype,
                )
                q_contact[env_id] = torch.as_tensor(
                    option.q_contact,
                    device=base.device,
                    dtype=current_q.dtype,
                )
                q_push[env_id] = torch.as_tensor(
                    option.q_push,
                    device=base.device,
                    dtype=current_q.dtype,
                )
                # Retreat outwards from the translated endpoint.  Reversing
                # the original pre-contact path can strike the object twice.
                q_retreat[env_id] = torch.as_tensor(
                    option.q_retreat,
                    device=base.device,
                    dtype=current_q.dtype,
                )
                bridges[env_id] = option.bridge
                selected_raw_yaw_response[env_id] = option.raw_yaw_response
                selected_push_direction[env_id] = torch.as_tensor(
                    option.push_direction_xy,
                    device=base.device,
                    dtype=selected_push_direction.dtype,
                )
                selected_push_distance[env_id] = option.push_distance_m
                selected_rank[env_id] = option.candidate_rank
                plan_enabled[env_id] = True

            if args_cli.physics_rollout_candidates == 0:
                for env_id in torch.nonzero(
                    active & ~exhausted, as_tuple=False
                ).flatten().tolist():
                    select_executable_option(
                        env_id, executable_options[env_id][0]
                    )
            else:
                rollout_count = args_cli.physics_rollout_candidates
                current_planar, current_height, current_rotation = _pose_errors(
                    base
                )
                rollout_planar = current_planar[:, None].repeat(1, rollout_count)
                rollout_height = current_height[:, None].repeat(1, rollout_count)
                rollout_rotation = current_rotation[:, None].repeat(
                    1, rollout_count
                )
                rollout_enabled = torch.zeros(
                    (base.num_envs, rollout_count),
                    device=base.device,
                    dtype=torch.bool,
                )
                rollout_contact = torch.zeros_like(rollout_enabled)
                rollout_c1 = torch.zeros_like(rollout_enabled)
                rollout_target_pose = torch.full(
                    (base.num_envs, rollout_count, 7),
                    torch.nan,
                    device=base.device,
                    dtype=target_position.dtype,
                )
                snapshot = base.scene.get_state(is_relative=False)
                expected_target_pose = base.scene["target"].data.root_pose_w.clone()
                for option_index in range(rollout_count):
                    enabled = torch.as_tensor(
                        [
                            bool(active[env_id])
                            and option_index < len(executable_options[env_id])
                            for env_id in range(base.num_envs)
                        ],
                        device=base.device,
                        dtype=torch.bool,
                    )
                    if not bool(enabled.any()):
                        continue
                    before_restore = restore_rollout_snapshot(
                        snapshot,
                        current_q,
                        expected_target_pose,
                    )
                    rollout_q_pre = current_q.clone()
                    rollout_q_contact = current_q.clone()
                    rollout_q_push = current_q.clone()
                    rollout_q_retreat = current_q.clone()
                    for env_id in torch.nonzero(
                        enabled, as_tuple=False
                    ).flatten().tolist():
                        option = executable_options[env_id][option_index]
                        rollout_q_pre[env_id] = torch.as_tensor(
                            option.q_precontact,
                            device=base.device,
                            dtype=current_q.dtype,
                        )
                        rollout_q_contact[env_id] = torch.as_tensor(
                            option.q_contact,
                            device=base.device,
                            dtype=current_q.dtype,
                        )
                        rollout_q_push[env_id] = torch.as_tensor(
                            option.q_push,
                            device=base.device,
                            dtype=current_q.dtype,
                        )
                        rollout_q_retreat[env_id] = torch.as_tensor(
                            option.q_retreat,
                            device=base.device,
                            dtype=current_q.dtype,
                        )
                    outcome = execute_physics_rollout(
                        q_start=current_q,
                        q_precontact=rollout_q_pre,
                        q_contact=rollout_q_contact,
                        q_push=rollout_q_push,
                        q_retreat=rollout_q_retreat,
                        enabled=enabled,
                    )
                    rollout_enabled[:, option_index] = enabled
                    rollout_contact[:, option_index] = outcome["contact_ready"]
                    rollout_c1[:, option_index] = outcome["c1_violation"]
                    rollout_planar[:, option_index] = outcome["planar_error"]
                    rollout_height[:, option_index] = outcome["height_error"]
                    rollout_rotation[:, option_index] = outcome["rotation_error"]
                    rollout_target_pose[:, option_index] = outcome["target_pose"]
                    rollout_candidate_evaluations += enabled.long()
                    rollout_legal_evaluations += (
                        enabled
                        & outcome["contact_ready"]
                        & ~outcome["c1_violation"]
                    ).long()
                    after_restore = restore_rollout_snapshot(
                        snapshot,
                        current_q,
                        expected_target_pose,
                    )
                    for env_id in torch.nonzero(
                        enabled, as_tuple=False
                    ).flatten().tolist():
                        diagnostic = executable_options[env_id][
                            option_index
                        ].diagnostic
                        diagnostic.update(
                            {
                                "physics_rollout_option": option_index,
                                "physics_rollout_contact": bool(
                                    outcome["contact_ready"][env_id].item()
                                ),
                                "physics_rollout_c1_violation": bool(
                                    outcome["c1_violation"][env_id].item()
                                ),
                                "physics_rollout_planar_error_m": float(
                                    outcome["planar_error"][env_id].item()
                                ),
                                "physics_rollout_height_error_m": float(
                                    outcome["height_error"][env_id].item()
                                ),
                                "physics_rollout_rotation_error_rad": float(
                                    outcome["rotation_error"][env_id].item()
                                ),
                                "restore_before_joint_error_rad": before_restore[0],
                                "restore_before_position_error_m": before_restore[1],
                                "restore_before_rotation_error_rad": before_restore[2],
                                "restore_after_joint_error_rad": after_restore[0],
                                "restore_after_position_error_m": after_restore[1],
                                "restore_after_rotation_error_rad": after_restore[2],
                            }
                        )

                raw_rollout_cost = joint_threshold_cost(
                    rollout_planar,
                    rollout_height,
                    rollout_rotation,
                    rollout_scoring_cfg,
                )
                best_option, has_improving, rollout_scores = (
                    rank_physics_rollouts(
                        current_planar_error=current_planar,
                        current_height_error=current_height,
                        current_rotation_error=current_rotation,
                        rollout_planar_error=rollout_planar,
                        rollout_height_error=rollout_height,
                        rollout_rotation_error=rollout_rotation,
                        enabled=rollout_enabled,
                        legal_safe_contact=rollout_contact,
                        c1_violation=rollout_c1,
                        cfg=rollout_scoring_cfg,
                    )
                )
                lookahead_first = torch.full_like(best_option, -1)
                lookahead_second = torch.full_like(best_option, -1)
                has_lookahead = torch.zeros_like(has_improving)
                lookahead_scores = torch.full(
                    (
                        base.num_envs,
                        rollout_count,
                        rollout_count,
                    ),
                    torch.inf,
                    device=base.device,
                    dtype=raw_rollout_cost.dtype,
                )
                if args_cli.rollout_lookahead_steps == 2:
                    current_target_pose = torch.cat(
                        (
                            target_position,
                            base.scene["target"].data.root_quat_w,
                        ),
                        dim=1,
                    )
                    finite_rollout_pose = torch.where(
                        rollout_enabled[..., None],
                        rollout_target_pose,
                        current_target_pose[:, None, :],
                    )
                    rollout_translation = finite_rollout_pose[..., :3]
                    local_translation_effect = (
                        rollout_translation - current_target_pose[:, None, :3]
                    )
                    predicted_pair_position = (
                        rollout_translation[:, :, None, :]
                        + local_translation_effect[:, None, :, :]
                    )
                    current_target_quaternion = current_target_pose[:, 3:7]
                    local_rotation_effect = _quaternion_multiply(
                        finite_rollout_pose[..., 3:7],
                        _quaternion_conjugate(
                            current_target_quaternion[:, None, :]
                        ),
                    )
                    predicted_pair_quaternion = _quaternion_multiply(
                        local_rotation_effect[:, None, :, :],
                        finite_rollout_pose[:, :, None, 3:7],
                    )
                    predicted_pair_planar = torch.linalg.vector_norm(
                        goal[:, None, None, :2]
                        - predicted_pair_position[..., :2],
                        dim=-1,
                    )
                    predicted_pair_height = torch.abs(
                        goal[:, None, None, 2]
                        - predicted_pair_position[..., 2]
                    )
                    predicted_pair_rotation = _quaternion_distance(
                        predicted_pair_quaternion,
                        goal[:, None, None, 3:7],
                    )
                    predicted_pair_cost = joint_threshold_cost(
                        predicted_pair_planar,
                        predicted_pair_height,
                        predicted_pair_rotation,
                        rollout_scoring_cfg,
                    )
                    current_rollout_cost = joint_threshold_cost(
                        current_planar,
                        current_height,
                        current_rotation,
                        rollout_scoring_cfg,
                    )
                    (
                        lookahead_first,
                        lookahead_second,
                        has_lookahead,
                        lookahead_scores,
                    ) = rank_physics_rollout_pairs(
                        current_cost=current_rollout_cost,
                        one_step_cost=raw_rollout_cost,
                        pair_cost=predicted_pair_cost,
                        legal_safe_contact=(
                            rollout_enabled & rollout_contact & ~rollout_c1
                        ),
                        minimum_cost_improvement=(
                            rollout_scoring_cfg.minimum_cost_improvement
                        ),
                        intermediate_cost_weight=(
                            args_cli.rollout_lookahead_intermediate_weight
                        ),
                    )
                use_lookahead = has_lookahead
                selected_option = torch.where(
                    use_lookahead, lookahead_first, best_option
                )
                has_selection = has_improving | use_lookahead
                print(
                    "M2_ROLLOUT",
                    f"replan={replan_index}",
                    f"evaluated={int(rollout_enabled.sum().item())}",
                    f"legal={int((rollout_enabled & rollout_contact & ~rollout_c1).sum().item())}",
                    f"one_step={int(has_improving.sum().item())}",
                    f"lookahead={int(use_lookahead.sum().item())}",
                    flush=True,
                )
                for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    for option_index, option in enumerate(
                        executable_options[env_id]
                    ):
                        option.diagnostic["physics_rollout_raw_cost"] = float(
                            raw_rollout_cost[env_id, option_index].item()
                        )
                        option.diagnostic["physics_rollout_improving"] = bool(
                            torch.isfinite(
                                rollout_scores[env_id, option_index]
                            ).item()
                        )
                    if not bool(has_selection[env_id]):
                        exhausted[env_id] = True
                        plan_diagnostics[env_id].append(
                            {
                                "replan": replan_index,
                                "failure": (
                                    "no_safe_physics_rollout_or_pair_improves_joint_pose"
                                ),
                                "physics_rollout_candidates": len(
                                    executable_options[env_id]
                                ),
                            }
                        )
                        continue
                    option_index = int(selected_option[env_id].item())
                    option = executable_options[env_id][option_index]
                    option.diagnostic["physics_rollout_selected"] = True
                    if bool(use_lookahead[env_id]):
                        second_index = int(lookahead_second[env_id].item())
                        option.diagnostic.update(
                            {
                                "physics_rollout_lookahead_selected": True,
                                "physics_rollout_lookahead_second_option": (
                                    second_index
                                ),
                                "physics_rollout_lookahead_score": float(
                                    lookahead_scores[
                                        env_id, option_index, second_index
                                    ].item()
                                ),
                            }
                        )
                    select_executable_option(env_id, option)
                    selected_rollout_target_pose[env_id] = rollout_target_pose[
                        env_id, option_index
                    ]
                    selected_rollout_score[env_id] = raw_rollout_cost[
                        env_id, option_index
                    ]
                    rollout_selected_actions[env_id] += 1

            if (
                video_markers is not None
                and bool(plan_enabled[0])
                and int(selected_rank[0].item()) >= 0
            ):
                rank = int(selected_rank[0].item())
                show_selected_plan(
                    video_markers,
                    env_origin=base.scene.env_origins[0],
                    contact=candidates.contact_tcp[0, rank],
                    precontact=candidates.precontact_tcp[0, rank],
                    push=candidates.push_tcp[0, rank],
                )

            print(
                (
                    "M2_REPLAN"
                    if args_cli.physics_rollout_candidates > 0
                    else "M1_REPLAN"
                ),
                f"index={replan_index}",
                f"active={int(active.sum().item())}",
                f"geometric={int((candidates.any_valid & active).sum().item())}",
                f"executable={int(plan_enabled.sum().item())}",
                flush=True,
            )
            planned_contact_attempts += plan_enabled.long()
            execute_phase(
                current_q,
                q_pre,
                args_cli.approach_steps,
                plan_enabled,
                args_cli.endpoint_hold_steps,
            )
            (
                contact_ready,
                q_contact_actual,
                safe_distance_at_gate,
                forbidden_distance_at_gate,
            ) = execute_until_safe_contact(
                q_pre,
                q_contact,
                args_cli.contact_steps,
                plan_enabled,
                args_cli.endpoint_hold_steps,
            )
            missed_contact = plan_enabled & ~contact_ready & ~forbidden_contact_ever
            selected_pushes += contact_ready.long()
            for env_id in torch.nonzero(plan_enabled, as_tuple=False).flatten().tolist():
                bridge_at_gate = bridges[env_id]
                if bridge_at_gate is None:
                    raise RuntimeError("missing Pinocchio bridge for enabled M1 plan")
                actual_q_at_gate = (
                    robot.data.joint_pos[env_id, joint_ids].detach().cpu().numpy()
                )
                commanded_q_at_gate = (
                    base.action_manager._terms["arm_action"]
                    .joint_position_target[env_id]
                    .detach()
                    .cpu()
                    .numpy()
                )
                actual_tcp_at_gate, actual_rotation_at_gate = ik.tcp_pose_in_env(
                    actual_q_at_gate, bridge_at_gate
                )
                desired_tcp_at_gate = (
                    candidates.contact_tcp[env_id, selected_rank[env_id]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                desired_rotation_at_gate = (
                    candidates.hand_rotation[env_id, selected_rank[env_id]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                plan_diagnostics[env_id].append(
                    {
                        "replan": replan_index,
                        "selected_rank": int(selected_rank[env_id].item()),
                        "contact_gate_passed": bool(contact_ready[env_id].item()),
                        "minimum_safe_distance_at_gate_m": float(
                            safe_distance_at_gate[env_id].item()
                        ),
                        "minimum_forbidden_distance_at_gate_m": float(
                            forbidden_distance_at_gate[env_id].item()
                        ),
                        "maximum_joint_error_at_gate_rad": float(
                            torch.max(
                                torch.abs(
                                    q_contact[env_id]
                                    - robot.data.joint_pos[env_id, joint_ids]
                                )
                            ).item()
                        ),
                        "joint_error_at_gate_rad": (
                            q_contact[env_id]
                            - robot.data.joint_pos[env_id, joint_ids]
                        )
                        .detach()
                        .cpu()
                        .tolist(),
                        "desired_joint_position_at_gate_rad": (
                            q_contact[env_id].detach().cpu().tolist()
                        ),
                        "actual_joint_position_at_gate_rad": actual_q_at_gate.tolist(),
                        "commanded_joint_target_at_gate_rad": (
                            commanded_q_at_gate.tolist()
                        ),
                        "tcp_position_error_at_gate_m": float(
                            np.linalg.norm(actual_tcp_at_gate - desired_tcp_at_gate)
                        ),
                        "tcp_rotation_error_at_gate_rad": float(
                            np.linalg.norm(
                                pin.log3(
                                    actual_rotation_at_gate.T
                                    @ desired_rotation_at_gate
                                )
                            )
                        ),
                    }
                )
            push_start_target_position = (
                base.scene["target"].data.root_pos_w[:, :3]
                - base.scene.env_origins
            ).clone()
            push_start_yaw_error = _signed_yaw_error(base).clone()
            execute_phase(
                q_contact_actual, q_push, args_cli.push_steps, contact_ready
            )
            retreat_start = torch.where(
                contact_ready[:, None], q_push, q_contact_actual
            )
            retreat_target = torch.where(
                contact_ready[:, None], q_retreat, q_pre
            )
            execute_phase(
                retreat_start,
                retreat_target,
                args_cli.retreat_steps,
                plan_enabled,
            )
            for _ in range(args_cli.inter_push_settle_steps):
                env.step(zero_action)
                update_metrics()
            push_end_target_position = (
                base.scene["target"].data.root_pos_w[:, :3]
                - base.scene.env_origins
            )
            push_end_yaw_error = _signed_yaw_error(base)
            if args_cli.physics_rollout_candidates > 0:
                actual_target_pose = torch.cat(
                    (
                        push_end_target_position,
                        base.scene["target"].data.root_quat_w,
                    ),
                    dim=1,
                )
                comparison_valid = plan_enabled & torch.isfinite(
                    selected_rollout_target_pose
                ).all(dim=1)
                transition_position_error = torch.linalg.vector_norm(
                    actual_target_pose[:, :3]
                    - selected_rollout_target_pose[:, :3],
                    dim=1,
                )
                transition_rotation_error = _quaternion_distance(
                    actual_target_pose[:, 3:7],
                    selected_rollout_target_pose[:, 3:7],
                )
                rollout_transition_position_error_sum += torch.where(
                    comparison_valid,
                    transition_position_error,
                    torch.zeros_like(transition_position_error),
                )
                rollout_transition_rotation_error_sum += torch.where(
                    comparison_valid,
                    transition_rotation_error,
                    torch.zeros_like(transition_rotation_error),
                )
                rollout_transition_comparisons += comparison_valid.long()
                for env_id in torch.nonzero(
                    comparison_valid, as_tuple=False
                ).flatten().tolist():
                    plan_diagnostics[env_id].append(
                        {
                            "replan": replan_index,
                            "event": "physics_rollout_real_agreement",
                            "selected_rollout_cost": float(
                                selected_rollout_score[env_id].item()
                            ),
                            "target_position_error_m": float(
                                transition_position_error[env_id].item()
                            ),
                            "target_rotation_error_rad": float(
                                transition_rotation_error[env_id].item()
                            ),
                        }
                    )
            observed_target_translation = (
                push_end_target_position[:, :2]
                - push_start_target_position[:, :2]
            )
            observed_target_yaw_delta = (
                push_start_yaw_error - push_end_yaw_error
            )
            valid_translation_update = contact_ready & (
                selected_push_distance > 1.0e-6
            )
            observed_translation_gain = torch.sum(
                observed_target_translation * selected_push_direction, dim=1
            ) / torch.clamp(selected_push_distance, min=1.0e-6)
            observed_translation_gain = torch.clamp(
                observed_translation_gain, min=0.02, max=1.50
            )
            adaptive_translation_gain.copy_(
                torch.where(
                    valid_translation_update,
                    (1.0 - args_cli.adaptive_dynamics_alpha)
                    * adaptive_translation_gain
                    + args_cli.adaptive_dynamics_alpha
                    * observed_translation_gain,
                    adaptive_translation_gain,
                )
            )
            valid_rotation_update = contact_ready & (
                torch.abs(selected_raw_yaw_response) > 1.0e-4
            )
            observed_rotation_gain = observed_target_yaw_delta / torch.where(
                torch.abs(selected_raw_yaw_response) > 1.0e-4,
                selected_raw_yaw_response,
                torch.ones_like(selected_raw_yaw_response),
            )
            observed_rotation_gain = torch.clamp(
                observed_rotation_gain, min=-2.0, max=2.0
            )
            adaptive_rotation_gain.copy_(
                torch.where(
                    valid_rotation_update,
                    (1.0 - args_cli.adaptive_dynamics_alpha)
                    * adaptive_rotation_gain
                    + args_cli.adaptive_dynamics_alpha * observed_rotation_gain,
                    adaptive_rotation_gain,
                )
            )
            for env_id in torch.nonzero(
                contact_ready, as_tuple=False
            ).flatten().tolist():
                plan_diagnostics[env_id].append(
                    {
                        "replan": replan_index,
                        "event": "adaptive_dynamics_update",
                        "observed_target_translation_m": observed_target_translation[
                            env_id
                        ]
                        .detach()
                        .cpu()
                        .tolist(),
                        "observed_target_yaw_delta_rad": float(
                            observed_target_yaw_delta[env_id].item()
                        ),
                        "updated_translation_gain": float(
                            adaptive_translation_gain[env_id].item()
                        ),
                        "updated_rotation_gain": float(
                            adaptive_rotation_gain[env_id].item()
                        ),
                    }
                )

        current_q = robot.data.joint_pos[:, joint_ids].clone()
        execute_phase(
            current_q,
            current_q,
            args_cli.final_hold_steps,
            ~forbidden_contact_ever,
        )

        final_planar, final_height, final_rotation = _pose_errors(base)
        final_yaw_error = _signed_yaw_error(base)
        constrained_success = success & ~forbidden_contact_ever
        rows: list[dict[str, object]] = []
        for env_id in range(base.num_envs):
            rows.append(
                {
                    "env_id": env_id,
                    "goal_direction_deg": float(initial_direction_deg[env_id].item()),
                    "goal_distance_m": float(
                        torch.linalg.vector_norm(initial_delta[env_id]).item()
                    ),
                    "initial_yaw_error_rad": float(
                        initial_yaw_error[env_id].item()
                    ),
                    "planner_candidate": bool(planner_candidate_ever[env_id].item()),
                    "ik_semantic_plan": bool(ik_plan_ever[env_id].item()),
                    "selected_pushes": int(selected_pushes[env_id].item()),
                    "planned_contact_attempts": int(
                        planned_contact_attempts[env_id].item()
                    ),
                    "contact_gate_success_rate": float(
                        selected_pushes[env_id].item()
                        / max(planned_contact_attempts[env_id].item(), 1)
                    ),
                    "physics_rollout_candidate_evaluations": int(
                        rollout_candidate_evaluations[env_id].item()
                    ),
                    "physics_rollout_legal_evaluations": int(
                        rollout_legal_evaluations[env_id].item()
                    ),
                    "physics_rollout_selected_actions": int(
                        rollout_selected_actions[env_id].item()
                    ),
                    "physics_rollout_real_position_error_mean_m": (
                        float(
                            rollout_transition_position_error_sum[env_id].item()
                            / rollout_transition_comparisons[env_id].item()
                        )
                        if rollout_transition_comparisons[env_id].item() > 0
                        else None
                    ),
                    "physics_rollout_real_rotation_error_mean_rad": (
                        float(
                            rollout_transition_rotation_error_sum[env_id].item()
                            / rollout_transition_comparisons[env_id].item()
                        )
                        if rollout_transition_comparisons[env_id].item() > 0
                        else None
                    ),
                    "safe_contact": bool(safe_contact_ever[env_id].item()),
                    "forbidden_contact": bool(forbidden_contact_ever[env_id].item()),
                    "forbidden_hand_contact": bool(
                        forbidden_hand_contact_ever[env_id].item()
                    ),
                    "arm_target_physical_contact": bool(
                        arm_target_contact_ever[env_id].item()
                    ),
                    "strict_pose": bool(strict_pose_ever[env_id].item()),
                    "constrained_success": bool(constrained_success[env_id].item()),
                    "planner_exhausted": bool(exhausted[env_id].item()),
                    "adaptive_translation_gain": float(
                        adaptive_translation_gain[env_id].item()
                    ),
                    "adaptive_rotation_gain": float(
                        adaptive_rotation_gain[env_id].item()
                    ),
                    "minimum_planar_error_m": float(minimum_planar_error[env_id].item()),
                    "minimum_height_error_m": float(minimum_height_error[env_id].item()),
                    "minimum_rotation_error_rad": float(
                        minimum_rotation_error[env_id].item()
                    ),
                    "minimum_planar_error_while_rotation_valid_m": (
                        _finite_float_or_none(
                            minimum_planar_while_rotation_valid[env_id]
                        )
                    ),
                    "minimum_rotation_error_while_planar_valid_rad": (
                        _finite_float_or_none(
                            minimum_rotation_while_planar_valid[env_id]
                        )
                    ),
                    "minimum_safe_distance_m": float(
                        minimum_safe_distance[env_id].item()
                    ),
                    "final_planar_error_m": float(final_planar[env_id].item()),
                    "final_height_error_m": float(final_height[env_id].item()),
                    "final_rotation_error_rad": float(final_rotation[env_id].item()),
                    "final_yaw_error_rad": float(final_yaw_error[env_id].item()),
                    "plans": plan_diagnostics[env_id],
                }
            )

        summary = {
            "scenes": base.num_envs,
            "planner_candidate_scenes": int(planner_candidate_ever.sum().item()),
            "ik_semantic_plan_scenes": int(ik_plan_ever.sum().item()),
            "safe_contact_scenes": int(safe_contact_ever.sum().item()),
            "c1_violation_scenes": int(forbidden_contact_ever.sum().item()),
            "forbidden_hand_contact_scenes": int(
                forbidden_hand_contact_ever.sum().item()
            ),
            "arm_target_contact_scenes": int(arm_target_contact_ever.sum().item()),
            "strict_pose_scenes": int(strict_pose_ever.sum().item()),
            "constrained_success_scenes": int(constrained_success.sum().item()),
            "constrained_success_rate": float(
                constrained_success.float().mean().item()
            ),
            "planned_contact_attempts": int(planned_contact_attempts.sum().item()),
            "legal_contact_gate_passes": int(selected_pushes.sum().item()),
            "legal_contact_gate_rate": float(
                selected_pushes.sum().item()
                / max(planned_contact_attempts.sum().item(), 1)
            ),
            "physics_rollout_candidate_evaluations": int(
                rollout_candidate_evaluations.sum().item()
            ),
            "physics_rollout_legal_evaluations": int(
                rollout_legal_evaluations.sum().item()
            ),
            "physics_rollout_selected_actions": int(
                rollout_selected_actions.sum().item()
            ),
            "physics_rollout_real_comparisons": int(
                rollout_transition_comparisons.sum().item()
            ),
            "physics_rollout_real_position_error_mean_m": (
                float(
                    rollout_transition_position_error_sum.sum().item()
                    / rollout_transition_comparisons.sum().item()
                )
                if rollout_transition_comparisons.sum().item() > 0
                else None
            ),
            "physics_rollout_real_rotation_error_mean_rad": (
                float(
                    rollout_transition_rotation_error_sum.sum().item()
                    / rollout_transition_comparisons.sum().item()
                )
                if rollout_transition_comparisons.sum().item() > 0
                else None
            ),
            "steps": global_step,
        }
        milestone = (
            "M2" if args_cli.physics_rollout_candidates > 0 else "M1"
        )
        payload = {
            "milestone": milestone,
            "scope": "single DOMINO hammer, no clutter, oracle pose and affordance",
            "task": args_cli.task,
            "seed": args_cli.seed,
            "planner": asdict(planner_cfg),
            "execution": {
                "max_replans": args_cli.max_replans,
                "approach_steps": args_cli.approach_steps,
                "contact_steps": args_cli.contact_steps,
                "endpoint_hold_steps": args_cli.endpoint_hold_steps,
                "servo_gain": args_cli.servo_gain,
                "joint_action_scale_rad": args_cli.joint_action_scale_rad,
                "gate_contact_distance_m": args_cli.gate_contact_distance_m,
                "adaptive_dynamics_alpha": args_cli.adaptive_dynamics_alpha,
                "inside_yaw_weight_m_per_rad": (
                    args_cli.inside_yaw_weight_m_per_rad
                ),
                "predicted_yaw_guard_rad": args_cli.predicted_yaw_guard_rad,
                "yaw_guard_penalty_m_per_rad": (
                    args_cli.yaw_guard_penalty_m_per_rad
                ),
                "push_steps": args_cli.push_steps,
                "retreat_steps": args_cli.retreat_steps,
                "dwell_steps": args_cli.dwell_steps,
                "physics_rollout_candidates": (
                    args_cli.physics_rollout_candidates
                ),
                "physics_rollout_lookahead_steps": (
                    args_cli.rollout_lookahead_steps
                ),
                "physics_rollout_lookahead_intermediate_weight": (
                    args_cli.rollout_lookahead_intermediate_weight
                ),
                "physics_rollout_scoring": asdict(rollout_scoring_cfg),
                "video": args_cli.video,
            },
            "summary": summary,
            "rows": rows,
        }
        output_path = args_cli.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"{milestone}_SUMMARY",
            json.dumps(summary, sort_keys=True),
            flush=True,
        )
        print(f"{milestone}_OUTPUT path={output_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
