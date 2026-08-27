#!/usr/bin/env python3
"""Scripted Cartesian push probe for the strict DOMINO hammer proof task."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AffordanceHammer-Pose-Franka-v0")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--approach-steps", type=int, default=40)
parser.add_argument("--push-steps", type=int, default=60)
parser.add_argument("--hold-steps", type=int, default=20)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--approach-distance", type=float, default=0.080)
parser.add_argument("--push-distance", type=float, default=0.100)
parser.add_argument("--ik-max-evaluations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--dwell-steps", type=int, default=5)
parser.add_argument(
    "--disable-reached-reset",
    action="store_true",
    help=(
        "keep scripted trajectories running after strict success and compute "
        "dwell locally; useful for deterministic physical-envelope audits"
    ),
)
parser.add_argument(
    "--contact-distance-m",
    type=float,
    default=0.010,
    help="semantic contact boundary; defaults to the current teacher C1 contract",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="optional JSON path for the per-environment signed-yaw sweep results",
)
parser.add_argument(
    "--freeze-on-strict",
    action="store_true",
    help="hold the current joints after first entering the strict pose region",
)
parser.add_argument(
    "--tcp-z",
    type=float,
    default=None,
    help="absolute TCP Z in each environment frame; preserve reset Z when omitted",
)
parser.add_argument(
    "--tcp-y-range",
    type=float,
    nargs=2,
    default=(0.0, 0.05),
    metavar=("MIN_Y", "MAX_Y"),
)
parser.add_argument(
    "--push-y-range",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("MIN_DY", "MAX_DY"),
    help="additional local-frame TCP Y displacement during the push phase",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import pinocchio as pin
import torch
from scipy.optimize import least_squares

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp


FRANKA_URDF = (
    "/data1/linsixu/IsaacLab-2.2.0/source/isaaclab/isaaclab/"
    "controllers/config/data/lula_franka_gen.urdf"
)


def _never_reached(env) -> torch.Tensor:
    """Keep the registered reached term observable without resetting the probe."""

    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


def main() -> None:
    if args_cli.num_envs < 2:
        raise ValueError("the scripted sweep requires at least two environments")
    if min(
        args_cli.settle_steps,
        args_cli.approach_steps,
        args_cli.push_steps,
        args_cli.hold_steps,
    ) < 0:
        raise ValueError("phase step counts must be non-negative")
    if args_cli.approach_steps == 0 or args_cli.push_steps == 0:
        raise ValueError("approach and push phases must contain at least one step")
    if args_cli.contact_distance_m <= 0.0:
        raise ValueError("contact distance must be positive")
    if args_cli.dwell_steps <= 0:
        raise ValueError("dwell steps must be positive")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    env_cfg.seed = args_cli.seed
    if args_cli.disable_reached_reset:
        env_cfg.terminations.reached.func = _never_reached
        env_cfg.terminations.reached.params = {}
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        robot = base.scene["robot"]
        robot_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        robot_cfg.resolve(base.scene)
        joint_ids = robot_cfg.joint_ids
        body_id = robot_cfg.body_ids[0]

        target = base.scene["target"]
        settle_start_position = (
            target.data.root_pos_w[:, :3] - base.scene.env_origins
        ).clone()
        settle_start_quaternion = target.data.root_quat_w.clone()
        for _ in range(args_cli.settle_steps):
            env.step(torch.zeros((base.num_envs, 7), device=base.device))
        settle_position = target.data.root_pos_w[:, :3] - base.scene.env_origins
        settle_quaternion = target.data.root_quat_w.clone()
        settle_translation = torch.linalg.vector_norm(
            settle_position - settle_start_position, dim=1
        )
        settle_dot = torch.sum(
            settle_quaternion * settle_start_quaternion, dim=1
        ).abs()
        settle_rotation = 2.0 * torch.acos(torch.clamp(settle_dot, max=1.0))
        print(
            "SCRIPTED_PUSH_SETTLE",
            f"steps={args_cli.settle_steps}",
            f"max_translation={settle_translation.max().item():.6f}",
            f"max_rotation={settle_rotation.max().item():.6f}",
            f"start_z={settle_start_position[0, 2].item():.6f}",
            f"settled_z={settle_position[0, 2].item():.6f}",
            flush=True,
        )

        initial_hand_pose = robot.data.body_state_w[:, body_id, :7].clone()
        initial_tcp = base.scene["ee_frame"].data.target_pos_w[:, 0].clone()
        initial_tcp_env = initial_tcp - base.scene.env_origins
        tcp_z_shift = (
            0.0
            if args_cli.tcp_z is None
            else float(args_cli.tcp_z - initial_tcp_env[0, 2].item())
        )
        desired_tcp_y = torch.linspace(
            args_cli.tcp_y_range[0],
            args_cli.tcp_y_range[1],
            base.num_envs,
            device=base.device,
        )
        push_tcp_y = torch.linspace(
            args_cli.push_y_range[0],
            args_cli.push_y_range[1],
            base.num_envs,
            device=base.device,
        )
        # The sweep is specified in each replicated environment's local frame.
        # Using world Y here would accidentally include the scene-grid offset
        # (roughly metres) in an otherwise centimetre-scale IK target.
        tcp_y_shift = desired_tcp_y - initial_tcp_env[:, 1]

        # Solve exact, collision-free kinematic endpoints offline.  Executing
        # interpolated joint targets avoids conflating task feasibility with a
        # separate online differential-IK tuning problem.
        model = pin.buildModelFromUrdf(FRANKA_URDF)
        frame_id = model.getFrameId("panda_hand")
        lower = model.lowerPositionLimit[:7] + 1.0e-4
        upper = model.upperPositionLimit[:7] - 1.0e-4
        q_initial_np = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()

        def forward(q_arm: np.ndarray) -> pin.SE3:
            data = model.createData()
            q_full = np.concatenate((q_arm, np.array((0.04, 0.04))))
            pin.forwardKinematics(model, data, q_full)
            pin.updateFramePlacements(model, data)
            return data.oMf[frame_id]

        initial_pin_pose = forward(q_initial_np)

        def solve_endpoint(
            seed: np.ndarray,
            x_displacement: float,
            y_displacement: float,
            z_displacement: float,
        ) -> tuple[np.ndarray, float, float]:
            target_position = initial_pin_pose.translation + np.array(
                (x_displacement, y_displacement, z_displacement)
            )

            def residual(q_arm: np.ndarray) -> np.ndarray:
                pose = forward(q_arm)
                return np.concatenate(
                    (
                        20.0 * (pose.translation - target_position),
                        pin.log3(pose.rotation.T @ initial_pin_pose.rotation),
                        1.0e-3 * (q_arm - seed),
                    )
                )

            result = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                max_nfev=args_cli.ik_max_evaluations,
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
            )
            pose = forward(result.x)
            position_error = float(np.linalg.norm(pose.translation - target_position))
            rotation_error = float(
                np.linalg.norm(pin.log3(pose.rotation.T @ initial_pin_pose.rotation))
            )
            if position_error > 1.0e-4 or rotation_error > 1.0e-3:
                raise RuntimeError(
                    "offline endpoint IK did not converge: "
                    f"position_error={position_error}, rotation_error={rotation_error}"
                )
            return result.x, position_error, rotation_error

        contact_targets = []
        final_targets = []
        for env_id in range(base.num_envs):
            y_shift = float(tcp_y_shift[env_id].item())
            q_contact, contact_pos_error, contact_rot_error = solve_endpoint(
                q_initial_np, args_cli.approach_distance, y_shift, tcp_z_shift
            )
            q_final, final_pos_error, final_rot_error = solve_endpoint(
                q_contact,
                args_cli.approach_distance + args_cli.push_distance,
                y_shift + float(push_tcp_y[env_id].item()),
                tcp_z_shift,
            )
            contact_targets.append(q_contact)
            final_targets.append(q_final)
            print(
                "SCRIPTED_PUSH_IK",
                f"env={env_id}",
                f"tcp_y={desired_tcp_y[env_id].item():.4f}",
                f"push_dy={push_tcp_y[env_id].item():.4f}",
                f"tcp_z={initial_tcp_env[env_id, 2].item() + tcp_z_shift:.4f}",
                f"contact_pos_error={contact_pos_error:.3e}",
                f"contact_rot_error={contact_rot_error:.3e}",
                f"final_pos_error={final_pos_error:.3e}",
                f"final_rot_error={final_rot_error:.3e}",
                flush=True,
            )
        q_initial = torch.as_tensor(
            q_initial_np, device=base.device, dtype=robot.data.joint_pos.dtype
        ).view(1, 7).repeat(base.num_envs, 1)
        q_contact = torch.as_tensor(
            np.stack(contact_targets),
            device=base.device,
            dtype=robot.data.joint_pos.dtype,
        )
        q_final = torch.as_tensor(
            np.stack(final_targets),
            device=base.device,
            dtype=robot.data.joint_pos.dtype,
        )

        safe_contact_ever = torch.zeros(
            base.num_envs, device=base.device, dtype=torch.bool
        )
        forbidden_contact_ever = torch.zeros_like(safe_contact_ever)
        navigation_contact_latched_ever = torch.zeros_like(safe_contact_ever)
        post_contact_pose_cost_latched_ever = torch.zeros_like(safe_contact_ever)
        post_contact_pose_improvement_latched_ever = torch.zeros_like(
            safe_contact_ever
        )
        first_safe_contact_step = torch.full(
            (base.num_envs,), -1, device=base.device, dtype=torch.long
        )
        first_forbidden_contact_step = torch.full_like(first_safe_contact_step, -1)
        first_navigation_latch_step = torch.full_like(first_safe_contact_step, -1)
        first_post_contact_pose_cost_latch_step = torch.full_like(
            first_safe_contact_step, -1
        )
        first_post_contact_pose_improvement_latch_step = torch.full_like(
            first_safe_contact_step, -1
        )
        pose_reference_cost_at_first_latch = torch.full(
            (base.num_envs,), torch.nan, device=base.device
        )
        yaw_at_first_safe_contact = torch.full(
            (base.num_envs,), torch.nan, device=base.device
        )
        yaw_at_first_forbidden_contact = torch.full_like(
            yaw_at_first_safe_contact, torch.nan
        )
        minimum_legal_yaw_after_safe_contact = torch.full(
            (base.num_envs,), torch.inf, device=base.device
        )
        maximum_legal_yaw_after_safe_contact = torch.full_like(
            minimum_legal_yaw_after_safe_contact, -torch.inf
        )
        strict_pose_ever = torch.zeros_like(safe_contact_ever)
        dwell_success_ever = torch.zeros_like(safe_contact_ever)
        minimum_planar_error = torch.full(
            (base.num_envs,), torch.inf, device=base.device
        )
        minimum_height_error = torch.full_like(minimum_planar_error, torch.inf)
        minimum_rotation_error = torch.full_like(minimum_planar_error, torch.inf)
        minimum_joint_error = torch.full_like(minimum_planar_error, torch.inf)
        best_joint_planar_error = torch.full_like(minimum_planar_error, torch.inf)
        best_joint_height_error = torch.full_like(minimum_planar_error, torch.inf)
        best_joint_rotation_error = torch.full_like(minimum_planar_error, torch.inf)
        maximum_target_dx = torch.full_like(minimum_planar_error, -torch.inf)
        minimum_target_yaw_delta = torch.full_like(minimum_planar_error, torch.inf)
        maximum_target_yaw_delta = torch.full_like(minimum_planar_error, -torch.inf)
        minimum_safe_distance = torch.full_like(minimum_planar_error, torch.inf)
        maximum_tcp_dx = torch.full_like(minimum_planar_error, -torch.inf)
        maximum_action = torch.zeros_like(minimum_planar_error)
        freeze_active = torch.zeros_like(safe_contact_ever)
        consecutive_strict_steps = torch.zeros_like(first_safe_contact_step)
        frozen_joint_position = torch.zeros_like(q_initial)
        initial_target_position = (
            base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        ).clone()

        total_steps = (
            args_cli.approach_steps + args_cli.push_steps + args_cli.hold_steps
        )
        action_scale = float(base.cfg.actions.arm_action.scale)
        for step in range(total_steps):
            if step < args_cli.approach_steps:
                alpha = (step + 1) / args_cli.approach_steps
                desired_joint_position = q_initial + alpha * (q_contact - q_initial)
            elif step < args_cli.approach_steps + args_cli.push_steps:
                push_step = step - args_cli.approach_steps + 1
                alpha = push_step / args_cli.push_steps
                desired_joint_position = q_contact + alpha * (
                    q_final - q_contact
                )
            else:
                desired_joint_position = q_final

            if args_cli.freeze_on_strict:
                desired_joint_position = torch.where(
                    freeze_active.unsqueeze(1),
                    frozen_joint_position,
                    desired_joint_position,
                )

            joint_position = robot.data.joint_pos[:, joint_ids]
            action = torch.clamp(
                (desired_joint_position - joint_position) / action_scale,
                min=-1.0,
                max=1.0,
            )
            maximum_action = torch.maximum(
                maximum_action, torch.amax(torch.abs(action), dim=1)
            )
            env.step(action)

            target = base.scene["target"]
            goal = base.command_manager.get_command("target_object_pose")
            target_position = target.data.root_pos_w[:, :3] - base.scene.env_origins
            relative_target_quaternion = quat_mul(
                target.data.root_quat_w, quat_conjugate(settle_quaternion)
            )
            relative_target_rotation = matrix_from_quat(relative_target_quaternion)
            target_yaw_delta = torch.atan2(
                relative_target_rotation[:, 1, 0],
                relative_target_rotation[:, 0, 0],
            )
            pose_delta = goal[:, :3] - target_position
            planar_error = torch.linalg.vector_norm(pose_delta[:, :2], dim=1)
            height_error = torch.abs(pose_delta[:, 2])
            quaternion_dot = torch.sum(target.data.root_quat_w * goal[:, 3:7], dim=1)
            rotation_error = 2.0 * torch.acos(
                torch.clamp(torch.abs(quaternion_dot), max=1.0)
            )
            joint_error = mdp.affordance_joint_pose_error(base)
            contact_state = mdp.domino_affordance_contact_state(
                base,
                contact_distance_m=args_cli.contact_distance_m,
                evaluate_protected=False,
            )
            strict_pose = (
                (planar_error < 0.02)
                & (height_error < 0.01)
                & (rotation_error < 0.10)
            )
            consecutive_strict_steps = torch.where(
                strict_pose,
                consecutive_strict_steps + 1,
                torch.zeros_like(consecutive_strict_steps),
            )
            if args_cli.disable_reached_reset:
                reached = consecutive_strict_steps >= args_cli.dwell_steps
            else:
                reached = base.termination_manager.get_term("reached")

            minimum_planar_error = torch.minimum(minimum_planar_error, planar_error)
            minimum_height_error = torch.minimum(minimum_height_error, height_error)
            minimum_rotation_error = torch.minimum(
                minimum_rotation_error, rotation_error
            )
            better_joint = joint_error < minimum_joint_error
            minimum_joint_error = torch.minimum(minimum_joint_error, joint_error)
            best_joint_planar_error = torch.where(
                better_joint, planar_error, best_joint_planar_error
            )
            best_joint_height_error = torch.where(
                better_joint, height_error, best_joint_height_error
            )
            best_joint_rotation_error = torch.where(
                better_joint, rotation_error, best_joint_rotation_error
            )
            maximum_target_dx = torch.maximum(
                maximum_target_dx,
                target_position[:, 0] - initial_target_position[:, 0],
            )
            minimum_target_yaw_delta = torch.minimum(
                minimum_target_yaw_delta, target_yaw_delta
            )
            maximum_target_yaw_delta = torch.maximum(
                maximum_target_yaw_delta, target_yaw_delta
            )
            current_tcp = base.scene["ee_frame"].data.target_pos_w[:, 0]
            maximum_tcp_dx = torch.maximum(
                maximum_tcp_dx, current_tcp[:, 0] - initial_tcp[:, 0]
            )
            minimum_safe_distance = torch.minimum(
                minimum_safe_distance, contact_state["minimum_safe_distance"]
            )
            safe_contact_now = contact_state["safe_robot_contact"]
            forbidden_contact_now = contact_state["forbidden_robot_contact"]
            navigation_contact_latched_now = getattr(
                base,
                "_affordance_navigation_contact_latched",
                torch.zeros_like(safe_contact_ever),
            )
            post_contact_pose_cost_latched_now = getattr(
                base,
                "_affordance_post_contact_pose_cost_latched",
                torch.zeros_like(safe_contact_ever),
            )
            post_contact_pose_improvement_latched_now = getattr(
                base,
                "_affordance_post_contact_pose_improvement_latched",
                torch.zeros_like(safe_contact_ever),
            )
            first_safe_now = (first_safe_contact_step < 0) & safe_contact_now
            first_forbidden_now = (
                first_forbidden_contact_step < 0
            ) & forbidden_contact_now
            first_navigation_latch_now = (
                first_navigation_latch_step < 0
            ) & navigation_contact_latched_now
            first_post_contact_pose_cost_latch_now = (
                first_post_contact_pose_cost_latch_step < 0
            ) & post_contact_pose_cost_latched_now
            first_post_contact_pose_improvement_latch_now = (
                first_post_contact_pose_improvement_latch_step < 0
            ) & post_contact_pose_improvement_latched_now
            first_safe_contact_step[first_safe_now] = step
            first_forbidden_contact_step[first_forbidden_now] = step
            first_navigation_latch_step[first_navigation_latch_now] = step
            first_post_contact_pose_cost_latch_step[
                first_post_contact_pose_cost_latch_now
            ] = step
            first_post_contact_pose_improvement_latch_step[
                first_post_contact_pose_improvement_latch_now
            ] = step
            reference_cost_now = getattr(
                base,
                "_affordance_post_contact_pose_reference_cost",
                torch.full_like(pose_reference_cost_at_first_latch, torch.nan),
            )
            pose_reference_cost_at_first_latch[
                first_post_contact_pose_improvement_latch_now
            ] = reference_cost_now[
                first_post_contact_pose_improvement_latch_now
            ]
            yaw_at_first_safe_contact[first_safe_now] = target_yaw_delta[first_safe_now]
            yaw_at_first_forbidden_contact[first_forbidden_now] = target_yaw_delta[
                first_forbidden_now
            ]
            safe_contact_ever |= safe_contact_now
            forbidden_contact_ever |= forbidden_contact_now
            navigation_contact_latched_ever |= navigation_contact_latched_now
            post_contact_pose_cost_latched_ever |= (
                post_contact_pose_cost_latched_now
            )
            post_contact_pose_improvement_latched_ever |= (
                post_contact_pose_improvement_latched_now
            )
            legal_after_safe_contact = safe_contact_ever & ~forbidden_contact_ever
            minimum_legal_yaw_after_safe_contact = torch.where(
                legal_after_safe_contact,
                torch.minimum(
                    minimum_legal_yaw_after_safe_contact, target_yaw_delta
                ),
                minimum_legal_yaw_after_safe_contact,
            )
            maximum_legal_yaw_after_safe_contact = torch.where(
                legal_after_safe_contact,
                torch.maximum(
                    maximum_legal_yaw_after_safe_contact, target_yaw_delta
                ),
                maximum_legal_yaw_after_safe_contact,
            )
            strict_pose_ever |= strict_pose
            dwell_success_ever |= reached
            if args_cli.freeze_on_strict:
                newly_strict = strict_pose & ~freeze_active
                frozen_joint_position[newly_strict] = robot.data.joint_pos[
                    newly_strict
                ][:, joint_ids]
                freeze_active |= strict_pose

        final_target_position = (
            base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        )
        final_goal = base.command_manager.get_command("target_object_pose")
        final_planar_error = torch.linalg.vector_norm(
            final_goal[:, :2] - final_target_position[:, :2], dim=1
        )
        final_height_error = torch.abs(
            final_goal[:, 2] - final_target_position[:, 2]
        )
        final_quaternion_dot = torch.sum(
            base.scene["target"].data.root_quat_w * final_goal[:, 3:7], dim=1
        )
        final_rotation_error = 2.0 * torch.acos(
            torch.clamp(torch.abs(final_quaternion_dot), max=1.0)
        )
        final_tcp = (
            base.scene["ee_frame"].data.target_pos_w[:, 0]
            - base.scene.env_origins
        )
        final_relative_target_quaternion = quat_mul(
            base.scene["target"].data.root_quat_w,
            quat_conjugate(settle_quaternion),
        )
        final_relative_target_rotation = matrix_from_quat(
            final_relative_target_quaternion
        )
        final_target_yaw_delta = torch.atan2(
            final_relative_target_rotation[:, 1, 0],
            final_relative_target_rotation[:, 0, 0],
        )
        print(
            "SCRIPTED_PUSH_SUMMARY",
            f"envs={base.num_envs}",
            f"safe_contact={int(safe_contact_ever.sum().item())}",
            f"forbidden_contact={int(forbidden_contact_ever.sum().item())}",
            f"navigation_latched={int(navigation_contact_latched_ever.sum().item())}",
            f"pose_cost_latched={int(post_contact_pose_cost_latched_ever.sum().item())}",
            f"pose_improvement_latched={int(post_contact_pose_improvement_latched_ever.sum().item())}",
            f"strict_pose={int(strict_pose_ever.sum().item())}",
            f"dwell_success={int(dwell_success_ever.sum().item())}",
            flush=True,
        )
        result_rows: list[dict[str, object]] = []
        for env_id in range(base.num_envs):
            result_row = {
                "env_id": env_id,
                "tcp_y_m": float(desired_tcp_y[env_id].item()),
                "push_dy_m": float(push_tcp_y[env_id].item()),
                "safe_contact": bool(safe_contact_ever[env_id].item()),
                "forbidden_contact": bool(forbidden_contact_ever[env_id].item()),
                "navigation_contact_latched": bool(
                    navigation_contact_latched_ever[env_id].item()
                ),
                "post_contact_pose_cost_latched": bool(
                    post_contact_pose_cost_latched_ever[env_id].item()
                ),
                "post_contact_pose_improvement_latched": bool(
                    post_contact_pose_improvement_latched_ever[env_id].item()
                ),
                "first_safe_contact_step": int(
                    first_safe_contact_step[env_id].item()
                ),
                "first_forbidden_contact_step": int(
                    first_forbidden_contact_step[env_id].item()
                ),
                "first_navigation_latch_step": int(
                    first_navigation_latch_step[env_id].item()
                ),
                "first_post_contact_pose_cost_latch_step": int(
                    first_post_contact_pose_cost_latch_step[env_id].item()
                ),
                "first_post_contact_pose_improvement_latch_step": int(
                    first_post_contact_pose_improvement_latch_step[env_id].item()
                ),
                "pose_reference_cost_at_first_latch": float(
                    pose_reference_cost_at_first_latch[env_id].item()
                ),
                "yaw_at_first_safe_contact_rad": float(
                    yaw_at_first_safe_contact[env_id].item()
                ),
                "yaw_at_first_forbidden_contact_rad": float(
                    yaw_at_first_forbidden_contact[env_id].item()
                ),
                "minimum_legal_yaw_after_safe_contact_rad": float(
                    minimum_legal_yaw_after_safe_contact[env_id].item()
                ),
                "maximum_legal_yaw_after_safe_contact_rad": float(
                    maximum_legal_yaw_after_safe_contact[env_id].item()
                ),
                "strict_pose": bool(strict_pose_ever[env_id].item()),
                "dwell_success": bool(dwell_success_ever[env_id].item()),
                "minimum_target_yaw_delta_rad": float(
                    minimum_target_yaw_delta[env_id].item()
                ),
                "maximum_target_yaw_delta_rad": float(
                    maximum_target_yaw_delta[env_id].item()
                ),
                "final_target_yaw_delta_rad": float(
                    final_target_yaw_delta[env_id].item()
                ),
                "maximum_target_dx_m": float(maximum_target_dx[env_id].item()),
                "minimum_safe_distance_m": float(
                    minimum_safe_distance[env_id].item()
                ),
                "minimum_planar_error_m": float(
                    minimum_planar_error[env_id].item()
                ),
                "minimum_height_error_m": float(
                    minimum_height_error[env_id].item()
                ),
                "minimum_rotation_error_rad": float(
                    minimum_rotation_error[env_id].item()
                ),
            }
            result_rows.append(result_row)
            print(
                "SCRIPTED_PUSH_ENV",
                f"env={env_id}",
                f"tcp_y={desired_tcp_y[env_id].item():.4f}",
                f"push_dy={push_tcp_y[env_id].item():.4f}",
                f"safe={int(safe_contact_ever[env_id].item())}",
                f"forbidden={int(forbidden_contact_ever[env_id].item())}",
                f"navigation_latched={int(navigation_contact_latched_ever[env_id].item())}",
                f"pose_cost_latched={int(post_contact_pose_cost_latched_ever[env_id].item())}",
                f"pose_improvement_latched={int(post_contact_pose_improvement_latched_ever[env_id].item())}",
                f"first_safe_step={first_safe_contact_step[env_id].item()}",
                f"first_forbidden_step={first_forbidden_contact_step[env_id].item()}",
                f"first_navigation_latch_step={first_navigation_latch_step[env_id].item()}",
                f"first_pose_cost_latch_step={first_post_contact_pose_cost_latch_step[env_id].item()}",
                f"first_pose_improvement_latch_step={first_post_contact_pose_improvement_latch_step[env_id].item()}",
                f"pose_reference_cost={pose_reference_cost_at_first_latch[env_id].item():.4f}",
                f"yaw_at_first_safe={yaw_at_first_safe_contact[env_id].item():.4f}",
                f"yaw_at_first_forbidden={yaw_at_first_forbidden_contact[env_id].item():.4f}",
                f"min_legal_yaw_after_safe={minimum_legal_yaw_after_safe_contact[env_id].item():.4f}",
                f"max_legal_yaw_after_safe={maximum_legal_yaw_after_safe_contact[env_id].item():.4f}",
                f"strict={int(strict_pose_ever[env_id].item())}",
                f"dwell={int(dwell_success_ever[env_id].item())}",
                f"max_dx={maximum_target_dx[env_id].item():.4f}",
                f"min_yaw_delta={minimum_target_yaw_delta[env_id].item():.4f}",
                f"max_yaw_delta={maximum_target_yaw_delta[env_id].item():.4f}",
                f"final_yaw_delta={final_target_yaw_delta[env_id].item():.4f}",
                f"max_tcp_dx={maximum_tcp_dx[env_id].item():.4f}",
                f"final_tcp={final_tcp[env_id].tolist()}",
                f"min_safe_distance={minimum_safe_distance[env_id].item():.4f}",
                f"max_action={maximum_action[env_id].item():.4f}",
                f"min_xy={minimum_planar_error[env_id].item():.4f}",
                f"min_z={minimum_height_error[env_id].item():.4f}",
                f"min_rot={minimum_rotation_error[env_id].item():.4f}",
                f"min_joint={minimum_joint_error[env_id].item():.4f}",
                f"best_joint_components={[best_joint_planar_error[env_id].item(), best_joint_height_error[env_id].item(), best_joint_rotation_error[env_id].item()]}",
                f"final_errors={[final_planar_error[env_id].item(), final_height_error[env_id].item(), final_rotation_error[env_id].item()]}",
                f"final_pos={final_target_position[env_id].tolist()}",
                flush=True,
            )
        if args_cli.output is not None:
            output_path = args_cli.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            legal_trajectory = safe_contact_ever & ~forbidden_contact_ever
            finite_legal_min = torch.isfinite(
                minimum_legal_yaw_after_safe_contact
            )
            finite_legal_max = torch.isfinite(
                maximum_legal_yaw_after_safe_contact
            )
            all_trajectory_minimum = (
                float(
                    minimum_legal_yaw_after_safe_contact[
                        finite_legal_min
                    ].min().item()
                )
                if bool(finite_legal_min.any())
                else None
            )
            all_trajectory_maximum = (
                float(
                    maximum_legal_yaw_after_safe_contact[
                        finite_legal_max
                    ].max().item()
                )
                if bool(finite_legal_max.any())
                else None
            )
            legal_minimum_mask = legal_trajectory & finite_legal_min
            legal_maximum_mask = legal_trajectory & finite_legal_max
            legal_trajectory_minimum = (
                float(
                    minimum_legal_yaw_after_safe_contact[
                        legal_minimum_mask
                    ].min().item()
                )
                if bool(legal_minimum_mask.any())
                else None
            )
            legal_trajectory_maximum = (
                float(
                    maximum_legal_yaw_after_safe_contact[
                        legal_maximum_mask
                    ].max().item()
                )
                if bool(legal_maximum_mask.any())
                else None
            )
            payload = {
                "task": args_cli.task,
                "num_envs": base.num_envs,
                "seed": args_cli.seed,
                "disable_reached_reset": args_cli.disable_reached_reset,
                "dwell_steps": args_cli.dwell_steps,
                "approach_distance_m": args_cli.approach_distance,
                "push_distance_m": args_cli.push_distance,
                "contact_distance_m": args_cli.contact_distance_m,
                "tcp_y_range_m": list(args_cli.tcp_y_range),
                "push_y_range_m": list(args_cli.push_y_range),
                "summary": {
                    "safe_contact_trajectories": int(
                        safe_contact_ever.sum().item()
                    ),
                    "forbidden_contact_trajectories": int(
                        forbidden_contact_ever.sum().item()
                    ),
                    "fully_legal_safe_contact_trajectories": int(
                        legal_trajectory.sum().item()
                    ),
                    "navigation_contact_latched_trajectories": int(
                        navigation_contact_latched_ever.sum().item()
                    ),
                    "post_contact_pose_cost_latched_trajectories": int(
                        post_contact_pose_cost_latched_ever.sum().item()
                    ),
                    "post_contact_pose_improvement_latched_trajectories": int(
                        post_contact_pose_improvement_latched_ever.sum().item()
                    ),
                    "synchronized_reward_latch_trajectories": int(
                        (
                            navigation_contact_latched_ever
                            & post_contact_pose_cost_latched_ever
                            & (
                                first_navigation_latch_step
                                == first_post_contact_pose_cost_latch_step
                            )
                        ).sum().item()
                    ),
                    "synchronized_improvement_latch_trajectories": int(
                        (
                            navigation_contact_latched_ever
                            & post_contact_pose_improvement_latched_ever
                            & (
                                first_navigation_latch_step
                                == first_post_contact_pose_improvement_latch_step
                            )
                        ).sum().item()
                    ),
                    "strict_pose_trajectories": int(
                        strict_pose_ever.sum().item()
                    ),
                    "dwell_success_trajectories": int(
                        dwell_success_ever.sum().item()
                    ),
                    "pre_violation_signed_yaw_envelope_rad": [
                        all_trajectory_minimum,
                        all_trajectory_maximum,
                    ],
                    "fully_legal_signed_yaw_envelope_rad": [
                        legal_trajectory_minimum,
                        legal_trajectory_maximum,
                    ],
                },
                "rows": result_rows,
            }
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"SCRIPTED_PUSH_OUTPUT path={output_path}", flush=True)
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
