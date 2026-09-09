#!/usr/bin/env python3
"""Execute a Push Anything C3+ trajectory inside the native Isaac Lab task.

This is a cross-simulator executor smoke test, not a success-rate benchmark.
The recorded C3+ TCP path is transformed to the Isaac support plane, solved
with Franka IK, and tracked by the native Isaac Lab joint controller.  The
result reports task geometry and oracle C1 contact independently.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", default="Isaac-AffordanceHammer-Pose-Franka-v0"
)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--trajectory-csv", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--domino-root", type=Path, default=Path("/data1/linsixu/DOMINO"))
parser.add_argument(
    "--domino-usd-root",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "data/domino_usd",
)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument(
    "--initialization-mode",
    choices=("teleport", "servo"),
    default="teleport",
    help=(
        "teleport matches the C3+ simulator initial robot state; servo is a "
        "diagnostic unplanned interpolation from the Isaac task reset pose"
    ),
)
parser.add_argument("--final-hold-steps", type=int, default=10)
parser.add_argument("--dwell-steps", type=int, default=5)
parser.add_argument(
    "--stop-on-success",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "stop formal execution once the unchanged strict pose gate has held "
        "for dwell-steps; --no-stop-on-success is an open-loop diagnostic"
    ),
)
parser.add_argument("--servo-gain", type=float, default=3.0)
parser.add_argument("--joint-action-scale-rad", type=float, default=0.12)
parser.add_argument("--table-height-offset-m", type=float, default=0.029)
parser.add_argument("--contact-distance-m", type=float, default=0.010)
parser.add_argument("--ik-max-evaluations", type=int, default=80)
parser.add_argument("--ik-position-tolerance-m", type=float, default=0.003)
parser.add_argument("--ik-rotation-tolerance-rad", type=float, default=0.03)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-folder", type=Path, default=Path("outputs/contact_planner_m3/isaaclab_videos"))
parser.add_argument("--video-name-prefix", default="push_anything_isaaclab_smoke")
parser.add_argument("--video-length", type=int, default=0)
parser.add_argument("--goal-ghost-opacity", type=float, default=0.68)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(1.15, 0.70, 0.82))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.48, 0.10, 0.06))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from dapl.contact_planner import (
    load_c3_joint_trajectory,
    resample_c3_joint_trajectory,
)
from dapl.contact_planner.franka_tcp_ik import FrankaTcpIK
from dapl.contact_planner.isaac_visualization import (
    M1MarkerUpdateWrapper,
    create_m1_video_markers,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp


FRANKA_URDF = str(
    Path(isaaclab.__file__).resolve().parent
    / "controllers/config/data/lula_franka_gen.urdf"
)


def _never_terminate(env) -> torch.Tensor:
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


def _quaternion_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(first, dim=-1)
    second = torch.nn.functional.normalize(second, dim=-1)
    chord = torch.minimum(
        torch.linalg.vector_norm(first - second, dim=-1),
        torch.linalg.vector_norm(first + second, dim=-1),
    )
    # For unit quaternions the sign-invariant chord is
    # ``2 sin(theta / 4)``, not ``2 sin(theta / 2)``.
    return 4.0 * torch.asin(torch.clamp(0.5 * chord, max=1.0))


def _pose_errors(base) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = base.scene["target"]
    goal = base.command_manager.get_command("target_object_pose")
    position = target.data.root_pos_w[:, :3] - base.scene.env_origins
    delta = goal[:, :3] - position
    return (
        torch.linalg.vector_norm(delta[:, :2], dim=1),
        torch.abs(delta[:, 2]),
        _quaternion_distance(target.data.root_quat_w, goal[:, 3:7]),
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> int:
    if min(args_cli.settle_steps, args_cli.warmup_steps, args_cli.final_hold_steps) < 0:
        raise ValueError("step counts must be non-negative")
    if args_cli.dwell_steps <= 0 or args_cli.servo_gain <= 0.0:
        raise ValueError("dwell-steps and servo-gain must be positive")
    manifest = args_cli.manifest.expanduser().resolve()
    trajectory_path = args_cli.trajectory_csv.expanduser().resolve()
    if not manifest.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("manifest and trajectory CSV must exist")

    os.environ["DAPL_CLUTTER_MANIFEST"] = str(manifest)
    os.environ["DAPL_CLUTTER_ASSET_SOURCE"] = "domino"
    os.environ["DOMINO_ROOT"] = str(args_cli.domino_root.expanduser().resolve())
    os.environ["DOMINO_USD_ROOT"] = str(
        args_cli.domino_usd_root.expanduser().resolve()
    )

    source_samples = load_c3_joint_trajectory(trajectory_path)
    source_samples = tuple(sample for sample in source_samples if sample.ee_xyz_m is not None)
    if len(source_samples) < 2:
        raise ValueError("trajectory has fewer than two samples with TCP positions")
    first_time = source_samples[0].time_s
    source_samples = tuple(
        replace(sample, time_s=sample.time_s - first_time) for sample in source_samples
    )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    env_cfg.seed = args_cli.seed
    env_cfg.disable_obs_noise = True
    env_cfg.actions.arm_action.scale = args_cli.joint_action_scale_rad
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
    env_cfg.episode_length_s = 180.0
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

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    try:
        env.reset()
        base = env.unwrapped
        control_period_s = float(base.step_dt)
        samples = resample_c3_joint_trajectory(source_samples, control_period_s)
        if args_cli.video:
            markers = create_m1_video_markers(
                env, goal_ghost_opacity=args_cli.goal_ghost_opacity
            )
            env = M1MarkerUpdateWrapper(env, markers)
            video_length = args_cli.video_length or (
                args_cli.settle_steps
                + (args_cli.warmup_steps if args_cli.initialization_mode == "servo" else 0)
                + len(samples)
                + args_cli.final_hold_steps
            )
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(args_cli.video_folder.expanduser().resolve()),
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                name_prefix=args_cli.video_name_prefix,
                disable_logger=True,
            )

        robot = base.scene["robot"]
        robot_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        robot_cfg.resolve(base.scene)
        joint_ids = robot_cfg.joint_ids
        zero_action = torch.zeros((1, 7), device=base.device)
        for _ in range(args_cli.settle_steps):
            env.step(zero_action)

        initial_target_pose = torch.cat(
            (
                base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins,
                base.scene["target"].data.root_quat_w,
            ),
            dim=1,
        ).clone()
        goal_pose = base.command_manager.get_command("target_object_pose").clone()
        initial_q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        tcp_position = (
            base.scene["ee_frame"].data.target_pos_w[0, 0]
            - base.scene.env_origins[0]
        ).detach().cpu().numpy()
        tcp_rotation = matrix_from_quat(
            base.scene["ee_frame"].data.target_quat_w[0, 0]
        ).detach().cpu().numpy()

        ik = FrankaTcpIK(
            FRANKA_URDF,
            max_evaluations=args_cli.ik_max_evaluations,
        )
        action_limit_margin = float(
            getattr(base.cfg.actions.arm_action, "joint_limit_margin", 0.0)
        )
        soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
        ik.lower = np.maximum(ik.lower, soft_limits[:, 0] + action_limit_margin + 1.0e-4)
        ik.upper = np.minimum(ik.upper, soft_limits[:, 1] - action_limit_margin - 1.0e-4)
        bridge = ik.bridge(initial_q, tcp_position, tcp_rotation)
        env_from_pin_rotation = bridge[1]

        desired_joint_positions: list[np.ndarray] = []
        ik_position_errors: list[float] = []
        ik_rotation_errors: list[float] = []
        seed = np.clip(np.asarray(samples[0].q), ik.lower, ik.upper)
        for index, sample in enumerate(samples):
            desired_position = np.asarray(sample.ee_xyz_m, dtype=float)
            desired_position[2] += args_cli.table_height_offset_m
            desired_rotation = env_from_pin_rotation @ ik.forward(
                np.asarray(sample.q, dtype=float)
            ).rotation
            solved, position_error, rotation_error = ik.solve(
                seed=seed,
                regularization_reference=np.asarray(sample.q, dtype=float),
                bridge=bridge,
                desired_tcp_env=desired_position,
                desired_rotation_env=desired_rotation,
            )
            desired_joint_positions.append(solved)
            ik_position_errors.append(position_error)
            ik_rotation_errors.append(rotation_error)
            seed = solved
            if index and index % 100 == 0:
                print(
                    "ISAAC_BRIDGE_IK",
                    f"sample={index}/{len(samples)}",
                    f"max_position_error_m={max(ik_position_errors):.6f}",
                    f"max_rotation_error_rad={max(ik_rotation_errors):.6f}",
                    flush=True,
                )

        action_scale = float(base.cfg.actions.arm_action.scale)
        safe_contact_ever = False
        forbidden_contact_ever = False
        forbidden_hand_contact_ever = False
        arm_target_contact_ever = False
        strict_pose_ever = False
        strict_dwell = 0
        maximum_strict_dwell = 0
        minimum_planar = math.inf
        minimum_height = math.inf
        minimum_rotation = math.inf
        minimum_safe_distance = math.inf
        minimum_forbidden_distance = math.inf
        q_tracking_errors: list[float] = []
        tcp_tracking_errors: list[float] = []
        execution_step = 0
        first_safe_contact_step: int | None = None
        first_forbidden_contact_step: int | None = None
        stopped_on_success = False
        success_stop_step: int | None = None
        executed_trajectory_steps = 0

        def observe(desired_q: np.ndarray | None, desired_tcp: np.ndarray | None) -> None:
            nonlocal safe_contact_ever, forbidden_contact_ever
            nonlocal forbidden_hand_contact_ever, arm_target_contact_ever
            nonlocal strict_pose_ever, strict_dwell, maximum_strict_dwell
            nonlocal minimum_planar, minimum_height, minimum_rotation
            nonlocal minimum_safe_distance, minimum_forbidden_distance
            nonlocal execution_step, first_safe_contact_step, first_forbidden_contact_step
            execution_step += 1
            planar, height, rotation = _pose_errors(base)
            p, h, r = float(planar[0]), float(height[0]), float(rotation[0])
            minimum_planar = min(minimum_planar, p)
            minimum_height = min(minimum_height, h)
            minimum_rotation = min(minimum_rotation, r)
            strict = p < 0.02 and h < 0.01 and r < 0.10
            strict_dwell = strict_dwell + 1 if strict else 0
            maximum_strict_dwell = max(maximum_strict_dwell, strict_dwell)
            strict_pose_ever = strict_pose_ever or strict
            contact = mdp.domino_affordance_contact_state(
                base,
                contact_distance_m=args_cli.contact_distance_m,
                evaluate_protected=False,
            )
            safe_contact_ever = safe_contact_ever or bool(contact["safe_robot_contact"][0])
            forbidden_contact_ever = forbidden_contact_ever or bool(
                contact["forbidden_robot_contact"][0]
            )
            if first_safe_contact_step is None and bool(contact["safe_robot_contact"][0]):
                first_safe_contact_step = execution_step
            if first_forbidden_contact_step is None and bool(
                contact["forbidden_robot_contact"][0]
            ):
                first_forbidden_contact_step = execution_step
            forbidden_hand_contact_ever = forbidden_hand_contact_ever or bool(
                contact["forbidden_hand_contact"][0]
            )
            arm_target_contact_ever = arm_target_contact_ever or bool(
                contact["arm_target_physical_contact"][0]
            )
            minimum_safe_distance = min(
                minimum_safe_distance, float(contact["minimum_safe_distance"][0])
            )
            minimum_forbidden_distance = min(
                minimum_forbidden_distance,
                float(contact["minimum_robot_forbidden_distance"][0]),
            )
            if desired_q is not None:
                actual_q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
                q_tracking_errors.append(float(np.linalg.norm(actual_q - desired_q)))
            if desired_tcp is not None:
                actual_tcp = (
                    base.scene["ee_frame"].data.target_pos_w[0, 0]
                    - base.scene.env_origins[0]
                ).detach().cpu().numpy()
                tcp_tracking_errors.append(float(np.linalg.norm(actual_tcp - desired_tcp)))

        first_q = torch.as_tensor(
            desired_joint_positions[0], device=base.device, dtype=robot.data.joint_pos.dtype
        ).unsqueeze(0)
        if args_cli.initialization_mode == "teleport":
            robot.write_joint_state_to_sim(
                first_q,
                torch.zeros_like(first_q),
                joint_ids=joint_ids,
            )
            base.sim.forward()
        else:
            warmup_start = robot.data.joint_pos[:, joint_ids].clone()
            for step in range(args_cli.warmup_steps):
                alpha = (step + 1) / max(args_cli.warmup_steps, 1)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                desired = warmup_start + alpha * (first_q - warmup_start)
                current = robot.data.joint_pos[:, joint_ids]
                action = torch.clamp(
                    args_cli.servo_gain * (desired - current) / action_scale, -1.0, 1.0
                )
                env.step(action)
                observe(desired_joint_positions[0], None)

        for sample, desired_q_np in zip(samples, desired_joint_positions):
            executed_trajectory_steps += 1
            desired_q = torch.as_tensor(
                desired_q_np, device=base.device, dtype=robot.data.joint_pos.dtype
            ).unsqueeze(0)
            current = robot.data.joint_pos[:, joint_ids]
            action = torch.clamp(
                args_cli.servo_gain * (desired_q - current) / action_scale, -1.0, 1.0
            )
            env.step(action)
            desired_tcp = np.asarray(sample.ee_xyz_m, dtype=float)
            desired_tcp[2] += args_cli.table_height_offset_m
            observe(desired_q_np, desired_tcp)
            if (
                args_cli.stop_on_success
                and strict_dwell >= args_cli.dwell_steps
                and safe_contact_ever
                and not forbidden_contact_ever
            ):
                stopped_on_success = True
                success_stop_step = execution_step
                break

        for _ in range(0 if stopped_on_success else args_cli.final_hold_steps):
            desired_q_np = desired_joint_positions[-1]
            desired_q = torch.as_tensor(
                desired_q_np, device=base.device, dtype=robot.data.joint_pos.dtype
            ).unsqueeze(0)
            current = robot.data.joint_pos[:, joint_ids]
            action = torch.clamp(
                args_cli.servo_gain * (desired_q - current) / action_scale, -1.0, 1.0
            )
            env.step(action)
            observe(desired_q_np, None)

        final_planar, final_height, final_rotation = _pose_errors(base)
        final_target_pose = torch.cat(
            (
                base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins,
                base.scene["target"].data.root_quat_w,
            ),
            dim=1,
        )
        final_position = final_target_pose[0, :3]
        initial_position = initial_target_pose[0, :3]
        strict_geometry = (
            float(final_planar[0]) < 0.02
            and float(final_height[0]) < 0.01
            and float(final_rotation[0]) < 0.10
        )
        result = {
            "schema": "nonprehensile.push_anything_isaaclab_smoke.v1",
            "executor_mode": "c3_tcp_path_to_isaaclab_ik_joint_servo",
            "deployment_ready": False,
            "deployment_blocker": "recorded_nominal_c3_trajectory_not_online_replanning",
            "initialization_mode": args_cli.initialization_mode,
            "task": args_cli.task,
            "manifest": str(manifest),
            "trajectory_csv": str(trajectory_path),
            "control_period_s": control_period_s,
            "source_joint_samples": len(source_samples),
            "planned_samples": len(samples),
            "executed_samples": executed_trajectory_steps,
            "table_height_offset_m": args_cli.table_height_offset_m,
            "ik_path_valid": bool(
                max(ik_position_errors) <= args_cli.ik_position_tolerance_m
                and max(ik_rotation_errors) <= args_cli.ik_rotation_tolerance_rad
            ),
            "ik_max_position_error_m": max(ik_position_errors),
            "ik_max_rotation_error_rad": max(ik_rotation_errors),
            "joint_tracking_rmse_rad": math.sqrt(
                sum(value * value for value in q_tracking_errors) / len(q_tracking_errors)
            ),
            "joint_tracking_max_l2_rad": max(q_tracking_errors),
            "tcp_tracking_rmse_m": math.sqrt(
                sum(value * value for value in tcp_tracking_errors) / len(tcp_tracking_errors)
            ),
            "tcp_tracking_max_m": max(tcp_tracking_errors),
            "initial_target_pose_wxyz": initial_target_pose[0].detach().cpu().tolist(),
            "goal_pose_wxyz": goal_pose[0].detach().cpu().tolist(),
            "final_target_pose_wxyz": final_target_pose[0].detach().cpu().tolist(),
            "target_translation_m": float(torch.linalg.vector_norm(final_position - initial_position)),
            "final_planar_error_m": float(final_planar[0]),
            "final_height_error_m": float(final_height[0]),
            "final_rotation_error_rad": float(final_rotation[0]),
            "minimum_planar_error_m": _finite_or_none(minimum_planar),
            "minimum_height_error_m": _finite_or_none(minimum_height),
            "minimum_rotation_error_rad": _finite_or_none(minimum_rotation),
            "maximum_strict_dwell_steps": maximum_strict_dwell,
            "stop_on_success": args_cli.stop_on_success,
            "stopped_on_success": stopped_on_success,
            "success_stop_step": success_stop_step,
            "strict_geometry_final": strict_geometry,
            "strict_pose_ever": strict_pose_ever,
            "safe_robot_contact_ever": safe_contact_ever,
            "forbidden_robot_contact_ever": forbidden_contact_ever,
            "forbidden_hand_contact_ever": forbidden_hand_contact_ever,
            "arm_target_physical_contact_ever": arm_target_contact_ever,
            "minimum_safe_distance_m": _finite_or_none(minimum_safe_distance),
            "minimum_robot_forbidden_distance_m": _finite_or_none(
                minimum_forbidden_distance
            ),
            "first_safe_contact_step": first_safe_contact_step,
            "first_forbidden_contact_step": first_forbidden_contact_step,
            "isaac_c1_pass": safe_contact_ever and not forbidden_contact_ever,
            "isaac_joint_acceptance": bool(
                strict_geometry
                and maximum_strict_dwell >= args_cli.dwell_steps
                and safe_contact_ever
                and not forbidden_contact_ever
            ),
        }
        output = args_cli.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("ISAAC_BRIDGE_RESULT", json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
