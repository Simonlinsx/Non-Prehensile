#!/usr/bin/env python3
"""Reject DOMINO scenes whose rigid bodies move under zero robot action."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AffordanceHammer-Pose-Franka-v0")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--settle-steps", type=int, default=30)
parser.add_argument(
    "--preserve-initial-task",
    action="store_true",
    help=(
        "audit the constructor-reset task used by eval.py instead of calling "
        "reset() once more and advancing every scene to its next task"
    ),
)
parser.add_argument("--translation-threshold", type=float, default=0.003)
parser.add_argument("--rotation-threshold", type=float, default=0.03)
parser.add_argument("--linear-speed-threshold", type=float, default=0.01)
parser.add_argument("--angular-speed-threshold", type=float, default=0.10)
parser.add_argument(
    "--physical-contact-force-threshold",
    type=float,
    default=None,
    help=(
        "Optionally override the task contact-sensor threshold during this "
        "audit. This is useful for stricter reset-admission checks without "
        "changing the training/evaluation task contract."
    ),
)
parser.add_argument(
    "--reject-constraint",
    action="append",
    choices=("c1", "c2", "c3", "target_obstacle_contact"),
    default=[],
    help=(
        "Reject a scene when the selected typed constraint is observed during "
        "zero-action settling. Repeat the flag to audit multiple constraints."
    ),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--filtered-manifest-output",
    type=Path,
    default=None,
    help="Optionally write the audited stable subset before returning failure.",
)
parser.add_argument(
    "--settled-manifest-output",
    type=Path,
    default=None,
    help=(
        "Optionally rewrite every active obstacle pose from its final PhysX "
        "state. This requires one environment per manifest scene and no "
        "episode reset during settling. The target task is left unchanged."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from dapl.scene import load_scene_manifest, write_scene_manifest


def _rotation_distance(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(start * end, dim=-1).abs()
    return 2.0 * torch.acos(torch.clamp(dot, max=1.0))


def _summary(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"maximum": 0.0, "mean": 0.0, "p95": 0.0}
    flat = values.reshape(-1).float()
    return {
        "maximum": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
    }


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.settle_steps <= 0:
        raise ValueError("num-envs and settle-steps must be positive")
    for name in (
        "translation_threshold",
        "rotation_threshold",
        "linear_speed_threshold",
        "angular_speed_threshold",
    ):
        if getattr(args_cli, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if (
        args_cli.physical_contact_force_threshold is not None
        and args_cli.physical_contact_force_threshold < 0.0
    ):
        raise ValueError("physical-contact-force-threshold must be non-negative")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    if args_cli.physical_contact_force_threshold is not None:
        env_cfg.physical_contact_force_threshold_n = float(
            args_cli.physical_contact_force_threshold
        )
    print(
        "[audit] parsed environment: "
        f"task={args_cli.task} "
        f"configured_active_obstacles={getattr(env_cfg, 'active_obstacle_count', None)} "
        f"manifest={getattr(env_cfg, 'clutter_manifest_path', None)}",
        flush=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        # Isaac Lab 2.2's raw ManagerBasedRLEnv does not run reset events in
        # gym.make().  Older versions did, which is why the original audit
        # treated env.reset() here as an optional *second* task advance.  Key
        # off the manifest reset cache instead: initialize exactly once when
        # needed, and only request another reset when the caller explicitly
        # does not preserve the first task.
        base = env.unwrapped
        performed_initial_reset = not hasattr(
            base, "_clutter_initial_obstacle_pose"
        )
        if performed_initial_reset or not args_cli.preserve_initial_task:
            env.reset()
        target = base.scene["target"]
        obstacles = base.scene["obstacles"]
        active_obstacles = int(
            getattr(base, "_clutter_active_obstacle_count", obstacles.num_objects)
        )
        spawned_obstacles = obstacles.num_objects
        print(
            "[audit] instantiated environment: "
            f"runtime_active_obstacles={active_obstacles} "
            f"spawned_obstacles={spawned_obstacles}",
            flush=True,
        )

        obstacle_cfgs = tuple(base.cfg.scene.obstacles.rigid_objects.values())
        obstacle_slot_cfg = []
        config_invalid = False
        kinematic_active_obstacles = bool(
            getattr(base.cfg, "kinematic_active_obstacles", False)
        )
        for obstacle_index, obstacle_cfg in enumerate(obstacle_cfgs):
            expected_enabled = obstacle_index < active_obstacles
            expected_kinematic = (
                not expected_enabled
            ) or kinematic_active_obstacles
            asset_cfgs = tuple(obstacle_cfg.spawn.assets_cfg)
            slot_valid = all(
                bool(asset_cfg.visible) == expected_enabled
                and bool(asset_cfg.collision_props.collision_enabled)
                == expected_enabled
                and bool(asset_cfg.rigid_props.disable_gravity)
                == expected_kinematic
                and bool(asset_cfg.rigid_props.kinematic_enabled)
                == expected_kinematic
                for asset_cfg in asset_cfgs
            )
            config_invalid |= not slot_valid
            obstacle_slot_cfg.append(
                {
                    "index": obstacle_index,
                    "expected_enabled": expected_enabled,
                    "asset_count": len(asset_cfgs),
                    "valid": slot_valid,
                }
            )

        target_start_position = (
            target.data.root_pos_w[:, :3] - base.scene.env_origins
        ).clone()
        target_start_quaternion = target.data.root_quat_w.clone()
        obstacle_start_position = (
            obstacles.data.object_pos_w
            - base.scene.env_origins.unsqueeze(1)
        ).clone()
        obstacle_start_quaternion = obstacles.data.object_quat_w.clone()
        expected_obstacle_pose = base._clutter_initial_obstacle_pose.clone()
        obstacle_start_pose_translation_error = torch.linalg.vector_norm(
            obstacle_start_position - expected_obstacle_pose[..., :3], dim=-1
        )
        obstacle_start_pose_rotation_error = _rotation_distance(
            obstacle_start_quaternion, expected_obstacle_pose[..., 3:7]
        )

        ended = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        constraint_state_keys = {
            "c1": "forbidden_robot_contact",
            "c2": "protected_obstacle_collision",
            "c3": "robot_obstacle_collision",
            # C2 is deliberately semantic: with the physical-contact gate on,
            # contact on a non-protected target point is not a C2 violation.
            # Scene admission still needs to reject *any* target-obstacle
            # interpenetration at reset, so expose the raw sensor event too.
            "target_obstacle_contact": "target_obstacle_physical_contact",
        }
        observed_contact_names = tuple(
            dict.fromkeys((*args_cli.reject_constraint, "target_obstacle_contact"))
        )
        zero_action_constraint_violation = {
            name: torch.zeros(
                base.num_envs, dtype=torch.bool, device=base.device
            )
            for name in observed_contact_names
        }
        actions = torch.zeros((base.num_envs, 7), device=base.device)
        for _ in range(args_cli.settle_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            ended |= terminated | truncated
            if zero_action_constraint_violation:
                state = base._metric_contact_state()
                for name, seen in zero_action_constraint_violation.items():
                    seen |= state[constraint_state_keys[name]]

        target_end_position = target.data.root_pos_w[:, :3] - base.scene.env_origins
        target_translation = torch.linalg.vector_norm(
            target_end_position - target_start_position, dim=-1
        )
        target_rotation = _rotation_distance(
            target_start_quaternion, target.data.root_quat_w
        )
        target_linear_speed = torch.linalg.vector_norm(
            target.data.root_com_lin_vel_w, dim=-1
        )
        target_angular_speed = torch.linalg.vector_norm(
            target.data.root_com_ang_vel_w, dim=-1
        )

        obstacle_end_position = (
            obstacles.data.object_pos_w
            - base.scene.env_origins.unsqueeze(1)
        )
        obstacle_translation = torch.linalg.vector_norm(
            obstacle_end_position - obstacle_start_position, dim=-1
        )
        obstacle_rotation = _rotation_distance(
            obstacle_start_quaternion,
            obstacles.data.object_quat_w,
        )
        obstacle_linear_speed = torch.linalg.vector_norm(
            obstacles.data.object_com_lin_vel_w, dim=-1
        )
        obstacle_angular_speed = torch.linalg.vector_norm(
            obstacles.data.object_com_ang_vel_w, dim=-1
        )
        obstacle_end_pose_translation_error = torch.linalg.vector_norm(
            obstacle_end_position - expected_obstacle_pose[..., :3], dim=-1
        )
        obstacle_end_pose_rotation_error = _rotation_distance(
            obstacles.data.object_quat_w, expected_obstacle_pose[..., 3:7]
        )

        target_invalid = (
            (target_translation > args_cli.translation_threshold)
            | (target_rotation > args_cli.rotation_threshold)
            | (target_linear_speed > args_cli.linear_speed_threshold)
            | (target_angular_speed > args_cli.angular_speed_threshold)
        )
        obstacle_invalid_by_slot = (
            (obstacle_translation > args_cli.translation_threshold)
            | (obstacle_rotation > args_cli.rotation_threshold)
            | (obstacle_linear_speed > args_cli.linear_speed_threshold)
            | (obstacle_angular_speed > args_cli.angular_speed_threshold)
            | (obstacle_start_pose_translation_error > args_cli.translation_threshold)
            | (obstacle_start_pose_rotation_error > args_cli.rotation_threshold)
            | (obstacle_end_pose_translation_error > args_cli.translation_threshold)
            | (obstacle_end_pose_rotation_error > args_cli.rotation_threshold)
        )
        obstacle_invalid = obstacle_invalid_by_slot.any(dim=1)
        constraint_invalid = torch.zeros_like(target_invalid)
        for name in args_cli.reject_constraint:
            constraint_invalid |= zero_action_constraint_violation[name]
        invalid = target_invalid | obstacle_invalid | ended | constraint_invalid
        if config_invalid:
            invalid |= True
        invalid_env_ids = torch.nonzero(invalid, as_tuple=False).flatten().tolist()
        invalid_obstacle_slots = [
            {"env_id": int(env_id), "obstacle_index": int(obstacle_index)}
            for env_id, obstacle_index in torch.nonzero(
                obstacle_invalid_by_slot, as_tuple=False
            ).tolist()
        ]

        active_slice = slice(0, active_obstacles)
        inactive_slice = slice(active_obstacles, spawned_obstacles)

        def obstacle_summary(object_slice: slice) -> dict:
            return {
                "translation_m": _summary(obstacle_translation[:, object_slice]),
                "rotation_rad": _summary(obstacle_rotation[:, object_slice]),
                "linear_speed_m_s": _summary(obstacle_linear_speed[:, object_slice]),
                "angular_speed_rad_s": _summary(obstacle_angular_speed[:, object_slice]),
                "start_pose_translation_error_m": _summary(
                    obstacle_start_pose_translation_error[:, object_slice]
                ),
                "start_pose_rotation_error_rad": _summary(
                    obstacle_start_pose_rotation_error[:, object_slice]
                ),
                "end_pose_translation_error_m": _summary(
                    obstacle_end_pose_translation_error[:, object_slice]
                ),
                "end_pose_rotation_error_rad": _summary(
                    obstacle_end_pose_rotation_error[:, object_slice]
                ),
            }

        report = {
            "task": args_cli.task,
            "num_envs": base.num_envs,
            "settle_steps": args_cli.settle_steps,
            "preserved_initial_task": bool(args_cli.preserve_initial_task),
            "performed_initial_reset": performed_initial_reset,
            "active_obstacle_count": active_obstacles,
            "kinematic_active_obstacles": kinematic_active_obstacles,
            "spawned_obstacle_count": spawned_obstacles,
            "obstacle_slot_config": obstacle_slot_cfg,
            "thresholds": {
                "translation_m": args_cli.translation_threshold,
                "rotation_rad": args_cli.rotation_threshold,
                "linear_speed_m_s": args_cli.linear_speed_threshold,
                "angular_speed_rad_s": args_cli.angular_speed_threshold,
                "physical_contact_force_n": float(
                    getattr(base.cfg, "physical_contact_force_threshold_n", 0.5)
                ),
            },
            "rejected_constraints": list(args_cli.reject_constraint),
            "zero_action_constraint_violations": {
                name: {
                    "count": int(seen.sum().item()),
                    "fraction": float(seen.float().mean().item()),
                    "env_ids": torch.nonzero(
                        seen, as_tuple=False
                    ).flatten().tolist(),
                }
                for name, seen in zero_action_constraint_violation.items()
            },
            "target": {
                "translation_m": _summary(target_translation),
                "rotation_rad": _summary(target_rotation),
                "linear_speed_m_s": _summary(target_linear_speed),
                "angular_speed_rad_s": _summary(target_angular_speed),
            },
            "obstacles": obstacle_summary(slice(0, spawned_obstacles)),
            "active_obstacles": obstacle_summary(active_slice),
            "inactive_obstacles": obstacle_summary(inactive_slice),
            "ended_during_settle_env_ids": torch.nonzero(
                ended, as_tuple=False
            ).flatten().tolist(),
            "invalid_obstacle_slots": invalid_obstacle_slots,
            "invalid_env_ids": invalid_env_ids,
            "passed": not invalid_env_ids and not config_invalid,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if (
            args_cli.filtered_manifest_output is not None
            or args_cli.settled_manifest_output is not None
        ):
            manifest_scenes = tuple(
                load_scene_manifest(base.cfg.clutter_manifest_path)
            )
            if len(manifest_scenes) != base.num_envs:
                raise ValueError(
                    "--filtered-manifest-output requires num-envs to equal "
                    "the source manifest scene count"
                )
            if args_cli.settled_manifest_output is not None:
                if bool(ended.any().item()):
                    raise RuntimeError(
                        "cannot export settled poses after an episode reset"
                    )
                scene_indices = base._clutter_scene_indices.tolist()
                if sorted(int(index) for index in scene_indices) != list(
                    range(len(manifest_scenes))
                ):
                    raise RuntimeError(
                        "settled pose export requires a one-to-one environment/scene mapping"
                    )
                settled_scenes = list(manifest_scenes)
                for env_id, scene_index_value in enumerate(scene_indices):
                    scene_index = int(scene_index_value)
                    scene = manifest_scenes[scene_index]
                    settled_obstacles = list(scene.obstacle_objects)
                    for obstacle_index in range(active_obstacles):
                        pose_values = torch.cat(
                            (
                                obstacle_end_position[env_id, obstacle_index],
                                obstacles.data.object_quat_w[
                                    env_id, obstacle_index
                                ],
                            )
                        ).tolist()
                        settled_obstacles[obstacle_index] = replace(
                            settled_obstacles[obstacle_index],
                            pose=tuple(float(value) for value in pose_values),
                        )
                    settled_scenes[scene_index] = replace(
                        scene,
                        objects=(scene.target_object, *settled_obstacles),
                    )
                write_scene_manifest(
                    args_cli.settled_manifest_output, tuple(settled_scenes)
                )
                report["settled_manifest_output"] = str(
                    args_cli.settled_manifest_output
                )
                report["settled_scene_count"] = len(settled_scenes)
            if args_cli.filtered_manifest_output is not None:
                invalid_set = set(invalid_env_ids)
                stable_scenes = tuple(
                    scene
                    for index, scene in enumerate(manifest_scenes)
                    if index not in invalid_set
                )
                if not stable_scenes:
                    raise RuntimeError("settling audit rejected every source scene")
                write_scene_manifest(args_cli.filtered_manifest_output, stable_scenes)
                report["filtered_manifest_output"] = str(
                    args_cli.filtered_manifest_output
                )
                report["filtered_scene_count"] = len(stable_scenes)
            args_cli.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if invalid_env_ids:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() may terminate the process before Python's
        # default exception hook flushes, hiding configuration/import errors
        # behind an apparent zero exit.  Emit the traceback first.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
