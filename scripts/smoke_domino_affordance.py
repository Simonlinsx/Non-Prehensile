#!/usr/bin/env python3
"""Bounded reset/step smoke test for the DOMINO affordance task."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AffordanceClutter6D-Franka-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument(
    "--robot-joint-pos",
    type=float,
    nargs=7,
    default=None,
    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
    help="Override the seven Franka arm reset joints before creating the scene.",
)
parser.add_argument(
    "--print-reset-geometry",
    action="store_true",
    help="Print target, TCP, fingertip, and semantic safe-region distances after reset.",
)
parser.add_argument(
    "--expected-active-obstacles",
    type=int,
    default=None,
    help="Assert the runtime active-obstacle count (use 0 for strict Stage 0/1).",
)
parser.add_argument(
    "--expected-max-episode-steps",
    type=int,
    default=None,
    help="Assert the runtime policy-step horizon (DAPL uses 300).",
)
parser.add_argument(
    "--print-prim-tree",
    action="store_true",
    help="Print env_0 robot/target/obstacle prim types for contact-sensor diagnostics.",
)
parser.add_argument(
    "--print-protected-geodesic",
    action="store_true",
    help="Print the optional C2 geodesic homotopy and route diagnostics.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[domino-smoke] Isaac Sim application launched", flush=True)


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401

print("[domino-smoke] task modules imported", flush=True)


def main() -> None:
    spec = gym.spec(args_cli.task)
    print(
        "[domino-smoke] registry entry",
        f"entry_point={spec.entry_point}",
        f"env_cfg_entry_point={spec.kwargs.get('env_cfg_entry_point')}",
        flush=True,
    )
    print("[domino-smoke] building environment config", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    if args_cli.robot_joint_pos is not None:
        joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
        joint_pos.update(
            {
                f"panda_joint{index + 1}": value
                for index, value in enumerate(args_cli.robot_joint_pos)
            }
        )
        env_cfg.scene.robot.init_state.joint_pos = joint_pos
    print("[domino-smoke] creating Gym environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if args_cli.print_prim_tree:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prefixes = (
                "/World/envs/env_0/Robot",
                "/World/envs/env_0/Target",
                "/World/envs/env_0/Obstacle_",
            )
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if path.startswith(prefixes):
                    applied = [str(schema) for schema in prim.GetAppliedSchemas()]
                    if prim.GetTypeName() or applied:
                        print(
                            "[prim-tree]",
                            path,
                            f"type={prim.GetTypeName()}",
                            f"schemas={applied}",
                            flush=True,
                        )
        observations, _ = env.reset()
        base = env.unwrapped
        if (
            args_cli.expected_max_episode_steps is not None
            and base.max_episode_length != args_cli.expected_max_episode_steps
        ):
            raise RuntimeError(
                "unexpected episode horizon: "
                f"expected {args_cli.expected_max_episode_steps}, "
                f"got {base.max_episode_length}"
            )
        policy = observations["policy"]
        expected_policy_dim = (
            4141 if args_cli.task.startswith("Isaac-AffordanceTeacher-") else 4146
        )
        if policy.shape != (args_cli.num_envs, expected_policy_dim):
            raise RuntimeError(
                f"unexpected policy observation shape: {tuple(policy.shape)}"
            )
        semantic_target = policy[:, :2560].reshape(args_cli.num_envs, 512, 5)
        obstacle_tokens = policy[:, 2560:4096].reshape(
            args_cli.num_envs, 512, 3
        )
        rel_goal = policy[:, 4126:4135]
        target_position = (
            base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        )
        target_goal = base.command_manager.get_command("target_object_pose")
        runtime_scenes = base._clutter_scenes_runtime
        runtime_scene_indices = base._clutter_scene_indices.detach().cpu().tolist()
        runtime_task_indices = base._clutter_task_indices.detach().cpu().tolist()
        expected_initial_pose = torch.tensor(
            [
                runtime_scenes[scene_index].tasks[task_index].initial_pose
                for scene_index, task_index in zip(
                    runtime_scene_indices, runtime_task_indices, strict=True
                )
            ],
            device=base.device,
            dtype=target_position.dtype,
        )
        expected_goal_pose = torch.tensor(
            [
                runtime_scenes[scene_index].tasks[task_index].goal_pose
                for scene_index, task_index in zip(
                    runtime_scene_indices, runtime_task_indices, strict=True
                )
            ],
            device=base.device,
            dtype=target_goal.dtype,
        )
        torch.testing.assert_close(
            target_position[:, :3], expected_initial_pose[:, :3], rtol=0.0, atol=1.0e-5
        )
        torch.testing.assert_close(
            base.scene["target"].data.root_quat_w,
            expected_initial_pose[:, 3:7],
            rtol=0.0,
            atol=1.0e-5,
        )
        torch.testing.assert_close(
            target_goal, expected_goal_pose, rtol=0.0, atol=1.0e-6
        )
        expected_rel_position = (target_goal[:, :3] - target_position) / torch.tensor(
            [0.10, 0.10, 0.02], device=base.device
        )
        torch.testing.assert_close(
            rel_goal[:, :3], expected_rel_position, rtol=1.0e-5, atol=1.0e-5
        )
        aligned = semantic_target[..., 3:5]
        if not torch.isfinite(policy).all():
            raise RuntimeError("non-finite observation after reset")
        if not torch.all(aligned.amax(dim=1) > 0.0):
            raise RuntimeError("one or more target affordance channels are empty")
        safe_mask = aligned[..., 0] >= 0.25
        protected_mask = aligned[..., 1] >= 0.25
        if torch.any(safe_mask & protected_mask):
            raise RuntimeError("safe and protected target masks overlap")
        safe_counts = safe_mask.sum(dim=1)
        protected_counts = protected_mask.sum(dim=1)
        if args_cli.print_reset_geometry:
            from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp

            ee_positions = (
                base.scene["ee_frame"].data.target_pos_w - base.scene.env_origins[:, None, :]
            )
            contact_state = mdp.domino_affordance_contact_state(
                base,
                evaluate_protected=base.cfg.curriculum_stage >= 2,
                evaluate_robot_obstacle=bool(
                    getattr(base.cfg, "evaluate_robot_obstacle", False)
                ),
                robot_target_sensor_name=getattr(
                    base.cfg, "robot_target_sensor_name", None
                ),
                robot_obstacle_sensor_name=getattr(
                    base.cfg, "robot_obstacle_sensor_name", None
                ),
                target_obstacle_sensor_name=getattr(
                    base.cfg, "target_obstacle_sensor_name", None
                ),
            )
            print(
                "[reset-geometry]",
                f"target0={target_position[0].tolist()}",
                "robot_joint0="
                f"{base.scene['robot'].data.joint_pos[0, :7].tolist()}",
                f"tcp0={ee_positions[0, 0].tolist()}",
                f"left_finger0={ee_positions[0, 1].tolist()}",
                f"right_finger0={ee_positions[0, 2].tolist()}",
                "minimum_safe_distance_range="
                f"({contact_state['minimum_safe_distance'].min().item():.6f},"
                f"{contact_state['minimum_safe_distance'].max().item():.6f})",
                "contact_sensors="
                f"robot-target:{bool(contact_state['robot_target_sensor_available'][0])},"
                f"target-obstacle:{bool(contact_state['target_obstacle_sensor_available'][0])},"
                f"robot-obstacle:{bool(contact_state['robot_obstacle_sensor_available'][0])}",
                flush=True,
            )
        active_obstacle_count = int(base._clutter_active_obstacle_count)
        if (
            args_cli.expected_active_obstacles is not None
            and active_obstacle_count != args_cli.expected_active_obstacles
        ):
            raise RuntimeError(
                "unexpected active obstacle count: "
                f"expected {args_cli.expected_active_obstacles}, "
                f"got {active_obstacle_count}"
            )
        if active_obstacle_count == 0:
            if torch.count_nonzero(obstacle_tokens).item() != 0:
                raise RuntimeError("single-object stage has non-zero obstacle tokens")
            obstacle_positions = (
                base.scene["obstacles"].data.object_pos_w
                - base.scene.env_origins[:, None, :]
            )
            # Inactive rigid bodies stay on a valid support face so they cannot
            # fall or roll if accidentally re-enabled, but they must remain
            # well outside the robot/target workspace and absent from policy
            # observations.
            if not torch.all(torch.linalg.vector_norm(obstacle_positions[..., :2], dim=-1) > 1.0):
                raise RuntimeError("inactive obstacle bodies were not parked outside the workspace")

        # A non-zero delta must be added exactly once at the policy boundary,
        # not once per physics substep.  Reset immediately afterwards so the
        # rest of the smoke test still checks a stationary zero-action rollout.
        robot = base.scene["robot"]
        q_before = robot.data.joint_pos[:, :7].clone()
        latch_probe = torch.zeros(env.action_space.shape, device=base.device)
        latch_probe[:, 0] = 0.5
        env.step(latch_probe)
        action_term = base.action_manager.get_term("arm_action")
        expected_target = q_before.clone()
        expected_target[:, 0] += 0.5 * float(base.cfg.actions.arm_action.scale)
        limits = robot.data.soft_joint_pos_limits[:, :7]
        margin = float(base.cfg.actions.arm_action.joint_limit_margin)
        expected_target = torch.maximum(
            torch.minimum(expected_target, limits[..., 1] - margin),
            limits[..., 0] + margin,
        )
        torch.testing.assert_close(
            action_term.joint_position_target,
            expected_target,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        observations, _ = env.reset()

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=base.device)
                observations, reward, _, _, _ = env.step(actions)
                if not torch.isfinite(observations["policy"]).all():
                    raise RuntimeError("non-finite policy observation")
                if not torch.isfinite(reward).all():
                    raise RuntimeError("non-finite reward")

        if args_cli.print_protected_geodesic:
            required = (
                "_protected_geodesic_detour_side",
                "_protected_geodesic_direct_clearance",
                "_protected_geodesic_route_length",
            )
            missing = [name for name in required if not hasattr(base, name)]
            if missing:
                raise RuntimeError(
                    "protected geodesic diagnostics were not produced: "
                    + ", ".join(missing)
                )
            sides = base._protected_geodesic_detour_side.detach().cpu().tolist()
            direct_clearance = (
                base._protected_geodesic_direct_clearance.detach().cpu().tolist()
            )
            route_length = (
                base._protected_geodesic_route_length.detach().cpu().tolist()
            )
            scene_indices = base._clutter_scene_indices.detach().cpu().tolist()
            for env_index, scene_index in enumerate(scene_indices):
                print(
                    "[protected-geodesic]",
                    f"env={env_index}",
                    f"scene_id={base._clutter_scenes_runtime[scene_index].scene_id}",
                    f"detour_side={sides[env_index]}",
                    f"direct_clearance_m={direct_clearance[env_index]:.6f}",
                    f"route_length_m={route_length[env_index]:.6f}",
                    flush=True,
                )

        print(
            "DOMINO_AFFORDANCE_SMOKE_OK",
            f"num_envs={base.num_envs}",
            f"max_episode_steps={base.max_episode_length}",
            f"step_dt={base.step_dt:.6f}",
            f"policy_shape={tuple(policy.shape)}",
            f"semantic_target_shape={tuple(semantic_target.shape)}",
            f"safe_score_max={aligned[..., 0].max().item():.4f}",
            f"protected_score_max={aligned[..., 1].max().item():.4f}",
            f"safe_count_range=({safe_counts.min().item()},{safe_counts.max().item()})",
            "protected_count_range="
            f"({protected_counts.min().item()},{protected_counts.max().item()})",
            f"active_obstacles={active_obstacle_count}",
            f"obstacle_token_max_abs={obstacle_tokens.abs().max().item():.6f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() can terminate Kit before Python renders an
        # uncaught exception, so emit it explicitly for a useful smoke log.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
