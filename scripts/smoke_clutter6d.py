"""Bounded reset/step smoke test for the manifest-backed Clutter6D task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Clutter6D-Franka-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        base = env.unwrapped
        assert base.scene["target"].data.root_pos_w.shape == (args_cli.num_envs, 3)
        assert base.scene["obstacles"].data.object_pos_w.shape[:2] == (
            args_cli.num_envs,
            3,
        )
        expected_scene = torch.arange(
            args_cli.num_envs, device=base.device, dtype=torch.long
        ) % len(base.cfg.clutter_scenes)
        torch.testing.assert_close(base._clutter_scene_indices, expected_scene)
        torch.testing.assert_close(
            base._clutter_task_indices,
            torch.zeros_like(base._clutter_task_indices),
        )
        scenes = base._clutter_scenes_runtime
        target = base.scene["target"]
        obstacles = base.scene["obstacles"]
        expected_target_mass = torch.tensor(
            [scenes[index % len(scenes)].target_object.mass_kg for index in range(base.num_envs)]
        ).unsqueeze(-1)
        expected_obstacle_mass = torch.tensor(
            [
                [item.mass_kg for item in scenes[index % len(scenes)].obstacle_objects]
                for index in range(base.num_envs)
            ]
        ).unsqueeze(-1)
        torch.testing.assert_close(
            target.data.default_mass.cpu(), expected_target_mass, rtol=1.0e-5, atol=1.0e-6
        )
        torch.testing.assert_close(
            obstacles.data.default_mass.cpu(),
            expected_obstacle_mass,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        if not torch.all(target.data.default_inertia[..., (0, 4, 8)] > 0.0):
            raise RuntimeError("target inertia was not computed from manifest mass")
        if not torch.all(obstacles.data.default_inertia[..., (0, 4, 8)] > 0.0):
            raise RuntimeError("obstacle inertia was not computed from manifest masses")

        target_materials = target.root_physx_view.get_material_properties()
        expected_target_materials = torch.tensor(
            [
                (
                    scenes[index % len(scenes)].target_object.static_friction,
                    scenes[index % len(scenes)].target_object.dynamic_friction,
                    scenes[index % len(scenes)].target_object.restitution,
                )
                for index in range(base.num_envs)
            ]
        ).unsqueeze(1).expand_as(target_materials)
        torch.testing.assert_close(target_materials, expected_target_materials)
        obstacle_materials_view = obstacles.root_physx_view.get_material_properties()
        obstacle_materials = obstacle_materials_view.reshape(
            obstacles.num_objects, obstacles.num_instances, -1, 3
        ).permute(1, 0, 2, 3)
        expected_obstacle_materials = torch.tensor(
            [
                [
                    (item.static_friction, item.dynamic_friction, item.restitution)
                    for item in scenes[index % len(scenes)].obstacle_objects
                ]
                for index in range(base.num_envs)
            ]
        ).unsqueeze(2).expand_as(obstacle_materials)
        torch.testing.assert_close(obstacle_materials, expected_obstacle_materials)
        policy = observations["policy"]
        if not torch.isfinite(policy).all():
            raise RuntimeError("non-finite policy observation after reset")
        physical_scene = observations["world_model"]["scene"]
        if physical_scene.shape != (args_cli.num_envs, 1280, 7):
            raise RuntimeError(
                f"unexpected DAPL physical scene shape: {tuple(physical_scene.shape)}"
            )
        if not torch.isfinite(physical_scene).all():
            raise RuntimeError("non-finite DAPL physical scene after reset")

        rewards = []
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=base.device)
                observations, reward, _, _, _ = env.step(actions)
                if not torch.isfinite(observations["policy"]).all():
                    raise RuntimeError("non-finite policy observation during stepping")
                if not torch.isfinite(observations["world_model"]["scene"]).all():
                    raise RuntimeError("non-finite DAPL physical scene during stepping")
                if not torch.isfinite(reward).all():
                    raise RuntimeError("non-finite reward during stepping")
                rewards.append(reward)

        stacked_rewards = torch.stack(rewards)
        print(
            "CLUTTER6D_SMOKE_OK",
            f"num_envs={base.num_envs}",
            f"scenes={len(base.cfg.clutter_scenes)}",
            f"observation_shape={tuple(policy.shape)}",
            f"world_model_shape={tuple(physical_scene.shape)}",
            f"hand_point_source={base._dapl_hand_point_source}",
            f"action_shape={tuple(env.action_space.shape)}",
            f"reward_range=({stacked_rewards.min().item():.6f},"
            f"{stacked_rewards.max().item():.6f})",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
