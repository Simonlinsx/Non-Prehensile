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
    print("[domino-smoke] creating Gym environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        base = env.unwrapped
        policy = observations["policy"]
        affordance = observations["world_model"]["target_affordance"]
        if affordance.shape != (args_cli.num_envs, 1024):
            raise RuntimeError(
                f"unexpected target affordance shape: {tuple(affordance.shape)}"
            )
        aligned = affordance.reshape(args_cli.num_envs, 512, 2)
        if not torch.isfinite(policy).all() or not torch.isfinite(aligned).all():
            raise RuntimeError("non-finite observation after reset")
        if not torch.all(aligned.amax(dim=1) > 0.0):
            raise RuntimeError("one or more target affordance channels are empty")

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=base.device)
                observations, reward, _, _, _ = env.step(actions)
                if not torch.isfinite(observations["policy"]).all():
                    raise RuntimeError("non-finite policy observation")
                if not torch.isfinite(reward).all():
                    raise RuntimeError("non-finite reward")

        print(
            "DOMINO_AFFORDANCE_SMOKE_OK",
            f"num_envs={base.num_envs}",
            f"policy_shape={tuple(policy.shape)}",
            f"affordance_shape={tuple(affordance.shape)}",
            f"safe_score_max={aligned[..., 0].max().item():.4f}",
            f"protected_score_max={aligned[..., 1].max().item():.4f}",
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
