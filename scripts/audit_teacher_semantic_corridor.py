#!/usr/bin/env python3
"""Audit whether the teacher's semantic corridor potential detects blocked routes."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-AffordanceTeacher-T0-SemanticCorridor-C1-Soft-Franka-v0",
)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--settle-steps", type=int, default=2)
parser.add_argument("--barrier-floor", type=float, default=None)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.affordances import (
    _goal_conditioned_semantic_corridor_potential,
)


def _direction_bin(angle_deg: float) -> str:
    if angle_deg < -70.0:
        return "[-90,-70)"
    if angle_deg < -35.0:
        return "[-70,-35)"
    if angle_deg < 0.0:
        return "[-35,0)"
    if angle_deg < 35.0:
        return "[0,35)"
    if angle_deg < 70.0:
        return "[35,70)"
    return "[70,90]"


def main() -> None:
    print("[semantic-corridor-audit] building environment", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        zero_action = torch.zeros((base.num_envs, 7), device=base.device)
        for _ in range(args_cli.settle_steps):
            env.step(zero_action)

        potential, distance, clearance = (
            _goal_conditioned_semantic_corridor_potential(
                base,
                normalization_distance_m=0.20,
                contact_distance_m=0.010,
                corridor_contact_clearance_m=0.010,
                corridor_activation_clearance_m=0.030,
                corridor_body_radius_m=0.030,
                corridor_barrier_floor=args_cli.barrier_floor,
                obstruction_weight=1.0,
                corridor_samples=9,
                corridor_start_fraction=0.10,
                corridor_end_fraction=0.85,
                command_name="target_object_pose",
                minimum_safe_score=0.25,
                side_band_m=0.015,
                minimum_goal_displacement_m=0.020,
                safe_radius_m=None,
                protected_radius_m=None,
                target_cfg=SceneEntityCfg("target"),
                obstacles_cfg=SceneEntityCfg("obstacles"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
            )
        )
        target_position = (
            base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        )
        goal = base.command_manager.get_command("target_object_pose")
        displacement = goal[:, :2] - target_position[:, :2]
        direction_deg = torch.rad2deg(
            torch.atan2(displacement[:, 1], displacement[:, 0])
        )

        bin_names = (
            "[-90,-70)",
            "[-70,-35)",
            "[-35,0)",
            "[0,35)",
            "[35,70)",
            "[70,90]",
        )
        per_bin: dict[str, dict[str, float | int]] = {}
        rows: list[dict[str, float | int | str | bool]] = []
        for env_id in range(base.num_envs):
            angle = float(direction_deg[env_id].item())
            rows.append(
                {
                    "environment_index": env_id,
                    "goal_direction_deg": angle,
                    "direction_bin": _direction_bin(angle),
                    "approach_distance_m": float(distance[env_id].item()),
                    "corridor_clearance_m": float(clearance[env_id].item()),
                    "corridor_obstructed": bool(clearance[env_id] < 0.030),
                    "corridor_contact_risk": bool(clearance[env_id] <= 0.010),
                    "navigation_potential": float(potential[env_id].item()),
                }
            )
        for name in bin_names:
            selected = [row for row in rows if row["direction_bin"] == name]
            count = len(selected)
            if count == 0:
                continue
            per_bin[name] = {
                "scenes": count,
                "approach_distance_mean_m": sum(
                    float(row["approach_distance_m"]) for row in selected
                )
                / count,
                "corridor_clearance_mean_m": sum(
                    float(row["corridor_clearance_m"]) for row in selected
                )
                / count,
                "corridor_obstruction_rate": sum(
                    bool(row["corridor_obstructed"]) for row in selected
                )
                / count,
                "corridor_contact_risk_rate": sum(
                    bool(row["corridor_contact_risk"]) for row in selected
                )
                / count,
                "navigation_potential_mean": sum(
                    float(row["navigation_potential"]) for row in selected
                )
                / count,
            }

        result = {
            "task": args_cli.task,
            "num_envs": base.num_envs,
            "corridor_definition": {
                "samples": 9,
                "fractions": [0.10, 0.85],
                "contact_clearance_m": 0.010,
                "activation_clearance_m": 0.030,
                "hand_sweep_radius_m": 0.030,
                "barrier_floor": args_cli.barrier_floor,
                "obstacles": "target points with safe score < 0.25",
            },
            "scope_note": (
                "Reset-state point-cloud diagnostic only; it does not certify a "
                "learned trajectory or proximal-link PhysX clearance."
            ),
            "per_direction_bin": per_bin,
            "per_scene": rows,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({"output": str(args_cli.output), **per_bin}, indent=2),
            flush=True,
        )
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
