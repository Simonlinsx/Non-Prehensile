#!/usr/bin/env python3
"""Audit the teacher's point-cloud semantic geodesic at reset."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-AffordanceTeacher-T0-SemanticGeodesic-C1-Soft-Franka-v0",
)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--settle-steps", type=int, default=2)
parser.add_argument("--direct-route-scale", type=float, default=0.15)
parser.add_argument(
    "--direct-route-activation-clearance-m", type=float, default=0.040
)
parser.add_argument("--recover-illegal-route", action="store_true", default=False)
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
    _goal_conditioned_safe_side_route,
    _goal_conditioned_semantic_geodesic_potential,
)
from dapl.metrics import clearance_conditioned_route_scale


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
    print("[semantic-geodesic-audit] building environment", flush=True)
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

        (
            potential,
            approach_distance,
            route_clearance,
            used_detour,
            direct_clearance,
            route_length,
            selected_direction,
        ) = _goal_conditioned_semantic_geodesic_potential(
            base,
            normalization_distance_m=0.20,
            contact_distance_m=0.010,
            route_contact_clearance_m=0.010,
            route_activation_clearance_m=0.030,
            route_body_radius_m=0.030,
            route_detour_margin_m=0.020,
            route_barrier_floor=0.01,
            obstruction_weight=1.0,
            route_candidates=12,
            route_segment_samples=7,
            route_obstacle_samples=96,
            command_name="target_object_pose",
            minimum_safe_score=0.25,
            side_band_m=0.015,
            minimum_goal_displacement_m=0.020,
            safe_radius_m=None,
            protected_radius_m=None,
            target_cfg=SceneEntityCfg("target"),
            obstacles_cfg=SceneEntityCfg("obstacles"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            recover_illegal_route=args_cli.recover_illegal_route,
        )
        _, route_start, route_end, _, _ = _goal_conditioned_safe_side_route(
            base,
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
        direct_direction = route_end - route_start
        direct_direction /= torch.clamp(
            torch.linalg.vector_norm(direct_direction, dim=1, keepdim=True),
            min=1.0e-6,
        )
        direct_alignment = torch.sum(selected_direction * direct_direction, dim=1)
        lateral_fraction = torch.sqrt(
            torch.clamp(1.0 - direct_alignment.square(), min=0.0)
        )
        route_scale = clearance_conditioned_route_scale(
            direct_clearance,
            contact_clearance=0.010,
            activation_clearance=args_cli.direct_route_activation_clearance_m,
            direct_route_scale=args_cli.direct_route_scale,
        )
        target_position = (
            base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        )
        goal = base.command_manager.get_command("target_object_pose")
        displacement = goal[:, :2] - target_position[:, :2]
        direction_deg = torch.rad2deg(torch.atan2(displacement[:, 1], displacement[:, 0]))

        rows: list[dict[str, float | int | str | bool]] = []
        for env_id in range(base.num_envs):
            angle = float(direction_deg[env_id].item())
            rows.append(
                {
                    "environment_index": env_id,
                    "goal_direction_deg": angle,
                    "direction_bin": _direction_bin(angle),
                    "approach_distance_m": float(approach_distance[env_id].item()),
                    "direct_clearance_m": float(direct_clearance[env_id].item()),
                    "selected_route_clearance_m": float(route_clearance[env_id].item()),
                    "selected_route_length_m": float(route_length[env_id].item()),
                    "direct_route_obstructed": bool(direct_clearance[env_id] < 0.030),
                    "selected_route_legal": bool(route_clearance[env_id] >= 0.010),
                    "used_detour": bool(used_detour[env_id]),
                    "field_direct_alignment": float(direct_alignment[env_id].item()),
                    "field_lateral_fraction": float(lateral_fraction[env_id].item()),
                    "clearance_conditioned_route_scale": float(
                        route_scale[env_id].item()
                    ),
                    "navigation_potential": float(potential[env_id].item()),
                }
            )

        per_bin: dict[str, dict[str, float | int]] = {}
        for name in (
            "[-90,-70)",
            "[-70,-35)",
            "[-35,0)",
            "[0,35)",
            "[35,70)",
            "[70,90]",
        ):
            selected = [row for row in rows if row["direction_bin"] == name]
            count = len(selected)
            if count == 0:
                continue
            per_bin[name] = {
                "scenes": count,
                "direct_obstruction_rate": sum(
                    bool(row["direct_route_obstructed"]) for row in selected
                ) / count,
                "detour_selection_rate": sum(
                    bool(row["used_detour"]) for row in selected
                ) / count,
                "legal_selected_route_rate": sum(
                    bool(row["selected_route_legal"]) for row in selected
                ) / count,
                "selected_route_length_mean_m": sum(
                    float(row["selected_route_length_m"]) for row in selected
                ) / count,
                "navigation_potential_mean": sum(
                    float(row["navigation_potential"]) for row in selected
                ) / count,
                "field_direct_alignment_mean": sum(
                    float(row["field_direct_alignment"]) for row in selected
                ) / count,
                "field_lateral_fraction_mean": sum(
                    float(row["field_lateral_fraction"]) for row in selected
                ) / count,
                "clearance_conditioned_route_scale_mean": sum(
                    float(row["clearance_conditioned_route_scale"])
                    for row in selected
                ) / count,
                "full_strength_route_rate": sum(
                    float(row["clearance_conditioned_route_scale"]) >= 0.999
                    for row in selected
                ) / count,
                "baseline_strength_route_rate": sum(
                    float(row["clearance_conditioned_route_scale"])
                    <= args_cli.direct_route_scale + 0.001
                    for row in selected
                ) / count,
                "detour_with_lateral_field_rate": sum(
                    bool(row["used_detour"])
                    and float(row["field_lateral_fraction"]) >= 0.10
                    for row in selected
                ) / max(
                    1, sum(bool(row["used_detour"]) for row in selected)
                ),
            }

        result = {
            "task": args_cli.task,
            "num_envs": base.num_envs,
            "geodesic_definition": {
                "candidate_routes": "direct plus 12 point-cloud support-ring detours",
                "segment_samples": 7,
                "obstacle_samples": 96,
                "contact_clearance_m": 0.010,
                "activation_clearance_m": 0.030,
                "hand_sweep_radius_m": 0.030,
                "detour_margin_m": 0.020,
                "obstacles": "target points with safe score < 0.25",
                "actor_route_input": False,
                "recover_illegal_route": args_cli.recover_illegal_route,
                "clearance_blend": {
                    "direct_route_scale": args_cli.direct_route_scale,
                    "activation_clearance_m": (
                        args_cli.direct_route_activation_clearance_m
                    ),
                },
                "field_signal": (
                    "reward-only signed hand displacement along the current "
                    "first legal free-space edge"
                ),
            },
            "scope_note": (
                "Reset-state point-cloud diagnostic only; it does not certify a learned "
                "trajectory or proximal-link PhysX clearance."
            ),
            "per_direction_bin": per_bin,
            "per_scene": rows,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args_cli.output), **per_bin}, indent=2), flush=True)
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
