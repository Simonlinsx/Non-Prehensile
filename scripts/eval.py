# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RSL-RL agent and report success rate."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL and report success rate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1248, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=2000, help="Number of episodes to evaluate.")
parser.add_argument("--max_episode_steps", type=int, default=300, help="Safety cap on episode length (steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation (num_envs must be 1).")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=1_000_000, help="Interval between videos (unused in eval, kept for parity).")
parser.add_argument("--video_folder", type=str, default=None, help="Optional directory for evaluation videos.")
parser.add_argument("--video_name_prefix", type=str, default="eval", help="Filename prefix for evaluation videos.")
parser.add_argument("--output_dir", type=str, default=None, help="Optional directory for JSON/CSV evaluation results.")
parser.add_argument(
    "--video_ground_color",
    type=float,
    nargs=3,
    default=(0.08, 0.20, 0.32),
    metavar=("R", "G", "B"),
    help="Video-only diffuse RGB color for the local support surface.",
)
parser.add_argument(
    "--video_dome_light_intensity",
    type=float,
    default=1200.0,
    help="Video-only dome-light intensity; quantitative evaluation is unchanged.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional world-frame viewer camera position for video evaluation.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional world-frame viewer camera target for video evaluation.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use the policy mean action instead of sampling training-time action noise.",
)
parser.add_argument(
    "--zero_actions",
    action="store_true",
    default=False,
    help=(
        "Diagnostic mode: replace policy output with zero actions while "
        "retaining the identical environment, manifest and episode accounting."
    ),
)
parser.add_argument(
    "--visualize_goal",
    action="store_true",
    default=False,
    help="Overlay a translucent copy of the target object at the commanded goal pose.",
)
parser.add_argument(
    "--goal_ghost_opacity",
    type=float,
    default=0.68,
    help="Opacity of the target-object ghost used by --visualize_goal (0 transparent, 1 opaque).",
)
parser.add_argument(
    "--show_goal_frame",
    action="store_true",
    default=False,
    help="Also show a small RGB local-pose frame inside the target-object ghost.",
)
parser.add_argument(
    "--visualize_affordance",
    action="store_true",
    default=False,
    help="Overlay DOMINO safe-contact points in green and protected points in red.",
)
parser.add_argument(
    "--trace_episode_ends",
    action="store_true",
    default=False,
    help="Record pre-step state, termination reasons, and post-reset state for every ended episode (single env only).",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible (single env).")
# checkpoint selection
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# append RSL-RL cli arguments (includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import time
import torch
from datetime import datetime
import json
import csv
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from dapl.metrics import flip_relative_goal_yaw_in_actor_observation


def _extract_asset_names_from_env(env) -> list[str]:
    """Extract asset base names from env config's MultiAssetSpawnerCfg.

    Assumes env.unwrapped.cfg.scene.object.spawn.assets_cfg exists and each has a usd_path like .../<name>/<name>.usd.
    """
    names: list[str] = []
    cfg = getattr(env, "unwrapped", env)
    scene = getattr(cfg, "cfg", None)
    if scene is None:
        scene = getattr(env, "cfg", None)
    if scene is None:
        return names
    scene_cfg = getattr(scene, "scene", None)
    if scene_cfg is None:
        return names
    object_cfg = getattr(scene_cfg, "object", None)
    spawn_cfg = getattr(object_cfg, "spawn", None) if object_cfg is not None else None
    assets_cfg = getattr(spawn_cfg, "assets_cfg", None) if spawn_cfg is not None else None
    if assets_cfg is None:
        return names
    for usd_cfg in assets_cfg:
        usd_path = getattr(usd_cfg, "usd_path", None)
        if isinstance(usd_path, str) and len(usd_path) > 0:
            base = os.path.basename(os.path.dirname(usd_path))
            names.append(base)
        else:
            names.append("unknown")
    return names


def _build_env_to_object_index(num_envs: int, num_assets: int) -> torch.Tensor:
    """Deterministic mapping from env_id to asset index when random_choice=False."""
    if num_assets <= 0:
        return torch.full((num_envs,), -1, dtype=torch.long)
    idx = torch.arange(num_envs, dtype=torch.long)
    return idx % num_assets


def _scalar_summary(values: list[float]) -> dict[str, float | None]:
    """Return compact terminal-error statistics without hiding empty data."""
    if not values:
        return {"minimum": None, "mean": None, "p95": None, "maximum": None}
    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return {"minimum": None, "mean": None, "p95": None, "maximum": None}
    return {
        "minimum": float(tensor.min().item()),
        "mean": float(tensor.mean().item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "maximum": float(tensor.max().item()),
    }


def _goal_yaw_delta_rad(initial_pose, goal_pose) -> float:
    """Return signed world-Z yaw of goal * inverse(initial)."""

    iw, ix, iy, iz = initial_pose[3:7]
    gw, gx, gy, gz = goal_pose[3:7]
    # goal quaternion multiplied by conjugate(initial quaternion).
    rw = gw * iw + gx * ix + gy * iy + gz * iz
    rx = -gw * ix + gx * iw - gy * iz + gz * iy
    ry = -gw * iy + gx * iz + gy * iw - gz * ix
    rz = -gw * iz - gx * iy + gy * ix + gz * iw
    return math.atan2(
        2.0 * (rw * rz + rx * ry),
        1.0 - 2.0 * (ry * ry + rz * rz),
    )


def _terminal_metric_value(env, name: str, env_id: int) -> float:
    values = getattr(env, name, None)
    return float("nan") if values is None else float(values[env_id].item())


def _clutter_diagnostic_snapshot(env, actions, rewards, step: int, initial_target_pose: torch.Tensor) -> dict:
    """Capture privileged single-environment diagnostics for a video frame."""
    base_env = env.unwrapped
    target = base_env.scene["target"]
    ee_frame = base_env.scene["ee_frame"]
    command = base_env.command_manager.get_command("target_object_pose")

    target_pos_env = target.data.root_pos_w[0, :3] - base_env.scene.env_origins[0]
    target_quat = target.data.root_quat_w[0]
    goal_pos = command[0, :3]
    goal_quat = command[0, 3:7]

    planar_error = torch.linalg.vector_norm(goal_pos[:2] - target_pos_env[:2])
    goal_dot = torch.clamp(torch.abs(torch.sum(target_quat * goal_quat)), max=1.0)
    rotation_error = 2.0 * torch.acos(goal_dot)

    ee_targets = ee_frame.data.target_pos_w[0]
    finger_distance = torch.minimum(
        torch.linalg.vector_norm(target.data.root_pos_w[0, :3] - ee_targets[1]),
        torch.linalg.vector_norm(target.data.root_pos_w[0, :3] - ee_targets[2]),
    )

    initial_pos = initial_target_pose[:3]
    initial_quat = initial_target_pose[3:7]
    target_translation = torch.linalg.vector_norm(target_pos_env[:2] - initial_pos[:2])
    initial_dot = torch.clamp(torch.abs(torch.sum(target_quat * initial_quat)), max=1.0)
    target_rotation = 2.0 * torch.acos(initial_dot)

    obstacle_translation = torch.tensor(0.0, device=base_env.device)
    obstacle_rotation = torch.tensor(0.0, device=base_env.device)
    active_obstacle_count = int(
        getattr(base_env, "_clutter_active_obstacle_count", 0)
    )
    if (
        active_obstacle_count > 0
        and hasattr(base_env, "_clutter_initial_obstacle_pose")
    ):
        obstacles = base_env.scene["obstacles"]
        obstacle_pos_env = (
            obstacles.data.object_pos_w[0, :active_obstacle_count]
            - base_env.scene.env_origins[0].unsqueeze(0)
        )
        initial_obstacles = base_env._clutter_initial_obstacle_pose[
            0, :active_obstacle_count
        ]
        obstacle_translation = torch.linalg.vector_norm(
            obstacle_pos_env - initial_obstacles[:, :3], dim=-1
        ).mean()
        obstacle_dot = torch.sum(
            obstacles.data.object_quat_w[0, :active_obstacle_count]
            * initial_obstacles[:, 3:7],
            dim=-1,
        )
        obstacle_rotation = (
            2.0 * torch.acos(torch.clamp(torch.abs(obstacle_dot), max=1.0))
        ).mean()

    semantic_route_diagnostic: dict[str, float | bool] = {}
    protected_geodesic_diagnostic: dict[str, float | int] = {}
    contact_diagnostic: dict[str, float | bool] = {}
    if base_env.num_envs == 1:
        # This is an evaluation-only diagnostic.  It makes route-mode changes
        # and the clearance-conditioned reward scale auditable in the same
        # single-scene trace as the rendered rollout; nothing is exposed to
        # the actor or used to select its action.
        from isaaclab.managers import SceneEntityCfg

        from dapl.metrics import (
            clearance_conditioned_route_scale,
            semantic_clearance_recovery_direction,
        )
        from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.affordances import (
            _goal_conditioned_safe_side_route,
            _goal_conditioned_semantic_geodesic_potential,
            domino_affordance_contact_state,
        )

        # Capture the exact semantic distances used by C1/C2.  These values
        # are evaluation-only and make it possible to distinguish an illegal
        # approach from a protected-part collision during the subsequent
        # object push.  They do not enter the actor observation or action.
        contact_state = domino_affordance_contact_state(
            base_env,
            evaluate_protected=active_obstacle_count > 0,
            evaluate_robot_obstacle=False,
        )
        contact_diagnostic = {
            "legal_safe_robot_contact": bool(
                contact_state["legal_safe_robot_contact"][0].item()
            ),
            "forbidden_hand_contact": bool(
                contact_state["forbidden_hand_contact"][0].item()
            ),
            "protected_hand_contact": bool(
                contact_state["protected_hand_contact"][0].item()
            ),
            "protected_obstacle_collision": bool(
                contact_state["protected_obstacle_collision"][0].item()
            ),
            "minimum_safe_distance_m": float(
                contact_state["minimum_safe_distance"][0].item()
            ),
            "minimum_robot_forbidden_distance_m": float(
                contact_state["minimum_robot_forbidden_distance"][0].item()
            ),
            "protected_obstacle_clearance_m": float(
                contact_state["protected_clearance"][0].item()
            ),
        }

        semantic_geodesic_cfg = getattr(
            getattr(base_env.cfg, "rewards", None),
            "semantic_geodesic_approach",
            None,
        )
        semantic_geodesic_params = (
            dict(semantic_geodesic_cfg.params)
            if semantic_geodesic_cfg is not None
            else {}
        )
        def _route_param(name: str, default):
            return semantic_geodesic_params.get(name, default)

        lexicographic_feasibility = bool(
            _route_param("lexicographic_feasibility", False)
        )
        (
            navigation_potential,
            _,
            selected_route_clearance,
            used_detour,
            direct_route_clearance,
            _,
            selected_direction,
        ) = _goal_conditioned_semantic_geodesic_potential(
            base_env,
            normalization_distance_m=float(
                _route_param("normalization_distance_m", 0.20)
            ),
            contact_distance_m=float(_route_param("contact_distance_m", 0.010)),
            route_contact_clearance_m=float(
                _route_param("route_contact_clearance_m", 0.010)
            ),
            route_activation_clearance_m=float(
                _route_param("route_activation_clearance_m", 0.030)
            ),
            route_body_radius_m=float(_route_param("route_body_radius_m", 0.030)),
            route_detour_margin_m=float(
                _route_param("route_detour_margin_m", 0.020)
            ),
            route_barrier_floor=float(_route_param("route_barrier_floor", 0.01)),
            obstruction_weight=float(_route_param("obstruction_weight", 1.0)),
            route_candidates=int(_route_param("route_candidates", 12)),
            route_segment_samples=int(_route_param("route_segment_samples", 7)),
            route_obstacle_samples=int(_route_param("route_obstacle_samples", 96)),
            command_name="target_object_pose",
            minimum_safe_score=float(_route_param("minimum_safe_score", 0.25)),
            side_band_m=float(_route_param("side_band_m", 0.015)),
            minimum_goal_displacement_m=float(
                _route_param("minimum_goal_displacement_m", 0.020)
            ),
            safe_radius_m=None,
            protected_radius_m=None,
            target_cfg=SceneEntityCfg("target"),
            obstacles_cfg=SceneEntityCfg("obstacles"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            recover_illegal_route=True,
            lexicographic_feasibility=lexicographic_feasibility,
            lexicographic_length_scale_m=float(
                _route_param("lexicographic_length_scale_m", 0.20)
            ),
            lexicographic_violation_scale_m=float(
                _route_param("lexicographic_violation_scale_m", 0.01)
            ),
            yaw_moment_weight=float(_route_param("yaw_moment_weight", 0.0)),
            yaw_activation_rad=float(_route_param("yaw_activation_rad", 0.10)),
        )
        _, route_start, route_end, target_points, forbidden_mask = (
            _goal_conditioned_safe_side_route(
                base_env,
                command_name="target_object_pose",
                minimum_safe_score=float(_route_param("minimum_safe_score", 0.25)),
                side_band_m=float(_route_param("side_band_m", 0.015)),
                minimum_goal_displacement_m=float(
                    _route_param("minimum_goal_displacement_m", 0.020)
                ),
                safe_radius_m=None,
                protected_radius_m=None,
                target_cfg=SceneEntityCfg("target"),
                obstacles_cfg=SceneEntityCfg("obstacles"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                yaw_moment_weight=float(
                    _route_param("yaw_moment_weight", 0.0)
                ),
                yaw_activation_rad=float(
                    _route_param("yaw_activation_rad", 0.10)
                ),
            )
        )
        direct_direction = route_end - route_start
        direct_direction /= torch.clamp(
            torch.linalg.vector_norm(direct_direction, dim=1, keepdim=True),
            min=1.0e-6,
        )
        route_scale = clearance_conditioned_route_scale(
            direct_route_clearance,
            contact_clearance=float(
                _route_param("route_contact_clearance_m", 0.010)
            ),
            activation_clearance=0.040,
            direct_route_scale=0.15,
        )
        outward_direction, _ = semantic_clearance_recovery_direction(
            route_start,
            target_points,
            obstacle_mask=forbidden_mask,
            safety_radius=float(
                _route_param("route_body_radius_m", 0.030)
                + _route_param("route_contact_clearance_m", 0.010)
            ),
        )
        outward_alignment = torch.sum(
            selected_direction[0] * outward_direction[0]
        )
        tangential_fraction = torch.sqrt(
            torch.clamp(1.0 - outward_alignment.square(), min=0.0, max=1.0)
        )
        semantic_route_diagnostic = {
            "semantic_navigation_potential": float(
                navigation_potential[0].item()
            ),
            "semantic_lexicographic_feasibility": lexicographic_feasibility,
            "semantic_route_used_detour": bool(used_detour[0].item()),
            "semantic_direct_route_clearance_m": float(
                direct_route_clearance[0].item()
            ),
            "semantic_selected_route_clearance_m": float(
                selected_route_clearance[0].item()
            ),
            "semantic_selected_route_legal": bool(
                selected_route_clearance[0].item()
                >= float(_route_param("route_contact_clearance_m", 0.010))
            ),
            "semantic_recovery_active": bool(
                used_detour[0].item()
                and selected_route_clearance[0].item()
                < float(_route_param("route_contact_clearance_m", 0.010))
            ),
            "semantic_clearance_conditioned_scale": float(route_scale[0].item()),
            "semantic_field_direct_alignment": float(
                torch.sum(selected_direction[0] * direct_direction[0]).item()
            ),
            "semantic_field_outward_alignment": float(outward_alignment.item()),
            "semantic_field_tangential_fraction": float(
                tangential_fraction.item()
            ),
        }

        # The protected-object geodesic is a reward-only teacher signal.  Keep
        # its latched homotopy and current scalar potential in evaluation
        # traces so we can audit whether the signal is present specifically
        # while a legal safe-region push moves the target.  These values are
        # never concatenated to the actor observation.
        protected_side = getattr(
            base_env, "_protected_geodesic_detour_side", None
        )
        protected_direct_clearance = getattr(
            base_env, "_protected_geodesic_direct_clearance", None
        )
        protected_route_length = getattr(
            base_env, "_protected_geodesic_route_length", None
        )
        if (
            protected_side is not None
            and protected_direct_clearance is not None
            and protected_route_length is not None
        ):
            protected_reward_rate = 0.0
            for term_name, term_values in (
                base_env.reward_manager.get_active_iterable_terms(0)
            ):
                if term_name == "protected_region_geodesic_progress":
                    protected_reward_rate = float(term_values[0])
                    break
            protected_geodesic_diagnostic = {
                "protected_geodesic_detour_side": int(
                    protected_side[0].item()
                ),
                "protected_geodesic_direct_clearance_m": float(
                    protected_direct_clearance[0].item()
                ),
                "protected_geodesic_route_length_m": float(
                    protected_route_length[0].item()
                ),
                "protected_geodesic_reward_weighted_rate": (
                    protected_reward_rate
                ),
            }

    return {
        "step": int(step),
        "reward": float(rewards[0].item()),
        "action_l2": float(torch.linalg.vector_norm(actions[0]).item()),
        "ee_target_distance_m": float(finger_distance.item()),
        "target_goal_planar_error_m": float(planar_error.item()),
        "target_goal_rotation_error_rad": float(rotation_error.item()),
        "target_translation_from_start_m": float(target_translation.item()),
        "target_rotation_from_start_rad": float(target_rotation.item()),
        "obstacle_mean_translation_m": float(obstacle_translation.item()),
        "obstacle_mean_rotation_rad": float(obstacle_rotation.item()),
        "active_obstacle_count": active_obstacle_count,
        "target_position_xy_m": [
            float(target_pos_env[0].item()),
            float(target_pos_env[1].item()),
        ],
        **contact_diagnostic,
        **semantic_route_diagnostic,
        **protected_geodesic_diagnostic,
    }


def _create_goal_object_ghost(env, opacity: float):
    """Spawn a non-physical translucent copy of the target at its goal pose."""
    import isaaclab.sim as sim_utils
    from isaacsim.core.prims import XFormPrim

    base_env = env.unwrapped
    if base_env.num_envs != 1:
        raise ValueError("target-object goal visualization currently requires one environment")
    if not 0.0 < opacity <= 1.0:
        raise ValueError("--goal_ghost_opacity must be in (0, 1]")

    target_spawn_cfg = base_env.scene["target"].cfg.spawn
    asset_cfgs = getattr(target_spawn_cfg, "assets_cfg", None)
    target_asset_cfg = asset_cfgs[0] if asset_cfgs else target_spawn_cfg
    if not hasattr(target_asset_cfg, "usd_path"):
        raise TypeError("target-object goal visualization requires a USD-backed target asset")

    command = base_env.command_manager.get_command("target_object_pose")
    goal_pos_w = command[0, :3] + base_env.scene.env_origins[0]
    goal_quat_w = command[0, 3:7]
    # Manager-based environments populate command buffers on their first
    # reset.  Before then, the quaternion is all zeros; seed the visual at the
    # physical target pose and move it to the goal once the command is valid.
    if torch.linalg.vector_norm(goal_quat_w).item() < 0.5:
        goal_pos_w = base_env.scene["target"].data.root_pos_w[0, :3]
        goal_quat_w = base_env.scene["target"].data.root_quat_w[0]
    prim_path = "/Visuals/Eval/GoalObjectGhost"
    ghost_cfg = sim_utils.UsdFileCfg(
        usd_path=target_asset_cfg.usd_path,
        scale=target_asset_cfg.scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material_path="GoalGhostMaterial",
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.75, 1.0),
            emissive_color=(0.0, 0.12, 0.2),
            roughness=0.35,
            opacity=opacity,
        ),
    )
    ghost_cfg.func(
        prim_path,
        ghost_cfg,
        translation=tuple(goal_pos_w.detach().cpu().tolist()),
        orientation=tuple(goal_quat_w.detach().cpu().tolist()),
    )
    return XFormPrim(
        prim_path,
        name="goal_object_ghost",
        reset_xform_properties=False,
        usd=True,
    )


def _create_clutter_video_markers(
    env,
    *,
    show_goal: bool,
    show_affordance: bool,
    goal_ghost_opacity: float,
    show_goal_frame: bool,
):
    """Create non-physical goal and semantic surface overlays."""
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG

    markers = {}
    if show_goal:
        markers["goal_ghost"] = _create_goal_object_ghost(
            env, goal_ghost_opacity
        )
        if show_goal_frame:
            goal_cfg = FRAME_MARKER_CFG.copy()
            goal_cfg.prim_path = "/Visuals/Eval/GoalPose"
            goal_cfg.markers["frame"].scale = (0.055, 0.055, 0.055)
            markers["goal_frame"] = VisualizationMarkers(goal_cfg)

    if show_affordance:
        safe_cfg = SPHERE_MARKER_CFG.copy()
        safe_cfg.prim_path = "/Visuals/Eval/SafeContact"
        safe_cfg.markers["sphere"].radius = 0.006
        safe_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
        markers["safe"] = VisualizationMarkers(safe_cfg)

        protected_cfg = SPHERE_MARKER_CFG.copy()
        protected_cfg.prim_path = "/Visuals/Eval/ProtectedFunctional"
        protected_cfg.markers["sphere"].radius = 0.006
        protected_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        markers["protected"] = VisualizationMarkers(protected_cfg)

    _update_clutter_video_markers(env, markers)
    return markers


def _subsample_masked_points(
    points: torch.Tensor, mask: torch.Tensor, maximum_points: int = 128
) -> torch.Tensor:
    """Deterministically retain a readable subset of one semantic region."""

    selected = points[mask]
    if selected.shape[0] <= maximum_points:
        return selected
    indices = torch.linspace(
        0,
        selected.shape[0] - 1,
        maximum_points,
        device=selected.device,
    ).long()
    return selected[indices]


def _update_clutter_video_markers(env, markers: dict) -> None:
    base_env = env.unwrapped
    if "goal_ghost" in markers or "goal_frame" in markers:
        command = base_env.command_manager.get_command("target_object_pose")
        goal_pos_w = command[:, :3] + base_env.scene.env_origins
        goal_quat_w = command[:, 3:7]
        # A zero quaternion means the first reset has not populated the
        # command yet.  Keep the ghost at its safe initialization pose until
        # the goal becomes valid.
        if torch.all(torch.linalg.vector_norm(goal_quat_w, dim=1) >= 0.5):
            if "goal_ghost" in markers:
                markers["goal_ghost"].set_world_poses(
                    positions=goal_pos_w,
                    orientations=goal_quat_w,
                    usd=True,
                )
            if "goal_frame" in markers:
                markers["goal_frame"].visualize(
                    translations=goal_pos_w, orientations=goal_quat_w
                )

    if "safe" in markers:
        from isaaclab.managers import SceneEntityCfg

        import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp

        target_points = mdp.get_object_pointcloud_in_env_frame(
            base_env, SceneEntityCfg("target")
        ).reshape(base_env.num_envs, 512, 3)
        semantics = mdp.domino_target_affordance(
            base_env, target_cfg=SceneEntityCfg("target")
        ).reshape(base_env.num_envs, 512, 2)
        points_w = target_points[0] + base_env.scene.env_origins[0]
        safe_points = _subsample_masked_points(
            points_w, semantics[0, :, 0] >= 0.25
        )
        protected_points = _subsample_masked_points(
            points_w, semantics[0, :, 1] >= 0.25
        )
        markers["safe"].visualize(translations=safe_points)
        markers["protected"].visualize(translations=protected_points)


class _MarkerUpdateWrapper(gym.Wrapper):
    """Update overlays after physics but before RecordVideo captures a frame."""

    def __init__(self, env, markers: dict):
        super().__init__(env)
        self._markers = markers

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        _update_clutter_video_markers(self.env, self._markers)
        return result

    def step(self, action):
        result = self.env.step(action)
        _update_clutter_video_markers(self.env, self._markers)
        return result


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate RSL-RL agent and report success rate over multiple episodes."""
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed

    if args_cli.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    if args_cli.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
    if args_cli.camera_eye is not None or args_cli.camera_lookat is not None:
        env_cfg.viewer.origin_type = "world"

    # The headless local-asset scene uses a large gray cuboid as its support
    # surface and a strong training light.  Under RTX/ACES tone mapping this
    # produces a nearly uniform white image (the gray floor and white Franka
    # differ by only a few pixel values).  Apply presentation-only styling
    # before scene creation so videos retain contrast without changing the
    # physics, policy observation, training task, or non-video evaluation.
    if args_cli.video:
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

    # Disable observation noise for evaluation
    env_cfg.disable_obs_noise = True

    # Limitations for video recording in evaluation
    if args_cli.video and env_cfg.scene.num_envs != 1:
        print("[WARN] Video recording in eval supports only num_envs=1. Overriding num_envs to 1.")
        env_cfg.scene.num_envs = 1
    if args_cli.trace_episode_ends and env_cfg.scene.num_envs != 1:
        raise ValueError("--trace_episode_ends requires --num_envs 1")

    # specify directory for loading experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # resolve resume path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # PreviewSurface opacity is otherwise omitted by the headless RTX path:
    # the ghost still casts a shadow but its translucent surface is invisible.
    if args_cli.visualize_goal:
        env_cfg.sim.render.enable_translucency = True

    # create isaac environment
    print(
        "[INFO] Clutter contract before gym.make: "
        f"configured_active_obstacles={getattr(env_cfg, 'active_obstacle_count', None)} "
        f"manifest={getattr(env_cfg, 'clutter_manifest_path', None)}"
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.trace_episode_ends:
        env.unwrapped._capture_affordance_reward_debug = True

    video_markers = (
        _create_clutter_video_markers(
            env,
            show_goal=args_cli.visualize_goal,
            show_affordance=args_cli.visualize_affordance,
            goal_ghost_opacity=args_cli.goal_ghost_opacity,
            show_goal_frame=args_cli.show_goal_frame,
        )
        if args_cli.visualize_goal or args_cli.visualize_affordance
        else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if video_markers is not None:
        env = _MarkerUpdateWrapper(env, video_markers)

    # wrap for video recording
    if args_cli.video:
        video_folder = args_cli.video_folder or os.path.join(log_dir, "videos", "eval")
        video_folder = os.path.abspath(video_folder)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,  # record first episode by default
            "video_length": min(args_cli.video_length, args_cli.max_episode_steps),
            "name_prefix": args_cli.video_name_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(
        "[INFO] Clutter contract after initial reset: "
        f"configured_active_obstacles={getattr(env.unwrapped.cfg, 'active_obstacle_count', None)} "
        f"runtime_active_obstacles={getattr(env.unwrapped, '_clutter_active_obstacle_count', None)} "
        f"spawned_obstacles={env.unwrapped.scene['obstacles'].num_objects}"
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Evaluation never updates PPO.  Skipping optimizer restoration also lets
    # actor-compatible legacy checkpoints be audited after adding a
    # training-only asymmetric critic branch.
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy_func = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy_obj = ppo_runner.alg.policy  # Get the actual policy object for act() method

    # evaluation loop (vectorized)
    num_envs = env.unwrapped.num_envs
    episodes_completed = 0
    total_successes = 0
    constrained_episodes = 0
    constrained_successes = 0
    affordance_violation_episodes = 0
    typed_violation_episodes = {label: 0 for label in ("c1", "c2", "c3")}
    c1_breakdown_episodes = {
        "hand_semantic": 0,
        "hand_neutral": 0,
        "hand_protected": 0,
        "arm_physical": 0,
    }
    # Give every parallel environment a fixed episode quota.  Without this,
    # environments that terminate successfully early can reset and contribute
    # multiple episodes before slower/time-out environments contribute even
    # one, which biases vectorized evaluation toward easy scenes.
    quota_base, quota_remainder = divmod(args_cli.num_episodes, num_envs)
    episode_quota_by_env = torch.full(
        (num_envs,), quota_base, device=env.unwrapped.device, dtype=torch.long
    )
    if quota_remainder:
        episode_quota_by_env[:quota_remainder] += 1
    episodes_counted_by_env = torch.zeros_like(episode_quota_by_env)

    # Validate that this is the instrumented task, but maintain exact local
    # counters below.  Environment-global counters include every environment
    # that finishes in the last vectorized step and can therefore overshoot
    # --num_episodes.
    if not hasattr(env.unwrapped, 'total_episodes') or not hasattr(env.unwrapped, 'total_successes'):
        raise AttributeError("Environment does not have total_episodes or total_successes. This eval script requires the NonPrehensileEnv with success tracking.")

    # per-object accounting
    asset_names = _extract_asset_names_from_env(env)
    num_assets = len(asset_names)
    env_to_obj_idx = _build_env_to_object_index(num_envs, num_assets)
    # counters stored on CPU for printing simplicity
    obj_episodes = {name: 0 for name in asset_names}
    obj_successes = {name: 0 for name in asset_names}
    runtime_scenes = tuple(
        getattr(env.unwrapped, "_clutter_scenes_runtime", ())
    )
    scene_stats = [
        {
            "scene_index": index,
            "scene_id": scene.scene_id,
            "goal_direction_deg": (
                math.degrees(
                    math.atan2(
                        scene.tasks[0].goal_pose[1]
                        - scene.tasks[0].initial_pose[1],
                        scene.tasks[0].goal_pose[0]
                        - scene.tasks[0].initial_pose[0],
                    )
                )
                if scene.tasks
                else None
            ),
            "goal_planar_displacement_m": (
                scene.tasks[0].planar_displacement if scene.tasks else None
            ),
            "goal_yaw_delta_rad": (
                _goal_yaw_delta_rad(
                    scene.tasks[0].initial_pose, scene.tasks[0].goal_pose
                )
                if scene.tasks
                else None
            ),
            "episodes": 0,
            "successes": 0,
            "constrained_successes": 0,
            "legal_safe_contact_episodes": 0,
            "typed_violation_episodes": {"c1": 0, "c2": 0, "c3": 0},
            "c1_breakdown_episodes": {
                "hand_semantic": 0,
                "hand_neutral": 0,
                "hand_protected": 0,
                "arm_physical": 0,
            },
            "terminal_planar_error_sum_m": 0.0,
            "terminal_height_error_sum_m": 0.0,
            "terminal_rotation_error_sum_rad": 0.0,
            "terminal_signed_yaw_error_sum_rad": 0.0,
            "terminal_yaw_progress_ratio_sum": 0.0,
            # Runtime episode identity and motion evidence.  Manifest scene
            # metadata alone is insufficient when a scene contains multiple
            # tasks or when a vectorized reset is replayed with one env.
            "task_indices": [],
            "initial_arm_joint_positions_rad": [],
            "initial_planar_error_sum_m": 0.0,
            "initial_height_error_sum_m": 0.0,
            "initial_rotation_error_sum_rad": 0.0,
            "initial_signed_yaw_error_sum_rad": 0.0,
            "maximum_target_translation_m": 0.0,
            "maximum_target_rotation_rad": 0.0,
        }
        for index, scene in enumerate(runtime_scenes)
    ]

    # reset environment
    obs, _ = env.get_observations()
    base_env = env.unwrapped
    target = base_env.scene["target"]
    initial_target_pose = torch.cat(
        (
            target.data.root_pos_w[0, :3] - base_env.scene.env_origins[0],
            target.data.root_quat_w[0],
        )
    ).clone()
    episode_initial_target_position = (
        target.data.root_pos_w[:, :3] - base_env.scene.env_origins
    ).clone()
    episode_initial_target_quaternion = target.data.root_quat_w.clone()
    episode_goal = base_env.command_manager.get_command(
        "target_object_pose"
    ).clone()
    episode_initial_position_delta = (
        episode_goal[:, :3] - episode_initial_target_position
    )
    episode_initial_planar_error = torch.linalg.vector_norm(
        episode_initial_position_delta[:, :2], dim=1
    )
    episode_initial_height_error = torch.abs(
        episode_initial_position_delta[:, 2]
    )
    episode_initial_quaternion_dot = torch.sum(
        episode_initial_target_quaternion * episode_goal[:, 3:7], dim=1
    )
    episode_initial_rotation_error = 2.0 * torch.acos(
        torch.clamp(torch.abs(episode_initial_quaternion_dot), max=1.0)
    )
    episode_initial_signed_yaw_error = (
        mdp.affordance_signed_yaw_goal_error(base_env).clone()
    )
    robot = base_env.scene["robot"]
    episode_initial_arm_joint_position = robot.data.joint_pos[:, :7].clone()
    episode_maximum_target_translation = torch.zeros(
        num_envs, device=base_env.device
    )
    episode_maximum_target_rotation = torch.zeros_like(
        episode_maximum_target_translation
    )
    diagnostic_trace: list[dict] = []
    episode_end_trace: list[dict] = []
    episode_reward_term_sums: dict[str, float] = {}
    terminal_planar_errors_m: list[float] = []
    terminal_height_errors_m: list[float] = []
    terminal_rotation_errors_rad: list[float] = []
    terminal_signed_yaw_errors_rad: list[float] = []
    terminal_yaw_progress_ratios: list[float] = []
    legal_safe_contact_episodes = 0
    yaw_counterfactual = {
        sign: {
            "samples": 0,
            "abs_yaw_sum_rad": 0.0,
            "action_l2_sum": 0.0,
            "delta_l2_sum": 0.0,
            "relative_delta_sum": 0.0,
            "delta_above_0p05": 0,
            "maximum_delta_l2": 0.0,
        }
        for sign in ("negative", "positive")
    }
    # Capture the first deterministic action for explicit C2 counterfactual
    # pairs.  The two scenes in a pair share target/task/robot state and differ
    # only in blocker XY, so this is a direct audit of whether the actor reacts
    # to blocker side before contact (rather than an indirect success metric).
    paired_side_initial_records: list[dict] = []

    # timing
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None

    # Initialize progress bar
    pbar = tqdm(total=args_cli.num_episodes, desc="Evaluating", unit="episodes")
    pbar.set_postfix({
        "Success Rate": "0.00%",
        "Episodes": 0,
        "Successes": 0
    })

    # Run until desired number of episodes are completed
    step_count = 0
    while episodes_completed < args_cli.num_episodes and simulation_app.is_running():
        start_time = time.time()
        step_count += 1

        # Sample before the action because ManagerBasedRLEnv auto-resets an
        # ended environment inside ``step``.  This retains motion through the
        # penultimate state without confusing it with the next task reset.
        current_target_position = (
            target.data.root_pos_w[:, :3] - base_env.scene.env_origins
        )
        current_target_translation = torch.linalg.vector_norm(
            current_target_position - episode_initial_target_position, dim=1
        )
        current_target_quaternion_dot = torch.sum(
            episode_initial_target_quaternion * target.data.root_quat_w,
            dim=1,
        )
        current_target_rotation = 2.0 * torch.acos(
            torch.clamp(
                torch.abs(current_target_quaternion_dot), max=1.0
            )
        )
        episode_maximum_target_translation = torch.maximum(
            episode_maximum_target_translation, current_target_translation
        )
        episode_maximum_target_rotation = torch.maximum(
            episode_maximum_target_rotation, current_target_rotation
        )

        pre_step_snapshot = None
        if args_cli.trace_episode_ends:
            zero_actions = torch.zeros(
                (1, env.unwrapped.action_manager.total_action_dim),
                device=env.unwrapped.device,
            )
            pre_step_snapshot = _clutter_diagnostic_snapshot(
                env,
                zero_actions,
                torch.zeros(1, device=env.unwrapped.device),
                step_count,
                initial_target_pose,
            )
        
        with torch.inference_mode():
            if args_cli.zero_actions:
                actions = torch.zeros(
                    (num_envs, env.unwrapped.action_manager.total_action_dim),
                    device=env.unwrapped.device,
                )
            elif args_cli.deterministic:
                actions = policy_func(obs)
                if (
                    step_count == 1
                    and isinstance(obs, torch.Tensor)
                    and runtime_scenes
                    and hasattr(base_env, "_clutter_scene_indices")
                    and obs.ndim == 2
                    and obs.shape[0] == num_envs
                    and obs.shape[1] >= 4096
                ):
                    scene_indices = base_env._clutter_scene_indices.detach().cpu()
                    initial_obs = obs.detach().cpu()
                    initial_actions = actions.detach().cpu()
                    paired_env_groups: dict[str, dict[str, int]] = {}
                    initial_record_by_env_id: dict[int, dict] = {}
                    for env_id, scene_index_value in enumerate(scene_indices.tolist()):
                        scene_index = int(scene_index_value)
                        if not 0 <= scene_index < len(runtime_scenes):
                            continue
                        scene_id = runtime_scenes[scene_index].scene_id
                        side = None
                        base_id = None
                        for candidate in ("positive", "negative"):
                            suffix = f"-c2-side-{candidate}"
                            if scene_id.endswith(suffix):
                                side = candidate
                                base_id = scene_id[: -len(suffix)]
                                break
                        if side is None:
                            continue
                        record = {
                            "base_id": base_id,
                            "side": side,
                            "scene_id": scene_id,
                            "target": initial_obs[env_id, :2560].clone(),
                            "obstacle": initial_obs[env_id, 2560:4096].clone(),
                            "state": initial_obs[env_id, 4096:].clone(),
                            "action": initial_actions[env_id].clone(),
                        }
                        paired_side_initial_records.append(record)
                        initial_record_by_env_id[env_id] = record
                        paired_env_groups.setdefault(base_id, {})[side] = env_id

                    # Causal C2 audit: keep every component of an environment's
                    # observation fixed and replace only its 512x3 obstacle
                    # point block with the block from the opposite-side scene.
                    # Comparing these actions is not confounded by independent
                    # simulator settling noise in robot/target state.
                    side_swapped_obs = obs.clone()
                    swapped_env_ids: set[int] = set()
                    for side_env_ids in paired_env_groups.values():
                        if set(side_env_ids) != {"positive", "negative"}:
                            continue
                        positive_env_id = side_env_ids["positive"]
                        negative_env_id = side_env_ids["negative"]
                        side_swapped_obs[positive_env_id, 2560:4096] = obs[
                            negative_env_id, 2560:4096
                        ]
                        side_swapped_obs[negative_env_id, 2560:4096] = obs[
                            positive_env_id, 2560:4096
                        ]
                        swapped_env_ids.update((positive_env_id, negative_env_id))
                    if swapped_env_ids:
                        side_swapped_actions = policy_func(side_swapped_obs).detach().cpu()
                        for env_id in swapped_env_ids:
                            initial_record_by_env_id[env_id][
                                "obstacle_swapped_action"
                            ] = side_swapped_actions[env_id].clone()
                previous_action_dim = int(actions.shape[1])
                rel_goal_start = 4096 + 9 + 14 + previous_action_dim
                if (
                    "AffordanceTeacher" in args_cli.task
                    and isinstance(obs, torch.Tensor)
                    and obs.ndim == 2
                    and obs.shape[1] >= rel_goal_start + 9
                ):
                    counterfactual_obs = (
                        flip_relative_goal_yaw_in_actor_observation(
                            obs, rel_goal_start=rel_goal_start
                        )
                    )
                    counterfactual_actions = policy_func(counterfactual_obs)
                    action_l2 = torch.linalg.vector_norm(actions, dim=1)
                    delta_l2 = torch.linalg.vector_norm(
                        counterfactual_actions - actions, dim=1
                    )
                    relative_delta = delta_l2 / torch.clamp(
                        action_l2, min=0.05
                    )
                    yaw = torch.atan2(
                        obs[:, rel_goal_start + 6],
                        obs[:, rel_goal_start + 3],
                    )
                    valid_yaw = torch.abs(yaw) > 1.0e-4
                    for sign, sign_mask in (
                        ("negative", yaw < 0.0),
                        ("positive", yaw >= 0.0),
                    ):
                        mask = valid_yaw & sign_mask
                        count = int(mask.sum().item())
                        if count == 0:
                            continue
                        stats = yaw_counterfactual[sign]
                        stats["samples"] += count
                        stats["abs_yaw_sum_rad"] += float(
                            torch.abs(yaw[mask]).sum().item()
                        )
                        stats["action_l2_sum"] += float(
                            action_l2[mask].sum().item()
                        )
                        stats["delta_l2_sum"] += float(
                            delta_l2[mask].sum().item()
                        )
                        stats["relative_delta_sum"] += float(
                            relative_delta[mask].sum().item()
                        )
                        stats["delta_above_0p05"] += int(
                            (delta_l2[mask] >= 0.05).sum().item()
                        )
                        stats["maximum_delta_l2"] = max(
                            float(stats["maximum_delta_l2"]),
                            float(delta_l2[mask].max().item()),
                        )
            else:
                # Match training-time exploration by sampling the action distribution.
                actions = policy_obj.act(obs)
            obs, rewards, dones, _ = env.step(actions)

        if args_cli.trace_episode_ends:
            for name, values in env.unwrapped.reward_manager.get_active_iterable_terms(0):
                episode_reward_term_sums[name] = (
                    episode_reward_term_sums.get(name, 0.0)
                    + float(values[0]) * float(env.unwrapped.step_dt)
                )

        sample_diagnostic = (
            step_count == 1
            or step_count % 30 == 0
            or (step_count + 1) % args_cli.max_episode_steps == 0
            # A protected-geodesic audit needs the contact transition and the
            # corresponding per-step potential delta, not a 30-step alias.
            # This attribute exists only for the experimental reward task, so
            # ordinary evaluation traces retain their compact cadence.
            or hasattr(
                env.unwrapped, "_protected_geodesic_route_length"
            )
        )
        # ManagerBasedRLEnv auto-resets ended environments inside step(), so a
        # post-done snapshot would describe the next episode rather than the
        # just-recorded video.
        if sample_diagnostic and not torch.any(dones):
            diagnostic_trace.append(
                _clutter_diagnostic_snapshot(env, actions, rewards, step_count, initial_target_pose)
            )

        # Use env's built-in success tracking - require it to be available
        if not hasattr(env.unwrapped, 'episode_success_buf'):
            raise AttributeError("Environment does not have episode_success_buf. This eval script requires the NonPrehensileEnv with success tracking.")

        # Use environment's episode ending signals directly
        # RslRlVecEnvWrapper already combines terminated | truncated into dones
        ended = dones.bool()
        

        if torch.any(ended):
            all_ended_ids = torch.where(ended)[0]
            eligible = (
                episodes_counted_by_env[all_ended_ids]
                < episode_quota_by_env[all_ended_ids]
            )
            # The simulator auto-resets all ended environments.  Only count a
            # terminal sample while that environment still has quota; this
            # yields a balanced, deterministic scene allocation.
            ended_ids = all_ended_ids[eligible]
            if args_cli.trace_episode_ends:
                termination_reasons = {
                    name: bool(env.unwrapped.termination_manager.get_term(name)[0].item())
                    for name in env.unwrapped.termination_manager.active_terms
                }
                # RewardManager retains the just-computed per-term values even
                # though ManagerBasedRLEnv has already auto-reset the ended
                # environment.  Persist them next to the terminal state so a
                # successful video is also an audit of the sparse success
                # signal received by PPO (values below include term weights;
                # ``step_contribution`` additionally includes the control dt).
                reward_terms = {
                    name: {
                        "weighted_rate": float(values[0]),
                        "step_contribution": float(values[0])
                        * float(env.unwrapped.step_dt),
                    }
                    for name, values in env.unwrapped.reward_manager.get_active_iterable_terms(0)
                }
                success_reward_predicates = {
                    name: bool(values[0].item())
                    for name, values in getattr(
                        env.unwrapped,
                        "_affordance_task_success_reward_debug",
                        {},
                    ).items()
                }
                post_reset_snapshot = _clutter_diagnostic_snapshot(
                    env, actions, rewards, step_count, initial_target_pose
                )
                terminal_state = {}
                if hasattr(
                    env.unwrapped, "_pre_reset_target_goal_planar_error"
                ):
                    terminal_state = {
                        "target_goal_planar_error_m": float(
                            env.unwrapped._pre_reset_target_goal_planar_error[0].item()
                        ),
                        "target_goal_rotation_error_rad": float(
                            env.unwrapped._pre_reset_target_goal_rotation_error[0].item()
                        ),
                        "target_goal_signed_yaw_error_rad": _terminal_metric_value(
                            env.unwrapped,
                            "_pre_reset_target_goal_signed_yaw_error",
                            0,
                        ),
                    }
                constrained_success = None
                affordance_violation = None
                typed_violations = {"c1": None, "c2": None, "c3": None}
                c1_breakdown = {
                    "hand_semantic": None,
                    "hand_neutral": None,
                    "hand_protected": None,
                    "arm_physical": None,
                }
                if hasattr(
                    env.unwrapped, "_episode_constrained_success_before_reset"
                ):
                    constrained_success = bool(
                        env.unwrapped._episode_constrained_success_before_reset[0].item()
                    )
                    affordance_violation = bool(
                        env.unwrapped._episode_affordance_violation_before_reset[0].item()
                    )
                    for label in typed_violations:
                        value = getattr(
                            env.unwrapped,
                            f"_episode_{label}_violation_before_reset",
                            None,
                        )
                        if value is not None:
                            typed_violations[label] = bool(value[0].item())
                    for label in c1_breakdown:
                        value = getattr(
                            env.unwrapped,
                            f"_episode_c1_{label}_violation_before_reset",
                            None,
                        )
                        if value is not None:
                            c1_breakdown[label] = bool(value[0].item())
                episode_end_trace.append(
                    {
                        "step": int(step_count),
                        "scene_index": int(
                            getattr(
                                env.unwrapped,
                                "_episode_clutter_scene_indices_before_reset",
                                env.unwrapped._clutter_scene_indices,
                            )[0].item()
                        ),
                        "scene_id": (
                            runtime_scenes[
                                int(
                                    getattr(
                                        env.unwrapped,
                                        "_episode_clutter_scene_indices_before_reset",
                                        env.unwrapped._clutter_scene_indices,
                                    )[0].item()
                                )
                            ].scene_id
                            if runtime_scenes
                            else None
                        ),
                        "termination_reasons": termination_reasons,
                        "reward_terms": reward_terms,
                        "episode_reward_term_sums": dict(
                            episode_reward_term_sums
                        ),
                        "success_reward_predicates": success_reward_predicates,
                        "constrained_success": constrained_success,
                        "affordance_violation": affordance_violation,
                        "typed_violations": typed_violations,
                        "c1_breakdown": c1_breakdown,
                        "pre_step": pre_step_snapshot,
                        "terminal_state": terminal_state,
                        "post_reset": post_reset_snapshot,
                    }
                )
                episode_reward_term_sums.clear()
            # Accumulate the selected terminal samples exactly once.
            for env_id in ended_ids.tolist():
                if hasattr(env.unwrapped, '_episode_success_before_reset'):
                    env_success = bool(
                        env.unwrapped._episode_success_before_reset[env_id].item()
                    )
                else:
                    env_success = bool(
                        env.unwrapped.episode_success_buf[env_id].item()
                    )
                env_constrained_success = bool(
                    getattr(
                        env.unwrapped,
                        "_episode_constrained_success_before_reset",
                        torch.zeros(num_envs, device=env.unwrapped.device),
                    )[env_id].item()
                )
                env_typed_violations = {
                    label: bool(
                        getattr(
                            env.unwrapped,
                            f"_episode_{label}_violation_before_reset",
                            torch.zeros(num_envs, device=env.unwrapped.device),
                        )[env_id].item()
                    )
                    for label in ("c1", "c2", "c3")
                }
                env_c1_breakdown = {
                    label: bool(
                        getattr(
                            env.unwrapped,
                            f"_episode_c1_{label}_violation_before_reset",
                            torch.zeros(num_envs, device=env.unwrapped.device),
                        )[env_id].item()
                    )
                    for label in (
                        "hand_semantic",
                        "hand_neutral",
                        "hand_protected",
                        "arm_physical",
                    )
                }
                env_affordance_violation = any(env_typed_violations.values())
                env_legal_safe_contact = bool(
                    getattr(
                        env.unwrapped,
                        "_episode_legal_safe_contact_before_reset",
                        torch.zeros(num_envs, device=env.unwrapped.device),
                    )[env_id].item()
                )
                env_terminal_planar_error = _terminal_metric_value(
                    env.unwrapped,
                    "_pre_reset_target_goal_planar_error",
                    env_id,
                )
                env_terminal_height_error = _terminal_metric_value(
                    env.unwrapped,
                    "_pre_reset_target_goal_height_error",
                    env_id,
                )
                env_terminal_rotation_error = _terminal_metric_value(
                    env.unwrapped,
                    "_pre_reset_target_goal_rotation_error",
                    env_id,
                )
                env_terminal_signed_yaw_error = _terminal_metric_value(
                    env.unwrapped,
                    "_pre_reset_target_goal_signed_yaw_error",
                    env_id,
                )
                terminal_planar_errors_m.append(env_terminal_planar_error)
                terminal_height_errors_m.append(env_terminal_height_error)
                terminal_rotation_errors_rad.append(env_terminal_rotation_error)
                terminal_signed_yaw_errors_rad.append(
                    env_terminal_signed_yaw_error
                )

                episodes_completed += 1
                total_successes += int(env_success)
                constrained_episodes += 1
                constrained_successes += int(env_constrained_success)
                legal_safe_contact_episodes += int(env_legal_safe_contact)
                affordance_violation_episodes += int(env_affordance_violation)
                for label, violated in env_typed_violations.items():
                    typed_violation_episodes[label] += int(violated)
                for label, violated in env_c1_breakdown.items():
                    c1_breakdown_episodes[label] += int(violated)
                episodes_counted_by_env[env_id] += 1

                if scene_stats and hasattr(env.unwrapped, "_clutter_scene_indices"):
                    scene_index = int(
                        getattr(
                            env.unwrapped,
                            "_episode_clutter_scene_indices_before_reset",
                            env.unwrapped._clutter_scene_indices,
                        )[env_id].item()
                    )
                    if 0 <= scene_index < len(scene_stats):
                        stats = scene_stats[scene_index]
                        stats["episodes"] += 1
                        stats["successes"] += int(env_success)
                        stats["constrained_successes"] += int(
                            env_constrained_success
                        )
                        stats["legal_safe_contact_episodes"] += int(
                            env_legal_safe_contact
                        )
                        for label, violated in env_typed_violations.items():
                            stats["typed_violation_episodes"][label] += int(
                                violated
                            )
                        for label, violated in env_c1_breakdown.items():
                            stats["c1_breakdown_episodes"][label] += int(violated)
                        stats["terminal_planar_error_sum_m"] += (
                            env_terminal_planar_error
                        )
                        stats["terminal_height_error_sum_m"] += (
                            env_terminal_height_error
                        )
                        stats["terminal_rotation_error_sum_rad"] += (
                            env_terminal_rotation_error
                        )
                        stats["terminal_signed_yaw_error_sum_rad"] += (
                            env_terminal_signed_yaw_error
                        )
                        runtime_task_index = int(
                            getattr(
                                env.unwrapped,
                                "_episode_clutter_task_indices_before_reset",
                                env.unwrapped._clutter_task_indices,
                            )[env_id].item()
                        )
                        stats["task_indices"].append(runtime_task_index)
                        stats["initial_arm_joint_positions_rad"].append(
                            episode_initial_arm_joint_position[env_id]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        stats["initial_planar_error_sum_m"] += float(
                            episode_initial_planar_error[env_id].item()
                        )
                        stats["initial_height_error_sum_m"] += float(
                            episode_initial_height_error[env_id].item()
                        )
                        stats["initial_rotation_error_sum_rad"] += float(
                            episode_initial_rotation_error[env_id].item()
                        )
                        initial_yaw_error = float(
                            episode_initial_signed_yaw_error[env_id].item()
                        )
                        stats["initial_signed_yaw_error_sum_rad"] += (
                            initial_yaw_error
                        )
                        stats["maximum_target_translation_m"] = max(
                            stats["maximum_target_translation_m"],
                            float(
                                episode_maximum_target_translation[env_id].item()
                            ),
                        )
                        stats["maximum_target_rotation_rad"] = max(
                            stats["maximum_target_rotation_rad"],
                            float(
                                episode_maximum_target_rotation[env_id].item()
                            ),
                        )
                        if (
                            initial_yaw_error is not None
                            and abs(initial_yaw_error) > 1.0e-6
                        ):
                            yaw_progress_ratio = 1.0 - (
                                env_terminal_signed_yaw_error
                                / initial_yaw_error
                            )
                            stats["terminal_yaw_progress_ratio_sum"] += (
                                yaw_progress_ratio
                            )
                            terminal_yaw_progress_ratios.append(
                                yaw_progress_ratio
                            )

                if 0 <= num_assets and num_assets > 0:
                    obj_idx = int(env_to_obj_idx[env_id].item()) if env_to_obj_idx.numel() == num_envs else -1
                    if 0 <= obj_idx < num_assets:
                        obj_name = asset_names[obj_idx]
                        obj_episodes[obj_name] = obj_episodes.get(obj_name, 0) + 1
                        if env_success:
                            obj_successes[obj_name] = obj_successes.get(obj_name, 0) + 1
                        # Debug: print first few per-object stats
                        if episodes_completed <= 5:
                            print(f"[DEBUG] Env {env_id} (obj {obj_name}): success={env_success}")

            # ``env.step`` has already reset every ended environment.  Seed
            # the motion reference for a possible next quota episode only
            # after the just-ended episode has been accounted for.
            if all_ended_ids.numel() > 0:
                reset_target_position = (
                    target.data.root_pos_w[:, :3]
                    - base_env.scene.env_origins
                )
                reset_goal = base_env.command_manager.get_command(
                    "target_object_pose"
                )
                reset_position_delta = reset_goal[:, :3] - reset_target_position
                reset_quaternion_dot = torch.sum(
                    target.data.root_quat_w * reset_goal[:, 3:7], dim=1
                )
                episode_initial_target_position[all_ended_ids] = (
                    reset_target_position[all_ended_ids]
                )
                episode_initial_target_quaternion[all_ended_ids] = (
                    target.data.root_quat_w[all_ended_ids]
                )
                episode_initial_planar_error[all_ended_ids] = (
                    torch.linalg.vector_norm(
                        reset_position_delta[all_ended_ids, :2], dim=1
                    )
                )
                episode_initial_height_error[all_ended_ids] = torch.abs(
                    reset_position_delta[all_ended_ids, 2]
                )
                episode_initial_rotation_error[all_ended_ids] = 2.0 * torch.acos(
                    torch.clamp(
                        torch.abs(reset_quaternion_dot[all_ended_ids]), max=1.0
                    )
                )
                episode_initial_signed_yaw_error[all_ended_ids] = (
                    mdp.affordance_signed_yaw_goal_error(base_env)[all_ended_ids]
                )
                episode_initial_arm_joint_position[all_ended_ids] = (
                    robot.data.joint_pos[all_ended_ids, :7]
                )
                episode_maximum_target_translation[all_ended_ids] = 0.0
                episode_maximum_target_rotation[all_ended_ids] = 0.0
            
            # Update progress bar
            current_success_rate = (total_successes / episodes_completed) * 100 if episodes_completed > 0 else 0.0
            pbar.update(ended_ids.numel())
            pbar.set_postfix({
                "Success Rate": f"{current_success_rate:.2f}%",
                "Episodes": episodes_completed,
                "Successes": total_successes
            })


        # optional real-time pacing
        if args_cli.real_time and (dt is not None):
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    success_rate = (total_successes / episodes_completed) if episodes_completed > 0 else 0.0

    # Close progress bar
    pbar.close()

    # persist results
    results_dir = os.path.abspath(args_cli.output_dir) if args_cli.output_dir else log_dir
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "eval_summary.json")
    per_object_path = os.path.join(results_dir, "eval_per_object.csv")
    per_scene_path = os.path.join(results_dir, "eval_per_scene.csv")
    yaw_counterfactual_summary = {}
    for sign, values in yaw_counterfactual.items():
        count = int(values["samples"])
        yaw_counterfactual_summary[sign] = {
            "samples": count,
            "mean_abs_yaw_rad": (
                float(values["abs_yaw_sum_rad"]) / count if count else None
            ),
            "mean_action_l2": (
                float(values["action_l2_sum"]) / count if count else None
            ),
            "mean_action_delta_l2": (
                float(values["delta_l2_sum"]) / count if count else None
            ),
            "mean_relative_action_delta": (
                float(values["relative_delta_sum"]) / count if count else None
            ),
            "fraction_action_delta_above_0p05": (
                int(values["delta_above_0p05"]) / count if count else None
            ),
            "maximum_action_delta_l2": (
                float(values["maximum_delta_l2"]) if count else None
            ),
        }
    paired_groups: dict[str, dict[str, dict]] = {}
    for record in paired_side_initial_records:
        paired_groups.setdefault(record["base_id"], {})[record["side"]] = record
    paired_side_diagnostics = []
    for base_id, sides in paired_groups.items():
        if set(sides) != {"positive", "negative"}:
            continue
        positive = sides["positive"]
        negative = sides["negative"]
        action_delta = positive["action"] - negative["action"]
        target_delta = positive["target"] - negative["target"]
        obstacle_delta = positive["obstacle"] - negative["obstacle"]
        state_delta = positive["state"] - negative["state"]
        positive_obstacle = positive["obstacle"].reshape(512, 3)
        negative_obstacle = negative["obstacle"].reshape(512, 3)
        centroid_delta = positive_obstacle.mean(dim=0) - negative_obstacle.mean(dim=0)
        pair_diagnostic = {
                "base_id": base_id,
                "positive_scene_id": positive["scene_id"],
                "negative_scene_id": negative["scene_id"],
                "action_positive": [float(value) for value in positive["action"].tolist()],
                "action_negative": [float(value) for value in negative["action"].tolist()],
                "action_delta_positive_minus_negative": [
                    float(value) for value in action_delta.tolist()
                ],
                "action_delta_l2": float(torch.linalg.vector_norm(action_delta).item()),
                "target_input_delta_l2": float(torch.linalg.vector_norm(target_delta).item()),
                "target_input_delta_max_abs": float(torch.abs(target_delta).max().item()),
                "obstacle_input_delta_l2": float(torch.linalg.vector_norm(obstacle_delta).item()),
                "obstacle_centroid_delta_positive_minus_negative_m": [
                    float(value) for value in centroid_delta.tolist()
                ],
                "state_input_delta_l2": float(torch.linalg.vector_norm(state_delta).item()),
                "state_input_delta_max_abs": float(torch.abs(state_delta).max().item()),
            }
        if (
            "obstacle_swapped_action" in positive
            and "obstacle_swapped_action" in negative
        ):
            positive_swap_delta = (
                positive["obstacle_swapped_action"] - positive["action"]
            )
            negative_swap_delta = (
                negative["obstacle_swapped_action"] - negative["action"]
            )
            pair_diagnostic.update(
                {
                    "positive_fixed_state_obstacle_swap_action": [
                        float(value)
                        for value in positive["obstacle_swapped_action"].tolist()
                    ],
                    "positive_fixed_state_obstacle_swap_delta": [
                        float(value) for value in positive_swap_delta.tolist()
                    ],
                    "positive_fixed_state_obstacle_swap_delta_l2": float(
                        torch.linalg.vector_norm(positive_swap_delta).item()
                    ),
                    "negative_fixed_state_obstacle_swap_action": [
                        float(value)
                        for value in negative["obstacle_swapped_action"].tolist()
                    ],
                    "negative_fixed_state_obstacle_swap_delta": [
                        float(value) for value in negative_swap_delta.tolist()
                    ],
                    "negative_fixed_state_obstacle_swap_delta_l2": float(
                        torch.linalg.vector_norm(negative_swap_delta).item()
                    ),
                }
            )
        paired_side_diagnostics.append(pair_diagnostic)
    paired_action_deltas = [
        item["action_delta_l2"] for item in paired_side_diagnostics
    ]
    paired_obstacle_only_action_deltas = [
        item[key]
        for item in paired_side_diagnostics
        for key in (
            "positive_fixed_state_obstacle_swap_delta_l2",
            "negative_fixed_state_obstacle_swap_delta_l2",
        )
        if key in item
    ]
    results_payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "clutter_manifest": os.environ.get("DAPL_CLUTTER_MANIFEST"),
        "seed": args_cli.seed,
        "deterministic": bool(args_cli.deterministic),
        "zero_actions": bool(args_cli.zero_actions),
        "episode_allocation": "balanced_per_environment",
        "episode_quota_min": int(episode_quota_by_env.min().item()),
        "episode_quota_max": int(episode_quota_by_env.max().item()),
        "episodes": int(episodes_completed),
        "successes": int(total_successes),
        "success_rate": float(success_rate),
        "constrained_episodes": constrained_episodes,
        "constrained_successes": constrained_successes,
        "constrained_success_rate": (
            constrained_successes / constrained_episodes
            if constrained_episodes > 0
            else None
        ),
        "legal_safe_contact_episodes": legal_safe_contact_episodes,
        "legal_safe_contact_episode_rate": (
            legal_safe_contact_episodes / constrained_episodes
            if constrained_episodes > 0
            else None
        ),
        "affordance_violation_episodes": affordance_violation_episodes,
        "affordance_violation_rate": (
            affordance_violation_episodes / constrained_episodes
            if constrained_episodes > 0
            else None
        ),
        "typed_violation_episodes": typed_violation_episodes,
        "typed_violation_rates": {
            label: (
                count / constrained_episodes if constrained_episodes > 0 else None
            )
            for label, count in typed_violation_episodes.items()
        },
        "c1_breakdown_episodes": c1_breakdown_episodes,
        "c1_breakdown_rates": {
            label: (
                count / constrained_episodes if constrained_episodes > 0 else None
            )
            for label, count in c1_breakdown_episodes.items()
        },
        "terminal_error_summary": {
            "planar_m": _scalar_summary(terminal_planar_errors_m),
            "height_m": _scalar_summary(terminal_height_errors_m),
            "rotation_rad": _scalar_summary(terminal_rotation_errors_rad),
            "signed_yaw_rad": _scalar_summary(
                terminal_signed_yaw_errors_rad
            ),
            "yaw_progress_ratio": _scalar_summary(
                terminal_yaw_progress_ratios
            ),
        },
        "counterfactual_yaw_action_sensitivity": {
            "intervention": "reflect only relative-goal R01/R10 (yaw -> -yaw)",
            "by_observed_yaw_sign": yaw_counterfactual_summary,
        },
        "paired_side_initial_action_sensitivity": {
            "contract": (
                "same target/task/robot state; blocker XY is the only intended "
                "counterfactual change"
            ),
            "complete_pair_count": len(paired_side_diagnostics),
            "action_delta_l2_summary": _scalar_summary(paired_action_deltas),
            "fixed_state_obstacle_only_intervention": (
                "replace only observation[2560:4096] with the opposite-side "
                "512x3 blocker point block"
            ),
            "fixed_state_obstacle_only_action_delta_l2_summary": _scalar_summary(
                paired_obstacle_only_action_deltas
            ),
            "pairs": paired_side_diagnostics,
        },
        "diagnostic_trace": diagnostic_trace,
        "episode_end_trace": episode_end_trace,
        "diagnostic_summary": {
            "min_ee_target_distance_m": min(
                (sample["ee_target_distance_m"] for sample in diagnostic_trace), default=None
            ),
            "min_target_goal_planar_error_m": min(
                (sample["target_goal_planar_error_m"] for sample in diagnostic_trace), default=None
            ),
            "min_target_goal_rotation_error_rad": min(
                (sample["target_goal_rotation_error_rad"] for sample in diagnostic_trace), default=None
            ),
            "max_target_translation_from_start_m": max(
                (sample["target_translation_from_start_m"] for sample in diagnostic_trace), default=None
            ),
            "max_obstacle_mean_translation_m": max(
                (sample["obstacle_mean_translation_m"] for sample in diagnostic_trace), default=None
            ),
            "min_semantic_clearance_conditioned_scale": min(
                (
                    sample["semantic_clearance_conditioned_scale"]
                    for sample in diagnostic_trace
                    if "semantic_clearance_conditioned_scale" in sample
                ),
                default=None,
            ),
            "max_semantic_clearance_conditioned_scale": max(
                (
                    sample["semantic_clearance_conditioned_scale"]
                    for sample in diagnostic_trace
                    if "semantic_clearance_conditioned_scale" in sample
                ),
                default=None,
            ),
            "semantic_detour_sample_fraction": (
                sum(
                    bool(sample["semantic_route_used_detour"])
                    for sample in diagnostic_trace
                    if "semantic_route_used_detour" in sample
                )
                / sum(
                    "semantic_route_used_detour" in sample
                    for sample in diagnostic_trace
                )
                if any(
                    "semantic_route_used_detour" in sample
                    for sample in diagnostic_trace
                )
                else None
            ),
            "mean_recovery_outward_alignment": (
                sum(
                    sample["semantic_field_outward_alignment"]
                    for sample in diagnostic_trace
                    if sample.get("semantic_recovery_active", False)
                )
                / sum(
                    1
                    for sample in diagnostic_trace
                    if sample.get("semantic_recovery_active", False)
                )
                if any(
                    sample.get("semantic_recovery_active", False)
                    for sample in diagnostic_trace
                )
                else None
            ),
            "mean_recovery_tangential_fraction": (
                sum(
                    sample["semantic_field_tangential_fraction"]
                    for sample in diagnostic_trace
                    if sample.get("semantic_recovery_active", False)
                )
                / sum(
                    1
                    for sample in diagnostic_trace
                    if sample.get("semantic_recovery_active", False)
                )
                if any(
                    sample.get("semantic_recovery_active", False)
                    for sample in diagnostic_trace
                )
                else None
            ),
            "min_terminal_target_goal_planar_error_m": min(
                (
                    item["terminal_state"]["target_goal_planar_error_m"]
                    for item in episode_end_trace
                    if item.get("terminal_state")
                ),
                default=None,
            ),
            "min_terminal_target_goal_rotation_error_rad": min(
                (
                    item["terminal_state"]["target_goal_rotation_error_rad"]
                    for item in episode_end_trace
                    if item.get("terminal_state")
                ),
                default=None,
            ),
            "final": diagnostic_trace[-1] if diagnostic_trace else None,
        },
        "per_object": [
            {
                "name": name,
                "episodes": int(obj_episodes.get(name, 0)),
                "successes": int(obj_successes.get(name, 0)),
                "success_rate": (float(obj_successes.get(name, 0)) / float(obj_episodes.get(name, 0))) if obj_episodes.get(name, 0) > 0 else 0.0,
            }
            for name in sorted(obj_episodes.keys())
        ],
        "per_scene": [
            {
                **stats,
                "success_rate": (
                    stats["successes"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "constrained_success_rate": (
                    stats["constrained_successes"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "legal_safe_contact_episode_rate": (
                    stats["legal_safe_contact_episodes"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "typed_violation_rates": {
                    label: (
                        count / stats["episodes"]
                        if stats["episodes"] > 0
                        else None
                    )
                    for label, count in stats[
                        "typed_violation_episodes"
                    ].items()
                },
                "c1_breakdown_rates": {
                    label: (
                        count / stats["episodes"]
                        if stats["episodes"] > 0
                        else None
                    )
                    for label, count in stats[
                        "c1_breakdown_episodes"
                    ].items()
                },
                "terminal_planar_error_mean_m": (
                    stats["terminal_planar_error_sum_m"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "terminal_height_error_mean_m": (
                    stats["terminal_height_error_sum_m"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "terminal_rotation_error_mean_rad": (
                    stats["terminal_rotation_error_sum_rad"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "terminal_signed_yaw_error_mean_rad": (
                    stats["terminal_signed_yaw_error_sum_rad"]
                    / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "terminal_yaw_progress_ratio_mean": (
                    stats["terminal_yaw_progress_ratio_sum"]
                    / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "initial_planar_error_mean_m": (
                    stats["initial_planar_error_sum_m"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "initial_height_error_mean_m": (
                    stats["initial_height_error_sum_m"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "initial_rotation_error_mean_rad": (
                    stats["initial_rotation_error_sum_rad"] / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
                "initial_signed_yaw_error_mean_rad": (
                    stats["initial_signed_yaw_error_sum_rad"]
                    / stats["episodes"]
                    if stats["episodes"] > 0
                    else None
                ),
            }
            for stats in scene_stats
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    # write CSV for per-object breakdown
    with open(per_object_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["name", "episodes", "successes", "success_rate"])
        for name in sorted(obj_episodes.keys()):
            ep = int(obj_episodes.get(name, 0))
            sc = int(obj_successes.get(name, 0))
            rate = (sc / ep) if ep > 0 else 0.0
            writer.writerow([name, ep, sc, rate])

    with open(per_scene_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "scene_index",
                "scene_id",
                "goal_direction_deg",
                "goal_planar_displacement_m",
                "goal_yaw_delta_rad",
                "task_indices",
                "episodes",
                "successes",
                "success_rate",
                "constrained_successes",
                "constrained_success_rate",
                "legal_safe_contact_episode_rate",
                "c1_violation_rate",
                "c1_hand_neutral_violation_rate",
                "c1_hand_protected_violation_rate",
                "c1_arm_physical_violation_rate",
                "c2_violation_rate",
                "c3_violation_rate",
                "terminal_planar_error_mean_m",
                "terminal_height_error_mean_m",
                "terminal_rotation_error_mean_rad",
                "terminal_signed_yaw_error_mean_rad",
                "terminal_yaw_progress_ratio_mean",
                "initial_planar_error_mean_m",
                "initial_height_error_mean_m",
                "initial_rotation_error_mean_rad",
                "initial_signed_yaw_error_mean_rad",
                "maximum_target_translation_m",
                "maximum_target_rotation_rad",
            ]
        )
        for stats in results_payload["per_scene"]:
            writer.writerow(
                [
                    stats["scene_index"],
                    stats["scene_id"],
                    stats["goal_direction_deg"],
                    stats["goal_planar_displacement_m"],
                    stats["goal_yaw_delta_rad"],
                    ";".join(str(value) for value in stats["task_indices"]),
                    stats["episodes"],
                    stats["successes"],
                    stats["success_rate"],
                    stats["constrained_successes"],
                    stats["constrained_success_rate"],
                    stats["legal_safe_contact_episode_rate"],
                    stats["typed_violation_rates"]["c1"],
                    stats["c1_breakdown_rates"]["hand_neutral"],
                    stats["c1_breakdown_rates"]["hand_protected"],
                    stats["c1_breakdown_rates"]["arm_physical"],
                    stats["typed_violation_rates"]["c2"],
                    stats["typed_violation_rates"]["c3"],
                    stats["terminal_planar_error_mean_m"],
                    stats["terminal_height_error_mean_m"],
                    stats["terminal_rotation_error_mean_rad"],
                    stats["terminal_signed_yaw_error_mean_rad"],
                    stats["terminal_yaw_progress_ratio_mean"],
                    stats["initial_planar_error_mean_m"],
                    stats["initial_height_error_mean_m"],
                    stats["initial_rotation_error_mean_rad"],
                    stats["initial_signed_yaw_error_mean_rad"],
                    stats["maximum_target_translation_m"],
                    stats["maximum_target_rotation_rad"],
                ]
            )

    # print summary
    print("\n========== Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {resume_path}")
    print(f"Episodes: {episodes_completed}")
    print(f"Successes: {total_successes}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    if constrained_episodes > 0:
        print(
            "Constrained Success Rate: "
            f"{100.0 * constrained_successes / constrained_episodes:.2f}%"
        )
        print(
            "Legal Safe-Contact Episode Rate: "
            f"{100.0 * legal_safe_contact_episodes / constrained_episodes:.2f}%"
        )
        print(
            "Affordance Violation Rate: "
            f"{100.0 * affordance_violation_episodes / constrained_episodes:.2f}%"
        )
        print(
            "Typed Violation Rates: "
            + ", ".join(
                f"{label.upper()}={100.0 * count / constrained_episodes:.2f}%"
                for label, count in typed_violation_episodes.items()
            )
        )
        print(
            "C1 Breakdown Rates: "
            + ", ".join(
                f"{label}={100.0 * count / constrained_episodes:.2f}%"
                for label, count in c1_breakdown_episodes.items()
            )
        )
    if any(
        values["samples"] > 0
        for values in yaw_counterfactual_summary.values()
    ):
        print(
            "Counterfactual yaw action delta: "
            + ", ".join(
                f"{sign}={values['mean_action_delta_l2']:.4f} "
                f"({100.0 * values['fraction_action_delta_above_0p05']:.1f}% >= 0.05)"
                for sign, values in yaw_counterfactual_summary.items()
                if values["samples"] > 0
            )
        )
    if diagnostic_trace:
        diagnostic_summary = results_payload["diagnostic_summary"]
        print(
            "Diagnostics: "
            f"min finger-target={diagnostic_summary['min_ee_target_distance_m']:.4f} m, "
            f"min planar goal error={diagnostic_summary['min_target_goal_planar_error_m']:.4f} m, "
            f"min rotation goal error={diagnostic_summary['min_target_goal_rotation_error_rad']:.4f} rad"
        )
        if diagnostic_summary["min_terminal_target_goal_planar_error_m"] is not None:
            print(
                "Terminal diagnostics: "
                f"planar goal error={diagnostic_summary['min_terminal_target_goal_planar_error_m']:.4f} m, "
                "rotation goal error="
                f"{diagnostic_summary['min_terminal_target_goal_rotation_error_rad']:.4f} rad"
            )
    print(f"Saved: {summary_path}")
    if len(obj_episodes) > 0:
        print(f"Saved: {per_object_path}")
    if scene_stats:
        print(f"Saved: {per_scene_path}")
    if num_assets > 0 and len(obj_episodes) > 0:
        print("\nPer-object success rates:")
        for name in sorted(obj_episodes.keys()):
            ep = obj_episodes[name]
            sc = obj_successes.get(name, 0)
            rate = (sc / ep) * 100.0 if ep > 0 else 0.0
            print(f"  - {name}: {sc}/{ep} ({rate:.2f}%)")
    print("=======================================\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
