"""Isaac Lab video overlays for the M1 oracle contact planner.

The visual layer is deliberately separate from :mod:`oracle_contact` so the
planner and its unit tests remain simulator-independent.  Every marker below
is non-physical and therefore cannot change planning or evaluation outcomes.
"""

from __future__ import annotations

import gymnasium as gym
import torch


def _create_goal_object_ghost(env, opacity: float):
    """Spawn a collision-free translucent copy of the target at the goal."""

    import isaaclab.sim as sim_utils
    from isaacsim.core.prims import XFormPrim

    base = env.unwrapped
    if base.num_envs != 1:
        raise ValueError("M1 video overlays require exactly one environment")
    if not 0.0 < opacity <= 1.0:
        raise ValueError("goal ghost opacity must be in (0, 1]")

    target_spawn_cfg = base.scene["target"].cfg.spawn
    asset_cfgs = getattr(target_spawn_cfg, "assets_cfg", None)
    target_asset_cfg = asset_cfgs[0] if asset_cfgs else target_spawn_cfg
    if not hasattr(target_asset_cfg, "usd_path"):
        raise TypeError("goal ghost requires a USD-backed target asset")

    command = base.command_manager.get_command("target_object_pose")
    goal_position = command[0, :3] + base.scene.env_origins[0]
    goal_quaternion = command[0, 3:7]
    if torch.linalg.vector_norm(goal_quaternion).item() < 0.5:
        goal_position = base.scene["target"].data.root_pos_w[0, :3]
        goal_quaternion = base.scene["target"].data.root_quat_w[0]

    prim_path = "/Visuals/ContactPlannerM1/GoalObjectGhost"
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
        translation=tuple(goal_position.detach().cpu().tolist()),
        orientation=tuple(goal_quaternion.detach().cpu().tolist()),
    )
    return XFormPrim(
        prim_path,
        name="m1_goal_object_ghost",
        reset_xform_properties=False,
        usd=True,
    )


def _sphere_marker(path: str, radius: float, color: tuple[float, float, float]):
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import SPHERE_MARKER_CFG

    cfg = SPHERE_MARKER_CFG.copy()
    cfg.prim_path = path
    cfg.markers["sphere"].radius = radius
    cfg.markers["sphere"].visual_material.diffuse_color = color
    return VisualizationMarkers(cfg)


def create_m1_video_markers(env, *, goal_ghost_opacity: float) -> dict:
    """Create goal, semantic, and selected-plan overlays for one M1 scene."""

    markers = {
        "goal_ghost": _create_goal_object_ghost(env, goal_ghost_opacity),
        "safe": _sphere_marker(
            "/Visuals/ContactPlannerM1/SafeContact", 0.006, (0.0, 1.0, 0.0)
        ),
        "protected": _sphere_marker(
            "/Visuals/ContactPlannerM1/ProtectedFunctional",
            0.006,
            (1.0, 0.0, 0.0),
        ),
        "contact": _sphere_marker(
            "/Visuals/ContactPlannerM1/SelectedContact", 0.009, (1.0, 0.9, 0.0)
        ),
        "precontact": _sphere_marker(
            "/Visuals/ContactPlannerM1/SelectedPrecontact",
            0.006,
            (1.0, 0.35, 0.0),
        ),
        "push": _sphere_marker(
            "/Visuals/ContactPlannerM1/SelectedPush", 0.006, (0.8, 0.0, 1.0)
        ),
    }
    update_m1_video_markers(env, markers)
    return markers


def _subsample(points: torch.Tensor, mask: torch.Tensor, limit: int = 128):
    selected = points[mask]
    if selected.shape[0] <= limit:
        return selected
    indices = torch.linspace(
        0, selected.shape[0] - 1, limit, device=selected.device
    ).long()
    return selected[indices]


def update_m1_video_markers(env, markers: dict) -> None:
    """Keep the goal ghost and oracle affordance overlays attached to state."""

    from isaaclab.managers import SceneEntityCfg

    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp

    base = env.unwrapped
    command = base.command_manager.get_command("target_object_pose")
    goal_position = command[:, :3] + base.scene.env_origins
    goal_quaternion = command[:, 3:7]
    if torch.all(torch.linalg.vector_norm(goal_quaternion, dim=1) >= 0.5):
        markers["goal_ghost"].set_world_poses(
            positions=goal_position,
            orientations=goal_quaternion,
            usd=True,
        )

    points = mdp.get_object_pointcloud_in_env_frame(
        base, SceneEntityCfg("target")
    ).reshape(base.num_envs, -1, 3)
    semantics = mdp.domino_target_affordance(
        base, target_cfg=SceneEntityCfg("target")
    ).reshape(base.num_envs, -1, 2)
    points_world = points[0] + base.scene.env_origins[0]
    markers["safe"].visualize(
        translations=_subsample(points_world, semantics[0, :, 0] >= 0.25)
    )
    markers["protected"].visualize(
        translations=_subsample(points_world, semantics[0, :, 1] >= 0.25)
    )


def show_selected_plan(
    markers: dict,
    *,
    env_origin: torch.Tensor,
    contact: torch.Tensor,
    precontact: torch.Tensor,
    push: torch.Tensor,
) -> None:
    """Display the selected TCP contact (yellow), staging, and push endpoints."""

    markers["contact"].visualize(translations=(contact + env_origin).reshape(1, 3))
    markers["precontact"].visualize(
        translations=(precontact + env_origin).reshape(1, 3)
    )
    markers["push"].visualize(translations=(push + env_origin).reshape(1, 3))


class M1MarkerUpdateWrapper(gym.Wrapper):
    """Refresh moving overlays before ``RecordVideo`` captures each frame."""

    def __init__(self, env, markers: dict):
        super().__init__(env)
        self._markers = markers

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        update_m1_video_markers(self.env, self._markers)
        return result

    def step(self, action):
        result = self.env.step(action)
        update_m1_video_markers(self.env, self._markers)
        return result
