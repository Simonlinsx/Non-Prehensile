"""Part-aware contact proxies derived from DOMINO sparse annotations.

DOMINO does not provide per-triangle semantic collision meshes.  These terms
therefore use the released contact/functional anchors to define metric regions
and point-cloud proximity to detect semantic-contact violations.  They are
useful for training and benchmark diagnostics, but are intentionally named and
documented as proxies rather than exact PhysX contact reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from dapl.domino import (
    DominoDataPaths,
    default_affordance_radius,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
)

from .observations import (
    get_end_effector_pointcloud_in_env_frame,
    get_object_pointcloud_in_env_frame,
    get_obstacle_pointclouds_in_env_frame,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass(frozen=True)
class _DominoAssetSemantics:
    features: torch.Tensor
    contact_anchors: torch.Tensor
    functional_anchors: torch.Tensor
    safe_radius_m: float
    protected_radius_m: float


def _radius_key(value: float | None) -> float | None:
    return None if value is None else round(float(value), 8)


def _asset_semantics(
    env: "ManagerBasedRLEnv",
    target_cfg: SceneEntityCfg,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
) -> tuple[_DominoAssetSemantics, ...]:
    """Load and device-cache semantics in target MultiAssetSpawner order."""

    target: RigidObject = env.scene[target_cfg.name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype
    key = (
        target_cfg.name,
        _radius_key(safe_radius_m),
        _radius_key(protected_radius_m),
        str(device),
        str(dtype),
    )
    cache = getattr(env, "_domino_asset_semantics", {})
    if key in cache:
        return cache[key]

    root = getattr(env.cfg, "clutter_asset_root", None)
    paths = DominoDataPaths.resolve(root)
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env import (
        get_cached_cloud,
    )

    result = []
    for asset_cfg in target.cfg.spawn.assets_cfg:
        asset_id = getattr(asset_cfg, "dapl_asset_id", None)
        if asset_id is None:
            raise RuntimeError(
                "target spawn config has no asset id; use the DOMINO manifest asset resolver"
            )
        source = paths.require_source_asset(asset_id)
        annotation = load_domino_affordance_annotation(source)
        if not annotation.contact_anchors or not annotation.functional_anchors:
            raise ValueError(
                f"DOMINO target {asset_id!r} needs both contact_points_pose and functional_matrix"
            )
        configured_scale = tuple(float(item) for item in (asset_cfg.scale or (1.0, 1.0, 1.0)))
        if any(
            abs(configured_scale[index] - annotation.scale[index]) > 1.0e-6
            for index in range(3)
        ):
            raise ValueError(
                f"manifest scale {configured_scale} for {asset_id!r} does not match "
                f"DOMINO annotation scale {annotation.scale}"
            )
        cloud = get_cached_cloud(asset_cfg.obj_path)
        canonical_points = torch.as_tensor(cloud.points, device=device, dtype=dtype)
        if canonical_points.shape != (512, 3):
            raise ValueError(
                f"DOMINO target cloud must contain 512 points, got {tuple(canonical_points.shape)}"
            )
        default_radius = default_affordance_radius(annotation)
        safe_radius = default_radius if safe_radius_m is None else float(safe_radius_m)
        protected_radius = (
            default_radius if protected_radius_m is None else float(protected_radius_m)
        )
        result.append(
            _DominoAssetSemantics(
                features=domino_point_affordance_features(
                    canonical_points,
                    annotation,
                    safe_radius_m=safe_radius,
                    protected_radius_m=protected_radius,
                ),
                contact_anchors=annotation.anchor_positions(
                    "contact", device=device, dtype=dtype
                ),
                functional_anchors=annotation.anchor_positions(
                    "functional", device=device, dtype=dtype
                ),
                safe_radius_m=safe_radius,
                protected_radius_m=protected_radius,
            )
        )
    cache[key] = tuple(result)
    env._domino_asset_semantics = cache
    return cache[key]


def _batched_semantics(
    env: "ManagerBasedRLEnv",
    target_cfg: SceneEntityCfg,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast per-asset labels/anchors to vectorized environments."""

    semantics = _asset_semantics(
        env, target_cfg, safe_radius_m, protected_radius_m
    )
    target: RigidObject = env.scene[target_cfg.name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype
    asset_indices = torch.remainder(
        torch.arange(env.num_envs, device=device, dtype=torch.long), len(semantics)
    )
    features = torch.stack([item.features for item in semantics], dim=0)[asset_indices]

    max_contact = max(len(item.contact_anchors) for item in semantics)
    max_functional = max(len(item.functional_anchors) for item in semantics)
    contact = torch.zeros((len(semantics), max_contact, 3), device=device, dtype=dtype)
    contact_mask = torch.zeros((len(semantics), max_contact), device=device, dtype=torch.bool)
    functional = torch.zeros((len(semantics), max_functional, 3), device=device, dtype=dtype)
    functional_mask = torch.zeros((len(semantics), max_functional), device=device, dtype=torch.bool)
    radii = torch.empty((len(semantics), 2), device=device, dtype=dtype)
    for index, item in enumerate(semantics):
        contact[index, : len(item.contact_anchors)] = item.contact_anchors
        contact_mask[index, : len(item.contact_anchors)] = True
        functional[index, : len(item.functional_anchors)] = item.functional_anchors
        functional_mask[index, : len(item.functional_anchors)] = True
        radii[index] = torch.tensor(
            (item.safe_radius_m, item.protected_radius_m), device=device, dtype=dtype
        )
    return (
        features,
        contact[asset_indices],
        contact_mask[asset_indices],
        functional[asset_indices],
        functional_mask[asset_indices],
        radii[asset_indices],
    )


def domino_target_affordance(
    env: "ManagerBasedRLEnv",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
) -> torch.Tensor:
    """Return 512 aligned ``[safe_contact, protected_functional]`` scores.

    The 1,024-D flattened result follows the exact canonical target point order
    used by ``get_object_pointcloud_in_env_frame`` and the DAPL physical scene.
    """

    features, *_ = _batched_semantics(
        env, target_cfg, safe_radius_m, protected_radius_m
    )
    return features.reshape(env.num_envs, -1)


def domino_affordance_contact_state(
    env: "ManagerBasedRLEnv",
    *,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> dict[str, torch.Tensor]:
    """Evaluate semantic robot contact and protected-part obstacle clearance.

    Robot contact is approximated by the closest pair between the 256-point
    end-effector cloud and the 512-point target cloud.  Functional-part
    collision is approximated by the closest distance between up to 64
    high-scoring protected target-surface points and all obstacle point clouds.
    """

    if contact_distance_m <= 0.0 or protected_clearance_m < 0.0:
        raise ValueError("contact distance must be positive and clearance non-negative")
    if not 0.0 <= minimum_safe_score <= 1.0 or not 0.0 <= minimum_protected_score <= 1.0:
        raise ValueError("minimum affordance scores must be in [0, 1]")
    if protected_point_count <= 0 or protected_point_count > 512:
        raise ValueError("protected_point_count must be in [1, 512]")
    cache_key = (
        int(getattr(env, "common_step_counter", -1)),
        round(float(contact_distance_m), 8),
        round(float(minimum_safe_score), 8),
        round(float(minimum_protected_score), 8),
        int(protected_point_count),
        round(float(protected_clearance_m), 8),
        _radius_key(safe_radius_m),
        _radius_key(protected_radius_m),
        target_cfg.name,
        obstacles_cfg.name,
        ee_frame_cfg.name,
    )
    cached = getattr(env, "_domino_affordance_state_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    (
        features,
        _contact_anchors,
        _contact_mask,
        _functional_anchors,
        _functional_mask,
        _radii,
    ) = _batched_semantics(env, target_cfg, safe_radius_m, protected_radius_m)

    target_points = get_object_pointcloud_in_env_frame(env, target_cfg).reshape(
        env.num_envs, 512, 3
    )
    end_effector_points = get_end_effector_pointcloud_in_env_frame(
        env, ee_frame_cfg
    )
    robot_target_distances = torch.cdist(end_effector_points, target_points)
    flat_robot_distance = robot_target_distances.reshape(env.num_envs, -1)
    minimum_robot_distance, closest_pair = flat_robot_distance.min(dim=1)
    closest_target_index = torch.remainder(closest_pair, target_points.shape[1])
    batch_index = torch.arange(env.num_envs, device=target_points.device)
    closest_safe_score = features[batch_index, closest_target_index, 0]
    robot_contact = minimum_robot_distance <= float(contact_distance_m)
    safe_robot_contact = robot_contact & (closest_safe_score >= minimum_safe_score)
    forbidden_robot_contact = robot_contact & ~safe_robot_contact

    protected_scores, protected_indices = torch.topk(
        features[..., 1], k=protected_point_count, dim=1, largest=True, sorted=False
    )
    protected_mask = protected_scores >= minimum_protected_score
    protected_points = torch.gather(
        target_points,
        1,
        protected_indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    obstacle_points = get_obstacle_pointclouds_in_env_frame(
        env, obstacles_cfg
    ).reshape(env.num_envs, -1, 3)
    functional_distances = torch.cdist(protected_points, obstacle_points)
    functional_distances = functional_distances.masked_fill(
        ~protected_mask.unsqueeze(-1), torch.inf
    )
    minimum_functional_distance = functional_distances.flatten(1).min(dim=1).values
    protected_obstacle_collision = minimum_functional_distance <= float(
        protected_clearance_m
    )

    result = {
        "robot_contact": robot_contact,
        "safe_robot_contact": safe_robot_contact,
        "forbidden_robot_contact": forbidden_robot_contact,
        "protected_obstacle_collision": protected_obstacle_collision,
        "minimum_robot_target_distance": minimum_robot_distance,
        "closest_safe_score": closest_safe_score,
        "protected_clearance": minimum_functional_distance,
    }
    env._domino_affordance_state_cache = (cache_key, result)
    return result


def _contact_term_state(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float,
    minimum_safe_score: float,
    minimum_protected_score: float,
    protected_point_count: int,
    protected_clearance_m: float,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Forward explicitly named manager parameters to the shared evaluator."""

    return domino_affordance_contact_state(
        env,
        contact_distance_m=contact_distance_m,
        minimum_safe_score=minimum_safe_score,
        minimum_protected_score=minimum_protected_score,
        protected_point_count=protected_point_count,
        protected_clearance_m=protected_clearance_m,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )


def safe_region_contact_reward(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward target contact whose closest semantic point is in the safe region."""

    return _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["safe_robot_contact"].float()


def forbidden_region_contact_penalty(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Positive indicator intended for use with a negative reward weight."""

    return _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["forbidden_robot_contact"].float()


def protected_region_collision_penalty(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Positive protected-part collision indicator for a negative reward weight."""

    return _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["protected_obstacle_collision"].float()


def forbidden_region_contact(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hard termination for robot contact outside the safe DOMINO region."""

    return _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["forbidden_robot_contact"]


def protected_region_collision(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hard termination for functional-region proximity to clutter."""

    return _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["protected_obstacle_collision"]
