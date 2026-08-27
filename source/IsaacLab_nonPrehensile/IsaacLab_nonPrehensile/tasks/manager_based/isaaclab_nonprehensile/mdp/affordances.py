"""Part-aware contact proxies derived from DOMINO annotations.

Audited assets use canonical part masks; remaining DOMINO assets use metric
regions expanded from the released contact/functional anchors.  Point-cloud
proximity detects semantic-contact violations.  These are training and
benchmark proxies rather than exact PhysX contact reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_conjugate, quat_mul

from dapl.domino import (
    DominoDataPaths,
    default_affordance_radius,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
)
from dapl.metrics import (
    axis_aligned_bounding_box_keypoints,
    bounded_joint_pose_tracking_cost,
    bounded_linear_distance_score,
    clearance_conditioned_route_scale,
    clearance_log_barrier,
    componentwise_progress_during_contact,
    dapl_combined_pose_error,
    dapl_multiscale_pose_score,
    dapl_tanh_proximity_reward,
    distance_progress_during_contact,
    discounted_potential_shaping,
    discounted_score_potential_shaping,
    dywa_exponential_keypoint_potential,
    gate_navigation_at_legal_contact,
    goal_swept_semantic_point_index,
    lexicographic_route_potential,
    near_goal_motion_cost,
    normalized_clearance_violation,
    normalized_contact_distance_excess,
    normalized_directional_displacement,
    normalized_distance_progress,
    planar_lateral_escape_axis,
    potential_consistent_progress,
    positive_distance_progress_during_contact,
    positive_reference_relative_component_improvement,
    positive_reference_relative_error_improvement,
    positive_reference_relative_pareto_pose_improvement,
    positive_reference_relative_score,
    reference_relative_pose_improvement,
    rigid_body_ring_route_aabb_clearance,
    route_conditioned_alignment,
    sampled_segment_minimum_clearance,
    semantic_clearance_recovery_direction,
    semantic_ring_route_geometry,
    semantic_tangential_recovery_direction,
    signed_reference_relative_error_improvement,
    smooth_max_normalized_pose_error,
    support_aware_pose_success,
    signed_yaw_contact_moment_score,
    update_route_detour_commitment,
    update_consecutive_success_count,
    wrench_aware_contact_support_score,
    weighted_componentwise_pose_progress,
    yaw_compatible_safe_point_mask,
)

from .observations import (
    get_end_effector_pointcloud_in_env_frame,
    get_object_pointcloud_in_env_frame,
    get_obstacle_pointclouds_in_env_frame,
)


ContactSensorNames = str | tuple[str, ...] | None

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


def _affordance_scene_geometry(
    env: "ManagerBasedRLEnv",
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target and obstacle clouds once per control step.

    Transforming every asset cloud is appreciable work at 1,024 environments.
    The observation and reward managers run in the same control step, so keep a
    small step-local cache and clear it whenever a manifest task is reset.
    """

    key = (
        int(getattr(env, "common_step_counter", -1)),
        target_cfg.name,
        obstacles_cfg.name,
    )
    cached = getattr(env, "_domino_affordance_geometry_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    target_points = get_object_pointcloud_in_env_frame(env, target_cfg).reshape(
        env.num_envs, 512, 3
    )
    configured_active_count = getattr(env.cfg, "active_obstacle_count", None)
    active_obstacle_count = getattr(
        env, "_clutter_active_obstacle_count", configured_active_count
    )
    if active_obstacle_count == 0:
        # Zero is the explicit no-clutter token consumed by the obstacle
        # PointNet.  Preserve 512x3 so all curriculum checkpoints are compatible.
        # Branch before querying the obstacle collection: T0 keeps inactive
        # obstacle slots for checkpoint compatibility, but transforming all of
        # their surface clouds every control step is pure overhead.
        obstacle_points = torch.zeros(
            (env.num_envs, 512, 3),
            device=target_points.device,
            dtype=target_points.dtype,
        )
    else:
        all_obstacle_points = get_obstacle_pointclouds_in_env_frame(
            env, obstacles_cfg
        )
        if active_obstacle_count is None:
            active_obstacle_count = all_obstacle_points.shape[1]
        active_obstacle_count = int(active_obstacle_count)
        obstacle_points = all_obstacle_points[:, :active_obstacle_count].reshape(
            env.num_envs, -1, 3
        )
    result = (target_points, obstacle_points)
    env._domino_affordance_geometry_cache = (key, result)
    return result


def domino_affordance_policy_scene(
    env: "ManagerBasedRLEnv",
    target_point_count: int = 512,
    obstacle_point_count: int = 512,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
) -> torch.Tensor:
    """Return the fixed policy prefix: target ``[xyz,safe,protected]`` + clutter xyz.

    The target scores remain aligned with their canonical surface points before
    PointNet pooling.  Obstacles use a deterministic, evenly-spaced subsample so
    the policy can reason about protected-part clearance without increasing the
    observation shape when the curriculum moves from sparse to cluttered scenes.
    """

    if target_point_count != 512:
        raise ValueError("DOMINO target_point_count must remain 512 for aligned labels")
    if obstacle_point_count <= 0:
        raise ValueError("obstacle_point_count must be positive")
    features, *_ = _batched_semantics(
        env, target_cfg, safe_radius_m, protected_radius_m
    )
    target_points, obstacle_points = _affordance_scene_geometry(
        env, target_cfg, obstacles_cfg
    )
    if obstacle_point_count > obstacle_points.shape[1]:
        raise ValueError(
            f"requested {obstacle_point_count} obstacle points from only "
            f"{obstacle_points.shape[1]} available points"
        )
    sample_indices = torch.div(
        torch.arange(
            obstacle_point_count,
            device=obstacle_points.device,
            dtype=torch.long,
        )
        * obstacle_points.shape[1],
        obstacle_point_count,
        rounding_mode="floor",
    )
    sampled_obstacles = obstacle_points[:, sample_indices]
    semantic_target = torch.cat((target_points, features), dim=-1)
    return torch.cat(
        (semantic_target.flatten(1), sampled_obstacles.flatten(1)), dim=-1
    )


def _chunked_min_right_per_left(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    right_mask: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each left point's nearest right point with bounded temporary memory."""

    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
        raise ValueError("left and right point sets must be batched 3-D tensors")
    best_distance = torch.full(
        left.shape[:2], torch.inf, device=left.device, dtype=left.dtype
    )
    best_right_index = torch.zeros(
        left.shape[:2], device=left.device, dtype=torch.long
    )
    for start in range(0, right.shape[1], chunk_size):
        stop = min(start + chunk_size, right.shape[1])
        distances = torch.cdist(left, right[:, start:stop])
        if right_mask is not None:
            distances = distances.masked_fill(
                ~right_mask[:, None, start:stop], torch.inf
            )
        chunk_distance, local_right_index = distances.min(dim=2)
        replace = chunk_distance < best_distance
        best_distance = torch.where(replace, chunk_distance, best_distance)
        best_right_index = torch.where(
            replace, local_right_index + start, best_right_index
        )
    return best_distance, best_right_index


def _chunked_closest_right_point(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    right_mask: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the globally closest pair without allocating ``B x L x R``."""

    per_left_distance, per_left_right_index = _chunked_min_right_per_left(
        left,
        right,
        right_mask=right_mask,
        chunk_size=chunk_size,
    )
    distance, left_index = per_left_distance.min(dim=1)
    right_index = torch.gather(per_left_right_index, 1, left_index[:, None]).squeeze(1)
    return distance, right_index


def _chunked_target_semantic_distances(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    semantic_masks: tuple[torch.Tensor, ...],
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Find nearest target points and all semantic-set minima in one scan.

    The previous contact evaluator called ``torch.cdist`` once for the
    unmasked target and once again for every semantic subset.  With 512 target
    points and 64-point chunks that meant 40 kernel launches per evaluator,
    multiplied again when reward terms used different contact-state cache
    keys.  Each distance block is independent of its semantic mask, so compute
    it once and reduce every requested subset before moving to the next block.
    """

    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
        raise ValueError("left and right point sets must be batched 3-D tensors")
    for mask in semantic_masks:
        if mask.shape != right.shape[:2] or mask.dtype != torch.bool:
            raise ValueError(
                "semantic masks must be boolean tensors matching right points"
            )

    best_distance = torch.full(
        left.shape[:2], torch.inf, device=left.device, dtype=left.dtype
    )
    best_right_index = torch.zeros(
        left.shape[:2], device=left.device, dtype=torch.long
    )
    semantic_minima = [
        torch.full(
            (left.shape[0],), torch.inf, device=left.device, dtype=left.dtype
        )
        for _ in semantic_masks
    ]
    for start in range(0, right.shape[1], chunk_size):
        stop = min(start + chunk_size, right.shape[1])
        distances = torch.cdist(left, right[:, start:stop])

        chunk_distance, local_right_index = distances.min(dim=2)
        replace = chunk_distance < best_distance
        best_distance = torch.where(replace, chunk_distance, best_distance)
        best_right_index = torch.where(
            replace, local_right_index + start, best_right_index
        )

        for index, mask in enumerate(semantic_masks):
            masked_distances = distances.masked_fill(
                ~mask[:, None, start:stop], torch.inf
            )
            chunk_minimum = masked_distances.flatten(1).min(dim=1).values
            semantic_minima[index] = torch.minimum(
                semantic_minima[index], chunk_minimum
            )

    return best_distance, best_right_index, tuple(semantic_minima)


def _robot_target_affordance_geometry(
    env: "ManagerBasedRLEnv",
    *,
    minimum_safe_score: float,
    minimum_protected_score: float,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Cache expensive robot-target geometry independently of contact rules."""

    key = (
        int(getattr(env, "common_step_counter", -1)),
        round(float(minimum_safe_score), 8),
        round(float(minimum_protected_score), 8),
        _radius_key(safe_radius_m),
        _radius_key(protected_radius_m),
        target_cfg.name,
        obstacles_cfg.name,
        ee_frame_cfg.name,
    )
    cached = getattr(env, "_domino_robot_target_geometry_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    features, *_ = _batched_semantics(
        env, target_cfg, safe_radius_m, protected_radius_m
    )
    target_points, obstacle_points = _affordance_scene_geometry(
        env, target_cfg, obstacles_cfg
    )
    end_effector_points = get_end_effector_pointcloud_in_env_frame(
        env, ee_frame_cfg
    )
    safe_mask = features[..., 0] >= minimum_safe_score
    protected_mask = features[..., 1] >= minimum_protected_score
    forbidden_mask = ~safe_mask
    neutral_mask = forbidden_mask & ~protected_mask
    (
        robot_point_distance,
        robot_target_index,
        semantic_minima,
    ) = _chunked_target_semantic_distances(
        end_effector_points,
        target_points,
        semantic_masks=(
            safe_mask,
            forbidden_mask,
            neutral_mask,
            protected_mask,
        ),
    )
    result = {
        "features": features,
        "target_points": target_points,
        "obstacle_points": obstacle_points,
        "end_effector_points": end_effector_points,
        "robot_point_distance": robot_point_distance,
        "robot_target_index": robot_target_index,
        "safe_mask": safe_mask,
        "protected_mask": protected_mask,
        "forbidden_mask": forbidden_mask,
        "neutral_mask": neutral_mask,
        "minimum_safe_distance": semantic_minima[0],
        "minimum_robot_forbidden_distance": semantic_minima[1],
        "minimum_robot_neutral_distance": semantic_minima[2],
        "minimum_robot_protected_distance": semantic_minima[3],
    }
    env._domino_robot_target_geometry_cache = (key, result)
    return result


def _robot_arm_proxy_points_in_env_frame(
    env: "ManagerBasedRLEnv",
    *,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    samples_per_segment: int = 5,
) -> torch.Tensor:
    """Sample the Franka link centerline for continuous whole-arm clearance.

    Filtered PhysX sensors remain the C3 collision authority.  This sparse,
    differentiable geometry proxy only supplies a useful pre-contact gradient
    for proximal links, which are absent from the end-effector point cloud.
    """

    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    robot = env.scene[robot_cfg.name]
    body_names = tuple(robot.body_names)
    expected_names = tuple(
        [f"panda_link{index}" for index in range(8)] + ["panda_hand"]
    )
    cache_key = (robot_cfg.name, body_names, int(samples_per_segment))
    index_cache = getattr(env, "_robot_arm_proxy_index_cache", {})
    if cache_key not in index_cache:
        missing = [name for name in expected_names if name not in body_names]
        if missing:
            raise RuntimeError(
                "whole-arm clearance proxy could not resolve Franka bodies: "
                + ", ".join(missing)
            )
        index_cache[cache_key] = torch.tensor(
            [body_names.index(name) for name in expected_names],
            device=robot.data.body_pos_w.device,
            dtype=torch.long,
        )
        env._robot_arm_proxy_index_cache = index_cache

    centers = robot.data.body_pos_w[:, index_cache[cache_key], :3]
    centers = centers - env.scene.env_origins[:, None, :]
    alpha = torch.linspace(
        0.0,
        1.0,
        samples_per_segment,
        device=centers.device,
        dtype=centers.dtype,
    ).view(1, 1, samples_per_segment, 1)
    segment_start = centers[:, :-1, None, :]
    segment_end = centers[:, 1:, None, :]
    return ((1.0 - alpha) * segment_start + alpha * segment_end).flatten(1, 2)


def _filtered_contact_event(
    env: "ManagerBasedRLEnv",
    sensor_name: ContactSensorNames,
    *,
    force_threshold_n: float,
    exclude_body_names: tuple[str, ...] = (),
) -> tuple[torch.Tensor, bool]:
    """Return a filtered PhysX contact event when a reporter is configured.

    The semantic region still comes from the aligned target point cloud, but
    this reporter supplies an independent physical-contact audit for whole-arm
    target contacts and robot/clutter contacts.  Existing tasks intentionally
    fall back to geometry when the optional teacher sensors are absent.
    """

    if force_threshold_n < 0.0:
        raise ValueError("contact force threshold must be non-negative")
    empty = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if not sensor_name:
        return empty, False
    sensors = getattr(env.scene, "sensors", {})
    sensor_names = (sensor_name,) if isinstance(sensor_name, str) else sensor_name
    event = empty
    available = False
    for name in sensor_names:
        if name not in sensors:
            continue
        sensor = sensors[name]
        forces = sensor.data.force_matrix_w
        if forces is None:
            continue
        active = torch.linalg.vector_norm(forces, dim=-1) > float(force_threshold_n)
        if exclude_body_names:
            include = torch.tensor(
                [body_name not in exclude_body_names for body_name in sensor.body_names],
                device=active.device,
                dtype=torch.bool,
            )
            active = active & include.view(1, -1, 1)
        event |= torch.any(active, dim=(1, 2))
        available = True
    return event, available


def domino_affordance_contact_state(
    env: "ManagerBasedRLEnv",
    *,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    robot_obstacle_clearance_m: float = 0.005,
    robot_link_proxy_radius_m: float = 0.045,
    robot_link_samples_per_segment: int = 5,
    physical_contact_force_threshold_n: float = 0.5,
    evaluate_protected: bool = True,
    evaluate_robot_obstacle: bool = False,
    require_physical_protected_contact: bool = False,
    robot_target_sensor_name: ContactSensorNames = None,
    robot_obstacle_sensor_name: ContactSensorNames = None,
    target_obstacle_sensor_name: ContactSensorNames = None,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Evaluate semantic robot contact and protected-part obstacle clearance.

    Robot contact is approximated by the closest pair between the 256-point
    end-effector cloud and the 512-point target cloud.  Safe-region shaping and
    protected-part clearance use every point that passes the corresponding
    semantic threshold.  ``protected_point_count`` remains in the signature
    only for compatibility with existing task configs and checkpoints.
    """

    if (
        contact_distance_m <= 0.0
        or protected_clearance_m < 0.0
        or robot_obstacle_clearance_m < 0.0
        or robot_link_proxy_radius_m < 0.0
    ):
        raise ValueError(
            "contact distance must be positive; clearance and proxy radius "
            "must be non-negative"
        )
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
        round(float(robot_obstacle_clearance_m), 8),
        round(float(robot_link_proxy_radius_m), 8),
        int(robot_link_samples_per_segment),
        round(float(physical_contact_force_threshold_n), 8),
        bool(evaluate_protected),
        bool(evaluate_robot_obstacle),
        bool(require_physical_protected_contact),
        robot_target_sensor_name,
        robot_obstacle_sensor_name,
        target_obstacle_sensor_name,
        _radius_key(safe_radius_m),
        _radius_key(protected_radius_m),
        target_cfg.name,
        obstacles_cfg.name,
        ee_frame_cfg.name,
        robot_cfg.name,
    )
    step = cache_key[0]
    if getattr(env, "_domino_affordance_state_cache_step", None) != step:
        env._domino_affordance_state_cache = {}
        env._domino_affordance_state_cache_step = step
    cache = getattr(env, "_domino_affordance_state_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    geometry = _robot_target_affordance_geometry(
        env,
        minimum_safe_score=minimum_safe_score,
        minimum_protected_score=minimum_protected_score,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    features = geometry["features"]
    target_points = geometry["target_points"]
    obstacle_points = geometry["obstacle_points"]
    end_effector_points = geometry["end_effector_points"]
    robot_point_distance = geometry["robot_point_distance"]
    robot_target_index = geometry["robot_target_index"]
    minimum_robot_distance, closest_robot_index = robot_point_distance.min(dim=1)
    closest_target_index = torch.gather(
        robot_target_index, 1, closest_robot_index[:, None]
    ).squeeze(1)
    batch_index = torch.arange(env.num_envs, device=target_points.device)
    closest_safe_score = features[batch_index, closest_target_index, 0]
    robot_point_safe_score = torch.gather(features[..., 0], 1, robot_target_index)
    robot_point_protected_score = torch.gather(
        features[..., 1], 1, robot_target_index
    )
    robot_point_contact = robot_point_distance <= float(contact_distance_m)
    robot_contact = torch.any(robot_point_contact, dim=1)
    safe_robot_contact = torch.any(
        robot_point_contact & (robot_point_safe_score >= minimum_safe_score), dim=1
    )
    forbidden_hand_contact = torch.any(
        robot_point_contact & (robot_point_safe_score < minimum_safe_score), dim=1
    )
    protected_hand_contact = torch.any(
        robot_point_contact
        & (robot_point_safe_score < minimum_safe_score)
        & (robot_point_protected_score >= minimum_protected_score),
        dim=1,
    )
    neutral_hand_contact = torch.any(
        robot_point_contact
        & (robot_point_safe_score < minimum_safe_score)
        & (robot_point_protected_score < minimum_protected_score),
        dim=1,
    )

    # The hand cloud classifies the intended contact region.  Any filtered
    # target contact made by a proximal Franka body is always forbidden.
    arm_target_physical_contact, robot_target_sensor_available = (
        _filtered_contact_event(
            env,
            robot_target_sensor_name,
            force_threshold_n=physical_contact_force_threshold_n,
            exclude_body_names=(
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ),
        )
    )
    forbidden_robot_contact = forbidden_hand_contact | arm_target_physical_contact
    # Soft exploration still needs pose-progress gradients when one hand point
    # reaches the safe patch while another point briefly violates C1.  Keep the
    # raw safe-contact predicate for those signed progress terms, but expose a
    # separate legal predicate for the one-time bonus and strict accounting.
    # Hard profiles terminate the same mixed-contact transition immediately.
    legal_safe_robot_contact = safe_robot_contact & ~forbidden_robot_contact

    safe_mask = geometry["safe_mask"]
    protected_mask = geometry["protected_mask"]
    minimum_safe_distance = geometry["minimum_safe_distance"]
    minimum_robot_forbidden_distance = geometry[
        "minimum_robot_forbidden_distance"
    ]
    minimum_robot_neutral_distance = geometry["minimum_robot_neutral_distance"]
    minimum_robot_protected_distance = geometry[
        "minimum_robot_protected_distance"
    ]
    if evaluate_robot_obstacle:
        minimum_hand_obstacle_distance, _ = _chunked_closest_right_point(
            end_effector_points, obstacle_points
        )
        arm_proxy_points = _robot_arm_proxy_points_in_env_frame(
            env,
            robot_cfg=robot_cfg,
            samples_per_segment=robot_link_samples_per_segment,
        )
        minimum_arm_centerline_distance, _ = _chunked_closest_right_point(
            arm_proxy_points, obstacle_points
        )
        minimum_arm_obstacle_distance = torch.clamp(
            minimum_arm_centerline_distance - float(robot_link_proxy_radius_m),
            min=0.0,
        )
        minimum_robot_obstacle_distance = torch.minimum(
            minimum_hand_obstacle_distance, minimum_arm_obstacle_distance
        )
    else:
        minimum_robot_obstacle_distance = torch.full_like(
            minimum_robot_distance, torch.inf
        )
    if evaluate_protected:
        target_obstacle_physical_contact, target_obstacle_sensor_available = (
            _filtered_contact_event(
                env,
                target_obstacle_sensor_name,
                force_threshold_n=physical_contact_force_threshold_n,
            )
        )
        minimum_functional_distance, _ = _chunked_closest_right_point(
            obstacle_points, target_points, right_mask=protected_mask
        )
        protected_obstacle_collision = minimum_functional_distance <= float(
            protected_clearance_m
        )
        if require_physical_protected_contact and target_obstacle_sensor_available:
            protected_obstacle_collision &= target_obstacle_physical_contact
    else:
        minimum_functional_distance = torch.full_like(minimum_robot_distance, torch.inf)
        protected_obstacle_collision = torch.zeros_like(robot_contact)
        target_obstacle_physical_contact = torch.zeros_like(robot_contact)
        target_obstacle_sensor_available = False

    if evaluate_robot_obstacle:
        robot_obstacle_physical_contact, robot_obstacle_sensor_available = (
            _filtered_contact_event(
                env,
                robot_obstacle_sensor_name,
                force_threshold_n=physical_contact_force_threshold_n,
            )
        )
        robot_obstacle_collision = (
            robot_obstacle_physical_contact
            if robot_obstacle_sensor_available
            else minimum_robot_obstacle_distance <= float(robot_obstacle_clearance_m)
        )
    else:
        robot_obstacle_collision = torch.zeros_like(robot_contact)
        robot_obstacle_physical_contact = torch.zeros_like(robot_contact)
        robot_obstacle_sensor_available = False

    result = {
        "robot_contact": robot_contact,
        "safe_robot_contact": safe_robot_contact,
        "legal_safe_robot_contact": legal_safe_robot_contact,
        "forbidden_robot_contact": forbidden_robot_contact,
        "forbidden_hand_contact": forbidden_hand_contact,
        "neutral_hand_contact": neutral_hand_contact,
        "protected_hand_contact": protected_hand_contact,
        "protected_obstacle_collision": protected_obstacle_collision,
        "robot_obstacle_collision": robot_obstacle_collision,
        "arm_target_physical_contact": arm_target_physical_contact,
        "target_obstacle_physical_contact": target_obstacle_physical_contact,
        "robot_obstacle_physical_contact": robot_obstacle_physical_contact,
        "robot_target_sensor_available": torch.full_like(
            robot_contact, robot_target_sensor_available
        ),
        "target_obstacle_sensor_available": torch.full_like(
            robot_contact, target_obstacle_sensor_available
        ),
        "robot_obstacle_sensor_available": torch.full_like(
            robot_contact, robot_obstacle_sensor_available
        ),
        "minimum_robot_target_distance": minimum_robot_distance,
        "minimum_safe_distance": minimum_safe_distance,
        "minimum_robot_forbidden_distance": minimum_robot_forbidden_distance,
        "minimum_robot_neutral_distance": minimum_robot_neutral_distance,
        "minimum_robot_protected_distance": minimum_robot_protected_distance,
        "closest_safe_score": closest_safe_score,
        "protected_clearance": minimum_functional_distance,
        "robot_obstacle_clearance": minimum_robot_obstacle_distance,
    }
    cache[cache_key] = result
    env._domino_affordance_state_cache = cache
    return result


def _contact_term_state(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float,
    minimum_safe_score: float,
    minimum_protected_score: float,
    protected_point_count: int,
    protected_clearance_m: float,
    evaluate_protected: bool,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    *,
    robot_obstacle_clearance_m: float = 0.005,
    physical_contact_force_threshold_n: float = 0.5,
    evaluate_robot_obstacle: bool = False,
    require_physical_protected_contact: bool = False,
    robot_target_sensor_name: ContactSensorNames = None,
    robot_obstacle_sensor_name: ContactSensorNames = None,
    target_obstacle_sensor_name: ContactSensorNames = None,
) -> dict[str, torch.Tensor]:
    """Forward explicitly named manager parameters to the shared evaluator."""

    return domino_affordance_contact_state(
        env,
        contact_distance_m=contact_distance_m,
        minimum_safe_score=minimum_safe_score,
        minimum_protected_score=minimum_protected_score,
        protected_point_count=protected_point_count,
        protected_clearance_m=protected_clearance_m,
        robot_obstacle_clearance_m=robot_obstacle_clearance_m,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        evaluate_protected=evaluate_protected,
        evaluate_robot_obstacle=evaluate_robot_obstacle,
        require_physical_protected_contact=require_physical_protected_contact,
        robot_target_sensor_name=robot_target_sensor_name,
        robot_obstacle_sensor_name=robot_obstacle_sensor_name,
        target_obstacle_sensor_name=target_obstacle_sensor_name,
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
    evaluate_protected: bool = True,
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
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["safe_robot_contact"].float()


def safe_region_distance_reward(
    env: "ManagerBasedRLEnv",
    maximum_distance_m: float = 0.50,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Linear shaping toward safe surface points with useful far-field slope."""

    state = _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )
    score = bounded_linear_distance_score(
        state["minimum_safe_distance"], maximum_distance=maximum_distance_m
    )
    # Approach shaping is useful before contact, but a persistent proximity
    # reward after contact encourages the policy to park at the handle.
    return score * (~state["safe_robot_contact"]).float()


def safe_region_ee_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """DAPL proximity shaping measured to the legal affordance set.

    This is the sole task-specific substitution in DAPL's contact reward:
    target-centroid distance is replaced by the minimum end-effector distance
    to a point labelled safe.  The tanh kernel and scale are unchanged.
    """

    state = _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )
    return dapl_tanh_proximity_reward(
        state["minimum_safe_distance"], standard_deviation=std
    )


def safe_region_gated_object_goal_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    safe_ee_distance_threshold: float = 0.1,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """DAPL full-pose goal reward gated by proximity to the safe region."""

    if safe_ee_distance_threshold <= 0.0:
        raise ValueError("safe_ee_distance_threshold must be positive")
    state = _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )
    target: RigidObject = env.scene[target_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_position = target.data.root_pos_w[:, :3] - env.scene.env_origins
    position_distance = torch.linalg.vector_norm(
        command[:, :3] - target_position, dim=-1
    )
    quaternion_dot = torch.sum(target.data.root_quat_w * command[:, 3:7], dim=-1)
    rotation_distance = 2.0 * torch.acos(
        torch.clamp(torch.abs(quaternion_dot), max=1.0)
    )
    pose_distance = dapl_combined_pose_error(
        position_distance, rotation_distance
    )
    near_safe_region = state["minimum_safe_distance"] < safe_ee_distance_threshold
    return near_safe_region * dapl_tanh_proximity_reward(
        pose_distance, standard_deviation=std
    )


def safe_region_distance_penalty(
    env: "ManagerBasedRLEnv",
    normalization_distance_m: float = 0.10,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Distance cost with zero cost at and inside the contact boundary."""

    state = _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )
    return normalized_contact_distance_excess(
        state["minimum_safe_distance"],
        contact_distance=contact_distance_m,
        normalization_distance=normalization_distance_m,
    )


def _goal_conditioned_safe_side_route(
    env: "ManagerBasedRLEnv",
    *,
    command_name: str,
    minimum_safe_score: float,
    side_band_m: float,
    minimum_goal_displacement_m: float,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    yaw_moment_weight: float = 0.0,
    yaw_activation_rad: float = 0.10,
    use_full_safe_region: bool = False,
    use_yaw_compatible_safe_region: bool = False,
    yaw_side_band_m: float = 0.010,
    yaw_compatible_selection_mode: str = "near_best",
    yaw_minimum_compatibility_m: float = 0.002,
    minimum_yaw_error_rad: float = 0.020,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve the closest hand-to-safe route compatible with the goal wrench.

    This is an object-centric set potential, not a world-frame waypoint.  For
    each environment, safe points are scored by their trailing-side support
    and, optionally, their signed moment arm for the remaining yaw.  Points
    within ``side_band_m`` of the best score form the admissible approach set.
    ``use_yaw_compatible_safe_region`` instead makes selection lexicographic:
    while yaw is materially wrong it retains a broad safe subset with the
    correct signed moment arm, then falls back to translation support.  Close
    to the goal, where the displacement direction is ill-conditioned, the
    translation-only set falls back to the complete safe region.
    """

    if not 0.0 <= minimum_safe_score <= 1.0:
        raise ValueError("minimum safe score must be in [0, 1]")
    if side_band_m < 0.0 or minimum_goal_displacement_m < 0.0:
        raise ValueError("side band and minimum goal displacement must be non-negative")
    if yaw_moment_weight < 0.0 or yaw_activation_rad <= 0.0:
        raise ValueError("yaw moment weight must be non-negative and activation positive")
    if (
        yaw_side_band_m < 0.0
        or yaw_minimum_compatibility_m < 0.0
        or minimum_yaw_error_rad < 0.0
    ):
        raise ValueError("yaw selection margins and minimum yaw error must be non-negative")
    if use_full_safe_region and use_yaw_compatible_safe_region:
        raise ValueError(
            "full-safe and yaw-compatible safe-region selection are mutually exclusive"
        )

    features, *_ = _batched_semantics(
        env, target_cfg, safe_radius_m, protected_radius_m
    )
    target_points, _ = _affordance_scene_geometry(env, target_cfg, obstacles_cfg)
    end_effector_points = get_end_effector_pointcloud_in_env_frame(
        env, ee_frame_cfg
    )

    safe_mask = features[..., 0] >= float(minimum_safe_score)
    # Audited assets always have a safe region.  Retain a finite fallback for
    # malformed annotations so a training batch cannot emit inf/NaN rewards.
    has_safe = safe_mask.any(dim=1, keepdim=True)
    safe_mask = torch.where(has_safe, safe_mask, torch.ones_like(safe_mask))
    target: RigidObject = env.scene[target_cfg.name]
    current_position = target.data.root_pos_w[:, :3] - env.scene.env_origins
    goal = env.command_manager.get_command(command_name)
    goal_displacement = goal[:, :2] - current_position[:, :2]
    displacement_norm = torch.linalg.vector_norm(goal_displacement, dim=1)
    relative_quat = quat_mul(
        goal[:, 3:7], quat_conjugate(target.data.root_quat_w)
    )
    relative_rotation = matrix_from_quat(relative_quat)
    yaw_error = torch.atan2(
        relative_rotation[:, 1, 0], relative_rotation[:, 0, 0]
    )
    point_offset_xy = target_points[:, :, :2] - current_position[:, None, :2]
    support_score = wrench_aware_contact_support_score(
        # Moment arms must be measured from the rigid body's root/COM rather
        # than from the centroid of the safe annotation.  For the legacy
        # translation-only projection this origin change is a batchwise
        # constant and therefore leaves the selected trailing subset exact.
        point_offset_xy,
        goal_displacement,
        yaw_error,
        yaw_moment_weight=yaw_moment_weight,
        yaw_activation_rad=yaw_activation_rad,
    )
    best_support_score = support_score.masked_fill(~safe_mask, -torch.inf).max(
        dim=1, keepdim=True
    ).values
    side_mask = safe_mask & (
        support_score >= best_support_score - float(side_band_m)
    )
    directional_goal = displacement_norm >= float(minimum_goal_displacement_m)
    if use_full_safe_region:
        # The policy receives the semantic cloud and recoverable relative
        # goal, so choosing *which* legal handle point to use is a policy
        # decision.  The approach reward only supplies the requested dense
        # distance-to-safe-region signal and must not impose a single-push
        # wrench that can be infeasible for one yaw sign.
        approach_mask = safe_mask
    elif use_yaw_compatible_safe_region:
        # Translation and yaw support are deliberately not added together.
        # A weighted sum created a cancellation basin in v52/v53 and selected
        # the wrong moment sign on the asymmetric hammer for some scenes.
        # This state-conditioned set is computed from current geometry and the
        # recoverable relative goal; it is not a waypoint or a hidden phase.
        yaw_score = signed_yaw_contact_moment_score(
            point_offset_xy,
            goal_displacement,
            yaw_error,
            yaw_activation_rad=yaw_activation_rad,
        )
        yaw_side_mask, yaw_selection_fallback = yaw_compatible_safe_point_mask(
            yaw_score,
            safe_mask,
            selection_mode=yaw_compatible_selection_mode,
            near_best_band_m=yaw_side_band_m,
            minimum_compatibility_m=yaw_minimum_compatibility_m,
        )
        translation_mask = torch.where(
            directional_goal[:, None], side_mask, safe_mask
        )
        yaw_active = torch.abs(yaw_error) >= float(minimum_yaw_error_rad)
        approach_mask = torch.where(
            yaw_active[:, None], yaw_side_mask, translation_mask
        )
        env._affordance_yaw_compatible_contact_active = yaw_active.detach()
        env._affordance_yaw_compatible_selected_count = approach_mask.sum(
            dim=1
        ).detach()
        env._affordance_yaw_compatible_safe_count = safe_mask.sum(dim=1).detach()
        env._affordance_yaw_compatible_selection_fallback = (
            yaw_selection_fallback.detach()
        )
    else:
        approach_mask = torch.where(
            directional_goal[:, None], side_mask, safe_mask
        )
    per_hand_distance, per_hand_target_index = _chunked_min_right_per_left(
        end_effector_points, target_points, right_mask=approach_mask
    )
    distance, hand_index = per_hand_distance.min(dim=1)
    target_index = torch.gather(
        per_hand_target_index, 1, hand_index[:, None]
    ).squeeze(1)
    batch_index = torch.arange(env.num_envs, device=target_points.device)
    route_start = end_effector_points[batch_index, hand_index]
    route_end = target_points[batch_index, target_index]
    return distance, route_start, route_end, target_points, ~safe_mask


def _goal_conditioned_safe_side_distance(
    env: "ManagerBasedRLEnv",
    *,
    command_name: str,
    minimum_safe_score: float,
    side_band_m: float,
    minimum_goal_displacement_m: float,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    yaw_moment_weight: float = 0.0,
    yaw_activation_rad: float = 0.10,
    use_full_safe_region: bool = False,
    use_yaw_compatible_safe_region: bool = False,
    yaw_side_band_m: float = 0.010,
    yaw_compatible_selection_mode: str = "near_best",
    yaw_minimum_compatibility_m: float = 0.002,
    minimum_yaw_error_rad: float = 0.020,
) -> torch.Tensor:
    """Distance to the safe surface subset that can push toward the goal."""

    return _goal_conditioned_safe_side_route(
        env,
        command_name=command_name,
        minimum_safe_score=minimum_safe_score,
        side_band_m=side_band_m,
        minimum_goal_displacement_m=minimum_goal_displacement_m,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
        yaw_moment_weight=yaw_moment_weight,
        yaw_activation_rad=yaw_activation_rad,
        use_full_safe_region=use_full_safe_region,
        use_yaw_compatible_safe_region=use_yaw_compatible_safe_region,
        yaw_side_band_m=yaw_side_band_m,
        yaw_compatible_selection_mode=yaw_compatible_selection_mode,
        yaw_minimum_compatibility_m=yaw_minimum_compatibility_m,
        minimum_yaw_error_rad=minimum_yaw_error_rad,
    )[0]


def _goal_conditioned_semantic_corridor_potential(
    env: "ManagerBasedRLEnv",
    *,
    normalization_distance_m: float,
    contact_distance_m: float,
    corridor_contact_clearance_m: float,
    corridor_activation_clearance_m: float,
    corridor_body_radius_m: float,
    corridor_barrier_floor: float | None,
    obstruction_weight: float,
    corridor_samples: int,
    corridor_start_fraction: float,
    corridor_end_fraction: float,
    command_name: str,
    minimum_safe_score: float,
    side_band_m: float,
    minimum_goal_displacement_m: float,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Point-cloud navigation potential for a legal safe-contact corridor.

    The first component attracts one hand surface point to the goal-compatible
    safe set.  The second component measures whether the open segment toward
    that contact is occluded by any non-safe target point.  Optimizing the
    scalar potential can discover either side of an obstacle, but it never
    supplies a waypoint, route label, or extra actor observation.
    """

    if normalization_distance_m <= 0.0:
        raise ValueError("normalization distance must be positive")
    if obstruction_weight < 0.0:
        raise ValueError("obstruction weight must be non-negative")
    if corridor_body_radius_m < 0.0:
        raise ValueError("corridor body radius must be non-negative")
    key = (
        int(getattr(env, "common_step_counter", -1)),
        round(float(normalization_distance_m), 8),
        round(float(contact_distance_m), 8),
        round(float(corridor_contact_clearance_m), 8),
        round(float(corridor_activation_clearance_m), 8),
        round(float(corridor_body_radius_m), 8),
        None
        if corridor_barrier_floor is None
        else round(float(corridor_barrier_floor), 8),
        round(float(obstruction_weight), 8),
        int(corridor_samples),
        round(float(corridor_start_fraction), 8),
        round(float(corridor_end_fraction), 8),
        command_name,
        round(float(minimum_safe_score), 8),
        round(float(side_band_m), 8),
        round(float(minimum_goal_displacement_m), 8),
        _radius_key(safe_radius_m),
        _radius_key(protected_radius_m),
        target_cfg.name,
        obstacles_cfg.name,
        ee_frame_cfg.name,
    )
    cached = getattr(env, "_domino_semantic_corridor_potential_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    distance, route_start, route_end, target_points, forbidden_mask = (
        _goal_conditioned_safe_side_route(
            env,
            command_name=command_name,
            minimum_safe_score=minimum_safe_score,
            side_band_m=side_band_m,
            minimum_goal_displacement_m=minimum_goal_displacement_m,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
    )
    point_clearance = sampled_segment_minimum_clearance(
        route_start,
        route_end,
        target_points,
        obstacle_mask=forbidden_mask,
        num_samples=corridor_samples,
        start_fraction=corridor_start_fraction,
        end_fraction=corridor_end_fraction,
    )
    # The selected surface point is a route anchor, not a point robot.  Inflate
    # non-safe geometry by a conservative hand sweep radius so obstruction is
    # visible before some other finger surface reaches the C1 boundary.
    corridor_clearance = torch.clamp(
        point_clearance - float(corridor_body_radius_m), min=0.0
    )
    if corridor_barrier_floor is None:
        obstruction = normalized_clearance_violation(
            corridor_clearance,
            contact_distance=corridor_contact_clearance_m,
            activation_distance=corridor_activation_clearance_m,
        )
    else:
        obstruction = clearance_log_barrier(
            corridor_clearance,
            contact_distance=corridor_contact_clearance_m,
            activation_distance=corridor_activation_clearance_m,
            minimum_free_fraction=corridor_barrier_floor,
        )
    # Keep a linear far-field slope.  The bounded contact-distance helper is
    # appropriate as a cost but would make all reset states beyond 20 cm
    # identical and remove the reaching progress signal.
    distance_cost = torch.clamp(
        (distance - float(contact_distance_m)) / float(normalization_distance_m),
        min=0.0,
    )
    potential = distance_cost + float(obstruction_weight) * obstruction
    # Once contact is made, push-progress rewards own the transition.  This
    # prevents a residual corridor term from paying the policy to park there.
    potential = torch.where(
        distance > float(contact_distance_m), potential, torch.zeros_like(potential)
    )
    result = (potential, distance, corridor_clearance)
    env._domino_semantic_corridor_potential_cache = (key, result)
    return result


def goal_conditioned_semantic_corridor_penalty(
    env: "ManagerBasedRLEnv",
    normalization_distance_m: float = 0.20,
    contact_distance_m: float = 0.010,
    corridor_contact_clearance_m: float = 0.010,
    corridor_activation_clearance_m: float = 0.030,
    corridor_body_radius_m: float = 0.030,
    corridor_barrier_floor: float | None = None,
    obstruction_weight: float = 1.0,
    corridor_samples: int = 9,
    corridor_start_fraction: float = 0.10,
    corridor_end_fraction: float = 0.85,
    minimum_safe_score: float = 0.25,
    side_band_m: float = 0.015,
    minimum_goal_displacement_m: float = 0.020,
    command_name: str = "target_object_pose",
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Continuous cost for distance plus non-safe corridor obstruction."""

    return _goal_conditioned_semantic_corridor_potential(
        env,
        normalization_distance_m=normalization_distance_m,
        contact_distance_m=contact_distance_m,
        corridor_contact_clearance_m=corridor_contact_clearance_m,
        corridor_activation_clearance_m=corridor_activation_clearance_m,
        corridor_body_radius_m=corridor_body_radius_m,
        corridor_barrier_floor=corridor_barrier_floor,
        obstruction_weight=obstruction_weight,
        corridor_samples=corridor_samples,
        corridor_start_fraction=corridor_start_fraction,
        corridor_end_fraction=corridor_end_fraction,
        command_name=command_name,
        minimum_safe_score=minimum_safe_score,
        side_band_m=side_band_m,
        minimum_goal_displacement_m=minimum_goal_displacement_m,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )[0]


def _goal_conditioned_semantic_geodesic_potential(
    env: "ManagerBasedRLEnv",
    *,
    normalization_distance_m: float,
    contact_distance_m: float,
    route_contact_clearance_m: float,
    route_activation_clearance_m: float,
    route_body_radius_m: float,
    route_detour_margin_m: float,
    route_barrier_floor: float,
    obstruction_weight: float,
    route_candidates: int,
    route_segment_samples: int,
    route_obstacle_samples: int,
    command_name: str,
    minimum_safe_score: float,
    side_band_m: float,
    minimum_goal_displacement_m: float,
    safe_radius_m: float | None,
    protected_radius_m: float | None,
    target_cfg: SceneEntityCfg,
    obstacles_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    recover_illegal_route: bool = False,
    lexicographic_feasibility: bool = False,
    lexicographic_length_scale_m: float = 0.20,
    lexicographic_violation_scale_m: float = 0.01,
    yaw_moment_weight: float = 0.0,
    yaw_activation_rad: float = 0.10,
    gate_on_legal_safe_contact: bool = False,
    use_full_safe_region: bool = False,
    use_yaw_compatible_safe_region: bool = False,
    yaw_side_band_m: float = 0.010,
    yaw_compatible_selection_mode: str = "near_best",
    yaw_minimum_compatibility_m: float = 0.002,
    minimum_yaw_error_rad: float = 0.020,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Shortest legal semantic route to a goal-compatible safe surface.

    Unlike the straight-corridor ablation, this potential derives a compact
    visibility graph from the current non-safe target cloud.  It evaluates a
    direct edge plus ring-support detours and selects the lowest-cost *legal*
    route whenever one exists.  The route is used only as a scalar reward; no
    candidate, waypoint index, or privileged route label enters the actor.
    """

    if normalization_distance_m <= 0.0:
        raise ValueError("normalization distance must be positive")
    if obstruction_weight < 0.0:
        raise ValueError("obstruction weight must be non-negative")
    key = (
        int(getattr(env, "common_step_counter", -1)),
        round(float(normalization_distance_m), 8),
        round(float(contact_distance_m), 8),
        round(float(route_contact_clearance_m), 8),
        round(float(route_activation_clearance_m), 8),
        round(float(route_body_radius_m), 8),
        round(float(route_detour_margin_m), 8),
        round(float(route_barrier_floor), 8),
        round(float(obstruction_weight), 8),
        int(route_candidates),
        int(route_segment_samples),
        int(route_obstacle_samples),
        command_name,
        round(float(minimum_safe_score), 8),
        round(float(side_band_m), 8),
        round(float(minimum_goal_displacement_m), 8),
        _radius_key(safe_radius_m),
        _radius_key(protected_radius_m),
        target_cfg.name,
        obstacles_cfg.name,
        ee_frame_cfg.name,
        bool(recover_illegal_route),
        bool(lexicographic_feasibility),
        round(float(lexicographic_length_scale_m), 8),
        round(float(lexicographic_violation_scale_m), 8),
        round(float(yaw_moment_weight), 8),
        round(float(yaw_activation_rad), 8),
        bool(gate_on_legal_safe_contact),
        bool(use_full_safe_region),
        bool(use_yaw_compatible_safe_region),
        round(float(yaw_side_band_m), 8),
        yaw_compatible_selection_mode,
        round(float(yaw_minimum_compatibility_m), 8),
        round(float(minimum_yaw_error_rad), 8),
    )
    cached = getattr(env, "_domino_semantic_geodesic_potential_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    distance, route_start, route_end, target_points, forbidden_mask = (
        _goal_conditioned_safe_side_route(
            env,
            command_name=command_name,
            minimum_safe_score=minimum_safe_score,
            side_band_m=side_band_m,
            minimum_goal_displacement_m=minimum_goal_displacement_m,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
            yaw_moment_weight=yaw_moment_weight,
            yaw_activation_rad=yaw_activation_rad,
            use_full_safe_region=use_full_safe_region,
            use_yaw_compatible_safe_region=use_yaw_compatible_safe_region,
            yaw_side_band_m=yaw_side_band_m,
            yaw_compatible_selection_mode=yaw_compatible_selection_mode,
            yaw_minimum_compatibility_m=yaw_minimum_compatibility_m,
            minimum_yaw_error_rad=minimum_yaw_error_rad,
        )
    )
    route_length, route_clearance, first_edge_target = semantic_ring_route_geometry(
        route_start,
        route_end,
        target_points,
        obstacle_mask=forbidden_mask,
        body_radius=route_body_radius_m,
        contact_clearance=route_contact_clearance_m,
        detour_margin=route_detour_margin_m,
        num_candidates=route_candidates,
        num_segment_samples=route_segment_samples,
        obstacle_sample_count=route_obstacle_samples,
    )
    if lexicographic_feasibility:
        # Safety is a feasibility predicate, not another objective.  Every
        # legal route is therefore preferable to every illegal route; among
        # legal routes only contact-path length matters.  This prevents the
        # old distance-plus-barrier optimum from retreating merely to collect
        # clearance beyond the audited C1 margin.
        selected_potential, selected_index, has_legal_route = (
            lexicographic_route_potential(
                route_length,
                route_clearance,
                required_clearance=route_contact_clearance_m,
                length_scale=lexicographic_length_scale_m,
                violation_scale=lexicographic_violation_scale_m,
            )
        )
    else:
        route_barrier = clearance_log_barrier(
            route_clearance,
            contact_distance=route_contact_clearance_m,
            activation_distance=route_activation_clearance_m,
            minimum_free_fraction=route_barrier_floor,
        )
        route_potential = route_length / float(normalization_distance_m)
        route_potential += float(obstruction_weight) * route_barrier

        legal_route = route_clearance >= float(route_contact_clearance_m)
        has_legal_route = legal_route.any(dim=1)
        legal_score = route_potential.masked_fill(~legal_route, torch.inf)
        selected_legal = legal_score.argmin(dim=1)
        # This fallback is reached only after the hand has already entered an
        # inflated obstacle or if an annotation has no sampled free route.
        # Keep it finite for PPO, while never letting an illegal shortcut beat
        # an available legal detour.
        selected_fallback = route_potential.argmin(dim=1)
        selected_index = torch.where(
            has_legal_route, selected_legal, selected_fallback
        )
        selected_potential = torch.gather(
            route_potential, 1, selected_index[:, None]
        ).squeeze(1)
    selected_clearance = torch.gather(
        route_clearance, 1, selected_index[:, None]
    ).squeeze(1)
    selected_length = torch.gather(
        route_length, 1, selected_index[:, None]
    ).squeeze(1)
    batch_index = torch.arange(env.num_envs, device=route_start.device)
    first_edge = first_edge_target[batch_index, selected_index] - route_start
    selected_direction = first_edge / torch.clamp(
        torch.linalg.vector_norm(first_edge, dim=1, keepdim=True), min=1.0e-6
    )
    recovery_required = torch.zeros_like(has_legal_route)
    if recover_illegal_route:
        recovery_direction, _ = (
            semantic_clearance_recovery_direction(
                route_start,
                target_points,
                obstacle_mask=forbidden_mask,
                safety_radius=float(
                    route_body_radius_m + route_contact_clearance_m
                ),
            )
        )
        recovery_direction = semantic_tangential_recovery_direction(
            route_start,
            route_end,
            recovery_direction,
            route_clearance,
            first_edge_target,
            contact_clearance=route_contact_clearance_m,
        )
        # An empty legal route set must never fall back to an unsafe direct
        # edge, whether or not the current hand center has already crossed the
        # body inflation.  The outward field restores feasibility first.
        recovery_required = ~has_legal_route
        selected_direction = torch.where(
            recovery_required[:, None], recovery_direction, selected_direction
        )
    selected_potential = torch.where(
        distance > float(contact_distance_m),
        selected_potential,
        torch.zeros_like(selected_potential),
    )
    if gate_on_legal_safe_contact:
        legal_contact = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            0.25,
            64,
            0.005,
            False,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )["legal_safe_robot_contact"]
        selected_potential, distance = gate_navigation_at_legal_contact(
            selected_potential, distance, legal_contact
        )
    result = (
        selected_potential,
        distance,
        selected_clearance,
        (selected_index > 0) | recovery_required,
        route_clearance[:, 0],
        selected_length,
        selected_direction,
    )
    env._domino_semantic_geodesic_potential_cache = (key, result)
    return result


def goal_conditioned_semantic_geodesic_penalty(
    env: "ManagerBasedRLEnv",
    normalization_distance_m: float = 0.20,
    contact_distance_m: float = 0.010,
    route_contact_clearance_m: float = 0.010,
    route_activation_clearance_m: float = 0.030,
    route_body_radius_m: float = 0.030,
    route_detour_margin_m: float = 0.020,
    route_barrier_floor: float = 0.01,
    obstruction_weight: float = 1.0,
    route_candidates: int = 12,
    route_segment_samples: int = 7,
    route_obstacle_samples: int = 96,
    minimum_safe_score: float = 0.25,
    side_band_m: float = 0.015,
    minimum_goal_displacement_m: float = 0.020,
    command_name: str = "target_object_pose",
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    lexicographic_feasibility: bool = False,
    lexicographic_length_scale_m: float = 0.20,
    lexicographic_violation_scale_m: float = 0.01,
    yaw_moment_weight: float = 0.0,
    yaw_activation_rad: float = 0.10,
    gate_on_legal_safe_contact: bool = False,
    use_full_safe_region: bool = False,
    use_yaw_compatible_safe_region: bool = False,
    yaw_side_band_m: float = 0.010,
    yaw_compatible_selection_mode: str = "near_best",
    yaw_minimum_compatibility_m: float = 0.002,
    minimum_yaw_error_rad: float = 0.020,
) -> torch.Tensor:
    """Continuous cost for the shortest available semantic free-space route."""

    return _goal_conditioned_semantic_geodesic_potential(
        env,
        normalization_distance_m=normalization_distance_m,
        contact_distance_m=contact_distance_m,
        route_contact_clearance_m=route_contact_clearance_m,
        route_activation_clearance_m=route_activation_clearance_m,
        route_body_radius_m=route_body_radius_m,
        route_detour_margin_m=route_detour_margin_m,
        route_barrier_floor=route_barrier_floor,
        obstruction_weight=obstruction_weight,
        route_candidates=route_candidates,
        route_segment_samples=route_segment_samples,
        route_obstacle_samples=route_obstacle_samples,
        command_name=command_name,
        minimum_safe_score=minimum_safe_score,
        side_band_m=side_band_m,
        minimum_goal_displacement_m=minimum_goal_displacement_m,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
        lexicographic_feasibility=lexicographic_feasibility,
        lexicographic_length_scale_m=lexicographic_length_scale_m,
        lexicographic_violation_scale_m=lexicographic_violation_scale_m,
        yaw_moment_weight=yaw_moment_weight,
        yaw_activation_rad=yaw_activation_rad,
        gate_on_legal_safe_contact=gate_on_legal_safe_contact,
        use_full_safe_region=use_full_safe_region,
        use_yaw_compatible_safe_region=use_yaw_compatible_safe_region,
        yaw_side_band_m=yaw_side_band_m,
        yaw_compatible_selection_mode=yaw_compatible_selection_mode,
        yaw_minimum_compatibility_m=yaw_minimum_compatibility_m,
        minimum_yaw_error_rad=minimum_yaw_error_rad,
    )[0]


class goal_conditioned_semantic_geodesic_progress_reward(ManagerTermBase):
    """Signed progress along the point-cloud semantic free-space route."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_potential = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_potential: float = 0.05,
        normalization_distance_m: float = 0.20,
        contact_distance_m: float = 0.010,
        route_contact_clearance_m: float = 0.010,
        route_activation_clearance_m: float = 0.030,
        route_body_radius_m: float = 0.030,
        route_detour_margin_m: float = 0.020,
        route_barrier_floor: float = 0.01,
        obstruction_weight: float = 1.0,
        route_candidates: int = 12,
        route_segment_samples: int = 7,
        route_obstacle_samples: int = 96,
        minimum_safe_score: float = 0.25,
        side_band_m: float = 0.015,
        minimum_goal_displacement_m: float = 0.020,
        command_name: str = "target_object_pose",
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        potential, distance, *_ = _goal_conditioned_semantic_geodesic_potential(
            env,
            normalization_distance_m=normalization_distance_m,
            contact_distance_m=contact_distance_m,
            route_contact_clearance_m=route_contact_clearance_m,
            route_activation_clearance_m=route_activation_clearance_m,
            route_body_radius_m=route_body_radius_m,
            route_detour_margin_m=route_detour_margin_m,
            route_barrier_floor=route_barrier_floor,
            obstruction_weight=obstruction_weight,
            route_candidates=route_candidates,
            route_segment_samples=route_segment_samples,
            route_obstacle_samples=route_obstacle_samples,
            command_name=command_name,
            minimum_safe_score=minimum_safe_score,
            side_band_m=side_band_m,
            minimum_goal_displacement_m=minimum_goal_displacement_m,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        valid_previous = torch.isfinite(self._previous_potential)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_potential, nan=0.0),
            potential,
            normalization_distance=normalization_potential,
        )
        self._previous_potential = potential.detach()
        return torch.where(
            valid_previous & (distance > float(contact_distance_m)),
            progress,
            torch.zeros_like(progress),
        )


class goal_conditioned_semantic_vector_field_progress_reward(ManagerTermBase):
    """Reward hand displacement along the current semantic free-space field.

    The field is recomputed from the oracle semantic cloud every step.  Only
    the scalar alignment reward reaches PPO; neither the route selection nor
    its first-edge direction is appended to the actor observation.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_hand_center = torch.full(
            (env.num_envs, 3), torch.nan, device=env.device
        )
        self._previous_direction = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self._previous_detour = torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        )
        self._detour_committed = torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        )
        self._previous_direct_clearance = torch.full(
            (env.num_envs,), torch.inf, device=env.device
        )
        self._previous_distance = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        self._previous_potential = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        self._legal_contact_latched = torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_hand_center[env_ids] = torch.nan
        self._previous_direction[env_ids] = 0.0
        self._previous_detour[env_ids] = False
        self._detour_committed[env_ids] = False
        self._previous_direct_clearance[env_ids] = torch.inf
        self._previous_distance[env_ids] = torch.nan
        self._previous_potential[env_ids] = torch.nan
        self._legal_contact_latched[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_displacement_m: float = 0.010,
        direct_route_scale: float = 0.0,
        latch_detour_until_contact: bool = False,
        direct_route_activation_clearance_m: float | None = None,
        recover_illegal_route: bool = False,
        require_potential_descent: bool = False,
        potential_shaping_discount: float | None = None,
        potential_scale: float = 1.0,
        descent_gate_floor: float = 0.25,
        lexicographic_feasibility: bool = False,
        lexicographic_length_scale_m: float = 0.20,
        lexicographic_violation_scale_m: float = 0.01,
        yaw_moment_weight: float = 0.0,
        yaw_activation_rad: float = 0.10,
        gate_on_legal_safe_contact: bool = False,
        latch_after_legal_safe_contact: bool = False,
        use_full_safe_region: bool = False,
        use_yaw_compatible_safe_region: bool = False,
        yaw_side_band_m: float = 0.010,
        yaw_compatible_selection_mode: str = "near_best",
        yaw_minimum_compatibility_m: float = 0.002,
        minimum_yaw_error_rad: float = 0.020,
        normalization_distance_m: float = 0.20,
        contact_distance_m: float = 0.010,
        route_contact_clearance_m: float = 0.010,
        route_activation_clearance_m: float = 0.030,
        route_body_radius_m: float = 0.030,
        route_detour_margin_m: float = 0.020,
        route_barrier_floor: float = 0.01,
        obstruction_weight: float = 1.0,
        route_candidates: int = 12,
        route_segment_samples: int = 7,
        route_obstacle_samples: int = 96,
        minimum_safe_score: float = 0.25,
        side_band_m: float = 0.015,
        minimum_goal_displacement_m: float = 0.020,
        command_name: str = "target_object_pose",
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if normalization_displacement_m <= 0.0:
            raise ValueError("normalization displacement must be positive")
        if not 0.0 <= direct_route_scale <= 1.0:
            raise ValueError("direct route scale must be in [0, 1]")
        (
            potential,
            distance,
            _,
            used_detour,
            direct_clearance,
            _,
            selected_direction,
        ) = _goal_conditioned_semantic_geodesic_potential(
            env,
            normalization_distance_m=normalization_distance_m,
            contact_distance_m=contact_distance_m,
            route_contact_clearance_m=route_contact_clearance_m,
            route_activation_clearance_m=route_activation_clearance_m,
            route_body_radius_m=route_body_radius_m,
            route_detour_margin_m=route_detour_margin_m,
            route_barrier_floor=route_barrier_floor,
            obstruction_weight=obstruction_weight,
            route_candidates=route_candidates,
            route_segment_samples=route_segment_samples,
            route_obstacle_samples=route_obstacle_samples,
            command_name=command_name,
            minimum_safe_score=minimum_safe_score,
            side_band_m=side_band_m,
            minimum_goal_displacement_m=minimum_goal_displacement_m,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
            recover_illegal_route=recover_illegal_route,
            lexicographic_feasibility=lexicographic_feasibility,
            lexicographic_length_scale_m=lexicographic_length_scale_m,
            lexicographic_violation_scale_m=lexicographic_violation_scale_m,
            yaw_moment_weight=yaw_moment_weight,
            yaw_activation_rad=yaw_activation_rad,
            gate_on_legal_safe_contact=gate_on_legal_safe_contact,
            use_full_safe_region=use_full_safe_region,
            use_yaw_compatible_safe_region=use_yaw_compatible_safe_region,
            yaw_side_band_m=yaw_side_band_m,
            yaw_compatible_selection_mode=yaw_compatible_selection_mode,
            yaw_minimum_compatibility_m=yaw_minimum_compatibility_m,
            minimum_yaw_error_rad=minimum_yaw_error_rad,
        )
        hand_center = get_end_effector_pointcloud_in_env_frame(
            env, ee_frame_cfg
        ).mean(dim=1)
        valid_previous = torch.isfinite(self._previous_hand_center).all(dim=1)
        displacement = hand_center - torch.nan_to_num(self._previous_hand_center)
        alignment = torch.sum(displacement * self._previous_direction, dim=1)
        alignment = torch.clamp(
            alignment / float(normalization_displacement_m), min=-1.0, max=1.0
        )
        if potential_shaping_discount is not None:
            route_alignment = discounted_potential_shaping(
                torch.nan_to_num(self._previous_potential, nan=0.0),
                potential,
                discount_factor=potential_shaping_discount,
            )
            valid_previous &= torch.isfinite(self._previous_potential)
        elif require_potential_descent:
            displacement_norm = torch.linalg.vector_norm(displacement, dim=1)
            field_alignment = torch.sum(
                displacement * self._previous_direction, dim=1
            ) / torch.clamp(displacement_norm, min=1.0e-6)
            route_alignment = potential_consistent_progress(
                torch.nan_to_num(self._previous_potential, nan=0.0),
                potential,
                field_alignment,
                potential_scale=potential_scale,
                descent_gate_floor=descent_gate_floor,
            )
            valid_previous &= torch.isfinite(self._previous_potential)
        elif direct_route_activation_clearance_m is None:
            reward_detour = (
                self._detour_committed
                if latch_detour_until_contact
                else self._previous_detour
            )
            route_alignment = route_conditioned_alignment(
                alignment,
                reward_detour,
                direct_route_scale=direct_route_scale,
            )
        else:
            route_scale = clearance_conditioned_route_scale(
                self._previous_direct_clearance,
                contact_clearance=route_contact_clearance_m,
                activation_clearance=direct_route_activation_clearance_m,
                direct_route_scale=direct_route_scale,
            )
            route_alignment = alignment * route_scale
        active = valid_previous & (
            self._previous_distance > float(contact_distance_m)
        )
        if latch_after_legal_safe_contact:
            # Keep the transition that first reaches legal contact, then turn
            # the pre-contact field off for the remainder of the episode.
            # Re-enabling it whenever contact is momentarily lost pulled v54
            # back to the same nearest handle point and suppressed the lateral
            # contact changes required for the opposite yaw sign.
            previously_latched = self._legal_contact_latched.clone()
            legal_contact = _contact_term_state(
                env,
                contact_distance_m,
                minimum_safe_score,
                0.25,
                64,
                0.005,
                False,
                safe_radius_m,
                protected_radius_m,
                target_cfg,
                obstacles_cfg,
                ee_frame_cfg,
            )["legal_safe_robot_contact"]
            self._legal_contact_latched |= legal_contact.detach()
            active &= ~previously_latched
            env._affordance_navigation_contact_latched = (
                self._legal_contact_latched.clone()
            )
        self._previous_hand_center = hand_center.detach()
        self._previous_direction = selected_direction.detach()
        self._previous_detour = used_detour.detach()
        self._detour_committed = update_route_detour_commitment(
            self._detour_committed, used_detour.detach()
        )
        self._previous_direct_clearance = direct_clearance.detach()
        self._previous_distance = distance.detach()
        self._previous_potential = potential.detach()
        return torch.where(active, route_alignment, torch.zeros_like(alignment))


class goal_conditioned_semantic_corridor_progress_reward(ManagerTermBase):
    """Signed progress on the semantic corridor navigation potential."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_potential = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_potential: float = 0.05,
        normalization_distance_m: float = 0.20,
        contact_distance_m: float = 0.010,
        corridor_contact_clearance_m: float = 0.010,
        corridor_activation_clearance_m: float = 0.030,
        corridor_body_radius_m: float = 0.030,
        corridor_barrier_floor: float | None = None,
        obstruction_weight: float = 1.0,
        corridor_samples: int = 9,
        corridor_start_fraction: float = 0.10,
        corridor_end_fraction: float = 0.85,
        minimum_safe_score: float = 0.25,
        side_band_m: float = 0.015,
        minimum_goal_displacement_m: float = 0.020,
        command_name: str = "target_object_pose",
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        potential, distance, _ = _goal_conditioned_semantic_corridor_potential(
            env,
            normalization_distance_m=normalization_distance_m,
            contact_distance_m=contact_distance_m,
            corridor_contact_clearance_m=corridor_contact_clearance_m,
            corridor_activation_clearance_m=corridor_activation_clearance_m,
            corridor_body_radius_m=corridor_body_radius_m,
            corridor_barrier_floor=corridor_barrier_floor,
            obstruction_weight=obstruction_weight,
            corridor_samples=corridor_samples,
            corridor_start_fraction=corridor_start_fraction,
            corridor_end_fraction=corridor_end_fraction,
            command_name=command_name,
            minimum_safe_score=minimum_safe_score,
            side_band_m=side_band_m,
            minimum_goal_displacement_m=minimum_goal_displacement_m,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        valid_previous = torch.isfinite(self._previous_potential)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_potential, nan=0.0),
            potential,
            normalization_distance=normalization_potential,
        )
        self._previous_potential = potential.detach()
        return torch.where(
            valid_previous & (distance > float(contact_distance_m)),
            progress,
            torch.zeros_like(progress),
        )


def goal_conditioned_safe_region_distance_penalty(
    env: "ManagerBasedRLEnv",
    normalization_distance_m: float = 0.20,
    contact_distance_m: float = 0.010,
    minimum_safe_score: float = 0.25,
    side_band_m: float = 0.015,
    minimum_goal_displacement_m: float = 0.020,
    command_name: str = "target_object_pose",
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Continuous approach cost for a goal-compatible subset of safe points."""

    distance = _goal_conditioned_safe_side_distance(
        env,
        command_name=command_name,
        minimum_safe_score=minimum_safe_score,
        side_band_m=side_band_m,
        minimum_goal_displacement_m=minimum_goal_displacement_m,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    return normalized_contact_distance_excess(
        distance,
        contact_distance=contact_distance_m,
        normalization_distance=normalization_distance_m,
    )


class goal_conditioned_safe_region_progress_reward(ManagerTermBase):
    """Signed progress toward the trailing-side safe set before contact."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_distance = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_distance[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_distance_m: float = 0.020,
        contact_distance_m: float = 0.010,
        minimum_safe_score: float = 0.25,
        side_band_m: float = 0.015,
        minimum_goal_displacement_m: float = 0.020,
        command_name: str = "target_object_pose",
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        current_distance = _goal_conditioned_safe_side_distance(
            env,
            command_name=command_name,
            minimum_safe_score=minimum_safe_score,
            side_band_m=side_band_m,
            minimum_goal_displacement_m=minimum_goal_displacement_m,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        valid_previous = torch.isfinite(self._previous_distance)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_distance, nan=0.0),
            current_distance,
            normalization_distance=normalization_distance_m,
        )
        self._previous_distance = current_distance.detach()
        return torch.where(
            valid_previous & (current_distance > float(contact_distance_m)),
            progress,
            torch.zeros_like(progress),
        )


class first_safe_region_contact_reward(ManagerTermBase):
    """Issue a one-time bonus on the first legal safe contact per episode."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._has_contacted = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._has_contacted[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        physical_contact_force_threshold_n: float = 0.5,
        robot_target_sensor_name: ContactSensorNames = None,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        safe_contact = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
            physical_contact_force_threshold_n=(
                physical_contact_force_threshold_n
            ),
            robot_target_sensor_name=robot_target_sensor_name,
        )["legal_safe_robot_contact"]
        first_contact = safe_contact & ~self._has_contacted
        self._has_contacted |= safe_contact
        return first_contact.float()


class safe_region_distance_progress_reward(ManagerTermBase):
    """Signed per-step progress toward safe points, with reset-safe state."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_distance = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_distance[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_distance_m: float = 0.02,
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        current_distance = state["minimum_safe_distance"]
        valid_previous = torch.isfinite(self._previous_distance)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_distance, nan=0.0),
            current_distance,
            normalization_distance=normalization_distance_m,
        )
        self._previous_distance = current_distance.detach()
        return torch.where(
            valid_previous & ~state["safe_robot_contact"],
            progress,
            torch.zeros_like(progress),
        )


class initial_relative_dapl_pose_score_reward(ManagerTermBase):
    """Persistent DAPL pose improvement with zero reset-pose payoff.

    The reference is one scalar score captured on the first reward evaluation
    after each reset.  It is neither a waypoint nor a phase flag: subtracting
    it only zero-centers the current-state DAPL score by an episode-constant
    baseline.  A stationary hammer therefore earns exactly zero even after
    the hand enters the safe-distance gate, while improvements remain
    positive on every subsequent step and regressions remain negative.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._reference_score = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._reference_score[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "target_object_pose",
        safe_ee_distance_threshold: float = 0.10,
        coarse_standard_deviation: float = 0.6,
        fine_standard_deviation: float = 0.3,
        coarse_weight: float = 5.0,
        fine_weight: float = 16.0,
        positive_only: bool = False,
        regression_scale: float = 1.0,
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if safe_ee_distance_threshold <= 0.0:
            raise ValueError("safe_ee_distance_threshold must be positive")
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        # A negative coarse weight is an internal profile selector rather than
        # another reward or state variable. Keeping the existing call
        # signature avoids OmegaConf recursively expanding a second stateful
        # reward schema; the DAPL branch still validates non-negative weights.
        joint_error_geometry = coarse_weight < 0.0
        if joint_error_geometry:
            current_score = affordance_joint_pose_error(
                env,
                planar_position_threshold=0.02,
                height_threshold=0.01,
                rotation_threshold=0.10,
                smooth_max_temperature=0.25,
                command_name=command_name,
                target_cfg=target_cfg,
            )
        else:
            planar, height, rotation = _affordance_goal_errors(
                env, command_name, target_cfg
            )
            position_distance = torch.sqrt(
                torch.square(planar) + torch.square(height)
            )
            current_score = dapl_multiscale_pose_score(
                position_distance,
                rotation,
                coarse_standard_deviation=coarse_standard_deviation,
                fine_standard_deviation=fine_standard_deviation,
                coarse_weight=coarse_weight,
                fine_weight=fine_weight,
            )
        missing_reference = ~torch.isfinite(self._reference_score)
        self._reference_score = torch.where(
            missing_reference,
            current_score.detach(),
            self._reference_score,
        )
        near_safe_region = (
            state["minimum_safe_distance"] < safe_ee_distance_threshold
        )
        if joint_error_geometry:
            if positive_only:
                relative_score = positive_reference_relative_error_improvement(
                    self._reference_score,
                    current_score,
                    reference_error_floor=1.0,
                )
            else:
                relative_score = signed_reference_relative_error_improvement(
                    self._reference_score,
                    current_score,
                    reference_error_floor=1.0,
                    regression_scale=regression_scale,
                )
        elif positive_only:
            relative_score = positive_reference_relative_score(
                self._reference_score, current_score
            )
        else:
            relative_score = current_score - self._reference_score
        return torch.where(
            near_safe_region,
            relative_score,
            torch.zeros_like(relative_score),
        )


def forbidden_region_contact_penalty(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    physical_contact_force_threshold_n: float = 0.5,
    robot_target_sensor_name: ContactSensorNames = None,
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
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        robot_target_sensor_name=robot_target_sensor_name,
    )["forbidden_robot_contact"].float()


def robot_forbidden_region_clearance_penalty(
    env: "ManagerBasedRLEnv",
    activation_distance_m: float = 0.02,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    physical_contact_force_threshold_n: float = 0.5,
    robot_target_sensor_name: ContactSensorNames = None,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense C1 cost aligned with the complete non-safe contact predicate.

    C1 forbids contact with both neutral and protected target points.  The
    earlier protected-only hinge left the neutral transition band unshaped.
    This cost uses the same safe-score partition as hard C1 and ramps from
    zero at ``activation_distance_m`` to one at the audited contact boundary.
    """

    if activation_distance_m <= contact_distance_m:
        raise ValueError(
            "activation_distance_m must exceed contact_distance_m"
        )
    state = _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        robot_target_sensor_name=robot_target_sensor_name,
    )
    distance = state["minimum_robot_forbidden_distance"]
    return normalized_clearance_violation(
        distance,
        contact_distance=float(contact_distance_m),
        activation_distance=float(activation_distance_m),
    )


def protected_region_collision_penalty(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    physical_contact_force_threshold_n: float = 0.5,
    require_physical_protected_contact: bool = False,
    target_obstacle_sensor_name: ContactSensorNames = None,
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
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        require_physical_protected_contact=require_physical_protected_contact,
        target_obstacle_sensor_name=target_obstacle_sensor_name,
    )["protected_obstacle_collision"].float()


def protected_region_clearance_penalty(
    env: "ManagerBasedRLEnv",
    activation_distance_m: float = 0.05,
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Continuous cost as protected target points approach clutter."""

    if activation_distance_m <= protected_clearance_m:
        raise ValueError("activation distance must exceed protected clearance")
    clearance = _contact_term_state(
        env,
        contact_distance_m,
        minimum_safe_score,
        minimum_protected_score,
        protected_point_count,
        protected_clearance_m,
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
    )["protected_clearance"]
    scale = activation_distance_m - protected_clearance_m
    return torch.clamp(
        (activation_distance_m - clearance) / scale, min=0.0, max=1.0
    )


class protected_region_clearance_progress_reward(ManagerTermBase):
    """Discounted potential progress toward C2-safe protected clearance.

    The existing absolute clearance cost says which states are unsafe, but a
    straight goal-distance potential can still penalize the first step of a
    necessary detour.  This term rewards transitions that increase protected
    clearance using the standard ``gamma * phi(s') - phi(s)`` construction.
    It introduces no waypoint, phase, action target, or privileged actor input.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_score = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_score[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        activation_distance_m: float = 0.05,
        contact_distance_m: float = 0.008,
        potential_discount: float = 0.99,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if activation_distance_m <= protected_clearance_m:
            raise ValueError("activation distance must exceed protected clearance")
        clearance = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )["protected_clearance"]
        clearance_score = 1.0 - normalized_clearance_violation(
            clearance,
            contact_distance=protected_clearance_m,
            activation_distance=activation_distance_m,
        )
        valid_previous = torch.isfinite(self._previous_score)
        shaping = discounted_score_potential_shaping(
            torch.nan_to_num(self._previous_score, nan=0.0),
            clearance_score,
            discount_factor=potential_discount,
        )
        self._previous_score = clearance_score.detach()
        return torch.where(valid_previous, shaping, torch.zeros_like(shaping))


class protected_region_geodesic_progress_reward(ManagerTermBase):
    """Reward C2-safe goal progress along a full-body semantic geodesic.

    At reset, the protected point whose current-to-goal sweep is closest to the
    isolated blocker identifies the collision-relevant route.  A conservative
    full-protected-cloud sweep then selects a feasible left/right homotopy.  At
    every later step the scalar shortest-path length is recomputed from the
    current point cloud; no route point, side label, or waypoint is added to
    the actor observation.  The actor already receives the same protected
    cloud, obstacle cloud, and relative goal needed to recover this signal.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._critical_point_index = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._detour_side = torch.zeros(
            (env.num_envs,), dtype=torch.int8, device=env.device
        )
        self._previous_route_length = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        self._previous_critical_point = torch.full(
            (env.num_envs, 3), torch.nan, device=env.device
        )
        self._lateral_axis = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self._previous_direct_clearance = torch.full(
            (env.num_envs,), torch.inf, device=env.device
        )
        self._detour_feasible = torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._critical_point_index[env_ids] = -1
        self._detour_side[env_ids] = 0
        self._previous_route_length[env_ids] = torch.nan
        self._previous_critical_point[env_ids] = torch.nan
        self._lateral_axis[env_ids] = 0.0
        self._previous_direct_clearance[env_ids] = torch.inf
        self._detour_feasible[env_ids] = False

    @staticmethod
    def _goal_points(
        env: "ManagerBasedRLEnv",
        current_points: torch.Tensor,
        command_name: str,
        target_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        target: RigidObject = env.scene[target_cfg.name]
        current_position = target.data.root_pos_w[:, :3] - env.scene.env_origins
        current_rotation = matrix_from_quat(target.data.root_quat_w)
        local_points = torch.bmm(
            current_points - current_position[:, None, :], current_rotation
        )
        goal = env.command_manager.get_command(command_name)
        goal_rotation = matrix_from_quat(goal[:, 3:7])
        return torch.bmm(local_points, goal_rotation.transpose(1, 2)) + goal[
            :, None, :3
        ]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        progress_mode: str = "route_length",
        normalization_distance_m: float = 0.01,
        route_contact_clearance_m: float = 0.005,
        route_detour_margin_m: float = 0.020,
        route_candidates: int = 12,
        route_segment_samples: int = 9,
        route_obstacle_samples: int = 128,
        critical_sweep_samples: int = 5,
        body_sweep_samples: int = 5,
        body_aabb_clearance_m: float = 0.0005,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if normalization_distance_m <= 0.0:
            raise ValueError("normalization_distance_m must be positive")
        if progress_mode not in {"route_length", "blocked_lateral_escape"}:
            raise ValueError(
                "progress_mode must be 'route_length' or "
                "'blocked_lateral_escape'"
            )
        if route_contact_clearance_m < 0.0 or body_aabb_clearance_m < 0.0:
            raise ValueError("route and body clearances must be non-negative")

        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        geometry = _robot_target_affordance_geometry(
            env,
            minimum_safe_score=minimum_safe_score,
            minimum_protected_score=minimum_protected_score,
            safe_radius_m=safe_radius_m,
            protected_radius_m=protected_radius_m,
            target_cfg=target_cfg,
            obstacles_cfg=obstacles_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        current_points = geometry["target_points"]
        goal_points = self._goal_points(
            env, current_points, command_name, target_cfg
        )
        protected_mask = geometry["protected_mask"]
        obstacle_points = geometry["obstacle_points"]

        uninitialized = self._critical_point_index < 0
        if torch.any(uninitialized):
            env_ids = torch.nonzero(uninitialized, as_tuple=False).squeeze(1)
            critical_index, _ = goal_swept_semantic_point_index(
                current_points[env_ids],
                goal_points[env_ids],
                obstacle_points[env_ids],
                point_mask=protected_mask[env_ids],
                num_samples=critical_sweep_samples,
            )
            self._critical_point_index[env_ids] = critical_index

        batch_index = torch.arange(env.num_envs, device=current_points.device)
        critical_start = current_points[
            batch_index, self._critical_point_index
        ]
        critical_end = goal_points[batch_index, self._critical_point_index]
        # The legacy route-length ablation recomputes every ring candidate at
        # every step.  The lateral escape mode needs the ring only when a new
        # episode selects a homotopy; later steps evaluate only the direct
        # protected-point sweep.  Besides removing the unsafe forward
        # component diagnosed in v70, this avoids its roughly 10x rollout
        # slowdown.
        route_length = route_clearance = first_edge_target = None
        if progress_mode == "route_length" or torch.any(uninitialized):
            route_length, route_clearance, first_edge_target = (
                semantic_ring_route_geometry(
                    critical_start,
                    critical_end,
                    obstacle_points,
                    body_radius=0.0,
                    contact_clearance=route_contact_clearance_m,
                    detour_margin=route_detour_margin_m,
                    num_candidates=route_candidates,
                    num_segment_samples=route_segment_samples,
                    obstacle_sample_count=route_obstacle_samples,
                )
            )

        if torch.any(uninitialized):
            assert (
                route_length is not None
                and route_clearance is not None
                and first_edge_target is not None
            )
            env_ids = torch.nonzero(uninitialized, as_tuple=False).squeeze(1)
            body_clearance = rigid_body_ring_route_aabb_clearance(
                current_points[env_ids],
                goal_points[env_ids],
                obstacle_points[env_ids],
                first_edge_target[env_ids],
                critical_start[env_ids],
                critical_end[env_ids],
                point_mask=protected_mask[env_ids],
                num_segment_samples=body_sweep_samples,
            )
            legal_body_route = (
                route_clearance[env_ids] >= float(route_contact_clearance_m)
            ) & (body_clearance >= float(body_aabb_clearance_m))
            legal_length = route_length[env_ids].masked_fill(
                ~legal_body_route, torch.inf
            )
            has_legal_body_route = legal_body_route.any(dim=1)
            selected_body_route = legal_length.argmin(dim=1)
            # If the conservative AABB screen rejects every candidate, use the
            # route with maximum body clearance, then shortest length for ties.
            best_body_clearance = body_clearance.amax(dim=1, keepdim=True)
            maximum_clearance = body_clearance == best_body_clearance
            fallback_length = route_length[env_ids].masked_fill(
                ~maximum_clearance, torch.inf
            )
            selected_body_route = torch.where(
                has_legal_body_route,
                selected_body_route,
                fallback_length.argmin(dim=1),
            )
            selected_first_edge = first_edge_target[
                env_ids, selected_body_route
            ] - critical_start[env_ids]
            direct = critical_end[env_ids] - critical_start[env_ids]
            cross_z = (
                direct[:, 0] * selected_first_edge[:, 1]
                - direct[:, 1] * selected_first_edge[:, 0]
            )
            selected_side = torch.sign(cross_z).to(torch.int8)
            selected_side = torch.where(
                selected_body_route == 0,
                torch.zeros_like(selected_side),
                selected_side,
            )
            self._detour_side[env_ids] = selected_side
            self._detour_feasible[env_ids] = has_legal_body_route
            self._lateral_axis[env_ids] = planar_lateral_escape_axis(
                critical_start[env_ids],
                critical_end[env_ids],
                selected_side,
            )

        if progress_mode == "blocked_lateral_escape":
            if route_clearance is not None:
                direct_clearance = route_clearance[:, 0]
            else:
                direct_clearance = sampled_segment_minimum_clearance(
                    critical_start,
                    critical_end,
                    obstacle_points,
                    num_samples=route_segment_samples,
                    start_fraction=0.0,
                    end_fraction=0.85,
                )
            valid_previous = torch.isfinite(
                self._previous_critical_point
            ).all(dim=1)
            lateral_progress = normalized_directional_displacement(
                torch.nan_to_num(self._previous_critical_point),
                critical_start,
                self._lateral_axis,
                normalization_distance=normalization_distance_m,
            )
            active = (
                valid_previous
                & self._detour_feasible
                & (self._detour_side != 0)
                & (
                    self._previous_direct_clearance
                    < float(route_contact_clearance_m)
                )
                & state["legal_safe_robot_contact"]
                & ~state["protected_obstacle_collision"]
            )
            self._previous_critical_point = critical_start.detach()
            self._previous_direct_clearance = direct_clearance.detach()
            env._protected_geodesic_detour_side = self._detour_side.clone()
            env._protected_geodesic_direct_clearance = (
                direct_clearance.detach()
            )
            env._protected_geodesic_route_length = torch.full_like(
                direct_clearance, torch.nan
            )
            env._protected_lateral_escape_axis = self._lateral_axis.clone()
            env._protected_lateral_escape_progress = lateral_progress.detach()
            return torch.where(
                active, lateral_progress, torch.zeros_like(lateral_progress)
            )

        assert (
            route_length is not None
            and route_clearance is not None
            and first_edge_target is not None
        )
        route_legal = route_clearance >= float(route_contact_clearance_m)
        first_edge = first_edge_target - critical_start[:, None, :]
        direct = critical_end - critical_start
        route_cross_z = (
            direct[:, None, 0] * first_edge[..., 1]
            - direct[:, None, 1] * first_edge[..., 0]
        )
        side_match = (
            route_cross_z * self._detour_side.to(route_cross_z.dtype)[:, None]
        ) > 0.0
        route_indices = torch.arange(
            route_length.shape[1], device=route_length.device
        )[None]
        direct_route = route_indices == 0
        committed = self._detour_side != 0
        side_allowed = torch.where(
            committed[:, None], direct_route | side_match, torch.ones_like(side_match)
        )
        allowed = route_legal & side_allowed
        has_allowed = allowed.any(dim=1)
        selected_route = route_length.masked_fill(~allowed, torch.inf).argmin(dim=1)
        _, fallback_route, _ = lexicographic_route_potential(
            route_length,
            route_clearance,
            required_clearance=route_contact_clearance_m,
            length_scale=0.10,
            violation_scale=max(route_contact_clearance_m, 1.0e-4),
        )
        selected_route = torch.where(has_allowed, selected_route, fallback_route)
        current_route_length = route_length[batch_index, selected_route]

        valid_previous = torch.isfinite(self._previous_route_length)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_route_length, nan=0.0),
            current_route_length,
            normalization_distance=normalization_distance_m,
        )
        active = valid_previous & state["legal_safe_robot_contact"] & has_allowed
        self._previous_route_length = current_route_length.detach()
        env._protected_geodesic_detour_side = self._detour_side.clone()
        env._protected_geodesic_direct_clearance = route_clearance[:, 0].detach()
        env._protected_geodesic_route_length = current_route_length.detach()
        return torch.where(active, progress, torch.zeros_like(progress))


def forbidden_region_contact(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    physical_contact_force_threshold_n: float = 0.5,
    robot_target_sensor_name: ContactSensorNames = None,
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
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        robot_target_sensor_name=robot_target_sensor_name,
    )["forbidden_robot_contact"]


def protected_region_collision(
    env: "ManagerBasedRLEnv",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    evaluate_protected: bool = True,
    physical_contact_force_threshold_n: float = 0.5,
    require_physical_protected_contact: bool = False,
    target_obstacle_sensor_name: ContactSensorNames = None,
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
        evaluate_protected,
        safe_radius_m,
        protected_radius_m,
        target_cfg,
        obstacles_cfg,
        ee_frame_cfg,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        require_physical_protected_contact=require_physical_protected_contact,
        target_obstacle_sensor_name=target_obstacle_sensor_name,
    )["protected_obstacle_collision"]


def robot_obstacle_collision_penalty(
    env: "ManagerBasedRLEnv",
    robot_obstacle_clearance_m: float = 0.005,
    robot_link_proxy_radius_m: float = 0.045,
    robot_link_samples_per_segment: int = 5,
    physical_contact_force_threshold_n: float = 0.5,
    robot_obstacle_sensor_name: ContactSensorNames = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive C3 collision indicator for a negative reward weight."""

    return domino_affordance_contact_state(
        env,
        robot_obstacle_clearance_m=robot_obstacle_clearance_m,
        robot_link_proxy_radius_m=robot_link_proxy_radius_m,
        robot_link_samples_per_segment=robot_link_samples_per_segment,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        evaluate_protected=False,
        evaluate_robot_obstacle=True,
        robot_obstacle_sensor_name=robot_obstacle_sensor_name,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
        robot_cfg=robot_cfg,
    )["robot_obstacle_collision"].float()


def robot_obstacle_clearance_penalty(
    env: "ManagerBasedRLEnv",
    activation_distance_m: float = 0.05,
    robot_obstacle_clearance_m: float = 0.005,
    robot_link_proxy_radius_m: float = 0.045,
    robot_link_samples_per_segment: int = 5,
    physical_contact_force_threshold_n: float = 0.5,
    robot_obstacle_sensor_name: ContactSensorNames = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous hand/clutter clearance cost used before hard C3 contact."""

    if activation_distance_m <= robot_obstacle_clearance_m:
        raise ValueError("activation distance must exceed robot-obstacle clearance")
    clearance = domino_affordance_contact_state(
        env,
        robot_obstacle_clearance_m=robot_obstacle_clearance_m,
        robot_link_proxy_radius_m=robot_link_proxy_radius_m,
        robot_link_samples_per_segment=robot_link_samples_per_segment,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        evaluate_protected=False,
        evaluate_robot_obstacle=True,
        robot_obstacle_sensor_name=robot_obstacle_sensor_name,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
        robot_cfg=robot_cfg,
    )["robot_obstacle_clearance"]
    scale = activation_distance_m - robot_obstacle_clearance_m
    return torch.clamp(
        (activation_distance_m - clearance) / scale, min=0.0, max=1.0
    )


def robot_obstacle_collision(
    env: "ManagerBasedRLEnv",
    robot_obstacle_clearance_m: float = 0.005,
    robot_link_proxy_radius_m: float = 0.045,
    robot_link_samples_per_segment: int = 5,
    physical_contact_force_threshold_n: float = 0.5,
    robot_obstacle_sensor_name: ContactSensorNames = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hard C3 termination using filtered whole-robot PhysX contacts."""

    return domino_affordance_contact_state(
        env,
        robot_obstacle_clearance_m=robot_obstacle_clearance_m,
        robot_link_proxy_radius_m=robot_link_proxy_radius_m,
        robot_link_samples_per_segment=robot_link_samples_per_segment,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        evaluate_protected=False,
        evaluate_robot_obstacle=True,
        robot_obstacle_sensor_name=robot_obstacle_sensor_name,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
        robot_cfg=robot_cfg,
    )["robot_obstacle_collision"]


def _affordance_goal_errors(
    env: "ManagerBasedRLEnv",
    command_name: str,
    target_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return planar, height, and full SO(3) target-pose errors."""

    target: RigidObject = env.scene[target_cfg.name]
    goal = env.command_manager.get_command(command_name)
    position_env = target.data.root_pos_w[:, :3] - env.scene.env_origins
    delta = goal[:, :3] - position_env
    planar_distance = torch.linalg.vector_norm(delta[:, :2], dim=1)
    height_distance = torch.abs(delta[:, 2])
    dot = torch.sum(target.data.root_quat_w * goal[:, 3:7], dim=1)
    rotation_distance = 2.0 * torch.acos(torch.clamp(torch.abs(dot), max=1.0))
    return planar_distance, height_distance, rotation_distance


def affordance_signed_yaw_goal_error(
    env: "ManagerBasedRLEnv",
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Return signed world-Z yaw still required to reach the goal pose.

    The complete SO(3) geodesic remains the success predicate.  This signed
    planar diagnostic is intentionally separate: it distinguishes undershoot
    from overshoot and prevents opposite yaw goals from cancelling in an
    aggregate mean during bidirectional-yaw teacher evaluation.
    """

    target: RigidObject = env.scene[target_cfg.name]
    goal = env.command_manager.get_command(command_name)
    relative_quat = quat_mul(
        goal[:, 3:7], quat_conjugate(target.data.root_quat_w)
    )
    relative_rotation = matrix_from_quat(relative_quat)
    return torch.atan2(
        relative_rotation[:, 1, 0], relative_rotation[:, 0, 0]
    )


def affordance_joint_pose_error(
    env: "ManagerBasedRLEnv",
    planar_position_threshold: float = 0.02,
    height_threshold: float = 0.01,
    rotation_threshold: float = 0.10,
    smooth_max_temperature: float = 0.25,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Joint XY/Z/SO(3) error expressed in success-threshold units."""

    planar, height, rotation = _affordance_goal_errors(
        env, command_name, target_cfg
    )
    return smooth_max_normalized_pose_error(
        planar,
        height,
        rotation,
        planar_threshold=planar_position_threshold,
        height_threshold=height_threshold,
        rotation_threshold=rotation_threshold,
        temperature=smooth_max_temperature,
    )


def affordance_joint_pose_reward(
    env: "ManagerBasedRLEnv",
    planar_position_threshold: float = 0.02,
    height_threshold: float = 0.01,
    rotation_threshold: float = 0.10,
    smooth_max_temperature: float = 0.25,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Dense reward that is high only when all pose components are good."""

    error = affordance_joint_pose_error(
        env,
        planar_position_threshold=planar_position_threshold,
        height_threshold=height_threshold,
        rotation_threshold=rotation_threshold,
        smooth_max_temperature=smooth_max_temperature,
        command_name=command_name,
        target_cfg=target_cfg,
    )
    return torch.exp(-0.5 * torch.square(error))


def affordance_joint_pose_tracking_cost(
    env: "ManagerBasedRLEnv",
    planar_scale_m: float = 0.08,
    height_scale_m: float = 0.01,
    rotation_scale_rad: float = 0.20,
    smooth_max_temperature: float = 0.25,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Continuous joint XY/Z/SO(3) state cost for non-prehensile control.

    These are shaping scales, not relaxed success thresholds.  The terminal
    predicate remains the strict 2 cm / 1 cm / 0.1 rad dwell contract.  The
    cost is bounded, high only when at least one component is poor, and has no
    waypoint, contact label, or simulator-only actor input.
    """

    planar, height, rotation = _affordance_goal_errors(
        env, command_name, target_cfg
    )
    return bounded_joint_pose_tracking_cost(
        planar,
        height,
        rotation,
        planar_scale=planar_scale_m,
        height_scale=height_scale_m,
        rotation_scale=rotation_scale_rad,
        temperature=smooth_max_temperature,
    )


class post_legal_safe_contact_joint_pose_tracking_cost(ManagerTermBase):
    """Activate the joint pose cost only after first legal safe contact.

    Before contact the robot cannot affect the hammer pose, so applying a
    persistent goal-pose cost there gives PPO an incentive to take unsafe
    shortcuts.  The observable legal-contact event is latched per episode;
    after that event the cost remains active even while the hand deliberately
    changes contact location.  This is a contact phase gate, not a waypoint,
    action label, relaxed terminal condition, or actor-only privileged input.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._contact_latched = torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contact_latched[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        planar_scale_m: float = 0.08,
        height_scale_m: float = 0.01,
        rotation_scale_rad: float = 0.20,
        smooth_max_temperature: float = 0.25,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.010,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        robot_obstacle_clearance_m: float = 0.005,
        physical_contact_force_threshold_n: float = 0.5,
        evaluate_robot_obstacle: bool = False,
        require_physical_protected_contact: bool = False,
        robot_target_sensor_name: ContactSensorNames = None,
        robot_obstacle_sensor_name: ContactSensorNames = None,
        target_obstacle_sensor_name: ContactSensorNames = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        contact_state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
            robot_obstacle_clearance_m=robot_obstacle_clearance_m,
            physical_contact_force_threshold_n=(
                physical_contact_force_threshold_n
            ),
            evaluate_robot_obstacle=evaluate_robot_obstacle,
            require_physical_protected_contact=(
                require_physical_protected_contact
            ),
            robot_target_sensor_name=robot_target_sensor_name,
            robot_obstacle_sensor_name=robot_obstacle_sensor_name,
            target_obstacle_sensor_name=target_obstacle_sensor_name,
        )
        self._contact_latched |= contact_state[
            "legal_safe_robot_contact"
        ].detach()
        cost = affordance_joint_pose_tracking_cost(
            env,
            planar_scale_m=planar_scale_m,
            height_scale_m=height_scale_m,
            rotation_scale_rad=rotation_scale_rad,
            smooth_max_temperature=smooth_max_temperature,
            command_name=command_name,
            target_cfg=target_cfg,
        )
        env._affordance_post_contact_pose_cost_latched = (
            self._contact_latched.clone()
        )
        return torch.where(
            self._contact_latched, cost, torch.zeros_like(cost)
        )


class post_legal_safe_contact_joint_pose_improvement_reward(ManagerTermBase):
    """Reward pose improvement relative to the first legal-contact state.

    The reference is captured on the first C1-legal safe contact.  The reward
    is exactly zero at that transition, positive after joint XY/Z/SO(3)
    improvement, and negative after regression.  Unlike an absolute
    post-contact cost, entering the controllable phase does not create a
    negative reward cliff that the policy can avoid by refusing contact.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._contact_latched = torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        )
        self._reference_cost = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contact_latched[env_ids] = False
        self._reference_cost[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        planar_scale_m: float = 0.08,
        height_scale_m: float = 0.01,
        rotation_scale_rad: float = 0.20,
        smooth_max_temperature: float = 0.25,
        normalization_cost: float = 0.25,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.010,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        robot_obstacle_clearance_m: float = 0.005,
        physical_contact_force_threshold_n: float = 0.5,
        evaluate_robot_obstacle: bool = False,
        require_physical_protected_contact: bool = False,
        robot_target_sensor_name: ContactSensorNames = None,
        robot_obstacle_sensor_name: ContactSensorNames = None,
        target_obstacle_sensor_name: ContactSensorNames = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        contact_state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
            robot_obstacle_clearance_m=robot_obstacle_clearance_m,
            physical_contact_force_threshold_n=physical_contact_force_threshold_n,
            evaluate_robot_obstacle=evaluate_robot_obstacle,
            require_physical_protected_contact=require_physical_protected_contact,
            robot_target_sensor_name=robot_target_sensor_name,
            robot_obstacle_sensor_name=robot_obstacle_sensor_name,
            target_obstacle_sensor_name=target_obstacle_sensor_name,
        )
        cost = affordance_joint_pose_tracking_cost(
            env,
            planar_scale_m=planar_scale_m,
            height_scale_m=height_scale_m,
            rotation_scale_rad=rotation_scale_rad,
            smooth_max_temperature=smooth_max_temperature,
            command_name=command_name,
            target_cfg=target_cfg,
        )
        legal_contact = contact_state["legal_safe_robot_contact"].detach()
        newly_latched = legal_contact & ~self._contact_latched
        self._reference_cost[newly_latched] = cost[newly_latched].detach()
        self._contact_latched |= legal_contact
        improvement = reference_relative_pose_improvement(
            torch.nan_to_num(self._reference_cost, nan=0.0),
            cost,
            normalization_cost=normalization_cost,
        )
        env._affordance_post_contact_pose_improvement_latched = (
            self._contact_latched.clone()
        )
        env._affordance_post_contact_pose_reference_cost = (
            self._reference_cost.clone()
        )
        return torch.where(
            self._contact_latched, improvement, torch.zeros_like(improvement)
        )


class legal_safe_contact_dapl_pose_progress_reward(ManagerTermBase):
    """Reward signed DAPL full-pose progress made during C1-legal contact.

    This is a transition potential, not an absolute goal-state reward: holding
    the hammer still inside the 10-cm gate earns exactly zero.  The potential
    uses DAPL's published full-3D position plus full-SO(3) error, while the
    contact gate enforces this task's safe-region semantics without a waypoint
    or a prescribed push direction.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_error = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_pose_error: float = 0.02,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        physical_contact_force_threshold_n: float = 0.5,
        robot_target_sensor_name: ContactSensorNames = None,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        planar_error, height_error, rotation_error = _affordance_goal_errors(
            env, command_name, target_cfg
        )
        position_error = torch.sqrt(
            torch.square(planar_error) + torch.square(height_error)
        )
        current_error = dapl_combined_pose_error(
            position_error, rotation_error
        )
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
            physical_contact_force_threshold_n=(
                physical_contact_force_threshold_n
            ),
            robot_target_sensor_name=robot_target_sensor_name,
        )
        valid_previous = torch.isfinite(self._previous_error)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_error, nan=0.0),
            current_error,
            normalization_distance=normalization_pose_error,
        )
        self._previous_error = current_error.detach()
        return torch.where(
            valid_previous & state["legal_safe_robot_contact"],
            progress,
            torch.zeros_like(progress),
        )


class affordance_joint_pose_progress_reward(ManagerTermBase):
    """Signed progress on one joint XY/Z/SO(3) potential."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_error = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        planar_position_threshold: float = 0.02,
        height_threshold: float = 0.01,
        rotation_threshold: float = 0.10,
        smooth_max_temperature: float = 0.25,
        command_name: str = "target_object_pose",
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ) -> torch.Tensor:
        current_error = affordance_joint_pose_error(
            env,
            planar_position_threshold=planar_position_threshold,
            height_threshold=height_threshold,
            rotation_threshold=rotation_threshold,
            smooth_max_temperature=smooth_max_temperature,
            command_name=command_name,
            target_cfg=target_cfg,
        )
        valid_previous = torch.isfinite(self._previous_error)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_error, nan=0.0),
            current_error,
            normalization_distance=1.0,
        )
        self._previous_error = current_error.detach()
        return torch.where(valid_previous, progress, torch.zeros_like(progress))


class affordance_weighted_component_pose_progress_reward(ManagerTermBase):
    """Signed XY/Z/SO(3) progress with independently visible components.

    The current safe-set distance only gates the reward to the same observable
    10-cm interaction neighborhood used by the preceding DAPL-aligned
    profiles.  It is not latched: leaving that neighborhood immediately turns
    the term off, and no phase state or waypoint is exposed to the policy.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_errors = torch.full(
            (env.num_envs, 3), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_errors[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        safe_ee_distance_threshold: float = 0.10,
        normalization_planar_distance_m: float = 0.01,
        normalization_height_m: float = 0.005,
        normalization_rotation_rad: float = 0.05,
        planar_weight: float = 20.0,
        height_weight: float = 4.0,
        rotation_weight: float = 8.0,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if safe_ee_distance_threshold <= 0.0:
            raise ValueError("safe_ee_distance_threshold must be positive")
        current_errors = torch.stack(
            _affordance_goal_errors(env, command_name, target_cfg), dim=-1
        )
        valid_previous = torch.isfinite(self._previous_errors).all(dim=-1)
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        near_safe_region = (
            state["minimum_safe_distance"] < safe_ee_distance_threshold
        )
        progress = weighted_componentwise_pose_progress(
            torch.nan_to_num(self._previous_errors, nan=0.0),
            current_errors,
            normalization_scales=(
                normalization_planar_distance_m,
                normalization_height_m,
                normalization_rotation_rad,
            ),
            component_weights=(
                planar_weight,
                height_weight,
                rotation_weight,
            ),
        )
        self._previous_errors = current_errors.detach()
        return torch.where(
            valid_previous & near_safe_region,
            progress,
            torch.zeros_like(progress),
        )


class affordance_positive_reference_component_pose_improvement_reward(
    ManagerTermBase
):
    """Positive reset-relative XY/Z/SO(3) improvement without max masking.

    This completes the minimal v13/v14 diagnostic pair: components are
    independently visible as in v14, while regression is rectified as in v13
    so exploratory contact is not assigned a sustained negative return.  The
    only state is the episode-constant reset reference; it is not a waypoint,
    contact latch, phase, or actor observation.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._reference_errors = torch.full(
            (env.num_envs, 3), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._reference_errors[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        safe_ee_distance_threshold: float = 0.10,
        reference_planar_error_floor_m: float = 0.02,
        reference_height_error_floor_m: float = 0.01,
        reference_rotation_error_floor_rad: float = 0.10,
        planar_weight: float = 20.0,
        height_weight: float = 4.0,
        rotation_weight: float = 8.0,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if safe_ee_distance_threshold <= 0.0:
            raise ValueError("safe_ee_distance_threshold must be positive")
        current_errors = torch.stack(
            _affordance_goal_errors(env, command_name, target_cfg), dim=-1
        )
        missing_reference = ~torch.isfinite(self._reference_errors).all(dim=-1)
        self._reference_errors = torch.where(
            missing_reference.unsqueeze(-1),
            current_errors.detach(),
            self._reference_errors,
        )
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        near_safe_region = (
            state["minimum_safe_distance"] < safe_ee_distance_threshold
        )
        improvement = positive_reference_relative_component_improvement(
            self._reference_errors,
            current_errors,
            reference_error_floors=(
                reference_planar_error_floor_m,
                reference_height_error_floor_m,
                reference_rotation_error_floor_rad,
            ),
            component_weights=(
                planar_weight,
                height_weight,
                rotation_weight,
            ),
        )
        return torch.where(
            near_safe_region,
            improvement,
            torch.zeros_like(improvement),
        )


class affordance_positive_reference_pareto_pose_improvement_reward(
    ManagerTermBase
):
    """Positive joint XY/SO(3) improvement inside the strict Z band.

    The episode-constant reference is the same non-policy state used by the
    preceding reset-relative diagnostics. The reward is instantaneous and
    contains no contact latch, stage, waypoint, or actor feature.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._reference_errors = torch.full(
            (env.num_envs, 3), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._reference_errors[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        safe_ee_distance_threshold: float = 0.10,
        reference_planar_error_floor_m: float = 0.02,
        reference_rotation_error_floor_rad: float = 0.10,
        support_height_tolerance_m: float = 0.01,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if safe_ee_distance_threshold <= 0.0:
            raise ValueError("safe_ee_distance_threshold must be positive")
        current_errors = torch.stack(
            _affordance_goal_errors(env, command_name, target_cfg), dim=-1
        )
        missing_reference = ~torch.isfinite(self._reference_errors).all(dim=-1)
        self._reference_errors = torch.where(
            missing_reference.unsqueeze(-1),
            current_errors.detach(),
            self._reference_errors,
        )
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        near_safe_region = (
            state["minimum_safe_distance"] < safe_ee_distance_threshold
        )
        improvement = positive_reference_relative_pareto_pose_improvement(
            self._reference_errors,
            current_errors,
            reference_planar_error_floor_m=reference_planar_error_floor_m,
            reference_rotation_error_floor_rad=reference_rotation_error_floor_rad,
            support_height_tolerance_m=support_height_tolerance_m,
        )
        return torch.where(
            near_safe_region,
            improvement,
            torch.zeros_like(improvement),
        )


class affordance_dywa_keypoint_pose_potential_reward(ManagerTermBase):
    """DyWA-style discounted potential over the hammer's surface keypoints.

    The same 512 canonical surface points represented in the actor input are
    transformed by the current and commanded object poses. Their corresponding
    distances jointly encode translation and full SO(3), including support
    height, without a component-wise trade, waypoint, contact gate, or phase
    state. Only the previous scalar potential is cached for the standard
    temporal-difference shaping transition.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_potential = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        potential_amplitude: float = 0.302,
        potential_distance_rate: float = 243.12,
        potential_exponential_base: float = 0.995,
        potential_discount: float = 0.99,
        use_bounding_box_keypoints: bool = False,
        command_name: str = "target_object_pose",
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ) -> torch.Tensor:
        target: RigidObject = env.scene[target_cfg.name]
        target_points, _ = _affordance_scene_geometry(
            env, target_cfg, obstacles_cfg
        )
        command = env.command_manager.get_command(command_name)
        current_position = target.data.root_pos_w[:, :3] - env.scene.env_origins
        current_quaternion = target.data.root_quat_w
        point_count = target_points.shape[1]

        current_offsets = target_points - current_position.unsqueeze(1)
        inverse_current = quat_conjugate(current_quaternion)
        canonical_offsets = quat_apply(
            inverse_current.unsqueeze(1)
            .expand(-1, point_count, -1)
            .reshape(-1, 4),
            current_offsets.reshape(-1, 3),
        ).reshape(env.num_envs, point_count, 3)

        if use_bounding_box_keypoints:
            canonical_offsets = axis_aligned_bounding_box_keypoints(canonical_offsets)
            point_count = canonical_offsets.shape[1]
            current_offsets = quat_apply(
                current_quaternion.unsqueeze(1)
                .expand(-1, point_count, -1)
                .reshape(-1, 4),
                canonical_offsets.reshape(-1, 3),
            ).reshape(env.num_envs, point_count, 3)
            current_keypoints = current_position.unsqueeze(1) + current_offsets
        else:
            current_keypoints = target_points

        goal_offsets = quat_apply(
            command[:, 3:7]
            .unsqueeze(1)
            .expand(-1, point_count, -1)
            .reshape(-1, 4),
            canonical_offsets.reshape(-1, 3),
        ).reshape(env.num_envs, point_count, 3)
        goal_points = command[:, :3].unsqueeze(1) + goal_offsets
        point_distances = torch.linalg.vector_norm(
            current_keypoints - goal_points, dim=-1
        )
        current_potential = dywa_exponential_keypoint_potential(
            point_distances,
            amplitude=potential_amplitude,
            distance_rate=potential_distance_rate,
            exponential_base=potential_exponential_base,
        )
        valid_previous = torch.isfinite(self._previous_potential)
        shaping = discounted_score_potential_shaping(
            torch.nan_to_num(self._previous_potential, nan=0.0),
            current_potential,
            discount_factor=potential_discount,
        )
        self._previous_potential = current_potential.detach()
        return torch.where(valid_previous, shaping, torch.zeros_like(shaping))


class safe_region_dywa_distance_potential_reward(ManagerTermBase):
    """DyWA temporal potential over distance to the safe affordance set.

    This is the affordance-aware analogue of DyWA's hand-to-object potential:
    centroid distance is replaced by the minimum distance to a safe surface
    point, while the exponential and temporal-difference structure remain the
    same.  It supplies continuous far-to-near shaping without a persistent
    absolute reaching cost competing with the object-goal potential.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_potential = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        potential_amplitude: float = 0.0604,
        potential_distance_rate: float = 243.12,
        potential_exponential_base: float = 0.995,
        potential_discount: float = 0.995,
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        current_potential = dywa_exponential_keypoint_potential(
            state["minimum_safe_distance"].unsqueeze(-1),
            amplitude=potential_amplitude,
            distance_rate=potential_distance_rate,
            exponential_base=potential_exponential_base,
        )
        valid_previous = torch.isfinite(self._previous_potential)
        shaping = discounted_score_potential_shaping(
            torch.nan_to_num(self._previous_potential, nan=0.0),
            current_potential,
            discount_factor=potential_discount,
        )
        self._previous_potential = current_potential.detach()
        return torch.where(valid_previous, shaping, torch.zeros_like(shaping))


def affordance_planar_goal_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.20,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Dense XY-only target progress, independent of end-effector contact."""

    if std <= 0.0:
        raise ValueError("planar goal std must be positive")
    distance, _, _ = _affordance_goal_errors(env, command_name, target_cfg)
    return 1.0 - torch.tanh(distance / std)


def affordance_orientation_goal_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.50,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Dense full-orientation reward active throughout pose training."""

    if std <= 0.0:
        raise ValueError("orientation goal std must be positive")
    _, _, angle = _affordance_goal_errors(env, command_name, target_cfg)
    return 1.0 - torch.tanh(angle / std)


class affordance_planar_goal_progress_reward(ManagerTermBase):
    """Signed potential progress of the target object toward its XY goal."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_error = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_distance_m: float = 0.01,
        command_name: str = "target_object_pose",
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ) -> torch.Tensor:
        current_error, _, _ = _affordance_goal_errors(
            env, command_name, target_cfg
        )
        valid_previous = torch.isfinite(self._previous_error)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_error, nan=0.0),
            current_error,
            normalization_distance=normalization_distance_m,
        )
        self._previous_error = current_error.detach()
        return torch.where(valid_previous, progress, torch.zeros_like(progress))


class affordance_orientation_goal_progress_reward(ManagerTermBase):
    """Signed potential progress toward the complete target orientation."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_error = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_angle_rad: float = 0.05,
        command_name: str = "target_object_pose",
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ) -> torch.Tensor:
        _, _, current_error = _affordance_goal_errors(
            env, command_name, target_cfg
        )
        valid_previous = torch.isfinite(self._previous_error)
        progress = normalized_distance_progress(
            torch.nan_to_num(self._previous_error, nan=0.0),
            current_error,
            normalization_distance=normalization_angle_rad,
        )
        self._previous_error = current_error.detach()
        return torch.where(valid_previous, progress, torch.zeros_like(progress))


class safe_contact_planar_goal_progress_reward(ManagerTermBase):
    """Reward safe contact only while it advances the object toward the goal."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_error = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        normalization_distance_m: float = 0.01,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        current_error, _, _ = _affordance_goal_errors(
            env, command_name, target_cfg
        )
        valid_previous = torch.isfinite(self._previous_error)
        safe_contact = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )["safe_robot_contact"]
        reward = positive_distance_progress_during_contact(
            torch.nan_to_num(self._previous_error, nan=0.0),
            current_error,
            valid_previous & safe_contact,
            normalization_distance=normalization_distance_m,
        )
        self._previous_error = current_error.detach()
        return reward


class safe_contact_joint_pose_progress_reward(ManagerTermBase):
    """Reward signed XY/Z/SO(3) improvement made during legal contact."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_error = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        planar_position_threshold: float = 0.02,
        height_threshold: float = 0.01,
        rotation_threshold: float = 0.10,
        smooth_max_temperature: float = 0.25,
        normalization_pose_error: float = 1.0,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        current_error = affordance_joint_pose_error(
            env,
            planar_position_threshold=planar_position_threshold,
            height_threshold=height_threshold,
            rotation_threshold=rotation_threshold,
            smooth_max_temperature=smooth_max_temperature,
            command_name=command_name,
            target_cfg=target_cfg,
        )
        valid_previous = torch.isfinite(self._previous_error)
        safe_contact = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )["safe_robot_contact"]
        reward = distance_progress_during_contact(
            torch.nan_to_num(self._previous_error, nan=0.0),
            current_error,
            valid_previous & safe_contact,
            normalization_distance=normalization_pose_error,
        )
        self._previous_error = current_error.detach()
        return reward


class safe_contact_pose_component_progress_reward(ManagerTermBase):
    """Reward independent XY, support-height, or SO(3) progress at safe contact."""

    _COMPONENT_INDICES = {"planar": 0, "height": 1, "rotation": 2}

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_errors = torch.full(
            (env.num_envs, 3), torch.nan, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_errors[env_ids] = torch.nan

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        component: str,
        normalization_planar_distance_m: float = 0.01,
        normalization_height_m: float = 0.005,
        normalization_rotation_rad: float = 0.05,
        command_name: str = "target_object_pose",
        contact_distance_m: float = 0.008,
        minimum_safe_score: float = 0.25,
        minimum_protected_score: float = 0.25,
        protected_point_count: int = 64,
        protected_clearance_m: float = 0.005,
        evaluate_protected: bool = True,
        require_legal_contact: bool = False,
        safe_radius_m: float | None = None,
        protected_radius_m: float | None = None,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if component not in self._COMPONENT_INDICES:
            raise ValueError(
                "pose progress component must be 'planar', 'height', or 'rotation'"
            )
        current_errors = torch.stack(
            _affordance_goal_errors(env, command_name, target_cfg), dim=-1
        )
        valid_previous = torch.isfinite(self._previous_errors).all(dim=-1)
        contact_state = _contact_term_state(
            env,
            contact_distance_m,
            minimum_safe_score,
            minimum_protected_score,
            protected_point_count,
            protected_clearance_m,
            evaluate_protected,
            safe_radius_m,
            protected_radius_m,
            target_cfg,
            obstacles_cfg,
            ee_frame_cfg,
        )
        safe_contact = contact_state[
            "legal_safe_robot_contact"
            if require_legal_contact
            else "safe_robot_contact"
        ]
        progress = componentwise_progress_during_contact(
            torch.nan_to_num(self._previous_errors, nan=0.0),
            current_errors,
            valid_previous & safe_contact,
            normalization_scales=(
                normalization_planar_distance_m,
                normalization_height_m,
                normalization_rotation_rad,
            ),
        )
        self._previous_errors = current_errors.detach()
        return progress[:, self._COMPONENT_INDICES[component]]


def near_goal_target_motion_penalty(
    env: "ManagerBasedRLEnv",
    activation_pose_error: float = 2.0,
    linear_speed_scale: float = 0.03,
    angular_speed_scale: float = 0.30,
    planar_position_threshold: float = 0.02,
    height_threshold: float = 0.01,
    rotation_threshold: float = 0.10,
    smooth_max_temperature: float = 0.25,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Penalize target motion only after the complete pose is near its goal."""

    target: RigidObject = env.scene[target_cfg.name]
    pose_error = affordance_joint_pose_error(
        env,
        planar_position_threshold=planar_position_threshold,
        height_threshold=height_threshold,
        rotation_threshold=rotation_threshold,
        smooth_max_temperature=smooth_max_temperature,
        command_name=command_name,
        target_cfg=target_cfg,
    )
    linear_speed = torch.linalg.vector_norm(target.data.root_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(target.data.root_ang_vel_w, dim=-1)
    return near_goal_motion_cost(
        pose_error,
        linear_speed,
        angular_speed,
        activation_pose_error=activation_pose_error,
        linear_speed_scale=linear_speed_scale,
        angular_speed_scale=angular_speed_scale,
    )


def affordance_height_goal_error_penalty(
    env: "ManagerBasedRLEnv",
    normalization_height_m: float = 0.01,
    command_name: str = "target_object_pose",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Penalize leaving the goal support height without adding a baseline reward."""

    if normalization_height_m <= 0.0:
        raise ValueError("height normalization must be positive")
    _, height_error, _ = _affordance_goal_errors(env, command_name, target_cfg)
    return torch.clamp(height_error / normalization_height_m, max=1.0)


def affordance_target_pose_valid(
    env: "ManagerBasedRLEnv",
    command_name: str = "target_object_pose",
    planar_position_threshold: float = 0.02,
    height_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Check XY, support height, and full orientation simultaneously."""

    target: RigidObject = env.scene[target_cfg.name]
    goal = env.command_manager.get_command(command_name)
    position_env = target.data.root_pos_w[:, :3] - env.scene.env_origins
    return support_aware_pose_success(
        position_env,
        target.data.root_quat_w,
        goal,
        planar_position_threshold=planar_position_threshold,
        height_threshold=height_threshold,
        rotation_threshold=rotation_threshold,
    )


class affordance_target_reached_goal(ManagerTermBase):
    """Terminate only after the full pose remains valid for several policy steps."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._consecutive_valid_steps = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_valid_steps[env_ids] = 0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "target_object_pose",
        planar_position_threshold: float = 0.02,
        height_threshold: float = 0.01,
        rotation_threshold: float = 0.1,
        consecutive_steps: int = 5,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ) -> torch.Tensor:
        if consecutive_steps <= 0:
            raise ValueError("consecutive_steps must be positive")
        pose_is_valid = affordance_target_pose_valid(
            env,
            command_name=command_name,
            planar_position_threshold=planar_position_threshold,
            height_threshold=height_threshold,
            rotation_threshold=rotation_threshold,
            target_cfg=target_cfg,
        )
        self._consecutive_valid_steps = update_consecutive_success_count(
            self._consecutive_valid_steps, pose_is_valid
        )
        return (self._consecutive_valid_steps >= consecutive_steps).clone()


def termination_success_reward(
    env: "ManagerBasedRLEnv",
    termination_term_name: str = "reached",
) -> torch.Tensor:
    """Reward a completed task without imposing an additional safety label.

    This is used only by explicitly named unconstrained diagnostic profiles.
    The termination term remains the source of truth for the pose thresholds
    and dwell requirement, while typed violations continue to be measured by
    the environment wrapper for a clean C1-on/C1-off comparison.
    """

    return env.termination_manager.get_term(termination_term_name).float()


def affordance_task_success_reward(
    env: "ManagerBasedRLEnv",
    termination_term_name: str = "reached",
    contact_distance_m: float = 0.008,
    minimum_safe_score: float = 0.25,
    minimum_protected_score: float = 0.25,
    protected_point_count: int = 64,
    protected_clearance_m: float = 0.005,
    robot_obstacle_clearance_m: float = 0.005,
    physical_contact_force_threshold_n: float = 0.5,
    evaluate_protected: bool = True,
    evaluate_robot_obstacle: bool = False,
    require_physical_protected_contact: bool = False,
    robot_target_sensor_name: ContactSensorNames = None,
    robot_obstacle_sensor_name: ContactSensorNames = None,
    target_obstacle_sensor_name: ContactSensorNames = None,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward only dwell-qualified goals reached without semantic violations."""

    reached = env.termination_manager.get_term(termination_term_name).bool()
    contact_state = domino_affordance_contact_state(
        env,
        contact_distance_m=contact_distance_m,
        minimum_safe_score=minimum_safe_score,
        minimum_protected_score=minimum_protected_score,
        protected_point_count=protected_point_count,
        protected_clearance_m=protected_clearance_m,
        robot_obstacle_clearance_m=robot_obstacle_clearance_m,
        physical_contact_force_threshold_n=physical_contact_force_threshold_n,
        evaluate_protected=evaluate_protected,
        evaluate_robot_obstacle=evaluate_robot_obstacle,
        require_physical_protected_contact=require_physical_protected_contact,
        robot_target_sensor_name=robot_target_sensor_name,
        robot_obstacle_sensor_name=robot_obstacle_sensor_name,
        target_obstacle_sensor_name=target_obstacle_sensor_name,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
        target_cfg=target_cfg,
        obstacles_cfg=obstacles_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    current_violation = (
        contact_state["forbidden_robot_contact"]
        | contact_state["protected_obstacle_collision"]
        | contact_state["robot_obstacle_collision"]
    )
    previous_violation = getattr(
        env,
        "episode_affordance_violation_buf",
        torch.zeros_like(current_violation),
    )
    success = reached & ~current_violation & ~previous_violation
    if getattr(env, "_capture_affordance_reward_debug", False):
        env._affordance_task_success_reward_debug = {
            "reached": reached.detach().clone(),
            "c1_forbidden_robot_contact": contact_state[
                "forbidden_robot_contact"
            ].detach().clone(),
            "c2_protected_obstacle_collision": contact_state[
                "protected_obstacle_collision"
            ].detach().clone(),
            "c3_robot_obstacle_collision": contact_state[
                "robot_obstacle_collision"
            ].detach().clone(),
            "current_violation": current_violation.detach().clone(),
            "previous_violation": previous_violation.detach().clone(),
            "success": success.detach().clone(),
        }
    return success.float()
