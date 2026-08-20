"""Deterministic, simulator-independent Clutter6D scene generation.

The public DAPL release currently contains object assets but not the scene
graphs/manifests used by the paper.  This module records the constraints that
are stated in the paper and generates replayable JSONL inputs for Isaac Lab.
It deliberately uses conservative 2-D footprint rejection: final physical
settling is still validated by the simulator before a manifest is promoted to
an evaluation split.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from .scene import ClutterScene, ClutterTrack, ManipulationTask, SceneObject


def _finite_tuple(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    result = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1.0e-12:
        raise ValueError("quaternion product has zero norm")
    return tuple(item / norm for item in result)


@dataclass(frozen=True)
class StablePose:
    """One stable mesh orientation and its table support geometry.

    ``support_height`` is the object-root height above the tabletop.
    ``footprint`` is ``(xmin, ymin, xmax, ymax)`` in that root frame after
    applying ``quaternion``.  An arbitrary yaw is applied during generation.
    """

    quaternion: tuple[float, ...]
    support_height: float
    footprint: tuple[float, ...]

    def __post_init__(self) -> None:
        quaternion = _finite_tuple(self.quaternion, 4, "quaternion")
        norm = math.sqrt(sum(item * item for item in quaternion))
        if not math.isclose(norm, 1.0, abs_tol=1.0e-3):
            raise ValueError("stable-pose quaternion must be normalized")
        object.__setattr__(self, "quaternion", quaternion)
        support_height = float(self.support_height)
        if not math.isfinite(support_height) or support_height < 0.0:
            raise ValueError("support_height must be finite and non-negative")
        object.__setattr__(self, "support_height", support_height)
        footprint = _finite_tuple(self.footprint, 4, "footprint")
        if footprint[0] >= footprint[2] or footprint[1] >= footprint[3]:
            raise ValueError("footprint must have positive width and depth")
        object.__setattr__(self, "footprint", footprint)


@dataclass(frozen=True)
class ClutterAsset:
    """Physical and support metadata needed to place one normalized asset."""

    asset_id: str
    scale: tuple[float, ...]
    mass_kg: float
    stable_poses: tuple[StablePose, ...]
    is_large: bool
    static_friction: float = 0.8
    dynamic_friction: float = 0.8
    restitution: float = 0.0

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        scale = _finite_tuple(self.scale, 3, "scale")
        if any(item <= 0.0 for item in scale):
            raise ValueError("asset scale must be positive")
        object.__setattr__(self, "scale", scale)
        stable_poses = tuple(self.stable_poses)
        if not stable_poses:
            raise ValueError("an asset needs at least one stable pose")
        object.__setattr__(self, "stable_poses", stable_poses)
        for name in ("mass_kg", "static_friction", "dynamic_friction", "restitution"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic friction must not exceed static friction")


@dataclass(frozen=True)
class ClutterGenerationConfig:
    """Paper constraints plus explicit tabletop placement bounds."""

    table_center: tuple[float, float] = (0.5, 0.0)
    table_x_range: tuple[float, float] = (0.18, 0.82)
    table_y_range: tuple[float, float] = (-0.45, 0.45)
    target_x_offset_range: tuple[float, float] = (-0.15, 0.15)
    target_y_offset_range: tuple[float, float] = (-0.30, 0.30)
    tasks_per_scene: int = 16
    minimum_planar_displacement: float = 0.15
    clearance: float = 0.008
    placement_attempts: int = 512
    scene_attempts: int = 64
    preserve_target_support_pose: bool = False

    def __post_init__(self) -> None:
        for name in (
            "table_center",
            "table_x_range",
            "table_y_range",
            "target_x_offset_range",
            "target_y_offset_range",
        ):
            value = _finite_tuple(getattr(self, name), 2, name)
            object.__setattr__(self, name, value)
        for name in ("table_x_range", "table_y_range", "target_x_offset_range", "target_y_offset_range"):
            lower, upper = getattr(self, name)
            if lower >= upper:
                raise ValueError(f"{name} must have lower < upper")
        if self.tasks_per_scene <= 0 or self.placement_attempts <= 0 or self.scene_attempts <= 0:
            raise ValueError("task and attempt counts must be positive")
        if self.minimum_planar_displacement <= 0.0 or self.clearance < 0.0:
            raise ValueError("displacement must be positive and clearance non-negative")


@dataclass(frozen=True)
class _PlacedPose:
    pose: tuple[float, ...]
    footprint: tuple[float, ...]


def _yaw_stable_pose(stable: StablePose, yaw: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Apply yaw and conservatively rotate a stable footprint AABB."""

    half_yaw = 0.5 * yaw
    yaw_quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    quaternion = _quat_multiply(yaw_quat, stable.quaternion)
    xmin, ymin, xmax, ymax = stable.footprint
    cosine, sine = math.cos(yaw), math.sin(yaw)
    corners = ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    rotated = tuple((cosine * x - sine * y, sine * x + cosine * y) for x, y in corners)
    xs, ys = zip(*rotated)
    return quaternion, (min(xs), min(ys), max(xs), max(ys))


def _overlap(left: Sequence[float], right: Sequence[float], clearance: float) -> bool:
    return not (
        left[2] + clearance <= right[0]
        or right[2] + clearance <= left[0]
        or left[3] + clearance <= right[1]
        or right[3] + clearance <= left[1]
    )


def _sample_pose(
    rng: random.Random,
    asset: ClutterAsset,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    occupied: Sequence[Sequence[float]],
    cfg: ClutterGenerationConfig,
    stable_pose: StablePose | None = None,
) -> _PlacedPose | None:
    for _ in range(cfg.placement_attempts):
        stable = stable_pose if stable_pose is not None else rng.choice(asset.stable_poses)
        quaternion, relative_footprint = _yaw_stable_pose(
            stable, rng.uniform(-math.pi, math.pi)
        )
        xmin, ymin, xmax, ymax = relative_footprint
        feasible_x = (max(x_range[0], cfg.table_x_range[0] - xmin), min(x_range[1], cfg.table_x_range[1] - xmax))
        feasible_y = (max(y_range[0], cfg.table_y_range[0] - ymin), min(y_range[1], cfg.table_y_range[1] - ymax))
        if feasible_x[0] > feasible_x[1] or feasible_y[0] > feasible_y[1]:
            continue
        root_x = rng.uniform(*feasible_x)
        root_y = rng.uniform(*feasible_y)
        footprint = (root_x + xmin, root_y + ymin, root_x + xmax, root_y + ymax)
        if any(_overlap(footprint, other, cfg.clearance) for other in occupied):
            continue
        return _PlacedPose(
            pose=(root_x, root_y, stable.support_height, *quaternion),
            footprint=footprint,
        )
    return None


def _sample_distinct_assets(
    rng: random.Random,
    candidates: Sequence[ClutterAsset],
    count: int,
    excluded: set[str],
) -> list[ClutterAsset]:
    available = [item for item in candidates if item.asset_id not in excluded]
    if len(available) < count:
        raise ValueError(f"need {count} distinct assets but only {len(available)} are available")
    selected = rng.sample(available, count)
    excluded.update(item.asset_id for item in selected)
    return selected


def _generate_one_scene(
    assets: Sequence[ClutterAsset],
    *,
    target_asset_ids: frozenset[str] | None,
    track: ClutterTrack,
    split: str,
    scene_index: int,
    seed: int,
    cfg: ClutterGenerationConfig,
) -> ClutterScene:
    rng = random.Random(seed)
    large_assets = [item for item in assets if item.is_large]
    small_assets = [item for item in assets if not item.is_large]
    if not assets:
        raise ValueError("asset catalog is empty")
    target_assets = (
        list(assets)
        if target_asset_ids is None
        else [item for item in assets if item.asset_id in target_asset_ids]
    )
    if not target_assets:
        raise ValueError("target_asset_ids does not select any catalog asset")

    for _ in range(cfg.scene_attempts):
        target = rng.choice(target_assets)
        excluded = {target.asset_id}
        large = _sample_distinct_assets(rng, large_assets, track.large_obstacle_count, excluded)
        small = _sample_distinct_assets(rng, small_assets, track.small_obstacle_count, excluded)
        obstacle_assets = large + small

        occupied: list[tuple[float, ...]] = []
        obstacle_placements: list[_PlacedPose] = []
        for asset in obstacle_assets:
            placement = _sample_pose(
                rng,
                asset,
                cfg.table_x_range,
                cfg.table_y_range,
                occupied,
                cfg,
            )
            if placement is None:
                break
            obstacle_placements.append(placement)
            occupied.append(placement.footprint)
        if len(obstacle_placements) != len(obstacle_assets):
            continue

        target_x_range = (
            cfg.table_center[0] + cfg.target_x_offset_range[0],
            cfg.table_center[0] + cfg.target_x_offset_range[1],
        )
        target_y_range = (
            cfg.table_center[1] + cfg.target_y_offset_range[0],
            cfg.table_center[1] + cfg.target_y_offset_range[1],
        )
        tasks: list[ManipulationTask] = []
        seen_pairs: set[tuple[float, ...]] = set()
        task_attempts = cfg.placement_attempts * cfg.tasks_per_scene
        for _ in range(task_attempts):
            if len(tasks) == cfg.tasks_per_scene:
                break
            shared_stable_pose = (
                rng.choice(target.stable_poses)
                if cfg.preserve_target_support_pose
                else None
            )
            initial = _sample_pose(
                rng,
                target,
                target_x_range,
                target_y_range,
                occupied,
                cfg,
                stable_pose=shared_stable_pose,
            )
            goal = _sample_pose(
                rng,
                target,
                target_x_range,
                target_y_range,
                occupied,
                cfg,
                stable_pose=shared_stable_pose,
            )
            if initial is None or goal is None:
                continue
            planar_displacement = math.hypot(
                goal.pose[0] - initial.pose[0], goal.pose[1] - initial.pose[1]
            )
            if planar_displacement + 1.0e-9 < cfg.minimum_planar_displacement:
                continue
            signature = tuple(round(item, 7) for item in (*initial.pose, *goal.pose))
            if signature in seen_pairs:
                continue
            seen_pairs.add(signature)
            tasks.append(
                ManipulationTask(
                    task_id=f"task-{len(tasks):02d}",
                    target_instance_id="target",
                    initial_pose=initial.pose,
                    goal_pose=goal.pose,
                )
            )
        if len(tasks) != cfg.tasks_per_scene:
            continue

        scene_objects = [
            SceneObject(
                instance_id="target",
                asset_id=target.asset_id,
                pose=tasks[0].initial_pose,
                scale=target.scale,
                mass_kg=target.mass_kg,
                static_friction=target.static_friction,
                dynamic_friction=target.dynamic_friction,
                restitution=target.restitution,
            )
        ]
        for obstacle_index, (asset, placement) in enumerate(
            zip(obstacle_assets, obstacle_placements, strict=True)
        ):
            cohort = "large" if asset.is_large else "small"
            scene_objects.append(
                SceneObject(
                    instance_id=f"{cohort}-{obstacle_index:02d}",
                    asset_id=asset.asset_id,
                    pose=placement.pose,
                    scale=asset.scale,
                    mass_kg=asset.mass_kg,
                    static_friction=asset.static_friction,
                    dynamic_friction=asset.dynamic_friction,
                    restitution=asset.restitution,
                )
            )

        scene = ClutterScene(
            scene_id=f"{split}-{track.value}-{scene_index:04d}",
            split=split,
            track=track,
            objects=tuple(scene_objects),
            tasks=tuple(tasks),
        )
        scene.validate_paper_contract(
            tasks_per_scene=cfg.tasks_per_scene,
            minimum_planar_displacement=cfg.minimum_planar_displacement,
        )
        return scene

    raise RuntimeError(
        f"failed to generate {track.value} scene {scene_index} after "
        f"{cfg.scene_attempts} scene attempts"
    )


def generate_clutter_scenes(
    assets: Sequence[ClutterAsset],
    *,
    track: ClutterTrack | str,
    split: str,
    scene_count: int,
    seed: int,
    config: ClutterGenerationConfig | None = None,
    target_asset_ids: Sequence[str] | None = None,
) -> tuple[ClutterScene, ...]:
    """Generate scenes whose index is reproducible independently of batch size.

    ``target_asset_ids`` optionally restricts the manipulated object while
    leaving all catalog assets available as clutter.  This is useful for
    part-aware benchmarks where only targets with semantic annotations should
    be selected.
    """

    track = ClutterTrack(track)
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    if scene_count <= 0:
        raise ValueError("scene_count must be positive")
    config = ClutterGenerationConfig() if config is None else config
    assets = tuple(assets)
    selected_target_ids = (
        None if target_asset_ids is None else frozenset(str(item) for item in target_asset_ids)
    )
    if selected_target_ids is not None:
        unknown = selected_target_ids - {item.asset_id for item in assets}
        if unknown:
            raise ValueError(f"target_asset_ids are absent from the catalog: {sorted(unknown)}")
    return tuple(
        _generate_one_scene(
            assets,
            target_asset_ids=selected_target_ids,
            track=track,
            split=split,
            scene_index=index,
            # Large odd stride prevents adjacent scenes from sharing RNG state
            # while keeping scene N invariant to the requested scene_count.
            seed=int(seed) + index * 1_000_003,
            cfg=config,
        )
        for index in range(scene_count)
    )
