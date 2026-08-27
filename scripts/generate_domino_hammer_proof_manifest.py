#!/usr/bin/env python3
"""Generate stable single-hammer scenes with controlled pose randomization."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import random

from dapl.catalog import stable_poses_from_mesh
from dapl.domino import DominoDataPaths
from dapl.scene import ClutterScene, ManipulationTask, load_scene_manifest, write_scene_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--scene-id-prefix",
        default="train-hammer-joint-pose-stable",
    )
    parser.add_argument("--domino-root", type=Path)
    parser.add_argument("--support-pose-index", type=int, default=0)
    parser.add_argument("--initial-yaw", type=float, default=-0.5 * math.pi)
    parser.add_argument("--initial-x", type=float, default=0.48)
    parser.add_argument("--initial-y", type=float, default=0.0)
    parser.add_argument("--goal-dx", type=float, default=0.08)
    parser.add_argument("--goal-dy", type=float, default=0.0)
    parser.add_argument("--goal-yaw-delta", type=float, default=-0.15)
    parser.add_argument("--initial-x-jitter", type=float, default=0.0)
    parser.add_argument("--initial-y-jitter", type=float, default=0.0)
    parser.add_argument("--initial-yaw-jitter", type=float, default=0.0)
    parser.add_argument("--goal-dx-jitter", type=float, default=0.0)
    parser.add_argument("--goal-dy-jitter", type=float, default=0.0)
    parser.add_argument("--goal-yaw-jitter", type=float, default=0.0)
    parser.add_argument(
        "--goal-angle-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_RAD", "MAX_RAD"),
        help=(
            "sample goal displacement directions over this angular interval; "
            "requires --goal-distance-range and replaces goal dx/dy sampling"
        ),
    )
    parser.add_argument(
        "--goal-distance-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--goal-direction-bins",
        type=int,
        default=8,
        help="stratify directional scenes across this many balanced angle bins",
    )
    parser.add_argument(
        "--workspace-x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="reject samples whose initial or goal X lies outside this range",
    )
    parser.add_argument(
        "--workspace-y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="reject samples whose initial or goal Y lies outside this range",
    )
    parser.add_argument(
        "--stable-obstacle-count",
        type=int,
        default=0,
        help="replace the first N obstacle poses with randomized stable supports",
    )
    parser.add_argument(
        "--obstacle-support-pose-index",
        type=int,
        default=0,
        help="stable-pose candidate used for every promoted obstacle",
    )
    parser.add_argument(
        "--obstacle-support-pose-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "per-slot stable-pose candidates; overrides "
            "--obstacle-support-pose-index"
        ),
    )
    parser.add_argument(
        "--obstacle-x-range",
        type=float,
        nargs=2,
        default=(0.42, 0.68),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--obstacle-y-range",
        type=float,
        nargs=2,
        default=(0.18, 0.24),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--obstacle-path-clearance",
        type=float,
        default=0.16,
        help="minimum obstacle-root distance from the initial-to-goal XY segment",
    )
    parser.add_argument(
        "--obstacle-path-max-distance",
        type=float,
        default=None,
        help=(
            "optional maximum root distance from the push segment; use this "
            "to put clutter near, but not directly on, the protected-part path"
        ),
    )
    parser.add_argument(
        "--obstacle-root-clearance",
        type=float,
        default=0.12,
        help="minimum root separation from target endpoints and other obstacles",
    )
    parser.add_argument(
        "--friction-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="optional shared static/dynamic friction range per scene",
    )
    parser.add_argument(
        "--target-mass-scale-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--obstacle-friction-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--obstacle-mass-scale-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
    )
    return parser.parse_args()


def _quat_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    result = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    norm = math.sqrt(sum(value * value for value in result))
    return tuple(value / norm for value in result)


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta = (end[0] - start[0], end[1] - start[1])
    squared_length = delta[0] * delta[0] + delta[1] * delta[1]
    if squared_length <= 1.0e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * delta[0]
        + (point[1] - start[1]) * delta[1]
    ) / squared_length
    projection = min(1.0, max(0.0, projection))
    closest = (
        start[0] + projection * delta[0],
        start[1] + projection * delta[1],
    )
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def _inside_optional_range(value: float, bounds: tuple[float, float] | None) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


def _sample_task_planar_pose(
    args: argparse.Namespace,
    rng: random.Random,
    scene_index: int,
) -> tuple[float, float, float, float]:
    """Sample reachable initial/goal XY with balanced displacement directions."""

    directional = args.goal_angle_range is not None
    for _ in range(2000):
        initial_x = args.initial_x + rng.uniform(
            -args.initial_x_jitter, args.initial_x_jitter
        )
        initial_y = args.initial_y + rng.uniform(
            -args.initial_y_jitter, args.initial_y_jitter
        )
        if directional:
            angle_min, angle_max = args.goal_angle_range
            bin_width = (angle_max - angle_min) / args.goal_direction_bins
            bin_index = scene_index % args.goal_direction_bins
            angle = rng.uniform(
                angle_min + bin_index * bin_width,
                angle_min + (bin_index + 1) * bin_width,
            )
            distance = rng.uniform(*args.goal_distance_range)
            goal_dx = distance * math.cos(angle)
            goal_dy = distance * math.sin(angle)
        else:
            goal_dx = args.goal_dx + rng.uniform(
                -args.goal_dx_jitter, args.goal_dx_jitter
            )
            goal_dy = args.goal_dy + rng.uniform(
                -args.goal_dy_jitter, args.goal_dy_jitter
            )
        goal_x = initial_x + goal_dx
        goal_y = initial_y + goal_dy
        if not _inside_optional_range(initial_x, args.workspace_x_range):
            continue
        if not _inside_optional_range(goal_x, args.workspace_x_range):
            continue
        if not _inside_optional_range(initial_y, args.workspace_y_range):
            continue
        if not _inside_optional_range(goal_y, args.workspace_y_range):
            continue
        return initial_x, initial_y, goal_dx, goal_dy
    raise RuntimeError(
        "could not sample initial/goal XY inside the requested workspace; "
        "widen the workspace or reduce displacement/randomization"
    )


def main() -> None:
    args = parse_args()
    if args.scene_count <= 0:
        raise ValueError("scene-count must be positive")
    if args.stable_obstacle_count < 0:
        raise ValueError("stable-obstacle-count must be non-negative")
    if args.obstacle_support_pose_index < 0:
        raise ValueError("obstacle-support-pose-index must be non-negative")
    if args.obstacle_path_clearance < 0.0:
        raise ValueError("obstacle-path-clearance must be non-negative")
    if args.obstacle_root_clearance < 0.0:
        raise ValueError("obstacle-root-clearance must be non-negative")
    if (
        args.obstacle_path_max_distance is not None
        and args.obstacle_path_max_distance < args.obstacle_path_clearance
    ):
        raise ValueError(
            "obstacle-path-max-distance must be at least obstacle-path-clearance"
        )
    if (args.goal_angle_range is None) != (args.goal_distance_range is None):
        raise ValueError(
            "goal-angle-range and goal-distance-range must be specified together"
        )
    if args.goal_direction_bins <= 0:
        raise ValueError("goal-direction-bins must be positive")
    for name in (
        "obstacle_x_range",
        "obstacle_y_range",
        "workspace_x_range",
        "workspace_y_range",
    ):
        if getattr(args, name) is None:
            continue
        lower, upper = getattr(args, name)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"{name.replace('_', '-')} must satisfy MIN <= MAX")
    if args.goal_angle_range is not None:
        angle_min, angle_max = args.goal_angle_range
        if not math.isfinite(angle_min) or not math.isfinite(angle_max) or angle_min >= angle_max:
            raise ValueError("goal-angle-range must satisfy finite MIN < MAX")
        distance_min, distance_max = args.goal_distance_range
        if not 0.0 < distance_min <= distance_max:
            raise ValueError("goal-distance-range must satisfy 0 < MIN <= MAX")
    for name in (
        "initial_x_jitter",
        "initial_y_jitter",
        "initial_yaw_jitter",
        "goal_dx_jitter",
        "goal_dy_jitter",
        "goal_yaw_jitter",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    for name in ("friction_range", "obstacle_friction_range"):
        bounds = getattr(args, name)
        if bounds is not None and not 0.0 < bounds[0] <= bounds[1]:
            raise ValueError(
                f"{name.replace('_', '-')} must satisfy 0 < MIN <= MAX"
            )
    for name in ("target_mass_scale_range", "obstacle_mass_scale_range"):
        bounds = getattr(args, name)
        if bounds is not None and not 0.0 < bounds[0] <= bounds[1]:
            raise ValueError(
                f"{name.replace('_', '-')} must satisfy 0 < MIN <= MAX"
            )
    if args.obstacle_support_pose_indices is not None:
        if len(args.obstacle_support_pose_indices) != args.stable_obstacle_count:
            raise ValueError(
                "obstacle-support-pose-indices must contain exactly one index "
                "per stable obstacle"
            )
        if any(index < 0 for index in args.obstacle_support_pose_indices):
            raise ValueError("obstacle support indices must be non-negative")
    base = next(load_scene_manifest(args.base_manifest))
    if base.target_object.asset_id != "020_hammer:0":
        raise ValueError("proof manifest requires 020_hammer:0 as target")

    paths = DominoDataPaths.resolve(args.domino_root)
    source = paths.require_source_asset(base.target_object.asset_id)
    stable_poses = stable_poses_from_mesh(
        source.collision_mesh,
        base.target_object.scale,
        max_candidates=max(1, args.support_pose_index + 1),
    )
    if not 0 <= args.support_pose_index < len(stable_poses):
        raise ValueError(
            f"support-pose-index {args.support_pose_index} is outside "
            f"the {len(stable_poses)} available poses"
        )
    # Candidate zero is trimesh's highest-probability support pose.  The old
    # proof copied a random, narrow support face from another manifest; under
    # PhysX it immediately tipped from z=6.69 cm to z=1.29 cm, making the
    # strict height/orientation goal impossible before the robot touched it.
    support = stable_poses[args.support_pose_index]
    obstacles = base.obstacle_objects
    if args.stable_obstacle_count > len(obstacles):
        raise ValueError(
            "stable-obstacle-count exceeds the base manifest obstacle count"
        )
    stable_obstacles = []
    obstacle_support_indices = (
        args.obstacle_support_pose_indices
        if args.obstacle_support_pose_indices is not None
        else [args.obstacle_support_pose_index] * args.stable_obstacle_count
    )
    for slot_index, item in enumerate(obstacles[: args.stable_obstacle_count]):
        support_index = obstacle_support_indices[slot_index]
        obstacle_source = paths.require_source_asset(item.asset_id)
        obstacle_supports = stable_poses_from_mesh(
            obstacle_source.collision_mesh,
            item.scale,
            max_candidates=support_index + 1,
        )
        if support_index >= len(obstacle_supports):
            raise ValueError(
                f"obstacle support index {support_index} is outside "
                f"the {len(obstacle_supports)} poses for {item.asset_id}"
            )
        obstacle_support = obstacle_supports[support_index]
        stable_obstacles.append((item, obstacle_support))
    rng = random.Random(args.seed)
    scenes = []
    initial_poses = []
    goal_poses = []
    for index in range(args.scene_count):
        initial_x, initial_y, goal_dx, goal_dy = _sample_task_planar_pose(
            args, rng, index
        )
        initial_yaw = args.initial_yaw + rng.uniform(
            -args.initial_yaw_jitter, args.initial_yaw_jitter
        )
        goal_yaw_delta = args.goal_yaw_delta + rng.uniform(
            -args.goal_yaw_jitter, args.goal_yaw_jitter
        )

        initial_yaw_quat = (
            math.cos(0.5 * initial_yaw),
            0.0,
            0.0,
            math.sin(0.5 * initial_yaw),
        )
        initial_quat = _quat_multiply(initial_yaw_quat, support.quaternion)
        initial_pose = (
            initial_x,
            initial_y,
            support.support_height,
            *initial_quat,
        )
        half_yaw = 0.5 * goal_yaw_delta
        yaw_quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        goal_quat = _quat_multiply(yaw_quat, initial_quat)
        goal_pose = (
            initial_x + goal_dx,
            initial_y + goal_dy,
            support.support_height,
            *goal_quat,
        )
        obstacle_poses = {}
        placed_obstacle_xy = []
        for obstacle, obstacle_support in stable_obstacles:
            obstacle_xy = None
            for _ in range(1000):
                candidate = (
                    rng.uniform(*args.obstacle_x_range),
                    rng.uniform(*args.obstacle_y_range),
                )
                path_distance = _point_segment_distance(
                    candidate,
                    initial_pose[:2],
                    goal_pose[:2],
                )
                if path_distance < args.obstacle_path_clearance:
                    continue
                if (
                    args.obstacle_path_max_distance is not None
                    and path_distance > args.obstacle_path_max_distance
                ):
                    continue
                if math.dist(candidate, initial_pose[:2]) < args.obstacle_root_clearance:
                    continue
                if math.dist(candidate, goal_pose[:2]) < args.obstacle_root_clearance:
                    continue
                if any(
                    math.dist(candidate, other) < args.obstacle_root_clearance
                    for other in placed_obstacle_xy
                ):
                    continue
                obstacle_xy = candidate
                break
            if obstacle_xy is None:
                raise RuntimeError(
                    f"could not place {obstacle.asset_id} with the requested "
                    "path-distance and root-clearance constraints"
                )
            placed_obstacle_xy.append(obstacle_xy)
            obstacle_yaw = rng.uniform(-math.pi, math.pi)
            obstacle_yaw_quat = (
                math.cos(0.5 * obstacle_yaw),
                0.0,
                0.0,
                math.sin(0.5 * obstacle_yaw),
            )
            obstacle_quat = _quat_multiply(
                obstacle_yaw_quat,
                obstacle_support.quaternion,
            )
            obstacle_poses[obstacle.instance_id] = (
                obstacle_xy[0],
                obstacle_xy[1],
                obstacle_support.support_height,
                *obstacle_quat,
            )
        target_friction = (
            None
            if args.friction_range is None
            else rng.uniform(*args.friction_range)
        )
        target_mass_scale = (
            1.0
            if args.target_mass_scale_range is None
            else rng.uniform(*args.target_mass_scale_range)
        )
        obstacle_randomization = {}
        for instance_id in obstacle_poses:
            obstacle_randomization[instance_id] = {
                "friction": (
                    None
                    if args.obstacle_friction_range is None
                    else rng.uniform(*args.obstacle_friction_range)
                ),
                "mass_scale": (
                    1.0
                    if args.obstacle_mass_scale_range is None
                    else rng.uniform(*args.obstacle_mass_scale_range)
                ),
            }
        objects = []
        for item in base.objects:
            if item.instance_id == base.target_instance_id:
                item = replace(
                    item,
                    pose=initial_pose,
                    mass_kg=item.mass_kg * target_mass_scale,
                    **(
                        {}
                        if target_friction is None
                        else {
                            "static_friction": target_friction,
                            "dynamic_friction": target_friction,
                        }
                    ),
                )
            elif item.instance_id in obstacle_poses:
                randomization = obstacle_randomization[item.instance_id]
                obstacle_friction = randomization["friction"]
                item = replace(
                    item,
                    pose=obstacle_poses[item.instance_id],
                    mass_kg=item.mass_kg * randomization["mass_scale"],
                    **(
                        {}
                        if obstacle_friction is None
                        else {
                            "static_friction": obstacle_friction,
                            "dynamic_friction": obstacle_friction,
                        }
                    ),
                )
            objects.append(item)
        task = ManipulationTask(
            task_id=(
                "joint-pose-directional-randomized"
                if args.goal_angle_range is not None
                else "joint-pose-randomized"
            ),
            target_instance_id=base.target_instance_id,
            initial_pose=initial_pose,
            goal_pose=goal_pose,
        )
        scenes.append(
            ClutterScene(
                scene_id=f"{args.scene_id_prefix}-{index:04d}",
                split=args.split,
                track=base.track,
                objects=tuple(objects),
                tasks=(task,),
            )
        )
        initial_poses.append(initial_pose)
        goal_poses.append(goal_pose)
    write_scene_manifest(args.output, scenes)
    print(
        f"wrote {len(scenes)} controlled hammer proof scenes to {args.output}\n"
        f"first_initial_pose={initial_poses[0]}\n"
        f"first_goal_pose={goal_poses[0]}\n"
        f"support_pose_index={args.support_pose_index} "
        f"support_height={support.support_height:.9f} seed={args.seed}\n"
        f"stable_obstacle_count={args.stable_obstacle_count} "
        f"obstacle_support_pose_indices={obstacle_support_indices}\n"
        f"goal_angle_range={args.goal_angle_range} "
        f"goal_distance_range={args.goal_distance_range} "
        f"goal_direction_bins={args.goal_direction_bins}"
    )


if __name__ == "__main__":
    main()
