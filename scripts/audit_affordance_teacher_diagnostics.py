#!/usr/bin/env python3
"""Audit that stable teacher scenes contain the intended typed challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

from dapl.domino import (
    DominoDataPaths,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
)
from dapl.scene import load_scene_manifest, write_scene_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domino-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("c2", "c3", "combined"), required=True)
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1817)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filtered-manifest-output", type=Path, default=None)
    parser.add_argument(
        "--c2-endpoint-clearance-m",
        type=float,
        default=0.005,
        help="Required whole-target and protected-region clearance at start/goal.",
    )
    parser.add_argument(
        "--c2-midpoint-blocked-distance-m",
        type=float,
        default=0.005,
        help="Maximum protected-region distance at the straight-path midpoint.",
    )
    parser.add_argument(
        "--c2-midpoint-clearance-m",
        type=float,
        default=None,
        help=(
            "If set, audit a matched non-conflicting C2 control by requiring "
            "at least this protected-region clearance at the straight-path "
            "midpoint instead of requiring a blocked midpoint."
        ),
    )
    parser.add_argument(
        "--c2-path-clearance-m",
        type=float,
        default=None,
        help=(
            "For a matched non-conflicting C2 control, require both the "
            "protected region and complete target to remain at least this far "
            "from the blocker over the sampled interpolated-pose sweep."
        ),
    )
    parser.add_argument(
        "--c2-path-samples",
        type=int,
        default=21,
        help="Number of uniformly spaced poses used by --c2-path-clearance-m.",
    )
    parser.add_argument(
        "--max-valid-scenes",
        type=int,
        default=None,
        help="Optionally cap the deterministic filtered subset after auditing.",
    )
    return parser.parse_args()


def _quat_matrix(quat: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(points: np.ndarray, pose: tuple[float, ...]) -> np.ndarray:
    return points @ _quat_matrix(pose[3:7]).T + np.asarray(pose[:3])


def _interpolate_pose(start: tuple[float, ...], goal: tuple[float, ...], alpha: float):
    position = (1.0 - alpha) * np.asarray(start[:3]) + alpha * np.asarray(goal[:3])
    left = np.asarray(start[3:7], dtype=np.float64)
    right = np.asarray(goal[3:7], dtype=np.float64)
    if np.dot(left, right) < 0.0:
        right = -right
    quat = (1.0 - alpha) * left + alpha * right
    quat /= np.linalg.norm(quat)
    return (*position.tolist(), *quat.tolist())


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _sample_asset(paths, item, count: int, rng: np.random.Generator) -> np.ndarray:
    asset = paths.require_source_asset(item.asset_id)
    scene = trimesh.load(asset.collision_mesh, force="scene")
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    # trimesh's sampler uses NumPy's global generator.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    points, _ = trimesh.sample.sample_surface(mesh, count)
    return points * np.asarray(item.scale, dtype=np.float64)


def _minimum_distance(left: np.ndarray, right: np.ndarray) -> float:
    # These clouds are deliberately small.  Spawning a full thread pool for
    # every scene/query was substantially slower than a single query worker
    # and made large rejection-sampling audits unnecessarily expensive.
    return float(cKDTree(right).query(left, k=1, workers=1)[0].min())


def main() -> None:
    args = parse_args()
    if args.sample_count < 1024:
        raise ValueError("sample-count must be at least 1024")
    if args.max_valid_scenes is not None and args.max_valid_scenes <= 0:
        raise ValueError("--max-valid-scenes must be positive")
    if args.c2_endpoint_clearance_m < 0.0:
        raise ValueError("--c2-endpoint-clearance-m must be non-negative")
    if args.c2_midpoint_blocked_distance_m < 0.0:
        raise ValueError("--c2-midpoint-blocked-distance-m must be non-negative")
    if (
        args.c2_midpoint_clearance_m is not None
        and args.c2_midpoint_clearance_m < 0.0
    ):
        raise ValueError("--c2-midpoint-clearance-m must be non-negative")
    if args.c2_path_clearance_m is not None and args.c2_path_clearance_m < 0.0:
        raise ValueError("--c2-path-clearance-m must be non-negative")
    if args.c2_path_samples < 3:
        raise ValueError("--c2-path-samples must be at least 3")
    scenes = tuple(load_scene_manifest(args.manifest))
    if not scenes:
        raise ValueError("manifest is empty")
    paths = DominoDataPaths.resolve(args.domino_root)
    rng = np.random.default_rng(args.seed)
    sampled_asset_cache: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}

    def sampled_asset(item) -> np.ndarray:
        key = (item.asset_id, tuple(float(value) for value in item.scale))
        points = sampled_asset_cache.get(key)
        if points is None:
            points = _sample_asset(paths, item, args.sample_count, rng)
            sampled_asset_cache[key] = points
        return points

    target_item = scenes[0].target_object
    target_asset = paths.require_source_asset(target_item.asset_id)
    annotation = load_domino_affordance_annotation(target_asset)
    # Sample in raw coordinates for canonical mask evaluation, then scale.
    raw_scene = trimesh.load(target_asset.collision_mesh, force="scene")
    raw_mesh = raw_scene.to_geometry() if isinstance(raw_scene, trimesh.Scene) else raw_scene
    np.random.seed(args.seed)
    raw_target, _ = trimesh.sample.sample_surface(raw_mesh, args.sample_count)
    features = domino_point_affordance_features(
        torch.from_numpy(raw_target).to(torch.float32), annotation
    ).numpy()
    target_points = raw_target * np.asarray(target_item.scale, dtype=np.float64)
    protected_points = target_points[features[:, 1] >= 0.25]

    c2_start: list[float] = []
    c2_mid: list[float] = []
    c2_goal: list[float] = []
    c2_target_start: list[float] = []
    c2_target_goal: list[float] = []
    c2_protected_sweep_min: list[float] = []
    c2_target_sweep_min: list[float] = []
    c3_corridor: list[float] = []
    c3_root_to_segment: list[float] = []
    c3_tcp_surface: list[float] = []
    c3_target_start: list[float] = []
    c3_root_to_tcp: list[float] = []
    c3_root_to_target: list[float] = []
    combined_obstacle_clearance: list[float] = []
    tcp = np.asarray((0.5198, 0.0, 0.2763), dtype=np.float64)

    for scene in scenes:
        task = scene.tasks[0]
        initial = task.initial_pose
        goal = task.goal_pose
        obstacle_index = 0 if args.profile in ("c2", "combined") else 0
        obstacle = scene.obstacle_objects[obstacle_index]
        obstacle_points = sampled_asset(obstacle)
        obstacle_world = _transform(obstacle_points, obstacle.pose)

        if args.profile in ("c2", "combined"):
            for alpha, destination in (
                (0.0, c2_start),
                (0.5, c2_mid),
                (1.0, c2_goal),
            ):
                protected_world = _transform(
                    protected_points, _interpolate_pose(initial, goal, alpha)
                )
                destination.append(_minimum_distance(protected_world, obstacle_world))
            c2_target_start.append(
                _minimum_distance(_transform(target_points, initial), obstacle_world)
            )
            c2_target_goal.append(
                _minimum_distance(_transform(target_points, goal), obstacle_world)
            )
            if args.c2_path_clearance_m is not None:
                protected_sweep_distances = []
                target_sweep_distances = []
                for alpha in np.linspace(0.0, 1.0, args.c2_path_samples):
                    pose = _interpolate_pose(initial, goal, float(alpha))
                    protected_sweep_distances.append(
                        _minimum_distance(
                            _transform(protected_points, pose), obstacle_world
                        )
                    )
                    target_sweep_distances.append(
                        _minimum_distance(
                            _transform(target_points, pose), obstacle_world
                        )
                    )
                c2_protected_sweep_min.append(min(protected_sweep_distances))
                c2_target_sweep_min.append(min(target_sweep_distances))

        if args.profile in ("c3", "combined"):
            c3_index = 1 if args.profile == "combined" else 0
            c3_obstacle = scene.obstacle_objects[c3_index]
            c3_points = sampled_asset(c3_obstacle)
            c3_world = _transform(c3_points, c3_obstacle.pose)
            target_contact = np.asarray(initial[:3], dtype=np.float64)
            line = np.linspace(tcp, target_contact, 96)
            c3_corridor.append(_minimum_distance(line, c3_world))
            c3_tcp_surface.append(_minimum_distance(tcp[None], c3_world))
            c3_target_start.append(
                _minimum_distance(_transform(target_points, initial), c3_world)
            )
            c3_root_to_tcp.append(
                float(np.linalg.norm(np.asarray(c3_obstacle.pose[:2]) - tcp[:2]))
            )
            c3_root_to_target.append(
                float(
                    np.linalg.norm(
                        np.asarray(c3_obstacle.pose[:2])
                        - np.asarray(initial[:2])
                    )
                )
            )
            delta = target_contact[:2] - tcp[:2]
            projection = np.dot(np.asarray(c3_obstacle.pose[:2]) - tcp[:2], delta)
            projection /= max(np.dot(delta, delta), 1.0e-12)
            projection = float(np.clip(projection, 0.0, 1.0))
            closest = tcp[:2] + projection * delta
            c3_root_to_segment.append(
                float(np.linalg.norm(np.asarray(c3_obstacle.pose[:2]) - closest))
            )
            if args.profile == "combined":
                combined_obstacle_clearance.append(
                    _minimum_distance(obstacle_world, c3_world)
                )

    report: dict[str, object] = {
        "manifest": str(args.manifest.resolve()),
        "profile": args.profile,
        "scene_count": len(scenes),
        "sample_count": args.sample_count,
    }
    if c2_mid:
        require_clear_midpoint = args.c2_midpoint_clearance_m is not None
        require_clear_sweep = args.c2_path_clearance_m is not None
        c2_feasible = []
        for index, (
            protected_start,
            middle,
            protected_goal,
            target_start,
            target_goal,
        ) in enumerate(
            zip(c2_start, c2_mid, c2_goal, c2_target_start, c2_target_goal)
        ):
            midpoint_valid = (
                middle >= args.c2_midpoint_clearance_m
                if require_clear_midpoint
                else middle <= args.c2_midpoint_blocked_distance_m
            )
            sweep_valid = (
                not require_clear_sweep
                or (
                    c2_protected_sweep_min[index] >= args.c2_path_clearance_m
                    and c2_target_sweep_min[index] >= args.c2_path_clearance_m
                )
            )
            c2_feasible.append(
                protected_start > args.c2_endpoint_clearance_m
                and protected_goal > args.c2_endpoint_clearance_m
                and target_start > args.c2_endpoint_clearance_m
                and target_goal > args.c2_endpoint_clearance_m
                and midpoint_valid
                and sweep_valid
            )
        report["c2_protected_to_blocker_distance_m"] = {
            "audit_mode": (
                "midpoint_clear" if require_clear_midpoint else "midpoint_blocked"
            ),
            "endpoint_clearance_threshold": args.c2_endpoint_clearance_m,
            "midpoint_blocked_distance_threshold": (
                args.c2_midpoint_blocked_distance_m
            ),
            "midpoint_clearance_threshold": args.c2_midpoint_clearance_m,
            "path_clearance_threshold": args.c2_path_clearance_m,
            "path_samples": (
                args.c2_path_samples if require_clear_sweep else None
            ),
            "start": _summary(c2_start),
            "straight_path_midpoint": _summary(c2_mid),
            "goal": _summary(c2_goal),
            "whole_target_start": _summary(c2_target_start),
            "whole_target_goal": _summary(c2_target_goal),
            "interpolated_pose_sweep_protected_min": (
                _summary(c2_protected_sweep_min)
                if require_clear_sweep
                else None
            ),
            "interpolated_pose_sweep_whole_target_min": (
                _summary(c2_target_sweep_min)
                if require_clear_sweep
                else None
            ),
            "midpoint_blocked_fraction": float(
                np.mean(
                    np.asarray(c2_mid)
                    <= args.c2_midpoint_blocked_distance_m
                )
            ),
            "midpoint_clear_fraction": (
                float(
                    np.mean(
                        np.asarray(c2_mid) >= args.c2_midpoint_clearance_m
                    )
                )
                if require_clear_midpoint
                else None
            ),
            "full_sweep_clear_fraction": (
                float(
                    np.mean(
                        (
                            np.asarray(c2_protected_sweep_min)
                            >= args.c2_path_clearance_m
                        )
                        & (
                            np.asarray(c2_target_sweep_min)
                            >= args.c2_path_clearance_m
                        )
                    )
                )
                if require_clear_sweep
                else None
            ),
            (
                "endpoint_clear_and_midpoint_clear_fraction"
                if require_clear_midpoint
                else "endpoint_clear_and_midpoint_blocked_fraction"
            ): float(np.mean(c2_feasible)),
        }
    if c3_corridor:
        c3_feasible = [
            0.02 <= root_distance <= 0.05
            and tcp_surface > 0.035
            and target_clearance > 0.005
            and root_to_tcp > 0.075
            and root_to_target > 0.075
            for root_distance, tcp_surface, target_clearance, root_to_tcp, root_to_target in zip(
                c3_root_to_segment,
                c3_tcp_surface,
                c3_target_start,
                c3_root_to_tcp,
                c3_root_to_target,
            )
        ]
        report["c3_tcp_to_safe_root_corridor"] = {
            "surface_distance_m": _summary(c3_corridor),
            "obstacle_root_to_segment_m": _summary(c3_root_to_segment),
            "tcp_to_obstacle_surface_m": _summary(c3_tcp_surface),
            "target_start_to_obstacle_surface_m": _summary(c3_target_start),
            "obstacle_root_to_tcp_m": _summary(c3_root_to_tcp),
            "obstacle_root_to_target_m": _summary(c3_root_to_target),
            "reset_clear_and_direct_corridor_blocked_fraction": float(
                np.mean(c3_feasible)
            ),
        }
    combined_feasible = [clearance > 0.005 for clearance in combined_obstacle_clearance]
    if combined_obstacle_clearance:
        report["combined_obstacle_pair_clearance_m"] = {
            **_summary(combined_obstacle_clearance),
            "clear_fraction": float(np.mean(combined_feasible)),
        }
    valid_indices = []
    for index in range(len(scenes)):
        c2_valid = not c2_mid or c2_feasible[index]
        c3_valid = not c3_corridor or c3_feasible[index]
        combined_valid = not combined_feasible or combined_feasible[index]
        if c2_valid and c3_valid and combined_valid:
            valid_indices.append(index)
    report["candidate_valid_scene_count"] = len(valid_indices)
    if args.max_valid_scenes is not None:
        valid_indices = valid_indices[: args.max_valid_scenes]
    report["valid_scene_indices"] = valid_indices
    report["valid_scene_count"] = len(valid_indices)
    if args.filtered_manifest_output is not None:
        if not valid_indices:
            raise RuntimeError("geometry audit rejected every diagnostic scene")
        write_scene_manifest(
            args.filtered_manifest_output,
            tuple(scenes[index] for index in valid_indices),
        )
        report["filtered_manifest_output"] = str(
            args.filtered_manifest_output
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
