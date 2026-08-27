#!/usr/bin/env python3
"""Generate typed C1/C2/C3/combined hammer diagnostic manifests."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random

from dapl.catalog import stable_poses_from_mesh
from dapl.domino import DominoDataPaths, load_domino_affordance_annotation
from dapl.scene import (
    ClutterScene,
    ManipulationTask,
    SceneObject,
    load_scene_manifest,
    write_scene_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domino-root", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1817)
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--goal-distance", type=float, default=0.22)
    parser.add_argument(
        "--c2-asset-id",
        default="077_phone:0",
        help=(
            "DOMINO blocker used for the protected-part sweep. Assets absent "
            "from the base manifest require --c2-scale."
        ),
    )
    parser.add_argument(
        "--c2-scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Scale for a C2 asset that is absent from the base manifest.",
    )
    parser.add_argument(
        "--c2-mass-kg",
        type=float,
        default=0.05,
        help="Mass for a C2 asset that is absent from the base manifest.",
    )
    parser.add_argument("--c2-stable-pose-index", type=int, default=0)
    parser.add_argument("--c2-lateral-offset", type=float, default=0.04)
    parser.add_argument("--c3-fraction", type=float, default=0.50)
    parser.add_argument("--c3-lateral-offset", type=float, default=0.035)
    return parser.parse_args()


def _quat_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    value = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    norm = math.sqrt(sum(item * item for item in value))
    return tuple(item / norm for item in value)


def _quat_rotate(quat: tuple[float, ...], vector: tuple[float, ...]) -> tuple[float, ...]:
    w, qx, qy, qz = quat
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def _yaw_quat(yaw: float) -> tuple[float, ...]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _stable_pose(paths: DominoDataPaths, item, index: int):
    asset = paths.require_source_asset(item.asset_id)
    poses = stable_poses_from_mesh(
        asset.collision_mesh,
        item.scale,
        max_candidates=index + 1,
    )
    if index >= len(poses):
        raise ValueError(f"{item.asset_id} has no stable pose {index}")
    return poses[index]


def _pose_on_support(
    support,
    x: float,
    y: float,
    yaw: float,
) -> tuple[float, ...]:
    quat = _quat_multiply(_yaw_quat(yaw), support.quaternion)
    return (x, y, support.support_height, *quat)


def _replace_pose(item, pose: tuple[float, ...], rng: random.Random):
    friction = rng.uniform(0.6, 1.0)
    mass_scale = rng.uniform(0.85, 1.15)
    return replace(
        item,
        pose=pose,
        mass_kg=item.mass_kg * mass_scale,
        static_friction=friction,
        dynamic_friction=friction,
    )


def _world_anchor(root_pose: tuple[float, ...], local: tuple[float, ...]) -> tuple[float, ...]:
    offset = _quat_rotate(root_pose[3:7], local)
    return tuple(root_pose[index] + offset[index] for index in range(3))


def _barrier_pose(
    support,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    fraction: float,
    lateral_offset: float,
    side: float,
) -> tuple[float, ...]:
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    length = math.hypot(dx, dy)
    if length < 1.0e-6:
        raise ValueError("barrier path must have non-zero length")
    ux, uy = dx / length, dy / length
    x = start_xy[0] + fraction * dx - side * uy * lateral_offset
    y = start_xy[1] + fraction * dy + side * ux * lateral_offset
    # Align the obstacle's long support axis across the nominal path.
    yaw = math.atan2(dy, dx) + 0.5 * math.pi
    return _pose_on_support(support, x, y, yaw)


def main() -> None:
    args = parse_args()
    if args.scene_count <= 0:
        raise ValueError("scene-count must be positive")
    if args.goal_distance < 0.12:
        raise ValueError("goal-distance must be at least 0.12 m for endpoint clearance")
    if not 0.0 <= args.c2_lateral_offset <= 0.08:
        raise ValueError("--c2-lateral-offset must be in [0, 0.08] m")
    if args.c2_stable_pose_index < 0:
        raise ValueError("--c2-stable-pose-index must be non-negative")
    if args.c2_mass_kg <= 0.0:
        raise ValueError("--c2-mass-kg must be positive")
    if args.c2_scale is not None and any(value <= 0.0 for value in args.c2_scale):
        raise ValueError("--c2-scale components must be positive")
    if not 0.2 <= args.c3_fraction <= 0.8:
        raise ValueError("--c3-fraction must be in [0.2, 0.8]")
    if not 0.0 <= args.c3_lateral_offset <= 0.08:
        raise ValueError("--c3-lateral-offset must be in [0, 0.08] m")

    base = next(load_scene_manifest(args.base_manifest))
    if base.target_object.asset_id != "020_hammer:0":
        raise ValueError("diagnostics require 020_hammer:0 as target")
    by_asset = {item.asset_id: item for item in base.obstacle_objects}
    required = ("039_mug:0", "084_woodenmallet:0")
    missing = [asset_id for asset_id in required if asset_id not in by_asset]
    if missing:
        raise ValueError(f"base manifest lacks diagnostic assets: {missing}")

    paths = DominoDataPaths.resolve(args.domino_root)
    target_item = base.target_object
    c2_item = by_asset.get(args.c2_asset_id)
    if c2_item is None:
        if args.c2_scale is None:
            raise ValueError(
                f"base manifest lacks C2 asset {args.c2_asset_id!r}; pass --c2-scale"
            )
        # The actual support pose replaces this placeholder before writing.
        c2_item = SceneObject(
            instance_id="c2-blocker-00",
            asset_id=args.c2_asset_id,
            pose=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            scale=tuple(args.c2_scale),
            mass_kg=args.c2_mass_kg,
        )
    mug_item = by_asset["039_mug:0"]
    mallet_item = by_asset["084_woodenmallet:0"]
    target_support = _stable_pose(paths, target_item, 0)
    c2_support = _stable_pose(paths, c2_item, args.c2_stable_pose_index)
    mug_support = _stable_pose(paths, mug_item, 3)
    mallet_support = _stable_pose(paths, mallet_item, 2)

    annotation = load_domino_affordance_annotation(
        paths.require_source_asset(target_item.asset_id)
    )
    functional_local = annotation.functional_anchors[0].scaled_position(
        annotation.scale
    )
    rng = random.Random(args.seed)
    profiles: dict[str, list[ClutterScene]] = {
        "t0_c1": [],
        "c2": [],
        "c3": [],
        "combined": [],
    }
    direction_centers = (-math.pi, -0.5 * math.pi, 0.0, 0.5 * math.pi)

    for index in range(args.scene_count):
        direction = direction_centers[index % len(direction_centers)] + rng.uniform(
            -0.18, 0.18
        )
        initial_x = rng.uniform(0.47, 0.54)
        # Keep the target away from the initial TCP's y=0 projection so C3
        # has a real, visible approach corridor.
        initial_y = (1.0 if (index // 4) % 2 == 0 else -1.0) * rng.uniform(
            0.16, 0.21
        )
        initial_yaw = rng.uniform(-math.pi, math.pi)
        # Typed diagnostics isolate the safety mechanism.  Keep the goal yaw
        # fixed so the C2 blocker challenges only the protected-part sweep;
        # randomized-yaw competence is measured on the 256-scene train/eval
        # manifest instead of being confounded with endpoint feasibility.
        goal_yaw_delta = 0.0
        initial_pose = _pose_on_support(
            target_support, initial_x, initial_y, initial_yaw
        )
        goal_x = initial_x + args.goal_distance * math.cos(direction)
        goal_y = initial_y + args.goal_distance * math.sin(direction)
        if not (0.30 <= goal_x <= 0.72 and -0.38 <= goal_y <= 0.38):
            # Reflect the displacement into the reachable table workspace.
            direction += math.pi
            goal_x = initial_x + args.goal_distance * math.cos(direction)
            goal_y = initial_y + args.goal_distance * math.sin(direction)
        goal_quat = _quat_multiply(_yaw_quat(goal_yaw_delta), initial_pose[3:7])
        goal_pose = (
            goal_x,
            goal_y,
            target_support.support_height,
            *goal_quat,
        )

        functional_start = _world_anchor(initial_pose, functional_local)
        functional_goal = _world_anchor(goal_pose, functional_local)
        side = -1.0 if index % 2 else 1.0
        c2_pose = _barrier_pose(
            c2_support,
            functional_start[:2],
            functional_goal[:2],
            fraction=0.5,
            lateral_offset=args.c2_lateral_offset,
            side=side,
        )
        c3_pose = _barrier_pose(
            mug_support,
            (0.5198, 0.0),
            initial_pose[:2],
            fraction=args.c3_fraction,
            lateral_offset=args.c3_lateral_offset,
            side=side,
        )
        parked_mallet = _pose_on_support(
            mallet_support, 0.82, 0.48 if side > 0 else -0.48, 0.0
        )

        target = _replace_pose(target_item, initial_pose, rng)
        c2_blocker = _replace_pose(c2_item, c2_pose, rng)
        mug = _replace_pose(mug_item, c3_pose, rng)
        mallet = _replace_pose(mallet_item, parked_mallet, rng)
        task = ManipulationTask(
            task_id="teacher-typed-diagnostic",
            target_instance_id=target.instance_id,
            initial_pose=initial_pose,
            goal_pose=goal_pose,
        )
        object_orders = {
            "t0_c1": (target, c2_blocker, mug, mallet),
            "c2": (target, c2_blocker, mug, mallet),
            "c3": (target, mug, c2_blocker, mallet),
            "combined": (target, c2_blocker, mug, mallet),
        }
        for profile, objects in object_orders.items():
            profiles[profile].append(
                ClutterScene(
                    scene_id=f"teacher-{profile}-{index:04d}",
                    split=args.split,
                    track=base.track,
                    objects=objects,
                    tasks=(task,),
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for profile, scenes in profiles.items():
        path = args.output_dir / f"hammer_teacher_{profile}_{args.scene_count}_seed{args.seed}.jsonl"
        write_scene_manifest(path, scenes)
        outputs[profile] = str(path)
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "split": args.split,
        "scene_count": args.scene_count,
        "goal_distance_m": args.goal_distance,
        "c2_asset_id": args.c2_asset_id,
        "c2_scale": list(c2_item.scale),
        "c2_mass_kg": c2_item.mass_kg,
        "c2_stable_pose_index": args.c2_stable_pose_index,
        "c2_lateral_offset_m": args.c2_lateral_offset,
        "c3_fraction": args.c3_fraction,
        "c3_lateral_offset_m": args.c3_lateral_offset,
        "profiles": outputs,
        "c2_design": (
            f"{args.c2_asset_id} crosses the protected-head straight-line sweep"
        ),
        "c3_design": "upright mug blocks the initial TCP-to-safe-root corridor",
        "combined_design": f"C2 {args.c2_asset_id} plus C3 mug",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    for profile, path in outputs.items():
        print(f"{profile}={path}")


if __name__ == "__main__":
    main()
