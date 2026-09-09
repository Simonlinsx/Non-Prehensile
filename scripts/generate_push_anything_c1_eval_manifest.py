#!/usr/bin/env python3
"""Generate a deterministic, balanced outward-push M3 C1 manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


INITIAL_X_VALUES = (0.39, 0.40, 0.41)
INITIAL_Y_VALUES = (0.18, 0.19, 0.20, 0.21, 0.22)
GOAL_DISTANCE_VALUES = (0.06, 0.07, 0.08, 0.09, 0.10)
GOAL_YAW_VALUES = (-10.0, -5.0, 0.0, 5.0, 10.0)
ROBOT_BASE_XY_M = (0.0, 0.0)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--relative-direction-limit-deg",
        type=float,
        default=90.0,
        help="symmetric goal-direction limit around the base-to-object ray",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_root
            / "data/manifests/contact_planner_m3"
            / "hammer_c1_outward180_eval50_seed20260902.jsonl"
        ),
    )
    parser.add_argument("--randomize-initial-yaw", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def balanced_values(
    values: tuple[float, ...], count: int, rng: random.Random
) -> list[float]:
    repeated = [values[index % len(values)] for index in range(count)]
    rng.shuffle(repeated)
    return repeated


def build_scenes(
    count: int, seed: int, relative_direction_limit_deg: float = 90.0,
    randomize_initial_yaw: bool = False,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 < relative_direction_limit_deg <= 90.0:
        raise ValueError("relative direction limit must be in (0, 90]")
    rng = random.Random(seed)
    # Stratification guarantees full front-hemisphere coverage rather than
    # relying on 50 independent samples to happen to cover the edge angles.
    relative_directions = [
        round(
            -relative_direction_limit_deg
            + (index + 0.5) * 2.0 * relative_direction_limit_deg / count,
            1,
        )
        for index in range(count)
    ]
    rng.shuffle(relative_directions)
    initial_x = balanced_values(INITIAL_X_VALUES, count, rng)
    initial_y = balanced_values(INITIAL_Y_VALUES, count, rng)
    distances = balanced_values(GOAL_DISTANCE_VALUES, count, rng)
    yaws = balanced_values(GOAL_YAW_VALUES, count, rng)
    sampling_seeds = rng.sample(range(1000, 1_000_000), count)

    initial_yaws = [-180.0 + (i + .5) * 360.0 / count for i in range(count)]
    if randomize_initial_yaw:
        rng.shuffle(initial_yaws)
    scenes = []
    for index in range(count):
        reference_direction_deg = math.degrees(
            math.atan2(
                initial_y[index] - ROBOT_BASE_XY_M[1],
                initial_x[index] - ROBOT_BASE_XY_M[0],
            )
        )
        world_direction_deg = reference_direction_deg + relative_directions[index]
        direction_rad = math.radians(world_direction_deg)
        goal_xy = [
            round(initial_x[index] + distances[index] * math.cos(direction_rad), 6),
            round(initial_y[index] + distances[index] * math.sin(direction_rad), 6),
        ]
        scenes.append(
            {
                "schema": "nonprehensile.push_anything_c1_scene.v2",
                "scene_id": f"scene{index:03d}",
                "manifest_seed": seed,
                "asset_id": "020_hammer:0",
                "support_pose_index": 0,
                "clutter_count": 0,
                "initial_xy_m": [initial_x[index], initial_y[index]],
                "goal_xy_m": goal_xy,
                "goal_distance_m": distances[index],
                "goal_direction_deg": round(world_direction_deg, 6),
                "goal_direction_relative_deg": relative_directions[index],
                "goal_direction_relative_limit_deg": (
                    relative_direction_limit_deg
                ),
                "goal_direction_reference_deg": round(reference_direction_deg, 6),
                "goal_direction_frame": "robot_base_to_initial_target",
                "robot_base_xy_m": list(ROBOT_BASE_XY_M),
                "goal_yaw_deg": yaws[index],
                "sampling_seed": sampling_seeds[index],
            }
        )
    if randomize_initial_yaw:
        for scene, yaw in zip(scenes, initial_yaws):
            scene["initial_yaw_deg"] = round(yaw, 6)
            scene["goal_yaw_delta_deg"] = scene["goal_yaw_deg"]
            scene["goal_yaw_deg"] = round((yaw + scene["goal_yaw_delta_deg"] + 180) % 360 - 180, 6)
    return scenes


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")
    scenes = build_scenes(args.count, args.seed, args.relative_direction_limit_deg, args.randomize_initial_yaw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for scene in scenes:
            stream.write(json.dumps(scene, sort_keys=True) + "\n")
    print(args.output)
    print(
        json.dumps(
            {
                "count": len(scenes),
                "seed": args.seed,
                "relative_direction_range_deg": [
                    min(scene["goal_direction_relative_deg"] for scene in scenes),
                    max(scene["goal_direction_relative_deg"] for scene in scenes),
                ],
                "relative_direction_limit_deg": args.relative_direction_limit_deg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
