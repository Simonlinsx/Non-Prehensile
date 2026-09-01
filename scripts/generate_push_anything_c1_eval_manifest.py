#!/usr/bin/env python3
"""Generate a deterministic, balanced 50-scene M3 C1 evaluation manifest."""

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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_root
            / "data/manifests/contact_planner_m3"
            / "hammer_c1_front180_eval50_seed20260901.jsonl"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def balanced_values(
    values: tuple[float, ...], count: int, rng: random.Random
) -> list[float]:
    repeated = [values[index % len(values)] for index in range(count)]
    rng.shuffle(repeated)
    return repeated


def build_scenes(count: int, seed: int) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    # Stratification guarantees full front-hemisphere coverage rather than
    # relying on 50 independent samples to happen to cover the edge angles.
    directions = [
        round(-90.0 + (index + 0.5) * 180.0 / count, 1)
        for index in range(count)
    ]
    rng.shuffle(directions)
    initial_x = balanced_values(INITIAL_X_VALUES, count, rng)
    initial_y = balanced_values(INITIAL_Y_VALUES, count, rng)
    distances = balanced_values(GOAL_DISTANCE_VALUES, count, rng)
    yaws = balanced_values(GOAL_YAW_VALUES, count, rng)
    sampling_seeds = rng.sample(range(1000, 1_000_000), count)

    scenes = []
    for index in range(count):
        direction_rad = math.radians(directions[index])
        goal_xy = [
            round(initial_x[index] + distances[index] * math.cos(direction_rad), 6),
            round(initial_y[index] + distances[index] * math.sin(direction_rad), 6),
        ]
        scenes.append(
            {
                "schema": "nonprehensile.push_anything_c1_scene.v1",
                "scene_id": f"scene{index:03d}",
                "manifest_seed": seed,
                "asset_id": "020_hammer:0",
                "support_pose_index": 0,
                "clutter_count": 0,
                "initial_xy_m": [initial_x[index], initial_y[index]],
                "goal_xy_m": goal_xy,
                "goal_distance_m": distances[index],
                "goal_direction_deg": directions[index],
                "goal_yaw_deg": yaws[index],
                "sampling_seed": sampling_seeds[index],
            }
        )
    return scenes


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")
    scenes = build_scenes(args.count, args.seed)
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
                "direction_range_deg": [
                    min(scene["goal_direction_deg"] for scene in scenes),
                    max(scene["goal_direction_deg"] for scene in scenes),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
