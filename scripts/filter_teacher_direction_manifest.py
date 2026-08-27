#!/usr/bin/env python3
"""Filter a teacher manifest by planar goal-displacement direction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from dapl.scene import load_scene_manifest, write_scene_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-absolute-angle-deg", type=float, required=True)
    parser.add_argument("--maximum-absolute-angle-deg", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_absolute_angle_deg <= args.maximum_absolute_angle_deg <= 180.0:
        raise ValueError("angle bounds must satisfy 0 <= minimum <= maximum <= 180")

    scenes = tuple(load_scene_manifest(args.input))
    selected = []
    negative = 0
    positive = 0
    observed_angles = []
    for scene in scenes:
        if not scene.tasks:
            raise ValueError(f"scene {scene.scene_id!r} has no manipulation task")
        task = scene.tasks[0]
        delta_x = task.goal_pose[0] - task.initial_pose[0]
        delta_y = task.goal_pose[1] - task.initial_pose[1]
        angle_deg = math.degrees(math.atan2(delta_y, delta_x))
        absolute_angle = abs(angle_deg)
        if not (
            args.minimum_absolute_angle_deg
            <= absolute_angle
            <= args.maximum_absolute_angle_deg
        ):
            continue
        selected.append(scene)
        observed_angles.append(angle_deg)
        negative += int(angle_deg < 0.0)
        positive += int(angle_deg >= 0.0)

    if not selected:
        raise RuntimeError("direction filter rejected every scene")
    write_scene_manifest(args.output, selected)
    print(
        f"selected {len(selected)}/{len(scenes)} scenes from {args.input}\n"
        f"output={args.output}\n"
        f"angle_range_deg=[{min(observed_angles):.3f}, {max(observed_angles):.3f}] "
        f"negative={negative} positive={positive}"
    )


if __name__ == "__main__":
    main()
