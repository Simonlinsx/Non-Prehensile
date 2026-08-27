#!/usr/bin/env python3
"""Make teacher goal-yaw signs balanced without changing scene physics.

The input manifests historically sampled a negative yaw delta plus jitter.
This tool preserves every initial pose, goal XYZ, material, mass, support face,
and obstacle placement.  It changes only the goal quaternion and stratifies
positive/negative yaw signs inside planar goal-direction bins, so the existing
reset/settling audit remains applicable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--direction-bins", type=int, default=8)
    parser.add_argument("--angle-range-deg", type=float, nargs=2, default=(-90.0, 90.0))
    return parser.parse_args()


def _quat_conjugate(quaternion: list[float]) -> tuple[float, float, float, float]:
    return (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])


def _quat_mul(left, right) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _yaw_from_quat(quaternion) -> float:
    w, x, y, z = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _direction_bin(angle_deg: float, minimum: float, maximum: float, bins: int) -> int:
    normalized = (angle_deg - minimum) / (maximum - minimum)
    return min(bins - 1, max(0, int(math.floor(normalized * bins))))


def main() -> None:
    args = parse_args()
    angle_min, angle_max = args.angle_range_deg
    if args.direction_bins <= 0:
        raise ValueError("direction-bins must be positive")
    if not angle_min < angle_max:
        raise ValueError("angle-range-deg must satisfy MIN < MAX")

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[int, list[int]] = {index: [] for index in range(args.direction_bins)}
    magnitudes: list[float] = []
    directions: list[float] = []
    for index, row in enumerate(rows):
        task = row["tasks"][0]
        initial = task["initial_pose"]
        goal = task["goal_pose"]
        direction = math.degrees(math.atan2(goal[1] - initial[1], goal[0] - initial[0]))
        if not angle_min - 1.0e-6 <= direction <= angle_max + 1.0e-6:
            raise ValueError(
                f"scene {row['scene_id']!r} direction {direction:.3f} is outside "
                f"[{angle_min}, {angle_max}]"
            )
        relative = _quat_mul(goal[3:7], _quat_conjugate(initial[3:7]))
        magnitude = abs(_yaw_from_quat(relative))
        if magnitude <= 1.0e-6:
            raise ValueError(f"scene {row['scene_id']!r} has zero goal-yaw magnitude")
        grouped[_direction_bin(direction, angle_min, angle_max, args.direction_bins)].append(index)
        magnitudes.append(magnitude)
        directions.append(direction)

    rng = random.Random(args.seed)
    assigned_sign = [0] * len(rows)
    bin_report = {}
    for bin_index, indices in grouped.items():
        rng.shuffle(indices)
        for local_index, scene_index in enumerate(indices):
            assigned_sign[scene_index] = 1 if local_index % 2 == 0 else -1
        positive = sum(assigned_sign[index] > 0 for index in indices)
        bin_report[str(bin_index)] = {
            "count": len(indices),
            "negative": len(indices) - positive,
            "positive": positive,
        }

    for index, row in enumerate(rows):
        task = row["tasks"][0]
        initial_quaternion = task["initial_pose"][3:7]
        yaw_delta = assigned_sign[index] * magnitudes[index]
        yaw_quaternion = (
            math.cos(0.5 * yaw_delta),
            0.0,
            0.0,
            math.sin(0.5 * yaw_delta),
        )
        task["goal_pose"][3:7] = list(_quat_mul(yaw_quaternion, initial_quaternion))
        task["task_id"] = "joint-pose-directional-bidirectional-yaw"
        row["scene_id"] = f"{row['scene_id']}-biyaw-{index:04d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "scene_count": len(rows),
        "direction_bins": args.direction_bins,
        "angle_range_deg": [angle_min, angle_max],
        "direction_observed_deg": [min(directions), max(directions)],
        "yaw_magnitude_rad": [min(magnitudes), max(magnitudes)],
        "negative_yaw_count": sum(sign < 0 for sign in assigned_sign),
        "positive_yaw_count": sum(sign > 0 for sign in assigned_sign),
        "per_direction_bin": bin_report,
        "preserved_fields": [
            "initial_pose",
            "goal_xyz",
            "objects",
            "materials",
            "masses",
            "support_face",
            "obstacles",
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.output)
    print(summary_path)


if __name__ == "__main__":
    main()
