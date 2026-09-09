#!/usr/bin/env python3
"""Summarize directional failure modes from a Push Anything C1 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any


UNSAFE_REJECTION_RE = re.compile(
    r"Previous repositioning target rejected by semantic C1 guard; "
    r"unsafe surface distance: ([0-9.eE+-]+) m"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--max-direction-deg",
        type=float,
        help="optionally keep only scenes whose goal direction is below this value",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def c3_episode_lengths(rows: list[dict[str, str]]) -> list[int]:
    lengths: list[int] = []
    current = 0
    for row in rows:
        if int(row["is_c3_mode"]):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def longest_joint_gate_run(
    rows: list[dict[str, str]],
    position_threshold_m: float,
    rotation_threshold_rad: float,
) -> int:
    longest = 0
    current = 0
    for row in rows:
        within = (
            float(row["position_error_m"]) < position_threshold_m
            and float(row["rotation_error_rad"]) < rotation_threshold_rad
        )
        current = current + 1 if within else 0
        longest = max(longest, current)
    return longest


def classify(record: dict[str, Any]) -> str:
    if record["accepted"]:
        return "accepted"
    if record["previous_target_semantic_rejections"] >= 100:
        return "semantic_guard_reposition_deadlock"
    if (
        record["best_position_error_m"] is not None
        and record["best_position_error_m"] <= 0.04
    ):
        return "near_position_gate_or_pose_coupling"
    return "insufficient_goal_progress"


def analyze_scene(scene_dir: Path, clearance_m: float) -> dict[str, Any]:
    scene = load_json(scene_dir / "scene_result.json")
    acceptance = load_json(scene_dir / "acceptance.json")
    with (scene_dir / "sampling_c3_debug.csv").open(encoding="utf-8") as stream:
        trajectory = [row for row in csv.DictReader(stream) if row["object_x_m"]]
    if not trajectory:
        raise ValueError(f"trajectory has no measured object poses: {scene_dir}")

    initial_xy = tuple(float(value) for value in scene["initial_xy_m"])
    goal_xy = tuple(float(value) for value in scene["goal_xy_m"])
    goal_distance = float(scene["goal_distance_m"])
    goal_unit = (
        (goal_xy[0] - initial_xy[0]) / goal_distance,
        (goal_xy[1] - initial_xy[1]) / goal_distance,
    )
    lateral_unit = (-goal_unit[1], goal_unit[0])
    measured_start = (
        float(trajectory[0]["object_x_m"]),
        float(trajectory[0]["object_y_m"]),
    )
    progress: list[float] = []
    cross_track: list[float] = []
    for row in trajectory:
        delta = (
            float(row["object_x_m"]) - measured_start[0],
            float(row["object_y_m"]) - measured_start[1],
        )
        progress.append(delta[0] * goal_unit[0] + delta[1] * goal_unit[1])
        cross_track.append(delta[0] * lateral_unit[0] + delta[1] * lateral_unit[1])

    controller_log = (scene_dir / "franka_sampling_c3_controller.log").read_text(
        encoding="utf-8", errors="replace"
    )
    rejection_distances = [
        float(value) for value in UNSAFE_REJECTION_RE.findall(controller_log)
    ]
    episode_lengths = c3_episode_lengths(trajectory)
    position_threshold_m = float(acceptance.get("position_threshold_m", 0.02))
    rotation_threshold_rad = float(
        acceptance.get("rotation_threshold_rad", 0.1)
    )
    record = {
        "scene_id": scene["scene_id"],
        "goal_direction_deg": float(scene["goal_direction_deg"]),
        "goal_direction_relative_deg": float(
            scene.get("goal_direction_relative_deg", scene["goal_direction_deg"])
        ),
        "goal_distance_m": goal_distance,
        "goal_yaw_deg": float(scene["goal_yaw_deg"]),
        "accepted": bool(scene["accepted"]),
        "best_position_error_m": acceptance.get("best_position_error_m"),
        "best_rotation_error_rad": acceptance.get("best_rotation_error_rad"),
        "final_position_error_m": acceptance.get("final_position_error_m"),
        "final_rotation_error_rad": acceptance.get("final_rotation_error_rad"),
        "max_goal_progress_m": max(progress),
        "final_goal_progress_m": progress[-1],
        "max_goal_progress_fraction": max(progress) / goal_distance,
        "max_abs_cross_track_m": max(abs(value) for value in cross_track),
        "trajectory_rows": len(trajectory),
        "c3_mode_rows": sum(int(row["is_c3_mode"]) for row in trajectory),
        "c3_episodes": len(episode_lengths),
        "median_c3_episode_rows": median(episode_lengths) if episode_lengths else 0,
        "joint_gate_rows": sum(
            float(row["position_error_m"]) < position_threshold_m
            and float(row["rotation_error_rad"]) < rotation_threshold_rad
            for row in trajectory
        ),
        "max_consecutive_joint_gate_rows": longest_joint_gate_run(
            trajectory, position_threshold_m, rotation_threshold_rad
        ),
        "previous_target_semantic_rejections": len(rejection_distances),
        "previous_target_rejections_below_clearance": sum(
            value <= clearance_m for value in rejection_distances
        ),
        "c3_trajectory_semantic_rejections": controller_log.count(
            "Semantic C1 guard rejected C3 trajectory"
        ),
    }
    record["diagnosis"] = classify(record)
    return record


def summarize(
    evaluation_root: Path, max_direction_deg: float | None = None
) -> dict[str, Any]:
    config_path = evaluation_root / "evaluation_config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    clearance_m = float(config.get("semantic_guard_clearance_m", 0.025))
    records = []
    for scene_dir in sorted(evaluation_root.glob("scene*")):
        required = (
            scene_dir / "scene_result.json",
            scene_dir / "acceptance.json",
            scene_dir / "sampling_c3_debug.csv",
            scene_dir / "franka_sampling_c3_controller.log",
        )
        if all(path.is_file() for path in required):
            record = analyze_scene(scene_dir, clearance_m)
            if (
                max_direction_deg is None
                or record["goal_direction_relative_deg"] < max_direction_deg
            ):
                records.append(record)
    diagnoses: dict[str, int] = {}
    for record in records:
        diagnosis = str(record["diagnosis"])
        diagnoses[diagnosis] = diagnoses.get(diagnosis, 0) + 1
    return {
        "schema": "nonprehensile.push_anything_directional_audit.v1",
        "evaluation_root": str(evaluation_root.resolve()),
        "semantic_guard_clearance_m": clearance_m,
        "max_direction_deg": max_direction_deg,
        "scene_count": len(records),
        "diagnosis_counts": diagnoses,
        "scenes": records,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Push Anything directional audit",
        "",
        "| scene | relative direction | accepted | best XY | final SO(3) | progress | joint run | guard rejects | diagnosis |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in summary["scenes"]:
        lines.append(
            "| {scene_id} | {goal_direction_relative_deg:+.1f}° | {accepted} | "
            "{best_position_error_m:.4f} m | {final_rotation_error_rad:.3f} rad | "
            "{max_goal_progress_fraction:.0%} | {max_consecutive_joint_gate_rows} | "
            "{previous_target_semantic_rejections} | {diagnosis} |".format(**record)
        )
    lines.extend(["", "Diagnosis counts:", ""])
    for name, count in sorted(summary["diagnosis_counts"].items()):
        lines.append(f"- `{name}`: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = summarize(args.evaluation_root.resolve(), args.max_direction_deg)
    rendered = markdown(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
