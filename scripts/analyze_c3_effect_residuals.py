#!/usr/bin/env python3
"""Compare C3 target-effect predictions with measured Isaac target motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--controller-log", type=Path, required=True)
    parser.add_argument("--scene-spec", type=Path, required=True)
    parser.add_argument(
        "--prediction-horizon-s",
        type=float,
        default=1.125,
        help="C3 N * planning_dt (15 * 0.075 s for the current scene)",
    )
    parser.add_argument("--minimum-effect", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def planar_yaw(world: list[float], support: list[float]) -> float:
    support_inverse = [support[0], -support[1], -support[2], -support[3]]
    planar = quaternion_multiply(world, support_inverse)
    return 2.0 * math.atan2(planar[3], planar[0])


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_effect_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "TARGET_EFFECT "
        if marker not in line:
            continue
        row: dict[str, float | str] = {}
        for token in line.split(marker, 1)[1].split():
            if "=" not in token:
                continue
            key, raw = token.split("=", 1)
            try:
                row[key] = float(raw)
            except ValueError:
                row[key] = raw
        if "state_time" in row:
            rows.append(row)
    return rows


def nearest_trace(trace: list[dict[str, object]], time_s: float) -> dict[str, object]:
    return min(trace, key=lambda row: abs(float(row["sim_time_s"]) - time_s))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rmse(values: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def main() -> None:
    args = parse_args()
    if args.prediction_horizon_s <= 0.0:
        raise ValueError("prediction horizon must be positive")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    spec = json.loads(args.scene_spec.read_text(encoding="utf-8"))
    trace = result.get("trace", [])
    if not trace:
        raise ValueError("result has no measured trace")
    effects = parse_effect_rows(args.controller_log)
    support = [float(value) for value in spec["objects"][0]["support_quaternion_wxyz"]]
    goal_pose = [float(value) for value in result["goal_pose_wxyz"]]
    goal_yaw = planar_yaw(goal_pose[3:7], support)

    comparisons = []
    for effect in effects:
        if effect.get("mode") != "c3":
            continue
        start_time = float(effect["state_time"])
        finish_time = start_time + args.prediction_horizon_s
        if finish_time > float(trace[-1]["sim_time_s"]):
            continue
        start = nearest_trace(trace, start_time)
        finish = nearest_trace(trace, finish_time)
        start_error = wrap_angle(
            goal_yaw
            - planar_yaw(
                [float(value) for value in start["target_quaternion_wxyz"]], support
            )
        )
        finish_error = wrap_angle(
            goal_yaw
            - planar_yaw(
                [float(value) for value in finish["target_quaternion_wxyz"]], support
            )
        )
        start_xy = [float(value) for value in start["target_position_m"][:2]]
        finish_xy = [float(value) for value in finish["target_position_m"][:2]]
        goal_xy = goal_pose[:2]
        start_planar = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
        finish_planar = math.hypot(
            goal_xy[0] - finish_xy[0], goal_xy[1] - finish_xy[1]
        )
        predicted_yaw_change = float(effect["predicted_yaw_change"])
        actual_yaw_change = wrap_angle(start_error - finish_error)
        predicted_planar_progress = float(effect["current_planar"]) - float(
            effect["predicted_planar"]
        )
        actual_planar_progress = start_planar - finish_planar
        contact_observed = any(
            bool(row["safe_robot_contact"])
            for row in trace
            if start_time <= float(row["sim_time_s"]) <= finish_time
        )
        comparisons.append(
            {
                "state_time_s": start_time,
                "measured_finish_time_s": float(finish["sim_time_s"]),
                "selected": int(float(effect["selected"])),
                "contact_observed": contact_observed,
                "predicted_yaw_change_rad": predicted_yaw_change,
                "actual_yaw_change_rad": actual_yaw_change,
                "yaw_residual_rad": actual_yaw_change - predicted_yaw_change,
                "predicted_planar_progress_m": predicted_planar_progress,
                "actual_planar_progress_m": actual_planar_progress,
                "planar_progress_residual_m": (
                    actual_planar_progress - predicted_planar_progress
                ),
                "goal_torque_proxy_m": float(effect["goal_torque_proxy"]),
                "predicted_torque_proxy_m": float(effect["predicted_torque_proxy"]),
            }
        )

    informative = [
        row
        for row in comparisons
        if abs(row["predicted_yaw_change_rad"]) >= args.minimum_effect
        or abs(row["actual_yaw_change_rad"]) >= args.minimum_effect
    ]
    contact_rows = [row for row in informative if row["contact_observed"]]
    sign_rows = [
        row
        for row in contact_rows
        if abs(row["predicted_yaw_change_rad"]) >= args.minimum_effect
        and abs(row["actual_yaw_change_rad"]) >= args.minimum_effect
    ]
    output = {
        "schema": "nonprehensile.c3_effect_residuals.v1",
        "result": str(args.result.resolve()),
        "controller_log": str(args.controller_log.resolve()),
        "scene_spec": str(args.scene_spec.resolve()),
        "prediction_horizon_s": args.prediction_horizon_s,
        "c3_comparison_count": len(comparisons),
        "informative_comparison_count": len(informative),
        "contact_comparison_count": len(contact_rows),
        "signed_yaw_direction_agreement": (
            sum(
                row["predicted_yaw_change_rad"] * row["actual_yaw_change_rad"] > 0.0
                for row in sign_rows
            )
            / len(sign_rows)
            if sign_rows
            else None
        ),
        "mean_predicted_yaw_change_rad": mean(
            [row["predicted_yaw_change_rad"] for row in contact_rows]
        ),
        "mean_actual_yaw_change_rad": mean(
            [row["actual_yaw_change_rad"] for row in contact_rows]
        ),
        "yaw_residual_rmse_rad": rmse(
            [row["yaw_residual_rad"] for row in contact_rows]
        ),
        "planar_progress_residual_rmse_m": rmse(
            [row["planar_progress_residual_m"] for row in contact_rows]
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
