#!/usr/bin/env python3
"""Measure C3+ one-step object-effect prediction error against simulator state."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-result", type=Path,
        help="optional Isaac executor result used to label physical-contact windows",
    )
    parser.add_argument(
        "--prediction-step-s",
        type=float,
        default=0.1,
        help="physical duration represented by the selected C3 knot",
    )
    parser.add_argument(
        "--prediction-knot-index",
        type=int,
        default=1,
        help=(
            "predicted knot compared with the future measured state; this "
            "must match --prediction-step-s (for example knot 5 at 10 ms "
            "per knot means a 0.05 s prediction step)"
        ),
    )
    parser.add_argument(
        "--maximum-state-time-error-s", type=float, default=0.03
    )
    parser.add_argument(
        "--minimum-predicted-motion-m", type=float, default=1.0e-4
    )
    return parser.parse_args()


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_yaw(quaternion_wxyz: list[float]) -> float:
    w, x, y, z = quaternion_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("cannot extract yaw from a zero quaternion")
    w, x, y, z = (component / norm for component in (w, x, y, z))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def planar_norm(vector: list[float]) -> float:
    return math.hypot(vector[0], vector[1])


def nearest_state(
    state_times_us: list[int], states: list[dict], query_us: int
) -> tuple[dict, int]:
    index = bisect.bisect_left(state_times_us, query_us)
    candidates = [candidate for candidate in (index - 1, index)
                  if 0 <= candidate < len(states)]
    best_index = min(candidates, key=lambda candidate: abs(
        state_times_us[candidate] - query_us))
    return states[best_index], abs(state_times_us[best_index] - query_us)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(
        fraction * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None,
                "maximum": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "maximum": max(values),
    }


def trace_in_relay_clock(result: dict) -> list[dict]:
    """Normalize legacy post-step Isaac samples to the relay clock.

    Isaac's online executor starts relay timestamps at 100 ms, and legacy
    sim_time_s labels the start of the step whose end-state was recorded.
    Leave generic traces alone; new executor traces carry an explicit stamp.
    """
    trace = result.get("trace", [])
    if result.get("schema") != "nonprehensile.c3_online_isaaclab_task.v1":
        return trace
    period_s = float(result["control_period_s"])
    return [
        dict(item, measurement_utime_us=item.get(
            "measurement_utime_us",
            100_000 + round((float(item["sim_time_s"]) + period_s) * 1.0e6),
        ))
        for item in trace
    ]


def trace_time_s(item: dict) -> float:
    if "measurement_utime_us" in item:
        return int(item["measurement_utime_us"]) * 1.0e-6
    return float(item.get("sim_time_s", -1.0))


def analyze(
    records: list[dict], prediction_step_s: float,
    maximum_state_time_error_s: float, minimum_predicted_motion_m: float,
    execution_trace: list[dict] | None = None,
    prediction_knot_index: int = 1,
) -> dict:
    states = [record for record in records
              if record.get("event") == "measured_state"]
    all_plans = [record for record in records
                 if record.get("event") == "object_plan"]
    plans = [record for record in all_plans if record.get("c3_mode")]
    states.sort(key=lambda item: item["utime_us"])
    state_times_us = [int(item["utime_us"]) for item in states]
    execution_trace = execution_trace or []
    maximum_time_error_us = round(maximum_state_time_error_s * 1.0e6)
    prediction_step_us = round(prediction_step_s * 1.0e6)
    all_samples = []
    for plan in all_plans:
        positions = plan.get("position_knots_m", [])
        quaternions = plan.get("quaternion_knots_wxyz", [])
        if (
            len(positions) <= prediction_knot_index
            or len(quaternions) <= prediction_knot_index
            or not states
        ):
            continue
        plan_time_us = int(plan["plan_utime_us"])
        try:
            state_start, start_time_error_us = nearest_state(
                state_times_us, states, plan_time_us)
            state_end, end_time_error_us = nearest_state(
                state_times_us, states, plan_time_us + prediction_step_us)
        except ValueError:
            continue
        if max(start_time_error_us, end_time_error_us) > maximum_time_error_us:
            continue
        window_start_s = plan_time_us * 1.0e-6
        window_end_s = window_start_s + prediction_step_s
        execution_window = [
            item for item in execution_trace
            if window_start_s - 1.0e-9 <= trace_time_s(item)
            <= window_end_s + 1.0e-9
        ]
        physical_safe_contact = any(
            bool(item.get("legal_safe_robot_contact"))
            or bool(item.get("safe_robot_contact"))
            for item in execution_window
        )
        committed_contact_execution = any(
            bool(item.get("contact_acquisition_commit_latched"))
            or item.get("local_control_phase") == "semantic_safe_contact_push"
            for item in execution_window
        )
        predicted_delta = [positions[prediction_knot_index][axis]
                           - positions[0][axis]
                           for axis in range(3)]
        actual_delta = [
            state_end["target_position_m"][axis]
            - state_start["target_position_m"][axis]
            for axis in range(3)
        ]
        predicted_motion = planar_norm(predicted_delta)
        actual_motion = planar_norm(actual_delta)
        residual = [actual_delta[axis] - predicted_delta[axis]
                    for axis in range(3)]
        predicted_yaw_delta = wrap_angle(
            quaternion_yaw(quaternions[prediction_knot_index])
            - quaternion_yaw(quaternions[0]))
        actual_yaw_delta = wrap_angle(
            quaternion_yaw(state_end["target_quaternion_wxyz"])
            - quaternion_yaw(state_start["target_quaternion_wxyz"]))
        cosine = None
        if predicted_motion > 1.0e-9 and actual_motion > 1.0e-9:
            cosine = sum(predicted_delta[axis] * actual_delta[axis]
                         for axis in range(2)) / (
                             predicted_motion * actual_motion)
        all_samples.append({
            "plan_utime_us": plan_time_us,
            "predicted_planar_delta_m": predicted_delta[:2],
            "actual_planar_delta_m": actual_delta[:2],
            "planar_effect_residual_m": planar_norm(residual),
            "predicted_planar_motion_m": predicted_motion,
            "actual_planar_motion_m": actual_motion,
            "planar_direction_cosine": cosine,
            "predicted_yaw_delta_rad": predicted_yaw_delta,
            "actual_yaw_delta_rad": actual_yaw_delta,
            "yaw_effect_residual_rad": abs(wrap_angle(
                actual_yaw_delta - predicted_yaw_delta)),
            "prediction_initial_xy_error_m": math.hypot(
                positions[0][0] - state_start["target_position_m"][0],
                positions[0][1] - state_start["target_position_m"][1]),
            "yaw_recovery": bool(plan.get("yaw_recovery")),
            "c3_mode": bool(plan.get("c3_mode")),
            "physical_safe_contact": physical_safe_contact,
            "committed_contact_execution": committed_contact_execution,
        })
    samples = [sample for sample in all_samples if sample["c3_mode"]]
    effectful = [sample for sample in samples
                 if sample["predicted_planar_motion_m"]
                 >= minimum_predicted_motion_m]
    cosines = [sample["planar_direction_cosine"] for sample in effectful
               if sample["planar_direction_cosine"] is not None]
    motion_ratios = [
        sample["actual_planar_motion_m"]
        / sample["predicted_planar_motion_m"]
        for sample in effectful
    ]
    yaw_sign_samples = [sample for sample in samples
                        if abs(sample["predicted_yaw_delta_rad"]) >= 1.0e-4]
    yaw_sign_agreement = None
    if yaw_sign_samples:
        yaw_sign_agreement = statistics.fmean(
            1.0 if sample["predicted_yaw_delta_rad"]
            * sample["actual_yaw_delta_rad"] > 0.0 else 0.0
            for sample in yaw_sign_samples)
    contact_effectful = [sample for sample in effectful
                         if sample["physical_safe_contact"]]
    no_contact_effectful = [sample for sample in effectful
                            if not sample["physical_safe_contact"]]
    contact_overlap = [sample for sample in all_samples
                       if sample["physical_safe_contact"]]
    committed_overlap = [sample for sample in all_samples
                         if sample["committed_contact_execution"]]

    def group_summary(group: list[dict]) -> dict:
        group_cosines = [sample["planar_direction_cosine"] for sample in group
                         if sample["planar_direction_cosine"] is not None]
        return {
            "count": len(group),
            "planar_effect_residual_m": distribution([
                sample["planar_effect_residual_m"] for sample in group]),
            "planar_direction_cosine": distribution(group_cosines),
            "actual_to_predicted_motion_ratio": distribution([
                sample["actual_planar_motion_m"]
                / sample["predicted_planar_motion_m"] for sample in group]),
            "yaw_effect_residual_rad": distribution([
                sample["yaw_effect_residual_rad"] for sample in group]),
        }
    return {
        "schema": "nonprehensile.c3_object_effect_audit_summary.v1",
        "prediction_step_s": prediction_step_s,
        "prediction_knot_index": prediction_knot_index,
        "maximum_state_time_error_s": maximum_state_time_error_s,
        "minimum_predicted_motion_m": minimum_predicted_motion_m,
        "measured_state_count": len(states),
        "object_plan_count": len(all_plans),
        "matched_plan_count_all_modes": len(all_samples),
        "c3_mode_plan_count": len(plans),
        "matched_plan_count": len(samples),
        "effectful_plan_count": len(effectful),
        "prediction_initial_xy_error_m": distribution([
            sample["prediction_initial_xy_error_m"] for sample in samples]),
        "planar_effect_residual_m": distribution([
            sample["planar_effect_residual_m"] for sample in effectful]),
        "yaw_effect_residual_rad": distribution([
            sample["yaw_effect_residual_rad"] for sample in samples]),
        "planar_direction_cosine": distribution(cosines),
        "actual_to_predicted_motion_ratio": distribution(motion_ratios),
        "yaw_sign_agreement_fraction": yaw_sign_agreement,
        "physical_contact_effectful": group_summary(contact_effectful),
        "no_physical_contact_effectful": group_summary(no_contact_effectful),
        "mode_contact_overlap": {
            "physical_contact_windows": len(contact_overlap),
            "physical_contact_windows_in_c3_mode": sum(
                1 for sample in contact_overlap if sample["c3_mode"]),
            "physical_contact_windows_outside_c3_mode": sum(
                1 for sample in contact_overlap if not sample["c3_mode"]),
            "committed_execution_windows": len(committed_overlap),
            "committed_execution_windows_in_c3_mode": sum(
                1 for sample in committed_overlap if sample["c3_mode"]),
            "committed_execution_windows_outside_c3_mode": sum(
                1 for sample in committed_overlap if not sample["c3_mode"]),
        },
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    if args.prediction_step_s <= 0.0:
        raise ValueError("prediction step must be positive")
    if args.prediction_knot_index <= 0:
        raise ValueError("prediction knot index must be positive")
    if args.maximum_state_time_error_s < 0.0:
        raise ValueError("maximum state time error must be non-negative")
    records = [json.loads(line) for line in
               args.audit.expanduser().resolve().read_text(encoding="utf-8").splitlines()
               if line.strip()]
    execution_trace = None
    if args.execution_result is not None:
        execution_result = json.loads(
            args.execution_result.expanduser().resolve().read_text(
                encoding="utf-8"))
        execution_trace = trace_in_relay_clock(execution_result)
    summary = analyze(
        records,
        args.prediction_step_s,
        args.maximum_state_time_error_s,
        args.minimum_predicted_motion_m,
        execution_trace,
        args.prediction_knot_index,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(
        "C3_EFFECT_AUDIT",
        f"matched={summary['matched_plan_count']}",
        f"effectful={summary['effectful_plan_count']}",
        f"xy_residual_median={summary['planar_effect_residual_m']['median']}",
        f"direction_cosine_median={summary['planar_direction_cosine']['median']}",
        f"yaw_residual_median={summary['yaw_effect_residual_rad']['median']}",
    )


if __name__ == "__main__":
    main()
