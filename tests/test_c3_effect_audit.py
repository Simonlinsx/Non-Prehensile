from __future__ import annotations

import math

from scripts.analyze_c3_effect_audit import analyze, trace_in_relay_clock


def yaw_quaternion(angle: float) -> list[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def test_contact_overlap_uses_relay_clock_for_post_step_isaac_trace():
    records = [
        {"event": "measured_state", "utime_us": t,
         "target_position_m": [x, 0.0, 0.0],
         "target_quaternion_wxyz": yaw_quaternion(0.0)}
        for t, x in [(200_000, 0.0), (300_000, 0.01)]
    ]
    records.append({
        "event": "object_plan", "plan_utime_us": 200_000, "c3_mode": True,
        "position_knots_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        "quaternion_knots_wxyz": [yaw_quaternion(0.0)] * 2,
    })
    result = {
        "schema": "nonprehensile.c3_online_isaaclab_task.v1",
        "control_period_s": 0.01,
        "trace": [{"sim_time_s": 0.1, "legal_safe_robot_contact": True}],
    }
    trace = trace_in_relay_clock(result)
    assert trace[0]["measurement_utime_us"] == 210_000
    assert "measurement_utime_us" not in result["trace"][0]
    summary = analyze(records, 0.1, 0.001, 1e-4, execution_trace=trace)
    assert summary["physical_contact_effectful"]["count"] == 1
    # An explicit timestamp is authoritative, even when the display clock
    # would put contact in a different window.
    trace[0]["measurement_utime_us"] = 410_000
    result["trace"] = trace
    summary = analyze(records, 0.1, 0.001, 1e-4,
                      execution_trace=trace_in_relay_clock(result))
    assert summary["physical_contact_effectful"]["count"] == 0


def test_effect_audit_measures_translation_and_yaw_residual() -> None:
    records = [
        {
            "event": "measured_state",
            "utime_us": 0,
            "target_position_m": [0.0, 0.0, 0.0],
            "target_quaternion_wxyz": yaw_quaternion(0.0),
        },
        {
            "event": "measured_state",
            "utime_us": 100_000,
            "target_position_m": [0.01, 0.0, 0.0],
            "target_quaternion_wxyz": yaw_quaternion(0.02),
        },
        {
            "event": "object_plan",
            "plan_utime_us": 0,
            "c3_mode": True,
            "yaw_recovery": False,
            "position_knots_m": [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
            "quaternion_knots_wxyz": [
                yaw_quaternion(0.0), yaw_quaternion(0.04)
            ],
        },
    ]

    summary = analyze(
        records, 0.1, 0.001, 1.0e-4,
        execution_trace=[{
            "sim_time_s": 0.1, "legal_safe_robot_contact": True
        }],
    )

    assert summary["matched_plan_count"] == 1
    assert summary["effectful_plan_count"] == 1
    assert math.isclose(
        summary["planar_effect_residual_m"]["median"], 0.01
    )
    assert math.isclose(
        summary["actual_to_predicted_motion_ratio"]["median"], 0.5
    )
    assert math.isclose(summary["planar_direction_cosine"]["median"], 1.0)
    assert math.isclose(summary["yaw_effect_residual_rad"]["median"], 0.02)
    assert summary["yaw_sign_agreement_fraction"] == 1.0
    assert summary["physical_contact_effectful"]["count"] == 1
    assert summary["no_physical_contact_effectful"]["count"] == 0


def test_effect_audit_excludes_non_c3_plans_and_unmatched_times() -> None:
    records = [
        {
            "event": "measured_state",
            "utime_us": 0,
            "target_position_m": [0.0, 0.0, 0.0],
            "target_quaternion_wxyz": yaw_quaternion(0.0),
        },
        {
            "event": "object_plan",
            "plan_utime_us": 0,
            "c3_mode": False,
            "position_knots_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            "quaternion_knots_wxyz": [
                yaw_quaternion(0.0), yaw_quaternion(0.0)
            ],
        },
    ]

    summary = analyze(records, 0.1, 0.01, 1.0e-4)

    assert summary["c3_mode_plan_count"] == 0
    assert summary["matched_plan_count"] == 0
    assert summary["planar_effect_residual_m"]["median"] is None


def test_effect_audit_compares_selected_knot_at_matching_physical_time() -> None:
    records = [
        {
            "event": "measured_state",
            "utime_us": 0,
            "target_position_m": [0.0, 0.0, 0.0],
            "target_quaternion_wxyz": yaw_quaternion(0.0),
        },
        {
            "event": "measured_state",
            "utime_us": 50_000,
            "target_position_m": [0.05, 0.0, 0.0],
            "target_quaternion_wxyz": yaw_quaternion(0.05),
        },
        {
            "event": "object_plan",
            "plan_utime_us": 0,
            "c3_mode": True,
            "yaw_recovery": False,
            "position_knots_m": [
                [0.01 * index, 0.0, 0.0] for index in range(6)
            ],
            # Deliberately scale the final quaternion.  Yaw extraction must
            # normalize it before comparing the predicted rotation.
            "quaternion_knots_wxyz": [
                yaw_quaternion(0.01 * index) if index < 5 else [
                    2.0 * value for value in yaw_quaternion(0.05)
                ]
                for index in range(6)
            ],
        },
    ]

    summary = analyze(
        records,
        prediction_step_s=0.05,
        maximum_state_time_error_s=0.001,
        minimum_predicted_motion_m=1.0e-4,
        prediction_knot_index=5,
    )

    assert summary["prediction_knot_index"] == 5
    assert math.isclose(summary["planar_effect_residual_m"]["median"], 0.0)
    assert math.isclose(summary["yaw_effect_residual_rad"]["median"], 0.0)
