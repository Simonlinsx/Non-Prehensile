from scripts.analyze_c3_execution_tracking import analyze


def test_distinguishes_reference_limit_from_servo_error_and_stale_contact():
    trace = [
        {
            "flags": 17, "c3_mode": True, "legal_safe_robot_contact": True,
            "raw_task_target_c3_m": [0.03, 0.0, 0.0],
            "governed_task_target_c3_m": [0.01, 0.0, 0.0],
            "planner_tip_position_m": [0.009, 0.0, 0.0],
        },
        {"flags": 19, "c3_mode": True, "legal_safe_robot_contact": True},
    ]
    result = analyze({"trace": trace, "trace_stride": 10, "control_period_s": 0.01})
    assert result["fresh_c3"]["sample_count"] == 1
    assert result["sampled_contact_outside_fresh_c3"] == 1
    import pytest
    assert result["fresh_c3"]["raw_to_governed_m"]["max"] == pytest.approx(0.02)
    assert result["fresh_c3"]["governed_to_measured_m"]["max"] == pytest.approx(0.001)


def test_missing_legacy_reference_fields_do_not_imply_zero_error():
    result = analyze({
        "trace": [{"flags": 17, "c3_mode": True, "legal_safe_robot_contact": False}],
        "trace_stride": 10, "control_period_s": 0.01,
    })
    assert result["fresh_c3"]["raw_to_governed_m"]["max"] is None
    assert result["fresh_c3"]["physical_safe_contact_sample_fraction"] == 0.0
    assert result["first_physical_contact_time_s"] is None


def test_contact_preserving_trace_does_not_claim_uniform_time_spacing():
    result = analyze({"trace": [], "trace_stride": 10, "control_period_s": .01,
                      "trace_sampling_policy": "stride_plus_all_physical_contact_steps"})
    assert result["sample_period_s"] is None
    assert result["nominal_stride_period_s"] == .1
