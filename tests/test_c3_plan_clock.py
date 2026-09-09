import pytest

from scripts.analyze_c3_plan_clock import analyze


def observation(lead=0.0, query=0.0):
    return {"event": "task_reference", "relay_state_utime_us": 2_000_000,
            "plan_start_time_s": 2 + lead, "query_time_s": 2 + query,
            "c3_mode": True}


def test_synchronous_clock_accepts_roundoff_but_rejects_wall_latency():
    assert analyze([observation(1e-7, 1e-7)])["passed"]
    assert not analyze([observation(.09, .09)])["passed"]
    assert not analyze([observation(-.05, 0)])["passed"]
    assert not analyze([observation(0, .05)])["passed"]


def test_synchronous_clock_rejects_missing_or_invalid_evidence():
    with pytest.raises(ValueError, match="No task-reference"):
        analyze([{"event": "measured_state"}])
    with pytest.raises(ValueError, match="Non-finite"):
        analyze([observation(float('nan'))])
