import copy

import pytest

from scripts.audit_fr3_finger_closure import audit_rows


def fixture():
    result = dict(executed_steps=3, control_period_s=.001, physics_dt_s=.001, control_decimation=1)
    rows = [dict(step=i, measurement_utime_us=101000 + i * 1000,
                 position_m=[.0001, .0002], target_m=[0., 0.]) for i in range(3)]
    return result, rows


def test_interior_opening_or_nonzero_command_is_not_hidden_by_closed_endpoints():
    result, rows = fixture()
    assert audit_rows(result, rows, .001)["closure_pass"]
    for key, value in (("position_m", [0., .0095]), ("target_m", [0., .01]),
                       ("position_m", [-.002, 0.])):
        modified = copy.deepcopy(rows)
        modified[1][key] = value
        report = audit_rows(result, modified, .001)
        assert not report["closure_pass"]
        assert report["first_violation_step"] == 1


def test_requires_exact_finite_complete_physics_measurements():
    result, rows = fixture()
    cases = [rows[::2], rows[:2], rows + [rows[-1]]]
    for key, value in (("measurement_utime_us", 101999), ("step", True),
                       ("position_m", [0., float("nan")])):
        modified = copy.deepcopy(rows)
        modified[1][key] = value
        cases.append(modified)
    for case in cases:
        with pytest.raises(ValueError):
            audit_rows(result, case, .001)
    with pytest.raises(ValueError):
        audit_rows(dict(result, control_decimation=2), rows, .001)
