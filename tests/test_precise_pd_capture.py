import copy

import pytest

from scripts.prepare_offline_pd_capture import extract


def fixture():
    x = [[0.]*19 for _ in range(3)]
    x[1][0],x[2][0] = .01,.02
    forces = [[1.,2.,3.],[4.,5.,6.]]
    native = dict(utime_us=2350000,dt_s=.075,force_tracking_enabled=True,raw_states=x,raw_forces=forces)
    packet = dict(event='task_reference',relay_state_utime_us=2350000,plan_utime_us=2349999,c3_mode=True,
                  position_knots_m=[row[:3] for row in x[:-1]],force_knots_n=copy.deepcopy(forces),knot_times_s=[0,.075])
    result = dict(trace=[dict(measurement_utime_us=2350000,measured_joint_position_rad=[.1]*7)])
    return result,[native],[packet]


def test_capture_matches_published_double_reference_and_joint_clock():
    args = fixture()
    output = extract(*args,2.35)
    assert output['executed_actor_positions_and_forces_match_exactly']
    assert output['N'] == 2
    assert output['u'] == [[1,2,3],[4,5,6]]


def test_force_mismatch_and_missing_joint_state_rejected():
    result,native,packets = fixture()
    packets[0]['force_knots_n'][0][0] += 1e-8
    with pytest.raises(ValueError,match='forces differ'):
        extract(result,native,packets,2.35)
    result,native,packets = fixture()
    result['trace'][0]['measurement_utime_us'] += 10000
    with pytest.raises(ValueError,match='match uniquely'):
        extract(result,native,packets,2.35)


def test_held_old_plan_is_not_an_executed_fresh_capture():
    result,native,packets = fixture()
    packets[0]['plan_utime_us'] -= 50000
    with pytest.raises(ValueError,match='match uniquely'):
        extract(result,native,packets,2.35)
