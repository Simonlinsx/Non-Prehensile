import copy
import math

import pytest

from scripts.audit_strict_pose_dwell import audit, pose_errors
from scripts.audit_simulation_acceptance import summarize


def measured_result():
    # Synthetic measurement contract: nonzero goal translation and yaw, with
    # quaternion sign flips to exercise rotation equivalence independently.
    goal = [.4, .2, .013, math.cos(.4), 0., 0., math.sin(.4)]
    trace = [dict(step=i, measurement_utime_us=100_000+(i+1)*10_000,
                  target_position_m=[.41, .2, .014],
                  target_quaternion_wxyz=[v*(-1 if i % 2 else 1) for v in goal[3:]],
                  forbidden_robot_contact=False, legal_safe_robot_contact=i == 10,
                  robot_target_contact_force_n_by_sensor={'target_hand_contacts': .2 if i == 10 else 0.})
             for i in range(50)]
    return dict(schema='nonprehensile.c3_online_isaaclab_task.v1', trace=trace,
                control_period_s=.01, executed_steps=50, executed_sim_time_s=.5,
                goal_pose_wxyz=goal, maximum_strict_dwell_steps=50,
                strict_pose_thresholds=dict(planar_m=.02, height_m=.01, rotation_rad=.105, dwell_time_s=.5),
                physical_end_effector='stock_franka_gripper_closed',
                controller_parameters={'native_goal_mode': 2}, forbidden_robot_contact_ever=False,
                online_closed_loop_success=True, stopped_reason='strict_pose_dwell_success',
                first_safe_contact_step=10, final_planar_error_m=.01,
                final_height_error_m=.001, final_rotation_error_rad=0.)


def test_certifies_all_fifty_real_timestamps_even_without_terminal_contact():
    report = audit(measured_result())
    assert report['terminal_dwell_certified']
    assert report['maximum_observed_consecutive_strict_steps'] == 50


def test_missing_or_non_strict_sample_breaks_dwell():
    r = measured_result()
    del r['trace'][25]
    assert not audit(r)['terminal_dwell_certified']
    r = measured_result()
    r['trace'][25]['target_position_m'][0] += .02
    assert not audit(r)['terminal_dwell_certified']
    r = measured_result()
    r['trace'][25]['target_position_m'][2] += .02
    assert not audit(r)['terminal_dwell_certified']


def test_rejects_duplicate_reordered_and_false_timestamps():
    r = measured_result()
    r['trace'].insert(10, copy.deepcopy(r['trace'][10]))
    with pytest.raises(ValueError): audit(r)
    r = measured_result()
    r['trace'][10]['measurement_utime_us'] -= 10_000
    with pytest.raises(ValueError): audit(r)
    r = measured_result()
    r['trace'][0]['target_quaternion_wxyz'][0] = float('nan')
    with pytest.raises(ValueError): audit(r)


def test_so3_detects_roll_even_with_zero_yaw_error():
    angle = .13
    errors = pose_errors([0., 0., 0.], [math.cos(angle/2), math.sin(angle/2), 0., 0.],
                         [0., 0., 0., 1., 0., 0., 0.])
    assert errors[2] == pytest.approx(angle)
    assert errors[2] > .105


def test_acceptance_cannot_trust_success_or_dwell_counter_without_measured_window():
    r = measured_result()
    p = dict(count=1, minimum_successes=1, success_rate_strictly_greater_than=.6,
             max_sim_time_s=180., strict_pose_thresholds=r['strict_pose_thresholds'],
             require_independent_dwell_trace=True)
    assert summarize(p, ['s'], {'s': r})['goal_passed']
    del r['trace'][25]
    report = summarize(p, ['s'], {'s': r})
    assert not report['goal_passed']
    assert report['invalid_scene_ids'] == ['s']
