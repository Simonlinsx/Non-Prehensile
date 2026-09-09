import copy
from scripts.audit_simulation_acceptance import summarize

P = {'count': 50, 'minimum_successes': 31, 'success_rate_strictly_greater_than': .6, 'max_sim_time_s': 180,
     'strict_pose_thresholds': {'planar_m': .02, 'height_m': .01, 'rotation_rad': .105, 'dwell_time_s': .5}}
IDS = [f'scene{i:03d}' for i in range(50)]


def results(successes):
    return {s: {'schema': 'nonprehensile.c3_online_isaaclab_task.v1',
                'executed_sim_time_s': 180., 'executed_steps': 18000,
                'physical_end_effector': 'stock_franka_gripper_closed', 'strict_pose_thresholds': copy.deepcopy(P['strict_pose_thresholds']),
                'controller_parameters': {'native_goal_mode': 2},
                'forbidden_robot_contact_ever': False, 'online_closed_loop_success': i < successes,
                'stopped_reason': 'strict_pose_dwell_success' if i < successes else 'maximum_sim_time',
                'final_planar_error_m': .01, 'final_height_error_m': .001, 'final_rotation_error_rad': .05,
                'maximum_strict_dwell_steps': 50 if i < successes else 0, 'control_period_s': .01,
                'first_safe_contact_step': 300,
                'trace': [{'legal_safe_robot_contact': True, 'forbidden_robot_contact': False,
                           'robot_target_contact_force_n_by_sensor': {'target_hand_contacts': .2}}]}
            for i, s in enumerate(IDS)}


def test_requires_strictly_more_than_sixty_percent_of_all_fifty():
    assert not summarize(P, IDS, results(30))['goal_passed']
    assert summarize(P, IDS, results(31))['goal_passed']
    r = results(31)
    del r[IDS[-1]]
    summary = summarize(P, IDS, r)
    assert summary['success_rate'] == .62
    assert not summary['goal_passed']


def test_rejects_unsafe_or_relaxed_gate_even_in_unsuccessful_scene():
    for key, value in [('forbidden_robot_contact_ever', True), ('strict_pose_thresholds', {})]:
        r = results(31)
        r[IDS[-1]][key] = value
        assert not summarize(P, IDS, r)['goal_passed']


def test_rechecks_pose_dwell_and_contact_instead_of_trusting_success_boolean():
    for key, value in [('final_planar_error_m', .03), ('final_rotation_error_rad', float('nan')),
                       ('first_safe_contact_step', None), ('maximum_strict_dwell_steps', 49)]:
        r = results(31)
        r[IDS[0]][key] = value
        summary = summarize(P, IDS, r)
        assert summary['successes'] == 30
        assert not summary['goal_passed']


def test_rejects_invalid_execution_evidence_even_for_failed_scene():
    for key, value in [('executed_sim_time_s', 181.), ('control_period_s', float('inf')),
                       ('maximum_strict_dwell_steps', 20000), ('first_safe_contact_step', -1),
                       ('executed_steps', True), ('schema', 'different'), ('controller_parameters', {'native_goal_mode': 0}), ('controller_parameters', []), ('diagnostic_reference_replay', True)]:
        r = results(31)
        r[IDS[-1]][key] = value
        summary = summarize(P, IDS, r)
        assert not summary['goal_passed']
        assert IDS[-1] in summary['invalid_scene_ids']


def test_requires_positive_hand_force_and_checks_trace_safety():
    r = results(31)
    r[IDS[0]]['trace'][0]['robot_target_contact_force_n_by_sensor']['target_hand_contacts'] = 0.
    assert summarize(P, IDS, r)['successes'] == 30
    r = results(31)
    r[IDS[-1]]['trace'][0]['forbidden_robot_contact'] = True
    summary = summarize(P, IDS, r)
    assert not summary['goal_passed']
    assert IDS[-1] in summary['c1_violations']


def test_actual_runner_failure_record_is_valid_but_never_a_success():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / 'fixtures/isaac_c1_real_failed_result_contract.json').read_text())
    actual = fixture['result']
    summary = summarize(P, ['diagnostic'], {'diagnostic': actual})
    assert summary['invalid_scene_ids'] == []
    assert summary['successes'] == 0
    assert not summary['goal_passed']
    # A top-level lookalike cannot override the real nested runtime setting.
    actual['controller_parameters']['native_goal_mode'] = 0
    actual['native_goal_mode'] = 2
    assert summarize(P, ['diagnostic'], {'diagnostic': actual})['invalid_scene_ids'] == ['diagnostic']
