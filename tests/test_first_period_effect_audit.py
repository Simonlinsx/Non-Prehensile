import copy
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('first_period', Path(__file__).parents[1]/'scripts/audit_c3_first_period_effects.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    start = dict(event='measured_state', utime_us=100000,
                 target_position_m=[0,0,0], target_quaternion_wxyz=[1,0,0,0])
    end = dict(start, utime_us=150000, target_position_m=[-.01,0,0])
    plan = dict(event='object_plan', c3_mode=True, relay_state_utime_us=100000,
                plan_utime_us=99999, position_knots_m=[[0,0,0],[.015,0,0]],
                quaternion_knots_wxyz=[[1,0,0,0],[1,0,0,0]])
    trace = [dict(measurement_utime_us=t, legal_safe_robot_contact=True,
                  robot_target_contact_force_n_by_sensor={'target_hand_contacts_left':1},
                  semantic_c1_guard_active=False, forbidden_robot_contact=False)
             for t in range(110000,150001,10000)]
    return [start,end,plan], dict(control_period_s=.01,trace=trace), [.04,0,0,1,0,0,0]


def test_exact_cycle_and_stale_plan_exclusion():
    records, result, goal = fixture()
    records.append(dict(records[-1], plan_utime_us=50000))
    report = module.analyze(records, result, goal)
    assert report['fresh_c3_matched'] == 1
    assert report['stale_c3_plans_excluded'] == 1
    assert report['predicted_improvement_actual_regression_windows'] == 1
    assert abs(report['rows'][0]['planar_residual_m']-.02) < 1e-12


def test_missing_servo_guard_or_endpoint_cannot_certify_unshielded_window():
    records, result, goal = fixture()
    missing = copy.deepcopy(result)
    missing['trace'].pop(2)
    assert module.analyze(records,missing,goal)['eligible_unshielded_contact_windows'] == 0
    guarded = copy.deepcopy(result)
    guarded['trace'][2]['semantic_c1_guard_active'] = True
    assert module.analyze(records,guarded,goal)['eligible_unshielded_contact_windows'] == 0
    unknown_guard = copy.deepcopy(result)
    unknown_guard['trace'][2].pop('semantic_c1_guard_active')
    assert module.analyze(records,unknown_guard,goal)['eligible_unshielded_contact_windows'] == 0
    records[1]['utime_us'] += 1
    report = module.analyze(records,result,goal)
    assert report['fresh_c3_matched'] == 0
    assert report['exact_endpoint_missing'] == 1
