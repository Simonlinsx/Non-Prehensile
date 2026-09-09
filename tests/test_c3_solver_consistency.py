import numpy as np
from scripts.analyze_c3_solver_consistency import analyze, paired_predictions


def test_common_knot_ignores_extra_pd_tail_and_excludes_fallback_eta():
    records = []
    for channel, xs in [('raw', [0, .01, .02]), ('pd', [0, .01, .02, 1.])]:
        records.append({'event': 'planner_diagnostic', 'channel': channel, 'utime_us': 5_000_000,
                        'trajectories': [{'name': 'object_position_target_0',
                                          'datapoints': [xs, [0] * len(xs), [0] * len(xs)]}]})
    x, u, lam, eta = [[0., .01]], [[1., 1.]], [[-.1, .1]], [[2., 0.]]
    records.append({'event': 'solver_diagnostic', 'utime_us': 5_000_000,
                    'x': x, 'u': u, 'lambda': lam, 'z': np.vstack([x, lam, u, eta]).tolist()})
    records.append({'event': 'solver_diagnostic', 'utime_us': 6_000_000,
                    'x': [[0., 0.]], 'u': [[0., 0.]], 'lambda': [[0., 0.]],
                    'z': [[0., 0.], [0., 0.], [0., 0.], [-100., 100.]]})
    result = analyze(records, knot=2)
    assert result['planar_prediction_m']['raw']['median'] == result['planar_prediction_m']['pd']['median'] == .02
    assert result['zero_input_frozen_solution_count'] == 1
    assert result['maximum_negative_lambda_magnitude']['maximum'] == .1
    assert result['maximum_lambda_eta_product']['maximum'] == .2


def test_pairs_current_plans_by_timestamp_and_interpolates_actual_horizon():
    records = [
        {'event': 'measured_state', 'utime_us': 4_000_000, 'target_position_m': [0., 0., 0.]},
        {'event': 'measured_state', 'utime_us': 4_100_000, 'target_position_m': [.04, 0., 0.]},
        {'event': 'planner_diagnostic', 'channel': 'C3_TRAJECTORY_OBJECT_CURR_PLAN', 'utime_us': 3_999_999,
         'trajectories': [{'name': 'object_position_target_0', 'datapoints': [[0., .1], [0., 0.], [0., 0.]]}]},
        {'event': 'object_plan', 'c3_mode': True, 'plan_utime_us': 4_000_000,
         'position_knots_m': [[0., 0., 0.], [.02, 0., 0.]]},
    ]
    result = paired_predictions(records, 4., 1, .05)
    assert result['matched_executed_c3_plans'] == 1
    assert result['raw_endpoint_xy_error_m']['median'] == .08
    assert result['pd_endpoint_xy_error_m']['median'] == 0.
    assert result['moving_actual_horizons']['actual_xy_displacement_m']['count'] == 1
    assert result['moving_actual_horizons']['actual_xy_displacement_m']['median'] == .02
    assert result['stationary_actual_horizons']['pd_endpoint_xy_error_m']['count'] == 0
    records[1]['target_position_m'] = [.001, 0., 0.]
    stationary = paired_predictions(records, 4., 1, .05)
    assert stationary['stationary_actual_horizons']['pd_endpoint_xy_error_m']['count'] == 1
    assert stationary['moving_actual_horizons']['pd_endpoint_xy_error_m']['count'] == 0
    assert paired_predictions(records, 4., 1, .2)['matched_executed_c3_plans'] == 0
    records[-1]['c3_mode'] = False
    assert paired_predictions(records, 4., 1, .05)['matched_executed_c3_plans'] == 0


def test_quaternion_interpolation_uses_short_arc_and_full_rotation():
    import pytest
    from scripts.analyze_c3_solver_consistency import interpolate_quaternion, rotation_error
    times = np.array([0, 100])
    qs = [[1., 0, 0, 0], [-np.sqrt(.5), -np.sqrt(.5), 0, 0]]
    midpoint = interpolate_quaternion(times, qs, 50)
    assert rotation_error(midpoint, [1, 0, 0, 0]) == pytest.approx(np.pi / 4)
    assert rotation_error(midpoint, -midpoint) == pytest.approx(0)
    with pytest.raises(ValueError):
        interpolate_quaternion(times, qs, 101)
    with pytest.raises(ValueError):
        rotation_error([0, 0, 0, 0], midpoint)
