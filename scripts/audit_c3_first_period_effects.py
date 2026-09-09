"""Compare fresh C3 predictions with the next exact 50 ms of measured motion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def pose_cost(position, quaternion, goal):
    q = np.asarray(quaternion, dtype=float)
    g = np.asarray(goal[3:], dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid predicted/measured quaternion')
    angle = 2*np.arccos(np.clip(abs(q @ g)/(np.linalg.norm(q)*np.linalg.norm(g)), 0, 1))
    xy = np.linalg.norm(np.asarray(position)[:2]-goal[:2])
    return float((xy/.02)**2+(angle/.105)**2)


def analyze(records, result, goal, knot_period=.075):
    if knot_period != .075 or result['control_period_s'] != .01:
        raise ValueError('Requires 75 ms planner knots and 10 ms servo steps')
    if len(goal) != 7 or not np.isfinite(goal).all() or np.linalg.norm(goal[3:]) < 1e-12:
        raise ValueError('Invalid fixed goal')
    states, plans, trace = {}, [], {}
    for r in records:
        if r.get('event') == 'measured_state':
            if r['utime_us'] in states:
                raise ValueError('Duplicate measured relay timestamp')
            states[r['utime_us']] = r
        elif r.get('event') == 'object_plan':
            plans.append(r)
    for row in result['trace']:
        t = row['measurement_utime_us']
        if t in trace:
            raise ValueError('Duplicate servo timestamp')
        trace[t] = row
    rows, stale, unmatched = [], 0, 0
    for plan in plans:
        if not plan['c3_mode']:
            continue
        stamp = plan['relay_state_utime_us']
        if abs(plan['plan_utime_us']-stamp) > 1:
            stale += 1
            continue
        if stamp not in states or stamp+50000 not in states:
            unmatched += 1
            continue
        start, end = states[stamp], states[stamp+50000]
        pos = np.asarray(plan['position_knots_m'][:2], dtype=float)
        quat = np.asarray(plan['quaternion_knots_wxyz'][:2], dtype=float)
        if pos.shape != (2, 3) or quat.shape != (2, 4) or not np.isfinite(pos).all():
            raise ValueError('Invalid published plan shape/values')
        predicted_position = pos[0]/3 + pos[1]*2/3
        predicted_quaternion = quat[0]/3 + quat[1]*2/3
        actual_delta = np.asarray(end['target_position_m'])-start['target_position_m']
        predicted_delta = predicted_position-pos[0]
        observed = [trace.get(stamp+i*10000) for i in range(1, 6)]
        complete = all(t is not None for t in observed)
        contact = any(t is not None and t.get('legal_safe_robot_contact') is True
                      and any(n.startswith('target_hand_contacts') and np.isfinite(f) and f > 0
                              for n, f in t.get('robot_target_contact_force_n_by_sensor', {}).items())
                      for t in observed)
        guarded = any(t is not None and t.get('semantic_c1_guard_active') is True for t in observed)
        unshielded_safe = complete and all(t.get('semantic_c1_guard_active') is False
                                           and t.get('forbidden_robot_contact') is False for t in observed)
        initial = pose_cost(start['target_position_m'], start['target_quaternion_wxyz'], goal)
        initial_prediction = pose_cost(pos[0], quat[0], goal)
        rows.append(dict(utime_us=stamp,
                         prediction_initial_xy_error_m=float(np.linalg.norm(pos[0, :2]-np.asarray(start['target_position_m'])[:2])),
                         planar_residual_m=float(np.linalg.norm(actual_delta[:2]-predicted_delta[:2])),
                         actual_motion_m=float(np.linalg.norm(actual_delta[:2])),
                         predicted_motion_m=float(np.linalg.norm(predicted_delta[:2])),
                         actual_delta_m=actual_delta.tolist(), predicted_delta_m=predicted_delta.tolist(),
                         predicted_task_cost_drop=initial_prediction-pose_cost(predicted_position, predicted_quaternion, goal),
                         actual_task_cost_drop=initial-pose_cost(end['target_position_m'], end['target_quaternion_wxyz'], goal),
                         complete_servo_coverage=complete, positive_legal_contact=contact,
                         any_recorded_guard=guarded,
                         eligible_unshielded_contact=unshielded_safe and contact))
    eligible = [r for r in rows if r['eligible_unshielded_contact']]
    contradictory = [r for r in eligible if r['predicted_task_cost_drop'] > .01 and r['actual_task_cost_drop'] < -.01]
    return dict(schema='nonprehensile.c3_first_period_effects.v1',
                fresh_c3_matched=len(rows), stale_c3_plans_excluded=stale,
                exact_endpoint_missing=unmatched, eligible_unshielded_contact_windows=len(eligible),
                predicted_improvement_actual_regression_windows=len(contradictory),
                top_contradictory_windows=sorted(contradictory, key=lambda r:r['planar_residual_m'], reverse=True)[:20],
                rows=rows,
                limitations=[
                    'Published 75ms knots are linearly interpolated at50ms; this is not an exact nonlinear50ms rollout.',
                    'Measured endpoints are exact relay samples; no nearest-time substitution or missing servo interpolation.',
                    'Task cost is XY^2/20mm^2 plus fullSO3^2/.105rad^2, not the native optimization objective.',
                    'Only windows with all5 servo observations, positive legal contact and no recorded guard are eligible.',
                    'Contradictions identify replay candidates; they do not isolate contact-model versus controller error.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.scene_directory
    result_path, audit_path = root/'result.json', root/'effect_audit.jsonl'
    result = json.loads(result_path.read_text())
    goal_path = Path(result['controller_parameters']['native_goal_artifact']['path'])
    if hashlib.sha256(goal_path.read_bytes()).hexdigest() != result['controller_parameters']['native_goal_artifact']['sha256']:
        raise ValueError('Native fixed goal changed')
    goal = yaml.safe_load(goal_path.read_text())
    if goal['goal_mode'] != 2:
        raise ValueError('Requires fixed goal')
    runtimes = list(root.glob('shared_runtime_*'))
    if len(runtimes) != 1:
        raise ValueError('Requires one isolated runtime')
    params_path = goal_path.parent/'sampling_c3_controller_params.yaml'
    params = yaml.safe_load(params_path.read_text())
    options_path = runtimes[0]/params['sampling_c3_options_file']
    options = yaml.safe_load(options_path.read_text())
    if any(options[k] != .075 for k in ('planning_dt_position', 'planning_dt_pose')):
        raise ValueError('Staged planning knots are not75ms')
    with audit_path.open() as stream:
        report = analyze((json.loads(line) for line in stream), result,
                         goal['fixed_target_positions'][0]+goal['fixed_target_orientations'][0])
    report['source_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (result_path, audit_path, goal_path, params_path, options_path, Path(__file__))}
    (root/'first_period_effect_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','top_contradictory_windows','source_sha256')}))


if __name__ == '__main__':
    main()
