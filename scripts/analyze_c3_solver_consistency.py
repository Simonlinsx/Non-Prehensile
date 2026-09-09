#!/usr/bin/env python3
"""Compare raw C3 plans with sample-ranking PD rollouts at the same knot."""
import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    return ({'count': len(values), 'median': float(np.median(values)),
             'p90': float(np.quantile(values, .9)), 'maximum': float(np.max(values))}
            if values else {'count': 0})


def unit_quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid prediction quaternion')
    return q / np.linalg.norm(q)


def rotation_error(a, b):
    return float(2 * np.arccos(np.clip(abs(unit_quaternion(a) @ unit_quaternion(b)), 0, 1)))


def interpolate_quaternion(times, quaternions, stamp):
    right = min(int(np.searchsorted(times, stamp, side='right')), len(times) - 1)
    left = max(0, right - 1)
    a, b = unit_quaternion(quaternions[left]), unit_quaternion(quaternions[right])
    if times[right] == times[left]:
        return a
    alpha = (stamp - times[left]) / (times[right] - times[left])
    if alpha < 0 or alpha > 1:
        raise ValueError('Quaternion interpolation would extrapolate')
    dot = float(a @ b)
    if dot < 0:
        b = -b
        dot = -dot
    if dot > .9995:
        return unit_quaternion((1 - alpha) * a + alpha * b)
    theta = np.arccos(np.clip(dot, 0, 1))
    return (np.sin((1 - alpha) * theta) * a + np.sin(alpha * theta) * b) / np.sin(theta)


def paired_predictions(records, minimum_time_s, knot, knot_period_s):
    raw = {r['utime_us']: r for r in records
           if r.get('channel') == 'C3_TRAJECTORY_OBJECT_CURR_PLAN'}
    states = sorted((r for r in records if r['event'] == 'measured_state'), key=lambda r: r['utime_us'])
    if not states:
        return {'matched_executed_c3_plans': 0}
    times = np.array([r['utime_us'] for r in states])
    positions = np.array([r['target_position_m'] for r in states])
    quaternions = [r.get('target_quaternion_wxyz') for r in states]
    have_measured_quaternions = all(q is not None for q in quaternions)
    raw_error, pd_error, disagreements, initial_mismatch = [], [], [], []
    pairs = []
    moving_threshold_m = .005
    for pd in records:
        if pd['event'] != 'object_plan' or not pd.get('c3_mode'):
            continue
        stamp = pd['plan_utime_us']
        native = next((raw[stamp + delta] for delta in (0, -1, 1) if stamp + delta in raw), None)
        endpoint = stamp + round(knot * knot_period_s * 1e6)
        if native is None or stamp < minimum_time_s * 1e6 or stamp < times[0] or endpoint > times[-1]:
            continue
        trajectory = next(t for t in native['trajectories'] if t['name'] == 'object_position_target_0')
        raw_points = np.asarray(trajectory['datapoints'])[:3].T
        pd_points = np.asarray(pd['position_knots_m'])
        if min(len(raw_points), len(pd_points)) <= knot:
            raise ValueError('Unavailable paired prediction knot')
        if not np.isfinite(raw_points).all() or not np.isfinite(pd_points).all() or not np.isfinite(positions).all():
            raise ValueError('Nonfinite paired prediction or measured state')
        measured_start = np.array([np.interp(stamp, times, positions[:, axis]) for axis in range(3)])
        measured = np.array([np.interp(endpoint, times, positions[:, axis]) for axis in range(3)])
        raw_error.append(float(np.linalg.norm(raw_points[knot, :2] - measured[:2])))
        pd_error.append(float(np.linalg.norm(pd_points[knot, :2] - measured[:2])))
        disagreements.append(float(np.linalg.norm(raw_points[knot, :2] - pd_points[knot, :2])))
        initial_mismatch.append(float(np.linalg.norm(raw_points[0] - pd_points[0])))
        pairs.append({
            'plan_utime_us': stamp,
            'actual_xy_displacement_m': float(np.linalg.norm(measured[:2] - measured_start[:2])),
            'raw_xy_displacement_m': float(np.linalg.norm(raw_points[knot, :2] - raw_points[0, :2])),
            'pd_xy_displacement_m': float(np.linalg.norm(pd_points[knot, :2] - pd_points[0, :2])),
            'raw_endpoint_xy_error_m': raw_error[-1], 'pd_endpoint_xy_error_m': pd_error[-1],
            'raw_initial_measured_xyz_error_m': float(np.linalg.norm(raw_points[0] - measured_start)),
            'pd_initial_measured_xyz_error_m': float(np.linalg.norm(pd_points[0] - measured_start)),
        })
        raw_quat_traj = next((t for t in native['trajectories'] if t['name'] == 'object_orientation_target_0'), None)
        if have_measured_quaternions and raw_quat_traj is not None and pd.get('quaternion_knots_wxyz'):
            raw_q = np.asarray(raw_quat_traj['datapoints']).T[knot]
            pd_q = pd['quaternion_knots_wxyz'][knot]
            actual_q = interpolate_quaternion(times, quaternions, endpoint)
            initial_q = interpolate_quaternion(times, quaternions, stamp)
            pairs[-1].update(
                actual_so3_displacement_rad=rotation_error(initial_q, actual_q),
                raw_initial_measured_so3_error_rad=rotation_error(np.asarray(raw_quat_traj["datapoints"]).T[0], initial_q),
                pd_initial_measured_so3_error_rad=rotation_error(pd["quaternion_knots_wxyz"][0], initial_q),
                raw_endpoint_so3_error_rad=rotation_error(raw_q, actual_q),
                pd_endpoint_so3_error_rad=rotation_error(pd_q, actual_q),
                raw_endpoint_quaternion_norm_error=abs(float(np.linalg.norm(raw_q))-1),
                pd_endpoint_quaternion_norm_error=abs(float(np.linalg.norm(pd_q))-1),
            )
    def summarize_subset(subset):
        return {key: distribution([row[key] for row in subset if key in row]) for key in (
            'actual_xy_displacement_m', 'raw_xy_displacement_m', 'pd_xy_displacement_m',
            'raw_endpoint_xy_error_m', 'pd_endpoint_xy_error_m',
            'raw_initial_measured_xyz_error_m', 'pd_initial_measured_xyz_error_m',
            'actual_so3_displacement_rad', 'raw_endpoint_so3_error_rad', 'pd_endpoint_so3_error_rad',
            'raw_initial_measured_so3_error_rad', 'pd_initial_measured_so3_error_rad',
            'raw_endpoint_quaternion_norm_error', 'pd_endpoint_quaternion_norm_error')}

    return {'matched_executed_c3_plans': len(raw_error), 'horizon_s': knot * knot_period_s,
            'moving_threshold_m': moving_threshold_m,
            'moving_actual_horizons': summarize_subset([r for r in pairs if r['actual_xy_displacement_m'] >= moving_threshold_m]),
            'stationary_actual_horizons': summarize_subset([r for r in pairs if r['actual_xy_displacement_m'] < moving_threshold_m]),
            'pairs': pairs,
            'raw_endpoint_xy_error_m': distribution(raw_error),
            'pd_endpoint_xy_error_m': distribution(pd_error),
            'raw_pd_endpoint_xy_disagreement_m': distribution(disagreements),
            'raw_pd_initial_state_disagreement_m': distribution(initial_mismatch),
            'actual_state_method': 'linear interpolation between measured states; no extrapolation',
            'interpretation': 'Closed-loop future includes subsequent replans; motion strata are diagnostic, not task-success criteria.'}


def analyze(records, minimum_time_s=4., knot=9, knot_period_s=.075):
    motions = {}
    negative, complementarity = [], []
    zero_input_frozen = 0
    for r in records:
        if r.get('utime_us', 0) < minimum_time_s * 1e6:
            continue
        if r['event'] == 'planner_diagnostic':
            t = next((t for t in r['trajectories'] if t['name'] == 'object_position_target_0'), None)
            if t is None:
                continue
            points = np.asarray(t['datapoints'], dtype=float)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] <= knot:
                raise ValueError('Malformed object trajectory or unavailable comparison knot')
            if not np.isfinite(points).all():
                raise ValueError('Nonfinite predicted trajectory')
            motions.setdefault(r['channel'], []).append(float(np.linalg.norm(points[:2, knot] - points[:2, 0])))
        elif r['event'] == 'solver_diagnostic':
            x, u, lam, z = [np.asarray(r[k], dtype=float) for k in ('x', 'u', 'lambda', 'z')]
            if any(v.ndim != 2 or not np.isfinite(v).all() for v in (x, u, lam, z)):
                raise ValueError('Malformed solver diagnostic')
            eta = z[len(x) + len(lam) + len(u):]
            if eta.shape != lam.shape:
                raise ValueError('Unexpected C3+ coordinate layout')
            # Fallback leaves stale eta in native C3+. Exclude its products.
            if np.max(np.abs(u)) == 0 and np.max(np.abs(x - x[:, :1])) == 0:
                zero_input_frozen += 1
                continue
            negative.append(float(max(0, -np.min(lam))))
            complementarity.append(float(np.max(np.abs(lam * eta))))
    return {'schema': 'nonprehensile.c3_solver_consistency.v1',
            'minimum_message_time_s': minimum_time_s, 'comparison_knot': knot,
            'modes': 'all controller modes; not restricted to executed C3 segments',
            'planar_prediction_m': {k: distribution(v) for k, v in motions.items()},
            'maximum_negative_lambda_magnitude': distribution(negative),
            'maximum_lambda_eta_product': distribution(complementarity),
            'zero_input_frozen_solution_count': zero_input_frozen,
            'paired_raw_pd_actual': paired_predictions(records, minimum_time_s, knot, knot_period_s),
            'limitations': ['Lambda uses native contact coordinates; eta remains LCS-scaled.',
                            'Products describe final QP only when end_on_qp_step is true.',
                            'Raw and PD channels are summarized separately at a common knot.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-time-s', type=float, default=4.)
    p.add_argument('--knot', type=int, default=9)
    p.add_argument('--knot-period-s', type=float, default=.075)
    args = p.parse_args()
    if args.knot < 1 or not np.isfinite(args.minimum_time_s) or not np.isfinite(args.knot_period_s) or args.knot_period_s <= 0:
        raise ValueError('Invalid comparison window')
    report = analyze([json.loads(line) for line in args.audit.read_text().splitlines() if line.strip()],
                     args.minimum_time_s, args.knot, args.knot_period_s)
    report['audit'] = str(args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
