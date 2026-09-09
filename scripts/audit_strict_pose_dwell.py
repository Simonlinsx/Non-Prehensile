#!/usr/bin/env python3
"""Reconstruct continuous strict pose dwell from measured, per-step poses."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def pose_errors(position, quaternion, goal):
    values = list(position) + list(quaternion) + list(goal)
    if len(position) != 3 or len(quaternion) != 4 or len(goal) != 7:
        raise ValueError('Invalid pose dimensions')
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        raise ValueError('Non-finite measured pose')
    norm = math.sqrt(sum(v*v for v in quaternion))
    goal_norm = math.sqrt(sum(v*v for v in goal[3:]))
    if min(norm, goal_norm) < 1e-12:
        raise ValueError('Zero quaternion')
    dot = abs(sum(a*b for a, b in zip(quaternion, goal[3:])) / (norm * goal_norm))
    return (math.hypot(position[0]-goal[0], position[1]-goal[1]),
            abs(position[2]-goal[2]), 2*math.acos(min(1., dot)))


def audit(result):
    period, steps = result['control_period_s'], result['executed_steps']
    if (type(steps) is not int or steps <= 0 or type(period) not in (int, float)
            or not math.isfinite(period) or period <= 0):
        raise ValueError('Invalid control timing')
    thresholds = result['strict_pose_thresholds']
    limits = [thresholds[k] for k in ('planar_m', 'height_m', 'rotation_rad')]
    duration = thresholds['dwell_time_s']
    if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in limits+[duration]):
        raise ValueError('Invalid strict thresholds')
    required = math.ceil(duration / period - 1e-12)
    previous, continuous, maximum = -1, 0, 0
    terminal = []
    for row in result['trace']:
        step = row['step']
        if type(step) is not int or not previous < step < steps:
            raise ValueError('Unordered, duplicate, or out-of-range measured step')
        expected_time = 100_000 + round((step + 1)*period*1e6)
        if row['measurement_utime_us'] != expected_time:
            raise ValueError('Measured pose timestamp does not match its control step')
        errors = pose_errors(row['target_position_m'], row['target_quaternion_wxyz'], result['goal_pose_wxyz'])
        strict = all(error < limit for error, limit in zip(errors, limits))
        continuous = (continuous + 1 if step == previous + 1 else 1) if strict else 0
        maximum = max(maximum, continuous)
        previous = step
        if step >= steps-required:
            terminal.append(dict(step=step, planar_m=errors[0], height_m=errors[1], rotation_rad=errors[2], strict=strict))
    complete = [r['step'] for r in terminal] == list(range(steps-required, steps))
    passed = complete and all(r['strict'] for r in terminal)
    return dict(schema='nonprehensile.strict_pose_dwell_audit.v1',
                terminal_dwell_certified=passed, required_consecutive_steps=required,
                complete_terminal_measurements=complete,
                maximum_observed_consecutive_strict_steps=maximum,
                maximum_observed_strict_duration_s=maximum*period,
                terminal_samples=terminal,
                limitations=['Certifies pose dwell only; physical contact and C1 are checked separately.',
                             'Missing or decimated steps break continuity; no interpolation fills gaps.',
                             'Full SO(3) is reconstructed from normalized measured and fixed-goal quaternions.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    report=audit(json.loads(args.result.read_text()))
    report['result_sha256']=hashlib.sha256(args.result.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('terminal_samples', 'limitations')}))
    return 0 if report['terminal_dwell_certified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
