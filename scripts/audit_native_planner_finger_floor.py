#!/usr/bin/env python3
"""Check native finger geometry floor against execution and constrained knots."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(r'^C3_PLANNER_FINGER_FLOOR utime_us=(\d+) minimum_actor_z=(\S+) table_z=(\S+) minimum_relative_z=(\S+)$')


def audit(result, records, native_text):
    clearance = result['controller_parameters'].get('native_planner_finger_table_clearance')
    if clearance is None or not np.isfinite(clearance):
        raise ValueError('Actual runtime does not declare a planner finger floor')
    floors = []
    for line in native_text.splitlines():
        match = PATTERN.match(line)
        if match:
            t, floor, table, relative = map(float, match.groups())
            if not np.isfinite([t, floor, table, relative]).all() or abs(floor - (table + clearance - relative)) > 1e-10:
                raise ValueError('Invalid native geometric floor')
            floors.append(dict(utime_us=int(t), floor_m=floor, table_z_m=table))
    if not floors or len({r['utime_us'] for r in floors}) != len(floors):
        raise ValueError('Missing or duplicate native floor records')
    times = np.array([r['utime_us'] for r in floors])

    def match_floor(stamp):
        indices = np.flatnonzero(np.abs(times - stamp) <= 1)
        return floors[indices[0]] if len(indices) == 1 else None

    alignment = []
    dt_us = round(result['control_period_s'] * 1e6)
    for row in result['trace']:
        # The stored execution floor is computed before this measured step.
        native = match_floor(row['measurement_utime_us'] - dt_us)
        if native is not None:
            difference = row['task_reference_floor_c3_m'] - native['floor_m']
            alignment.append(dict(utime_us=native['utime_us'], difference_m=difference))
    if not alignment:
        raise ValueError('No exactly synchronized execution geometry samples')
    plans = []
    for row in records:
        if row['event'] != 'solver_diagnostic':
            continue
        native = match_floor(row['utime_us'])
        if native is None:
            continue
        x = np.asarray(row['x'])
        if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError('Invalid current C3 state trajectory')
        # Original STATE API excludes x0; the unexecuted synthetic xN is absent.
        violation = max(0., native['floor_m'] - float(x[2, 1:].min()))
        plans.append(dict(utime_us=row['utime_us'], published_future_knots=x.shape[1]-1, maximum_violation_m=violation))
    if not plans:
        raise ValueError('No native plans matched to height constraints')
    maximum_alignment = max(abs(r['difference_m']) for r in alignment)
    maximum_violation = max(r['maximum_violation_m'] for r in plans)
    return dict(schema='nonprehensile.native_planner_finger_floor_audit.v1',
                native_floor_records=len(floors), exact_geometry_matches=len(alignment),
                maximum_native_vs_execution_floor_difference_m=maximum_alignment,
                geometry_alignment_pass=maximum_alignment < 1e-5,
                matched_current_plans=len(plans), maximum_published_knot_floor_violation_m=maximum_violation,
                published_knot_floor_pass=maximum_violation <= 1e-5,
                plans_violating_by_more_than_10um=sum(r['maximum_violation_m'] > 1e-5 for r in plans),
                measured_initial_state_excluded=True, alignment=alignment, plans=plans,
                limitations=['Geometry comparison is synchronized to the command interval, not the subsequent measured pose.',
                             'Solver messages contain float32 state values; 10 micrometres is the reporting tolerance.',
                             'This verifies reference geometry and current-plan constraints, not task acceptance or all robot/table contacts.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--native-log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=audit(json.loads(args.result.read_text()),[json.loads(s) for s in args.audit.read_text().splitlines()],args.native_log.read_text())
    report['artifact_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.result,args.audit,args.native_log]}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['alignment','plans','artifact_sha256']}))
    return 0 if report['geometry_alignment_pass'] and report['published_knot_floor_pass'] else 2


if __name__=='__main__':raise SystemExit(main())
