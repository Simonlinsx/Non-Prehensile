#!/usr/bin/env python3
"""Extract an executed, captured C3 reference for command-free native replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def extract(result, records, native_records):
    if result.get('diagnostic_reference_replay') is not True or result.get('error') is not None:
        raise ValueError('Requires a completed physical reference replay')
    references = [r for r in records if r.get('event') == 'task_reference'
                  and r.get('diagnostic_reference_replay_active')]
    if not references:
        raise ValueError('No captured reference')
    reference = references[0]
    stamp = reference['plan_utime_us']
    matched = [r for r in native_records if abs(r['utime_us']-stamp) <= 1]
    measured = [r for r in result['trace'] if abs(r['measurement_utime_us']-stamp) <= 1]
    if len(matched) != 1 or len(measured) != 1:
        raise ValueError('Native reference and measured arm pose must match uniquely')
    native, row = matched[0], measured[0]
    if not reference['c3_mode'] or native['force_tracking_enabled'] is not True:
        raise ValueError('Requires an executed C3 reference with physical force tracking')
    x, u, position = map(np.asarray, (native['raw_states'], reference['force_knots_n'], reference['position_knots_m']))
    arm_q = np.asarray(row['measured_joint_position_rad'])
    N = len(u)
    if (x.shape != (N+1, 19) or u.shape != (N, 3) or position.shape != (N, 3)
            or arm_q.shape != (7,) or not all(np.isfinite(v).all() for v in (x, u, position, arm_q))):
        raise ValueError('Invalid reference or measured arm dimensions/values')
    if not np.array_equal(x[:-1, :3], position):
        raise ValueError('Published executed actor reference differs from full-precision native plan')
    dt = native['dt_s']
    if not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(reference['knot_times_s']), dt, atol=1e-10, rtol=0):
        raise ValueError('Native and executed knot durations differ')
    # Each reported reference within the captured window must be identical.
    for other in references:
        if any(other[k] != reference[k] for k in ('force_knots_n', 'position_knots_m', 'plan_utime_us')):
            raise ValueError('Physical replay changed its captured reference')
    return dict(capture_utime_us=stamp, dt_s=dt, N=N, arm_q=arm_q.tolist(),
                x=x.tolist(), u=u.tolist(), executed_reference_exactly_matches_native_actor_plan=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[4, 8, 15, 30])
    args=parser.parse_args()
    if any(not 1 <= n <= 64 for n in args.resolutions):
        raise ValueError('Resolution must be in 1..64')
    source=args.scene_directory.resolve()
    paths=[source/'result.json', source/'effect_audit.jsonl', source/'controller_online.log']
    result=json.loads(paths[0].read_text())
    records=[json.loads(s) for s in paths[1].read_text().splitlines()]
    native=[json.loads(s.split(' ',1)[1]) for s in paths[2].read_text().splitlines()
            if s.startswith('C3_PD_REFERENCE_COMPARISON ')]
    extracted=extract(result, records, native)
    output=args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files={}
    for resolution in args.resolutions:
        fields=[extracted['N'], 19, 3, extracted['dt_s'], resolution, 1]
        fields += extracted['arm_q']
        fields += [v for row in extracted['x'] for v in row]
        fields += [v for row in extracted['u'] for v in row]
        path=output/f'resolution{resolution}.txt'
        path.write_text(' '.join(map(str,fields))+'\n')
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata=dict(extracted, source_scene_directory=str(source),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  input_sha256=files,
                  scope='Offline predictions of an already executed reference; no new task success claims.')
    (output/'input_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('x','u','source_sha256')}))


if __name__ == '__main__': main()
