"""Extract full-precision inputs for replay of one executed native C3 period."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def extract(result, captures, references, requested_s):
    requested = round(requested_s*1e6)
    matched = [r for r in captures if abs(r['utime_us']-requested) <= 1]
    packets = [r for r in references if r.get('event') == 'task_reference'
               and abs(r['relay_state_utime_us']-requested) <= 1
               and abs(r['plan_utime_us']-requested) <= 1]
    measured = [r for r in result['trace'] if abs(r['measurement_utime_us']-requested) <= 1]
    if len(matched) != 1 or len(packets) != 1 or len(measured) != 1:
        raise ValueError('Capture, fresh command and measured joint pose must match uniquely')
    native, packet, row = matched[0], packets[0], measured[0]
    if native['force_tracking_enabled'] is not True or packet['c3_mode'] is not True:
        raise ValueError('Capture must execute a force-tracking C3 command')
    x, u, q = map(np.asarray, (native['raw_states'], native['raw_forces'], row['measured_joint_position_rad']))
    N = len(u)
    if (x.shape != (N+1,19) or u.shape != (N,3) or q.shape != (7,)
            or not all(np.isfinite(v).all() for v in (x,u,q))):
        raise ValueError('Invalid full-precision capture shape or values')
    if not np.array_equal(x[:-1,:3], packet['position_knots_m']):
        raise ValueError('Captured actor states differ from executed position reference')
    if not np.array_equal(u, packet['force_knots_n']):
        raise ValueError('Captured forces differ from executed force reference')
    dt = native['dt_s']
    if dt != .075 or not np.allclose(np.diff(packet['knot_times_s']),dt,rtol=0,atol=1e-10):
        raise ValueError('Capture and reference must use matching75ms knots')
    return dict(capture_utime_us=native['utime_us'], dt_s=dt,N=N,arm_q=q.tolist(),x=x.tolist(),u=u.tolist(),
                executed_actor_positions_and_forces_match_exactly=True,
                measured_joint_pose_timestamp_us=row['measurement_utime_us'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory',type=Path,required=True)
    parser.add_argument('--capture-time-s',type=float,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--resolutions',type=int,nargs='+',default=[4,30])
    args = parser.parse_args()
    if not np.isfinite(args.capture_time_s) or args.capture_time_s < 0 or any(not 1 <= r <= 64 for r in args.resolutions):
        raise ValueError('Invalid capture time or replay resolution')
    root = args.scene_directory.resolve()
    paths = [root/'result.json',root/'effect_audit.jsonl',root/'controller_online.log',root.parent/'evaluation_config.json']
    result = json.loads(paths[0].read_text())
    config = json.loads(paths[3].read_text())
    if config.get('native_pd_capture_time_s') != args.capture_time_s:
        raise ValueError('Capture time differs from recorded configuration')
    captures = [json.loads(l.split(' ',1)[1]) for l in paths[2].read_text().splitlines() if l.startswith('C3_PD_REPLAY_CAPTURE ')]
    with paths[1].open() as stream:
        references = [r for l in stream if (r:=json.loads(l)).get('event') == 'task_reference']
    extracted = extract(result,captures,references,args.capture_time_s)
    output = args.output_directory.resolve()
    output.mkdir(parents=True,exist_ok=False)
    files = {}
    for resolution in args.resolutions:
        values = [extracted['N'],19,3,extracted['dt_s'],resolution,1]+extracted['arm_q']
        values += [v for row in extracted['x'] for v in row]+[v for row in extracted['u'] for v in row]
        f = output/f'resolution{resolution}.txt'
        f.write_text(' '.join(map(str,values))+'\n')
        files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    metadata = dict(extracted,source_scene_directory=str(root),
                    input_sha256=files,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},
                    scope='Only first50ms is executed before replanning; longer offline predictions are not an observed open-loop rollout.')
    (output/'input_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('x','u','source_sha256')}))


if __name__ == '__main__':
    main()
