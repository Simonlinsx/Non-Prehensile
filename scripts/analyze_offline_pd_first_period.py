"""Compare exact offline fine states with the first executed 50 ms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .audit_strict_pose_dwell import pose_errors
except ImportError:
    from audit_strict_pose_dwell import pose_errors


def unique_at(rows, key, stamp):
    found = [r for r in rows if abs(r[key]-stamp) <= 1]
    if len(found) != 1:
        raise ValueError(f'Missing or ambiguous exact timestamp {stamp}')
    return found[0]


def analyze(metadata, result, measured_states, fine, references, replay):
    if result['control_period_s'] != .01 or replay.get('sampled_reference_clocks') is not True:
        raise ValueError('Requires sampled reference model and100Hz physical servo')
    dt = replay['dt_s']/replay['resolution']
    times = np.asarray([r['time_s'] for r in fine])
    expected = np.arange(1,round(.05/dt)+1)*dt
    if (len(times) != len(expected) or len(times) == 0
            or not np.allclose(times,expected,atol=1e-12,rtol=0) or abs(times[-1]-.05)>1e-12):
        raise ValueError('Fine trajectory must end exactly at50ms without missing steps')
    predictions = np.asarray([r['state'] for r in fine])
    if predictions.shape != (len(times),19) or not np.isfinite(predictions).all():
        raise ValueError('Invalid fine predicted state')
    stamp = metadata['capture_utime_us']
    start = unique_at(measured_states,'utime_us',stamp)
    end = unique_at(measured_states,'utime_us',stamp+50000)
    initial = np.asarray(metadata['x'][0])
    initial_errors = pose_errors(initial[7:10].tolist(),initial[3:7].tolist(),start['target_position_m']+start['target_quaternion_wxyz'])
    if max(initial_errors) > 2e-6:
        raise ValueError('Offline initial object pose differs from executed start')
    errors = pose_errors(predictions[-1,7:10].tolist(),predictions[-1,3:7].tolist(),end['target_position_m']+end['target_quaternion_wxyz'])
    observed, missing = [], []
    for i in range(1,6):
        t = stamp+i*10000
        matches = [r for r in result['trace'] if abs(r['measurement_utime_us']-t)<=1]
        if len(matches)>1:
            raise ValueError('Ambiguous servo sample')
        if not matches:
            missing.append(t)
            continue
        row = matches[0]
        native = [r for r in references if abs(r['time_s']-(i-1)*.01)<1e-12]
        if len(native)!=1:
            raise ValueError('Missing sampled native reference')
        native = native[0]
        observed.append(dict(measurement_utime_us=row['measurement_utime_us'],
            raw_position_error_m=float(np.linalg.norm(np.asarray(row['raw_task_target_c3_m'])-native['position_m'])),
            governed_position_error_m=float(np.linalg.norm(np.asarray(row['governed_task_target_c3_m'])-native['position_m'])),
            velocity_error_m_s=float(np.linalg.norm(np.asarray(row['osc_reference_tip_velocity_m_s'])-native['velocity_m_s'])),
            applied_force_error_n=float(np.linalg.norm(np.asarray(row['applied_feedforward_force_n'])-native['external_force_n'])),
            unshielded_safe=row.get('semantic_c1_guard_active') is False and row.get('forbidden_robot_contact') is False,
            positive_legal_contact=row.get('legal_safe_robot_contact') is True and any(
                name.startswith('target_hand_contacts') and np.isfinite(force) and force>0
                for name,force in row.get('robot_target_contact_force_n_by_sensor',{}).items())))
    return dict(schema='nonprehensile.offline_pd_first_period_comparison.v1',
        capture_utime_us=stamp,model_fine_dt_s=dt,model_state_count=len(fine),
        actual_start_utime_us=start['utime_us'],actual_end_utime_us=end['utime_us'],
        initial_pose_errors=dict(zip(('xy_m','height_m','so3_rad'),initial_errors)),
        final_prediction_errors=dict(zip(('xy_m','height_m','so3_rad'),errors)),
        predicted_target_delta_m=(predictions[-1,7:10]-initial[7:10]).tolist(),
        actual_target_delta_m=(np.asarray(end['target_position_m'])-start['target_position_m']).tolist(),
        missing_servo_timestamps=missing,servo_reference_comparisons=observed,
        eligible_unshielded_contact=len(observed)==5 and all(r['unshielded_safe'] for r in observed)
            and any(r['positive_legal_contact'] for r in observed),
        limitations=['Exact native fine-step endpoint, not interpolation of75ms knots.',
            'Offline model uses sampled20Hz/100Hz references; it does not include measured governor changes or full robot dynamics.',
            'This50ms prediction comparison is not a task-success or generalization result.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-provenance',type=Path,required=True)
    parser.add_argument('--native-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    metadata=json.loads(args.input_provenance.read_text())
    source=Path(metadata['source_scene_directory'])
    paths=[source/'result.json',source/'effect_audit.jsonl']
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['source_sha256'][str(path)]:
            raise ValueError(f'Physical source changed: {path}')
    records={name:[] for name in ('C3_OFFLINE_PD_FINE_STATE','C3_OFFLINE_PD_REFERENCE','C3_OFFLINE_PD_REPLAY')}
    for line in args.native_log.read_text().splitlines():
        key=line.split(' ',1)[0]
        if key in records: records[key].append(json.loads(line.split(' ',1)[1]))
    if len(records['C3_OFFLINE_PD_REPLAY'])!=1 or records['C3_OFFLINE_PD_REPLAY'][0].get('failed'):
        raise ValueError('Requires successful native replay')
    with paths[1].open() as stream:
        states=[r for line in stream if (r:=json.loads(line)).get('event')=='measured_state']
    report=analyze(metadata,json.loads(paths[0].read_text()),states,records['C3_OFFLINE_PD_FINE_STATE'],
                   records['C3_OFFLINE_PD_REFERENCE'],records['C3_OFFLINE_PD_REPLAY'][0])
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[args.input_provenance,args.native_log,Path(__file__)]}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','servo_reference_comparisons')}))


if __name__=='__main__':
    main()
