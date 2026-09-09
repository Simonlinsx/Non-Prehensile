#!/usr/bin/env python3
"""Compare offline prediction resolutions against a captured physical replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .audit_strict_pose_dwell import pose_errors
except ImportError:
    from audit_strict_pose_dwell import pose_errors


def tagged(path, prefix):
    return [json.loads(s[len(prefix):]) for s in path.read_text().splitlines() if s.startswith(prefix)]


def audit_sampled_references(path, result, capture_us, duration):
    """Match pre-step commands to post-step trace timestamps; never fill gaps."""
    rows = tagged(path, 'C3_OFFLINE_PD_REFERENCE ')
    if not rows:
        raise ValueError('Sampled-clock prediction has no reference audit')
    period_us = round(result['control_period_s'] * 1e6)
    if period_us != 10000:
        raise ValueError('Sampled model requires the recorded 100 Hz executor')
    fields = [('position_m', 'raw_task_target_c3_m', 1),
              ('velocity_m_s', 'task_velocity_m_s', 1),
              ('external_force_n', 'feedforward_force_n', -1)]
    maximum = {name: 0.0 for name, _, _ in fields}
    applied_maximum = dict(position_m=0.0, velocity_m_s=0.0, external_force_n=0.0)
    matched, missing = [], []
    for row in rows:
        time = row['time_s']
        if time >= duration - 1e-9:
            continue
        candidates = [t for t in result['trace'] if abs(
            t['measurement_utime_us'] - period_us - capture_us - round(time * 1e6)) <= 1]
        if not candidates:
            missing.append(time)
            continue
        if len(candidates) != 1:
            raise ValueError('Duplicate measured command timestamp')
        measured = candidates[0]
        for name, key, sign in fields:
            delta = np.asarray(row[name]) - sign * np.asarray(measured[key])
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite command comparison')
            maximum[name] = max(maximum[name], float(np.max(np.abs(delta))))
        for name, key in [('position_m', 'governed_task_target_c3_m'),
                          ('velocity_m_s', 'osc_reference_tip_velocity_m_s'),
                          ('external_force_n', 'applied_feedforward_force_n')]:
            delta = np.asarray(row[name]) - np.asarray(measured[key])
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite applied command comparison')
            applied_maximum[name] = max(applied_maximum[name], float(np.max(np.abs(delta))))
        matched.append(time)
    if not matched or max(maximum.values()) > 1e-10:
        raise ValueError('Sampled-clock references differ from recorded raw commands')
    return dict(matched_samples=len(matched), matched_times_s=matched,
                unavailable_trace_times_s=missing, raw_reference_maximum_absolute_error=maximum,
                applied_reference_maximum_absolute_error=applied_maximum,
                all_available_raw_references_match=True, complete_trace_coverage=not missing,
                limitation='Sparse historical trace: missing timestamps are not verified. '
                           'Position governor and physical dynamics are not modeled.')


def analyze(directory, baseline_log):
    provenance=directory/'input_provenance.json'
    meta=json.loads(provenance.read_text())
    for filename,digest in meta['input_sha256'].items():
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Offline input changed')
    for filename,digest in meta['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Physical replay source changed')
    source=Path(meta['source_scene_directory'])
    result=json.loads((source/'result.json').read_text())
    eligibility_path=source/'reference_replay_audit.json'
    eligibility=json.loads(eligibility_path.read_text())
    if (eligibility['same_reference_through_window'] is not True
            or eligibility['semantic_guard_active_samples'] or eligibility['c1_violation_samples']
            or eligibility['positive_legal_hand_force_samples'] <= 0):
        raise ValueError('Physical replay window is not an unshielded contact comparison')
    duration=eligibility['duration_s']
    knot=round(duration/meta['dt_s'])
    if abs(knot*meta['dt_s']-duration) > 1e-9 or not 0 < knot <= meta['N']:
        raise ValueError('Physical endpoint must coincide with a published model knot')
    stamp=meta['capture_utime_us']
    records=[json.loads(s) for s in (source/'effect_audit.jsonl').read_text().splitlines()]
    measured=[r for r in records if r['event']=='measured_state'
              and abs(r['utime_us']-(stamp+round(duration*1e6))) <= 1]
    if len(measured) != 1:
        raise ValueError('Missing unique measured physical endpoint')
    target=measured[0]['target_position_m']+measured[0]['target_quaternion_wxyz']
    raw=[r for r in tagged(baseline_log,'C3_PD_REFERENCE_COMPARISON ')
         if abs(r['utime_us']-stamp) <= 1]
    baseline=[r for r in tagged(baseline_log,'C3_PD_CALIBRATION ')
              if abs(r['utime_us']-stamp) <= 1 and r['actor_mass_kg']==1
              and r['contact_geometry']=='relinearized' and 'states' in r]
    if len(raw)!=1 or len(baseline)!=1 or not np.array_equal(raw[0]['raw_states'],meta['x']):
        raise ValueError('Baseline does not correspond to the exact supplied native state plan')
    predictions={}
    for filename in meta['input_sha256']:
        resolution=int(Path(filename).stem.removeprefix('resolution'))
        path=directory/f'resolution{resolution}.log'
        rows=tagged(path,'C3_OFFLINE_PD_REPLAY ')
        if len(rows)!=1:
            raise ValueError('Missing or duplicate offline prediction')
        r=rows[0]
        execution_path = directory / f'resolution{resolution}_execution.json'
        execution = json.loads(execution_path.read_text())
        options = tagged(path, 'C3_OFFLINE_PD_LCP_OPTIONS ')
        expected_tolerance = execution.get('env_override', {}).get('PUSH_ANYTHING_OFFLINE_PD_LCP_ZERO_TOL')
        if (len(options) > 1 or bool(options) != bool(expected_tolerance)
                or options and options[0]['zero_tol'] != float(expected_tolerance)):
            raise ValueError('Recorded LCP tolerance differs from executed diagnostic')
        tolerance = options[0]['zero_tol'] if options else 'original_default'
        if r.get('failed'):
            predictions[resolution]=dict(prediction_valid=False,error=r['error'], lcp_zero_tolerance=tolerance)
            continue
        states=np.asarray(r['states'])
        if (r['resolution']!=resolution or r['dt_s']!=meta['dt_s']
                or states.shape!=(meta['N']+1,19) or not np.isfinite(states).all()
                or not np.array_equal(states[0],meta['x'][0])):
            raise ValueError('Invalid offline prediction contract')
        errors=pose_errors(states[knot,7:10].tolist(),states[knot,3:7].tolist(),target)
        predictions[resolution]=dict(prediction_valid=True,fine_dt_s=meta['dt_s']/resolution,
                                    sampled_reference_clocks=r.get('sampled_reference_clocks', False),
                                    lcp_zero_tolerance=tolerance,
                                    endpoint_xy_error_m=errors[0],endpoint_height_error_m=errors[1],
                                    endpoint_so3_error_rad=errors[2],
                                    predicted_xy_displacement_m=float(np.linalg.norm(states[knot,7:9]-states[0,7:9])),
                                    output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if r.get('sampled_reference_clocks', False):
            predictions[resolution]['reference_clock_audit'] = audit_sampled_references(
                path, result, stamp, duration)
        inertia_rows = tagged(path, 'C3_OFFLINE_PD_INERTIA ')
        if inertia_rows:
            inertia_path = directory / 'inertia_provenance.json'
            inertia_meta = json.loads(inertia_path.read_text())
            if (len(inertia_rows) != 1 or not r.get('sampled_reference_clocks')
                    or not np.array_equal(inertia_rows[0]['matrix_kg'], inertia_meta['matrix_kg'])
                    or hashlib.sha256((directory / 'inertia.txt').read_bytes()).hexdigest()
                    != inertia_meta['inertia_input_sha256']):
                raise ValueError('Tensor prediction differs from recorded inertia input')
            for filename, digest in inertia_meta['source_sha256'].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                    raise ValueError('Recorded inertia source changed')
            checks = tagged(path, 'C3_OFFLINE_PD_INERTIA_TRANSFORM ')
            if (len(checks) != meta['N'] or [c['step'] for c in checks] != list(range(0, meta['N']*resolution, resolution))
                    or any(not np.isfinite([c['maximum_coefficient_error'], c['maximum_lcp_state_error']]).all()
                           or c['maximum_coefficient_error'] >= 1e-8 or c['maximum_lcp_state_error'] >= 1e-7
                           for c in checks)):
                raise ValueError('Missing or failed scalar-inertia factory equivalence checks')
            predictions[resolution]['inertia_model'] = dict(
                mode='frozen_measured_translational_tensor', matrix_kg=inertia_rows[0]['matrix_kg'],
                factory_equivalence_checks=len(checks),
                maximum_coefficient_error=max(c['maximum_coefficient_error'] for c in checks),
                maximum_lcp_state_error=max(c['maximum_lcp_state_error'] for c in checks),
                provenance=str(inertia_path), limitation=inertia_meta['limitation'])
        else:
            predictions[resolution]['inertia_model'] = dict(mode='scalar', mass_kg=1.)
        if resolution==4:
            difference=float(np.max(np.abs(states-np.asarray(baseline[0]['states']))))
            predictions[resolution]['maximum_state_difference_from_online_baseline']=difference
            if difference != 0:
                raise ValueError('Original-resolution offline predictions do not exactly reproduce online baseline')
    if 4 not in predictions or not predictions[4]['prediction_valid']:
        raise ValueError('A valid original-resolution baseline is required')
    return dict(schema='nonprehensile.offline_pd_resolution_comparison.v1',
                acceptance_eligible=False,capture_utime_us=stamp,duration_s=duration,
                original_resolution_exactly_reproduced=True,predictions=predictions,
                physical_replay_audit=str(eligibility_path),
                artifact_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (provenance,eligibility_path,baseline_log)},
                limitations=['A counterfactual model comparison, not another physical trial.',
                             'Nearby overlapping windows do not constitute independent held-out scenes.',
                             'All variants retain supplied knots and force convention; sampled_reference_clocks and inertia_model explicitly identify model changes.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--baseline-native-log',type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.directory,args.baseline_native_log)
    (args.directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['predictions']))


if __name__=='__main__':main()
