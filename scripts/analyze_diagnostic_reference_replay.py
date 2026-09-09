#!/usr/bin/env python3
"""Audit a single captured native reference; never reports task acceptance."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_c3_solver_consistency import distribution, interpolate_quaternion, rotation_error


def analyze_replay(records, result):
    starts = [r for r in records if r['event'] == 'diagnostic_reference_replay_start']
    ends = [r for r in records if r['event'] == 'diagnostic_reference_replay_end']
    if len(starts) != 1 or len(ends) != 1 or result.get('diagnostic_reference_replay') is not True:
        raise ValueError('Need exactly one complete, explicitly marked diagnostic replay')
    start = starts[0]
    first, last = start['utime_us'], start['end_utime_us']
    if ends[0]['planned_end_utime_us'] != last or ends[0]['utime_us'] < last:
        raise ValueError('Replay end does not match capture')
    references = [r for r in records if r['event'] == 'task_reference' and first <= r['relay_state_utime_us'] < last]
    if (not references or references[0]['relay_state_utime_us'] != first or
            any(not r.get('diagnostic_reference_replay_active') or r['plan_utime_us'] != start['plan_utime_us'] for r in references)):
        raise ValueError('Reference changed or missing during diagnostic window')
    knots = np.asarray(references[0]['knot_times_s'])
    positions = np.asarray(references[0]['position_knots_m'])
    forces = np.asarray(references[0]['force_knots_n'])
    if (len(knots) < 2 or len(knots) != len(positions) or len(knots) != len(forces)
            or np.any(np.diff(knots) <= 0) or not np.allclose(np.diff(knots), np.diff(knots)[0])):
        raise ValueError('Invalid or nonuniform native reference knots')
    if any(r['knot_times_s'] != references[0]['knot_times_s'] or r['position_knots_m'] != references[0]['position_knots_m'] or
           r['force_knots_n'] != references[0]['force_knots_n'] for r in references):
        raise ValueError('Captured reference bytes changed during replay')
    states = sorted((r for r in records if r['event'] == 'measured_state'), key=lambda r:r['utime_us'])
    times = np.array([r['utime_us'] for r in states])
    reference_times = np.array([r['relay_state_utime_us'] for r in references])
    state_period = np.median(np.diff(np.unique(times)))
    if (not np.isfinite(state_period) or state_period <= 0
            or np.any(np.diff(reference_times) > 1.5 * state_period)
            or reference_times[-1] + 1.5 * state_period < last):
        raise ValueError('Missing reference samples within replay horizon')
    measured_p = np.array([r['target_position_m'] for r in states])
    measured_q = np.array([r['target_quaternion_wxyz'] for r in states])
    if not all(np.isfinite(a).all() for a in (knots,positions,forces,times,measured_p,measured_q)):
        raise ValueError('Nonfinite replay data')
    if not times[0] <= first < last <= times[-1]:
        raise ValueError('Measured state does not bracket replay horizon')
    interp = lambda t, ts, values: np.array([np.interp(t, ts, values[:,axis]) for axis in range(values.shape[1])])
    actual_start = interp(first,times,measured_p)
    actual_end = interp(last,times,measured_p)
    actual_q_start = interpolate_quaternion(times,measured_q,first)
    actual_q_end = interpolate_quaternion(times,measured_q,last)
    pd = {t['name']: t for t in start['object_trajectories']}
    pp = pd['object_position_target_0']; pq = pd['object_orientation_target_0']
    period = np.diff(knots)[0]
    pred_times = first + np.asarray(pp['time_vec']) * period * 1e6
    pred_positions = np.asarray(pp['datapoints']).T[:,:3]
    pred_quaternions = np.asarray(pq['datapoints']).T
    if not pred_times[0] <= first < last <= pred_times[-1]:
        raise ValueError('Prediction does not cover actual replay duration')
    predicted_end = interp(last,pred_times,pred_positions)
    predicted_q_end = interpolate_quaternion(pred_times,pred_quaternions,last)
    # Trace reference belongs to the servo interval BEFORE its measured pose.
    dt_us = round(result['control_period_s'] * 1e6)
    rows = [r for r in result['trace'] if first <= r['measurement_utime_us'] - dt_us < last]
    raw_errors, governor_errors, force_errors, tracking_errors = [],[],[],[]
    force_sign = result['controller_parameters']['c3_force_action_sign']
    for row in rows:
        stamp = (row['measurement_utime_us'] - dt_us) / 1e6
        ideal_p = interp(stamp,knots,positions)
        ideal_f = force_sign * interp(stamp,knots,forces)
        raw_errors.append(float(np.linalg.norm(np.asarray(row['raw_task_target_c3_m'])-ideal_p)))
        governor_errors.append(float(np.linalg.norm(np.asarray(row['governed_task_target_c3_m'])-row['raw_task_target_c3_m'])))
        force_errors.append(float(np.linalg.norm(np.asarray(row['applied_feedforward_force_n'])-ideal_f)))
        tracking_errors.append(float(row['task_tracking_error_m']))
    legal_forces = [max([float(v) for k,v in row['robot_target_contact_force_n_by_sensor'].items() if k.startswith('target_hand_contacts')]+[0.])
                    for row in rows if row['legal_safe_robot_contact']]
    window_times = np.unique(np.r_[first,times[(times>first)&(times<last)],last])
    curve = [{'utime_us':float(t), 'actual_position_m':interp(t,times,measured_p).tolist(),
              'predicted_position_m':interp(t,pred_times,pred_positions).tolist()} for t in window_times]
    return dict(schema='nonprehensile.diagnostic_reference_replay_audit.v1', acceptance_eligible=False,
        plan_utime_us=start['plan_utime_us'], start_utime_us=first,end_utime_us=last,
        actual_resume_utime_us=ends[0]['utime_us'],duration_s=(last-first)/1e6,
        matched_reference_count=len(references),trace_sample_count=len(rows),
        same_reference_through_window=True,
        actual_xy_displacement_m=float(np.linalg.norm(actual_end[:2]-actual_start[:2])),
        predicted_xy_displacement_m=float(np.linalg.norm(predicted_end[:2]-pred_positions[0,:2])),
        endpoint_xy_error_m=float(np.linalg.norm(predicted_end[:2]-actual_end[:2])),
        endpoint_so3_error_rad=rotation_error(predicted_q_end,actual_q_end),
        actual_so3_displacement_rad=rotation_error(actual_q_start,actual_q_end),
        initial_xyz_error_m=float(np.linalg.norm(pred_positions[0]-actual_start)),
        initial_so3_error_rad=rotation_error(pred_quaternions[0],actual_q_start),
        servo_raw_reference_vs_ideal_foh_m=distribution(raw_errors),
        servo_governor_correction_m=distribution(governor_errors),
        applied_force_vs_signed_ideal_foh_n=distribution(force_errors),
        task_tracking_error_m=distribution(tracking_errors),
        semantic_guard_active_samples=sum(bool(row['semantic_c1_guard_active']) for row in rows),
        c1_violation_samples=sum(bool(row['forbidden_robot_contact']) for row in rows),
        positive_legal_hand_force_samples=sum(f>0 for f in legal_forces),
        legal_hand_force_n=distribution(legal_forces),curve=curve,
        limitations=['Diagnostic replay is not receding-horizon task acceptance.',
            'Endpoint between measured states uses interpolation; no extrapolation.',
            'Trace is sampled, with additional contact rows; reference/force differences quantify downstream changes.',
            'The servo extrapolates 20 Hz command position with local velocity and holds force between packets.',
            'Existing height/speed/force limits and C1 shielding remain enabled.'])


def compare_dual_predictions(native_records, records, result):
    if result.get('diagnostic_compare_pd_references') is not True:
        raise ValueError('Result does not declare read-only dual prediction diagnostics')
    start = next(r for r in records if r['event'] == 'diagnostic_reference_replay_start')
    matches = [r for r in native_records if abs(r['utime_us'] - start['plan_utime_us']) <= 1]
    if len(matches) != 1:
        raise ValueError('Need one native comparison for the captured plan')
    native = matches[0]
    stamp, end = start['utime_us'], start['end_utime_us']
    pd = {r['name']: np.asarray(r['datapoints']).T for r in start['object_trajectories']}
    recorded_pd = np.hstack([pd['object_orientation_target_0'],pd['object_position_target_0']])
    zoh = np.asarray(native['zoh_states'])
    if zoh.ndim != 2 or zoh.shape[1] != 19 or zoh.shape[0] != len(recorded_pd):
        raise ValueError('Expected target-only 19-dimensional native state')
    selected_mode=result['controller_parameters']['native_pd_rollout_interpolation']
    if selected_mode not in ('zoh','foh'):
        raise ValueError('Unknown recorded PD reference interpolation')
    selected=np.asarray(native[selected_mode+'_states'])
    difference = float(np.max(np.abs(selected[:,3:10]-recorded_pd)))
    if difference > 1e-6:
        raise ValueError('Read-only selected prediction differs from captured executed-plan prediction')
    states = sorted((r for r in records if r['event']=='measured_state'),key=lambda r:r['utime_us'])
    times=np.array([r['utime_us'] for r in states]);positions=np.array([r['target_position_m'] for r in states])
    quat=[r['target_quaternion_wxyz'] for r in states]
    interp=lambda t,ts,ps:np.array([np.interp(t,ts,ps[:,i]) for i in range(ps.shape[1])])
    actual=interp(end,times,positions);actual_q=interpolate_quaternion(times,quat,end)
    trace=sorted(result['trace'],key=lambda r:r['measurement_utime_us'])
    tip_times=np.array([r['measurement_utime_us'] for r in trace]);tip_positions=np.array([r['planner_tip_position_m'] for r in trace])
    if not tip_times[0] <= stamp < end <= tip_times[-1]:
        raise ValueError('Tip measurements do not bracket comparison window')
    actual_tip=interp(end,tip_times,tip_positions)
    predicted_times=stamp+np.arange(len(zoh))*native['dt_s']*1e6
    modes={}
    for mode in ['raw','zoh','foh']:
        x=np.asarray(native[mode+'_states'])
        if x.shape != zoh.shape or not np.isfinite(x).all():
            raise ValueError('Invalid native comparison state matrix')
        p=interp(end,predicted_times,x[:,7:10]);q=interpolate_quaternion(predicted_times,x[:,3:7],end)
        tip=interp(end,predicted_times,x[:,:3])
        modes[mode]={'target_endpoint_xy_error_m':float(np.linalg.norm(p[:2]-actual[:2])),
                     'target_endpoint_so3_error_rad':rotation_error(q,actual_q),
                     'target_predicted_xy_motion_m':float(np.linalg.norm(p[:2]-x[0,7:9])),
                     'actor_endpoint_xyz_error_m':float(np.linalg.norm(tip-actual_tip)),
                     'actor_predicted_position_m':tip.tolist(),
                     'target_predicted_position_m':p.tolist()}
    return {'native_plan_utime_us':native['utime_us'],'recorded_selected_pd_max_state_difference':difference,'selected_interpolation':selected_mode,
            'force_tracking_enabled':native['force_tracking_enabled'],
            'actual_actor_position_m':actual_tip.tolist(),'actual_target_position_m':actual.tolist(),
            'modes':modes,'interpretation':'Both PD predictions use identical raw plan, fine LCS, initial state and gains; only reference interpolation differs.'}



def compare_calibrated_predictions(calibrations, native_records, records, result):
    if result.get('diagnostic_pd_inertia_calibration') is not True:
        raise ValueError('Result does not declare inertia calibration diagnostics')
    if result['controller_parameters']['native_pd_rollout_interpolation'] != 'zoh':
        raise ValueError('Calibration comparison currently requires unchanged ZOH selection')
    start = next(r for r in records if r['event'] == 'diagnostic_reference_replay_start')
    native = [r for r in native_records if abs(r['utime_us'] - start['plan_utime_us']) <= 1]
    matches = [r for r in calibrations if abs(r['utime_us'] - start['plan_utime_us']) <= 1]
    geometries = ['frozen', 'relinearized'] if result.get('diagnostic_pd_contact_relinearization') else ['frozen']
    expected = sorted((mass,geometry) for mass in [.057,1.] for geometry in geometries)
    if len(native) != 1 or sorted((r['actor_mass_kg'],r.get('contact_geometry','frozen')) for r in matches) != expected:
        raise ValueError('Need each declared calibrated prediction and one unchanged native plan')
    output = []
    for row in matches:
        if 'error' in row:
            output.append(dict(row, prediction_valid=False))
            continue
        if (row['dt_s'] != native[0]['dt_s'] or row['force_sign'] != -1 or
                not row['gravity_compensated'] or row['acceleration_kp'] != 200 or row['acceleration_kd'] != 20 or
                row['force_tracking_enabled'] != native[0]['force_tracking_enabled'] or
                not np.array_equal(row['states'][0], native[0]['raw_states'][0])):
            raise ValueError('Calibration timing, input convention or initial state does not match')
        # Reuse the independently checked measured-state/SO(3) interpolation and
        # unchanged ZOH-to-LCM consistency check; the counterfactual replaces FOH only.
        comparison = compare_dual_predictions([dict(native[0], foh_states=row['states'])], records, result)
        output.append({k:v for k,v in row.items() if k != 'states'} | comparison['modes']['foh'] | {'prediction_valid': True})
    return {'variants': output, 'acceptance_eligible': False,
            'limitations': ['Scalar inertia does not reproduce measured anisotropic OSC inertia.',
                           'FOH force differs from the 20 Hz held execution force.',
                           'Only prediction changes; active C3 cost and control remain unchanged.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-log',type=Path)
    args=parser.parse_args()
    records=[json.loads(l) for l in args.audit.read_text().splitlines() if l.strip()]
    result=json.loads(args.result.read_text())
    report=analyze_replay(records,result)
    if args.native_log is not None:
        native_records=[json.loads(line.split(' ',1)[1]) for line in args.native_log.read_text().splitlines()
                        if line.startswith('C3_PD_REFERENCE_COMPARISON ')]
        report['dual_pd_comparison']=compare_dual_predictions(native_records,records,result)
        report['native_log_sha256']=hashlib.sha256(args.native_log.read_bytes()).hexdigest()
        if result.get('diagnostic_pd_inertia_calibration'):
            calibrations=[json.loads(line.split(' ',1)[1]) for line in args.native_log.read_text().splitlines()
                          if line.startswith('C3_PD_CALIBRATION ')]
            report['calibrated_pd_comparison']=compare_calibrated_predictions(calibrations,native_records,records,result)

    report.update(result=str(args.result),result_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest(),audit=str(args.audit),audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='curve'}))


if __name__=='__main__': main()
