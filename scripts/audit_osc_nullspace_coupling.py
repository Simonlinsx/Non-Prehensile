"""Reconstruct the recorded OSC nullspace contribution and a full-inertia alternative."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(result):
    cfg = result['controller_parameters']
    if cfg['osc_nullspace_target'] != 'push_anything_joint2':
        raise ValueError('Requires the recorded joint2-only posture objective')
    period_us = round(result['control_period_s'] * 1e6)
    kp = cfg['osc_nullspace_stiffness']
    kd = 2 * cfg['osc_nullspace_damping_ratio'] * np.sqrt(kp)
    trace = {t['measurement_utime_us']: t for t in result['trace']}
    if len(trace) != len(result['trace']):
        raise ValueError('Duplicate measured trace time')
    conditions, rows, missing = [], [], 0
    for stamp, row in trace.items():
        state = row.get('osc_dynamics_audit')
        if state is None:
            continue
        if (state['frame'] != 'robot_root_axes_at_task_point'
                or state['inertial_dynamics_decoupling'] is not True
                or state['partial_inertial_dynamics_decoupling'] is not True):
            raise ValueError('Requires the recorded partial-inertia controller')
        J = np.asarray(state['jacobian_b'])
        M = np.asarray(state['joint_mass_matrix_kg_m2'])
        q = np.asarray(state['joint_position_rad'])
        qd = np.asarray(state['joint_velocity_rad_s'])
        if (J.shape != (6, 7) or M.shape != (7, 7) or q.shape != (7,) or qd.shape != (7,)
                or not all(np.isfinite(a).all() for a in (J, M, q, qd))):
            raise ValueError('Invalid recorded arm dynamics')
        inverse_mass = np.linalg.inv(M)
        task_inverse_mass = J @ inverse_mass @ J.T
        conditions.append(float(np.linalg.cond(task_inverse_mass)))
        previous = trace.get(stamp-period_us)
        if previous is None:
            missing += 1
            continue
        # The runner sets this target once before env.step; other joints use
        # the measured position at the immediately preceding servo boundary.
        target = np.asarray(previous['measured_joint_position_rad']).copy()
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Invalid previous servo target state')
        target[1] = 1.1
        desired_joint_acceleration = kp*(target-q)-kd*qd
        kinematic_projector = np.eye(7)-J.T @ np.linalg.pinv(J).T
        full_projector = np.eye(7)-J.T @ np.linalg.inv(task_inverse_mass) @ J @ inverse_mass
        null_torque = kinematic_projector @ M @ desired_joint_acceleration
        partial_acceleration = J @ inverse_mass @ null_torque
        full_acceleration = J @ inverse_mass @ full_projector @ M @ desired_joint_acceleration
        rows.append(dict(measurement_utime_us=stamp, null_torque_nm=null_torque.tolist(),
                         partial_task_acceleration=partial_acceleration.tolist(),
                         partial_translation_norm_m_s2=float(np.linalg.norm(partial_acceleration[:3])),
                         partial_rotation_norm_rad_s2=float(np.linalg.norm(partial_acceleration[3:])),
                         full_task_acceleration_residual_norm=float(np.linalg.norm(full_acceleration))))
    if not rows:
        raise ValueError('No consecutive measured records with cached dynamics')
    def quantiles(values):
        return dict(zip(('minimum', 'median', 'p95', 'maximum'), map(float, np.quantile(values, [0, .5, .95, 1]))))
    return dict(schema='nonprehensile.osc_nullspace_coupling.v1',
                dynamics_records=len(conditions), consecutive_servo_pairs=len(rows),
                missing_previous_servo_records=missing, nullspace_kp=kp, nullspace_kd=float(kd),
                full_task_inverse_inertia_condition=quantiles(conditions),
                partial_null_translation_acceleration_m_s2=quantiles([r['partial_translation_norm_m_s2'] for r in rows]),
                partial_null_rotation_acceleration_rad_s2=quantiles([r['partial_rotation_norm_rad_s2'] for r in rows]),
                maximum_full_null_residual=float(max(r['full_task_acceleration_residual_norm'] for r in rows)),
                rows=rows, limitations=[
                    'Reconstructed instantaneous acceleration contribution, not measured total acceleration or task success.',
                    'Uses recorded cached dynamics before the final physics substep and the previous measured servo target.',
                    'Float64 algebra; physical float32 arithmetic, contacts and torque saturation require simulation validation.',
                    'This one debug trajectory does not establish conditioning across random initial poses.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.result.read_text()))
    report['source_result_sha256'] = hashlib.sha256(args.result.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
