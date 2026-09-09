"""Compare the actual imported PhysX FR3 with original OSC at identical q."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .fr3_robot_model_contract import load_contract
except ImportError:
    from fr3_robot_model_contract import load_contract


def audit(result_path, native_binary, runtime, output):
    result = json.loads(result_path.read_text())
    contract = load_contract(result['robot_model_contract']['manifest_path'])
    if contract['manifest_sha256'] != result['robot_model_contract']['manifest_sha256']:
        raise ValueError('Robot model changed since the run')
    q_path = output.with_suffix('.q.txt')
    q_path.write_text(' '.join(format(x, '.17g') for x in result['initial_robot_joint_position_rad'])+'\n')
    run = subprocess.run([str(native_binary.resolve()), '--offline_robot_dynamics_input='+str(q_path.resolve())],
                         cwd=runtime, env=dict(os.environ, PUSH_ANYTHING_ROBOT_MODEL=contract['native_urdf']),
                         text=True, capture_output=True, check=True)
    output.with_suffix('.native.log').write_text(run.stdout+run.stderr)
    native = {}
    rows = []
    for line in run.stdout.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'OSC_MODEL_GRAVITY':
            native['gravity'] = np.array(words[1:], dtype=float)
        elif words[0] == 'OSC_MODEL_TIP':
            native['tip'] = np.array(words[1:], dtype=float)
        elif words[0] == 'OSC_MODEL_MASS_ROW':
            assert int(words[1]) == len(rows)
            rows.append([float(x) for x in words[2:]])
    native['mass'] = np.array(rows)
    assert native['mass'].shape == (7, 7)
    actual = result['initial_robot_dynamics']
    indices = [actual['joint_names'].index(f'panda_joint{i}') for i in range(1, 8)]
    actual_armature = np.array(actual['dof_armatures_kg_m2'])[indices]
    # PhysX tensor API M excludes added armature; native Drake M includes it.
    actual_mass = np.array(actual['arm_mass_matrix']) + np.diag(actual_armature)
    body_checks = {}
    model = ET.parse(contract['simulation_urdf']).getroot()
    for link in model.findall('link'):
        name = link.get('name')
        index = actual['body_names'].index(name)
        inertial = link.find('inertial')
        origin = inertial.find('origin')
        com = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
        orientation = Rotation.from_euler('xyz', [float(x) for x in origin.get('rpy', '0 0 0').split()]).as_matrix()
        inertia = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
        matrix = np.array([[inertia['ixx'],inertia['ixy'],inertia['ixz']],
                           [inertia['ixy'],inertia['iyy'],inertia['iyz']],
                           [inertia['ixz'],inertia['iyz'],inertia['izz']]])
        matrix = orientation @ matrix @ orientation.T
        actual_com = np.array(actual['body_com_pose_xyzw'][index])
        # Tensor API already returns inertia about COM in body axes, in
        # column-major order. get_coms rotation is the principal-axis frame;
        # applying that rotation here would incorrectly rotate it twice.
        imported_matrix = np.array(actual['body_inertias_kg_m2'][index]).reshape(3,3,order='F')
        body_checks[name] = dict(
            mass_error_kg=abs(actual['body_masses_kg'][index]-float(inertial.find('mass').get('value'))),
            com_error_m=float(np.max(np.abs(actual_com[:3]-com))),
            inertia_error_kg_m2=float(np.max(np.abs(imported_matrix-matrix))))
    base_position = np.array(actual['base_position_world_m'])
    base_rotation = Rotation.from_quat(np.roll(actual['base_quaternion_wxyz'], -1))
    expected_tcp = base_position + base_rotation.apply(native['tip'][:3])
    actual_tip_rotation = Rotation.from_quat(np.roll(actual['hand_quaternion_wxyz'], -1)) * Rotation.from_euler('xyz', [np.pi,0,np.pi/4])
    expected_rotation = base_rotation * Rotation.from_quat(native['tip'][3:])
    initial_tcp = np.array(result['initial_tcp_position_m'])
    drift = np.linalg.norm(np.array([r['tcp_position_m'] for r in result['trace']])-initial_tcp, axis=1)
    metrics = dict(
        maximum_body_mass_error_kg=max(v['mass_error_kg'] for v in body_checks.values()),
        maximum_body_com_error_m=max(v['com_error_m'] for v in body_checks.values()),
        maximum_body_inertia_error_kg_m2=max(v['inertia_error_kg_m2'] for v in body_checks.values()),
        maximum_gravity_error_nm=float(np.max(np.abs(np.array(actual['gravity_compensation_nm'])-native['gravity']))),
        maximum_mass_matrix_error_kg_m2=float(np.max(np.abs(actual_mass-native['mass']))),
        maximum_armature_error_kg_m2=float(np.max(np.abs(actual_armature-contract['armature_kg_m2']))),
        tcp_position_error_m=float(np.linalg.norm(expected_tcp-initial_tcp)),
        tcp_orientation_error_rad=float((expected_rotation.inv()*actual_tip_rotation).magnitude()))
    tolerances = dict(maximum_body_mass_error_kg=2e-5,maximum_body_com_error_m=2e-6,
                      maximum_body_inertia_error_kg_m2=2e-6,maximum_gravity_error_nm=5e-4,
                      maximum_mass_matrix_error_kg_m2=5e-5,maximum_armature_error_kg_m2=1e-6,
                      tcp_position_error_m=2e-6,tcp_orientation_error_rad=2e-5)
    checks = {name: value <= tolerances[name] for name, value in metrics.items()}
    # Static M/g/FK equality cannot detect friction or hidden joint drives.
    # Require backend values, not actuator configuration intent.
    friction_ok = (
        'dof_friction_properties' in actual
        and np.all(np.array(actual['dof_friction_properties'])[indices] == 0)
        and np.all(np.array(actual['dof_friction_coefficients'])[indices] == 0))
    drives_ok = ('dof_stiffness' in actual
        and np.all(np.array(actual['dof_stiffness'])[indices] == 0)
        and np.allclose(np.array(actual['dof_damping'])[indices],
            [contract['joint_dynamics'][f'panda_joint{i}']['damping'] for i in range(1,8)],
            atol=1e-7, rtol=0))
    finger_indices = [actual['joint_names'].index(f'panda_finger_joint{i}') for i in (1, 2)]
    finger_limits = [contract['joint_limits'][f'panda_finger_joint{i}'] for i in (1, 2)]
    finger_checks = {}
    for key, expected in (
        ('dof_position_limits', [[x['lower'], x['upper']] for x in finger_limits]),
        ('dof_velocity_limits', [x['velocity'] for x in finger_limits]),
        ('dof_effort_limits', [x['effort'] for x in finger_limits]),
    ):
        finger_checks[key] = bool(key in actual and np.allclose(
            np.array(actual[key])[finger_indices], expected, atol=1e-7, rtol=0))
    fingers_ok = all(finger_checks.values())
    report = dict(schema='nonprehensile.fr3_native_model_alignment.v1',
        scope='Nominal robot model and backend actuator configuration check, not randomized pushing acceptance.',
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        native_binary_sha256=hashlib.sha256(native_binary.read_bytes()).hexdigest(),
        robot_model_manifest_sha256=contract['manifest_sha256'],
        metrics=metrics,tolerances=tolerances,checks=checks,
        rigid_body_alignment_pass=all(checks.values()),
        friction_alignment_pass=bool(friction_ok), drive_alignment_pass=bool(drives_ok),
        finger_limit_checks=finger_checks, finger_limits_alignment_pass=fingers_ok,
        model_alignment_pass=bool(all(checks.values()) and friction_ok and drives_ok and fingers_ok),
        body_checks=body_checks,actual_armature_kg_m2=actual_armature.tolist(),
        inertia_api_reference='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_inertias',
        mass_matrix_convention='PhysX get_generalized_mass_matrices + diag(actual get_dof_armatures) vs Drake CalcMassMatrix',
        trajectory_motion=dict(duration_s=result['executed_sim_time_s'],
            maximum_tcp_displacement_m=float(drift.max()),terminal_tcp_displacement_m=float(drift[-1]),
            command_counts=result['command_counts'],forbidden_contact_ever=result['forbidden_robot_contact_ever']),
        formal_acceptance_pass=False)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('model_alignment_pass','metrics','trajectory_motion')},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--native-binary', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.result, args.native_binary, args.runtime, args.output)
    raise SystemExit(0 if report['model_alignment_pass'] else 2)
