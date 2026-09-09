import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import yaml
from scripts.validate_shared_contact_geometry import compare
from scripts.stage_shared_physx_contact_model import target_inertial_in_support_frame


@pytest.mark.parametrize('scaling_mode', ['ee', 'all'])
def test_shared_shapes_preserve_geometry_and_leave_source_runtime_unchanged(tmp_path, scaling_mode):
    upstream = tmp_path / "upstream"
    asset = upstream / "examples/sampling_c3/urdf/DOMINO_020_hammer_safe"
    params = upstream / "examples/sampling_c3/anything/parameters"
    asset.mkdir(parents=True)
    params.mkdir(parents=True)
    (upstream / 'solvers').mkdir()
    (upstream / 'solvers/osqp_options_default.yaml').write_text('int_options: {max_iter: 200}\n')
    (params / "sampling_c3_controller_params.yaml").write_text(
        "object_models: [test.sdf]\nprogress_params_file: examples/sampling_c3/anything/parameters/progress_params_c3plus.yaml\n"
        "use_relinearized_pd_cost: true\nuse_osc_matched_coarse_model: true\nplanner_finger_table_clearance: 0.002\n")
    (params / "progress_params_c3plus.yaml").write_text(
        "track_c3_progress_via: 3\nprogress_enforced_over_n_loops: 35\nprogress_enforced_cost_drop: 0.5\n")
    (params / "sampling_params.yaml").write_text("buffer_distance: 0.0235\n")
    for name in ("sampling_c3_options.yaml", "sampling_c3plus_options.yaml"):
        (params / name).write_text("mu_per_pair_type: [0.823, 0.42, 0.46, 0.375]\n"
                                  "contact_model: anitescu\nresolve_contacts_to_lists: [[0, 1, 3, 1]]\n"
                                  "num_contacts_index: 0\nresolve_as_planar_contacts_list: [0, 0, 0, 1]\n"
                                  "num_friction_directions: 2\nfinal_augmented_cost_contact_indices: [0, 1, 2, 3]\n")
    volume = '<collision name="old"><geometry><mesh><uri>old.obj</uri></mesh></geometry></collision>'
    supports = ''.join(f'<collision name="support{i}"><geometry><sphere><radius>0.001</radius></sphere></geometry></collision>' for i in range(3))
    for suffix, contacts in [(".sdf", volume), ("_controller.sdf", volume + supports)]:
        (asset / ("DOMINO_020_hammer_safe" + suffix)).write_text('<sdf version="1.7"><model name="hammer"><link name="hammer">' + contacts + '</link></model></sdf>')
    source_hashes = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest() for p in upstream.rglob('*') if p.is_file()}
    box = trimesh.creation.box(extents=[.02, .03, .04])
    record = {"vertices_body_m": box.vertices.tolist(), "face_vertex_counts": [3] * len(box.faces), "face_vertex_indices": box.faces.reshape(-1).tolist()}
    hand_r = Rotation.from_euler('xyz', [.3, -.2, .7])
    support = Rotation.from_euler('xyz', [.5, .1, -.6])
    hand = np.array([.4, .2, .1])
    finger = hand + hand_r.apply([0, .01, .0584])
    poses = [p.tolist() + np.roll(hand_r.as_quat(), 1).tolist() for p in (hand, finger, finger)]
    live = {"joint_position_rad_or_m": [0.] * 9,
            "body_names": ["panda_hand", "panda_leftfinger", "panda_rightfinger"], "body_poses_w": poses,
            "colliders": [{"body_path": '/' + name, "cooked_convexes": [record]} for name in ["Target", "panda_leftfinger", "panda_rightfinger"]]}
    live['target_dynamics'] = {
        'mass_kg': .05, 'com_position_body_m': [.003, -.005, .001],
        'inertia_frame': 'body_axes_about_com',
        'inertia_body_about_com_kg_m2': np.diag([2e-5, 3e-5, 4e-5]).tolist(),
    }
    live['target_support'] = {'sphere_radius_m': .001,
                              'support_sphere_centers_body_m': [[-.02, -.01, -.009], [.02, -.01, -.009], [0, .02, -.009]]}
    live['contact_friction'] = {'ee_object': .4, 'object_ground': .3}
    export, stage, runtime = tmp_path/'export.json', tmp_path/'stage.json', tmp_path/'runtime'
    export.write_text(json.dumps(live))
    stage.write_text(json.dumps({"end_effector": {"mode": "closed-gripper-proxy", "reference_offset_m": .104279112}, "support_quaternion_wxyz": np.roll(support.as_quat(), 1).tolist()}))
    command = [sys.executable, 'scripts/stage_shared_physx_contact_model.py', '--export', str(export), '--stage-manifest', str(stage), '--upstream-root', str(upstream), '--runtime', str(runtime), '--enforce-actor-workspace', '--execution-height-mode', 'optimized']
    command += ['--spatial-safe-sampling', '--c3-admm-iterations', '6', '--c3-final-contact-scaling-mode', scaling_mode]
    command += ['--c3-qp-max-iterations', '2000']
    command += ['--c3-nonnegative-contact-forces', '--c3-pd-rollout-interpolation', 'foh']
    command += ['--c3-pd-rollout-kp', '11.4', '11.4', '11.4',
                '--c3-pd-rollout-kd', '1.14', '1.14', '1.14']
    command += ['--c3-progress-window-loops', '5', '--c3-progress-cost-drop', '0.07831035905913464']
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
    for relative, digest in source_hashes.items():
        assert hashlib.sha256((upstream/relative).read_bytes()).hexdigest() == digest
    meshes = runtime/'examples/sampling_c3/urdf/shared_physx_contact'
    expected_finger = box.vertices + [0, .01, .0584 - .104279112]
    for filename, expected in [('target_00.obj', support.apply(box.vertices)), ('panda_leftfinger.obj', expected_finger)]:
        actual = trimesh.load(meshes/filename, force='mesh').vertices
        assert np.max(cKDTree(actual).query(expected)[0]) < 1e-10
    sdf = ET.parse(runtime/'examples/sampling_c3/urdf/DOMINO_020_hammer_safe/DOMINO_020_hammer_safe_controller.sdf')
    contacts = sdf.getroot().findall('model/link/collision')
    assert len(contacts) == 4
    assert contacts[0].find('geometry/mesh/{uri:drake}declare_convex') is not None
    assert all(c.find('geometry/sphere') is not None for c in contacts[-3:])
    for name in ("sampling_c3_options.yaml", "sampling_c3plus_options.yaml"):
        staged = yaml.safe_load((runtime/'examples/sampling_c3/anything/parameters'/name).read_text())
        assert staged['mu_per_pair_type'] == [.823, .4, .3, .375]
        assert staged['planar_demo'] is False
        assert staged['admm_iter'] == 6
        assert staged['Kp_for_ee_pd_rollout'] == [11.4] * 3
        assert staged['Kd_for_ee_pd_rollout'] == [1.14] * 3
        assert staged['final_augmented_cost_contact_indices'] == list(range(18 if scaling_mode == 'all' else 4))
    assert yaml.safe_load((runtime/'examples/sampling_c3/anything/parameters/sampling_params.yaml').read_text())['gen_planar_samples'] is False
    progress = yaml.safe_load((runtime/'examples/sampling_c3/anything/parameters/progress_params_c3plus.yaml').read_text())
    assert progress['progress_enforced_over_n_loops'] == 5
    assert progress['progress_enforced_cost_drop'] == pytest.approx(1-.5**(4/34))
    assert yaml.safe_load((runtime/'solvers/osqp_options_default.yaml').read_text())['int_options']['max_iter'] == 2000
    assert not (runtime/'solvers/osqp_options_default.yaml').is_symlink()
    assert yaml.safe_load((runtime/'examples/sampling_c3/anything/parameters/sampling_c3_controller_params.yaml').read_text())['enforce_nonnegative_contact_forces'] is True
    assert yaml.safe_load((runtime/'examples/sampling_c3/anything/parameters/sampling_c3_controller_params.yaml').read_text())['use_foh_pd_rollout'] is True
    staged_params = yaml.safe_load((runtime/'examples/sampling_c3/anything/parameters/sampling_c3_controller_params.yaml').read_text())
    for disabled in ('use_relinearized_pd_cost', 'use_osc_matched_coarse_model', 'planner_finger_table_clearance'):
        assert disabled not in staged_params  # Also clear inherited enabled keys.
    for contact, center in zip(contacts[-3:], support.apply(live['target_support']['support_sphere_centers_body_m'])):
        assert np.allclose(list(map(float, contact.find('pose').text.split()))[:3], center, atol=1e-12)
        assert float(contact.find('geometry/sphere/radius').text) == .001
    expected_com = support.apply([.003, -.005, .001])
    R = support.as_matrix()
    expected_inertia = R @ np.diag([2e-5, 3e-5, 4e-5]) @ R.T
    for suffix in ('_controller.sdf', '.sdf'):
        tree = ET.parse(runtime/'examples/sampling_c3/urdf/DOMINO_020_hammer_safe'/('DOMINO_020_hammer_safe'+suffix))
        inertial = tree.getroot().find('model/link/inertial')
        assert float(inertial.find('mass').text) == .05
        assert np.allclose(list(map(float, inertial.find('pose').text.split())), [*expected_com, 0, 0, 0], atol=1e-12)
        for key, i, j in [('ixx', 0, 0), ('iyy', 1, 1), ('izz', 2, 2), ('ixy', 0, 1), ('ixz', 0, 2), ('iyz', 1, 2)]:
            assert float(inertial.find('inertia/'+key).text) == pytest.approx(expected_inertia[i,j], abs=1e-14)
    # Existing runtime evidence must not be silently overwritten.
    repeated = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True)
    assert repeated.returncode != 0


@pytest.mark.parametrize('field,value', [
    ('mass_kg', -1), ('com_position_body_m', [0, float('nan'), 0]),
    ('inertia_frame', 'principal_axes'),
    ('inertia_body_about_com_kg_m2', [[1, 0, 0], [0, 1, 0], [0, 0, 3]]),
])
def test_target_dynamics_rejects_invalid_contract(field, value):
    dynamics = {'mass_kg': .05, 'com_position_body_m': [0, 0, 0],
                'inertia_frame': 'body_axes_about_com',
                'inertia_body_about_com_kg_m2': np.eye(3).tolist()}
    dynamics[field] = value
    with pytest.raises(ValueError):
        target_inertial_in_support_frame(dynamics, Rotation.identity())


def test_normal_validation_rejects_wrong_direction_and_distance():
    physical = {"body_names": ["panda_hand"], "reconstructed_contact_poses": [{
        "result": "reference", "step": 1, "body_poses_w": [[0, 0, 0, 1, 0, 0, 0]],
        "one_step_contact_probe": {"contacts": [{"normal_w": [0, 0, 1], "separation_m": .00002}]},
    }]}
    valid = "CLOSED_GRIPPER_GEOMETRY_AUDIT 0 0.00002 0 0 -1 1 0 0 0"
    assert compare(physical, valid)["geometry_check_passed"]
    assert not compare(physical, valid.replace("0 0 -1", "1 0 0"))["geometry_check_passed"]
    assert not compare(physical, valid.replace("0.00002", "0.002"))["geometry_check_passed"]
    with pytest.raises(ValueError, match="count"):
        compare(physical, "")
