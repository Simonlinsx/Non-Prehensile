import copy
import hashlib
import json

import pytest
import yaml

from scripts.prepare_fr3_osc_scenes import SCENE_KEYS, scene_parameters
from scripts.fr3_runtime_inputs import inventory


def test_scene_preparation_changes_only_prescribed_pose_and_seed():
    template = {"sim_params.yaml": dict(q_init_objects=[], q_init_franka=[1., 2.], dt=.001),
                "goal_params.yaml": dict(goal_mode=2, fixed_target_positions=[], fixed_target_orientations=[],
                                         orientation_success_threshold=.1),
                "sampling_params.yaml": dict(random_seed=2, buffer_distance=.0235)}
    original = copy.deepcopy(template)
    scene = dict(initial_yaw_deg=-179., goal_yaw_deg=179., initial_xy_m=[.39, .2],
                 goal_xy_m=[.45, .21], sampling_seed=123)
    out = scene_parameters(template, scene, dict(support_height_m=.013), .029)
    assert template == original
    assert out['sim_params.yaml']['q_init_objects'][0][4:] == pytest.approx([.39, .2, -.016])
    assert out['goal_params.yaml']['fixed_target_positions'][0] == pytest.approx([.45, .21, -.016])
    assert out['sampling_params.yaml']['random_seed'] == 123
    for name, values in original.items():
        assert {k: v for k, v in out[name].items() if k not in SCENE_KEYS[name]} == {
            k: v for k, v in values.items() if k not in SCENE_KEYS[name]}
    template['goal_params.yaml']['goal_mode'] = 0
    with pytest.raises(ValueError, match='fixed goal'):
        scene_parameters(template, scene, dict(support_height_m=.013), .029)


def test_runtime_signature_detects_retuned_gain_and_changed_contact_mesh(tmp_path):
    parameters = tmp_path / 'examples/sampling_c3/anything/parameters'
    parameters.mkdir(parents=True)
    goal = parameters / 'goal_params.yaml'
    goal.write_text(yaml.safe_dump(dict(goal_mode=2, fixed_target_positions=[[0, 0, 0]], gain=200)))
    models = tmp_path / 'examples/sampling_c3/urdf'
    models.mkdir()
    for name in ('ground', 'platform', 'end_effector_simple_model'):
        (models / (name + '.urdf')).write_text('<robot name="test"/>')
    mesh = models / 'finger.obj'
    mesh.write_text('v 0 0 0\n')
    (tmp_path / 'shared_contact_model.json').write_text(json.dumps(dict(files={
        str(mesh.relative_to(tmp_path)): hashlib.sha256(mesh.read_bytes()).hexdigest()})))
    baseline = inventory(tmp_path)['controller_signature_sha256']
    goal.write_text(yaml.safe_dump(dict(goal_mode=2, fixed_target_positions=[[1, 1, 1]], gain=200)))
    assert inventory(tmp_path)['controller_signature_sha256'] == baseline
    goal.write_text(yaml.safe_dump(dict(goal_mode=2, fixed_target_positions=[[1, 1, 1]], gain=201)))
    assert inventory(tmp_path)['controller_signature_sha256'] != baseline
    mesh.write_text('v 0 1 0\n')
    with pytest.raises(ValueError, match='checksum changed'):
        inventory(tmp_path)
