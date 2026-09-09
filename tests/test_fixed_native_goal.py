import yaml
import hashlib
import pytest

from scripts.stage_domino_hammer_push_anything import configure_fixed_task_goal
from scripts.audit_simulation_acceptance import verify_native_goal_artifact


def test_random_template_becomes_fixed_without_changing_goal_or_thresholds(tmp_path):
    path = tmp_path / 'goal_params.yaml'
    path.write_text('goal_mode: 0\nposition_success_threshold: 0.02\n'
                    'orientation_success_threshold: 0.1\nfixed_target_positions: [[0,0,0]]\n')
    goal = [.45, .2, -.016]
    quat = [-.6, 0, 0, .8]
    configure_fixed_task_goal(path, goal, quat)
    content = path.read_text()
    configure_fixed_task_goal(path, goal, quat)
    assert path.read_text() == content
    params = yaml.safe_load(content)
    assert params['goal_mode'] == 2
    assert params['fixed_target_positions'] == [goal]
    assert params['fixed_target_orientations'] == [quat]
    assert params['resting_object_heights'] == [goal[2]]
    assert params['position_success_threshold'] == .02
    assert params['orientation_success_threshold'] == .1


def test_acceptance_checks_actual_goal_file_including_hash_and_random_mode(tmp_path):
    scene = tmp_path / 'scene001'
    scene.mkdir()
    goal = scene / 'goal_params.yaml'
    goal.write_text('goal_mode: 2\n')
    result = {'controller_parameters': {'native_goal_artifact': {'path': str(goal),
        'sha256': hashlib.sha256(goal.read_bytes()).hexdigest()}}}
    verify_native_goal_artifact(result, scene)
    goal.write_text('goal_mode: 0\n')
    with pytest.raises(ValueError):
        verify_native_goal_artifact(result, scene)
    result['controller_parameters']['native_goal_artifact']['sha256'] = hashlib.sha256(goal.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_native_goal_artifact(result, scene)
    goal.write_text('goal_mode: 2\n')
    result['controller_parameters']['native_goal_artifact']['sha256'] = hashlib.sha256(goal.read_bytes()).hexdigest()
    unrelated = tmp_path / 'other_scene'
    unrelated.mkdir()
    with pytest.raises(ValueError):
        verify_native_goal_artifact(result, unrelated)
