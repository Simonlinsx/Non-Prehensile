import json
from pathlib import Path

import pytest

from scripts.audit_scene_pose_contract import audit


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/strict_success_scene_pose_contract.json').read_text())


def test_real_isaac_and_native_goal_poses_match_scene_and_baked_support():
    f = fixture()
    report = audit(f['scene'], f['result'], f['native_goal'], f['semantic'])
    assert report['scene_pose_contract_pass']
    assert report['errors']['native_goal']['planar_m'] < 1e-6


@pytest.mark.parametrize('change', ['initial_yaw', 'goal_xy', 'goal_yaw', 'native_yaw', 'support', 'base_height'])
def test_rejects_wrong_executed_scene_even_with_fixed_native_goal(change):
    f = fixture()
    if change == 'initial_yaw': f['scene']['initial_yaw_deg'] = 45.
    elif change == 'goal_xy': f['scene']['goal_xy_m'][0] += .01
    elif change == 'goal_yaw': f['scene']['goal_yaw_deg'] += 5.
    elif change == 'native_yaw': f['native_goal']['fixed_target_orientations'][0] = [1., 0., 0., 0.]
    elif change == 'support': f['semantic']['support_quaternion_wxyz'] = [1., 0., 0., 0.]
    else: f['result']['franka_base_height_m'] = .03
    with pytest.raises(ValueError):
        audit(f['scene'], f['result'], f['native_goal'], f['semantic'])
