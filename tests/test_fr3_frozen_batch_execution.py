import copy
import json
from pathlib import Path

import pytest

from scripts.audit_fr3_osc_batch import execution_checks
from scripts.run_fr3_osc_batch_worker import command_for, execution_env


def result():
    return dict(schema="nonprehensile.c3_online_isaaclab.v1",
                executor_mode="measured_isaac_state_to_online_c3_osc_effort", robot_model="FR3",
                physical_end_effector="stock_franka_gripper_closed", executed_steps=1000,
                executed_sim_time_s=1., control_period_s=.001, physics_dt_s=.001, control_decimation=1,
                strict_pose_thresholds=dict(planar_m=.02, height_m=.01, rotation_rad=.105, dwell_time_s=.5),
                forbidden_robot_contact_ever=False,
                command_counts=dict(fresh=1000, ready=1000, stale=0, watchdog=0, planner_failure=0),
                maximum_command_age_s=0.,
                controller_parameters=dict(native_osc=True, force_c3_on_contact=False, native_goal_mode=2, planner_period_ms="50"),
                final_target_pose_wxyz=[0., 0., 0., 1., 0., 0., 0.], goal_pose_wxyz=[0., 0., 0., 1., 0., 0., 0.],
                final_planar_error_m=0., final_height_error_m=0., final_rotation_error_rad=0.)


def test_native_gate_rejects_relabelled_panda_stale_or_relaxed_results():
    r = result()
    protocol = dict(strict_pose_thresholds=r['strict_pose_thresholds'])
    assert execution_checks(r, protocol) == (0., 0., 0.)
    for key, value in (('robot_model', 'Panda'), ('schema', 'nonprehensile.c3_online_isaaclab_task.v1'),
                       ('final_rotation_error_rad', .1), ('control_period_s', .01),
                       ('maximum_command_age_s', .01), ('executed_steps', True),
                       ('command_counts', dict(fresh=999, ready=1000, stale=1, watchdog=0, planner_failure=0))):
        changed = copy.deepcopy(r)
        changed[key] = value
        with pytest.raises(ValueError):
            execution_checks(changed, protocol)


def test_worker_clears_inherited_controller_overrides_and_uses_frozen_source(monkeypatch):
    monkeypatch.setenv('PUSH_ANYTHING_DIAGNOSTIC_OSC_HOLD', '1')
    monkeypatch.setenv('PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S', '3')
    monkeypatch.setenv('DAPL_LOCAL_FRANKA_URDF', '/wrong.urdf')
    root, scene = Path('/batch'), Path('/batch/scene000')
    env = execution_env(root, scene, dict(python_executable='/python', robot_model_manifest='/model.json'), '2', 21000)
    assert env['PUSH_ANYTHING_DIAGNOSTIC_OSC_HOLD'] == '0'
    assert 'PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S' not in env
    assert 'DAPL_LOCAL_FRANKA_URDF' not in env
    assert env['PUSH_ANYTHING_EXECUTOR_MODE'] == 'effort'
    command = command_for(root, scene, 21000)
    assert command[1] == '/batch/execution_source/scripts/run_push_anything_isaaclab_online.sh'
    assert command[command.index('--max-sim-time-s')+1] == '180'


def test_batch_requires_twenty_one_of_all_fifty_and_rejects_unsafe_invalid_scene(tmp_path, monkeypatch):
    import scripts.audit_fr3_osc_batch as module
    scenes = [dict(scene_id=f'scene{i:03d}') for i in range(50)]
    monkeypatch.setattr(module, 'verify_batch', lambda *args: ({}, {}, scenes, {}))
    for scene in scenes:
        root = tmp_path / scene['scene_id']
        root.mkdir()
        (root / 'execution.json').write_text(json.dumps(dict(status='finished')))
        (root / 'result.json').write_text(json.dumps(dict(forbidden_robot_contact_ever=False)))
    successes = 20

    def scene_audit(root, freeze, protocol, scene, semantic, digest):
        if scene.get('invalid'):
            raise ValueError('invalid evidence')
        return dict(scene_id=scene['scene_id'], success=int(scene['scene_id'][5:]) < successes, c1_violation=False)

    monkeypatch.setattr(module, 'audit_scene', scene_audit)
    report = module.audit_batch(tmp_path, 'freeze', tmp_path / 'report.json')
    assert report['success_rate'] == .4
    assert not report['simulation_acceptance_pass']
    successes = 21
    assert module.audit_batch(tmp_path, 'freeze', tmp_path / 'report.json')['simulation_acceptance_pass']
    (tmp_path / 'scene049/execution.json').unlink()
    report = module.audit_batch(tmp_path, 'freeze', tmp_path / 'report.json')
    assert report['success_rate'] == .42
    assert not report['simulation_acceptance_pass']
    (tmp_path / 'scene049/execution.json').write_text(json.dumps(dict(status='finished')))
    (tmp_path / 'scene049/result.json').write_text(json.dumps(dict(forbidden_robot_contact_ever=True)))
    scenes[-1]['invalid'] = True
    report = module.audit_batch(tmp_path, 'freeze', tmp_path / 'report.json')
    assert not report['simulation_acceptance_pass']
    assert report['c1_violation_scene_ids'] == ['scene049']
    assert 'scene049' in report['invalid_scene_ids']
