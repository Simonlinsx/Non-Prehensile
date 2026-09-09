"""Exercise actual relay selection code without loading the native LCM bindings."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

source = Path(__file__).resolve().parents[1] / 'third_party/push_anything/online_bridge_relay.py'
tree = ast.parse(source.read_text())
needed = {'trajectory_by_name', 'DiagnosticReferenceReplay'}
namespace = {'copy': copy, 'math': math}
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in needed], type_ignores=[]), str(source), 'exec'), namespace)
Replay = namespace['DiagnosticReferenceReplay']


def messages(stamp=8_000_000):
    times = [stamp/1e6, stamp/1e6+.675]
    trajectories = [NS(trajectory_name=name, time_vec=times.copy(), datapoints=[[0., 1.]]*3,
                       num_points=2) for name in ('end_effector_position_target', 'end_effector_force_target')]
    actor = NS(utime=stamp, saved_traj=NS(trajectories=trajectories))
    mode = {'message': NS(utime=stamp), 'value': True, 'yaw_recovery': False}
    pd = NS(utime=stamp, saved_traj=NS(trajectories=[NS(trajectory_name='object_position_target_0', time_vec=[0, 9], datapoints=[[0., 1.]]*3)]))
    return actor, mode, pd


def test_default_disabled_preserves_native_reference_and_mode():
    actor, mode, pd = messages()
    result = Replay(None).select(8_000_000, actor, mode, pd)
    assert result == (actor, True, False, False, None)
    assert Replay(None).select(8_001_000, actor, mode, pd)[1] is False


def test_capture_requires_current_c3_actor_and_matching_pd_prediction():
    actor, mode, pd = messages()
    replay = Replay(8.)
    assert not replay.select(7_999_000, actor, mode, pd)[3]
    pd.utime -= 1_000
    assert not replay.select(8_000_000, actor, mode, pd)[3]
    pd.utime += 1_000
    mode['value'] = False
    assert not replay.select(8_000_000, actor, mode, pd)[3]
    mode['value'] = True
    selected, c3, yaw, active, event = replay.select(8_000_000, actor, mode, pd)
    assert active and c3 and not yaw
    assert selected is not actor and selected.utime == actor.utime
    assert event['end_utime_us'] == 8_675_000
    assert event['object_plan_utime_us'] == actor.utime
    actor.saved_traj.trajectories[0].datapoints[0][0] = 99.
    newer, new_mode, new_pd = messages(8_050_000)
    new_mode['value'] = False
    result = replay.select(8_050_000, newer, new_mode, new_pd)
    assert result[0].utime == 8_000_000 and result[1] and result[3]
    assert result[0].saved_traj.trajectories[0].datapoints[0][0] == 0
    result = replay.select(8_680_000, newer, new_mode, new_pd)
    assert result[0] is newer and not result[3]
    assert result[4]['event'] == 'diagnostic_reference_replay_end'
    again = messages(9_000_000)
    assert not replay.select(9_000_000, *again)[3]


def test_replay_rejects_invalid_timing_and_never_extrapolates():
    for start, duration in [(-1,.5), (float('nan'),.5), (0,1), (0,0)]:
        with pytest.raises(ValueError): Replay(start,duration)
    actor, mode, pd = messages()
    actor.saved_traj.trajectories[0].time_vec[-1] = 8.5
    assert not Replay(8).select(8_000_000, actor, mode, pd)[3]
    actor.saved_traj.trajectories[0].time_vec[-1] = 8.
    with pytest.raises(ValueError): Replay(8).select(8_000_000, actor, mode, pd)
