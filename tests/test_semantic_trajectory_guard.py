from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from dapl.contact_planner.semantic_trajectory_guard import SemanticTrajectoryGuard


def scene():
    mesh = trimesh.Trimesh(vertices=[[0, -1, -1], [0, 1, -1], [0, 0, 1]],
                           faces=[[0, 1, 2]], process=False)
    state = SimpleNamespace(position_m=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0])
    return SemanticTrajectoryGuard([mesh], .055), state


def test_hold_command_retreat_measured_and_pass_safe_reference():
    guard, state = scene()
    target, held, d = guard.apply([.08, 0, 0], [.03, 0, 0], [state])
    assert held and d == pytest.approx(.03)
    assert target == pytest.approx([.08, 0, 0])
    target, held, d = guard.apply([.025, 0, 0], [.08, 0, 0], [state])
    assert held and target == pytest.approx([.058, 0, 0])
    target, held, d = guard.apply([.08, 0, 0], [.09, .01, 0], [state])
    assert not held and target == pytest.approx([.09, .01, 0])


def test_rotated_translated_object_uses_wxyz_and_rejects_missing_state():
    guard, state = scene()
    state.position_m = [1, 2, 3]
    state.quaternion_wxyz = [-2**.5, 0, 0, -2**.5]  # Rz(pi/2), scaled negative
    target, held, d = guard.apply([1, 2.025, 3], [1, 2.08, 3], [state])
    assert held and d == pytest.approx(.025)
    assert target == pytest.approx([1, 2.058, 3])
    with pytest.raises(ValueError):
        guard.apply([0, 0, 0], [0, 0, 0], [])
    state.quaternion_wxyz = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        guard.apply([0, 0, 0], [0, 0, 0], [state])
