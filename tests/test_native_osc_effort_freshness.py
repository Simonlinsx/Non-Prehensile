import ast
from pathlib import Path
from types import SimpleNamespace
import math
import pytest


def current_check():
    # Load the protocol predicate without requiring the native LCM bindings.
    path = Path(__file__).resolve().parents[1] / 'third_party/push_anything/online_bridge_relay.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'effort_is_current')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['effort_is_current']


def test_old_and_future_torques_are_rejected():
    check = current_check()
    assert not check(None, 150000, 100)
    for timestamp in (149000, 149899, 150101, 151000):
        assert not check(SimpleNamespace(utime=timestamp), 150000, 100)


def test_native_timestamp_truncation_does_not_drop_current_torque():
    check = current_check()
    for timestamp in (149900, 149999, 150000, 150100):
        assert check(SimpleNamespace(utime=timestamp), 150000, 100)


def test_separate_planner_clock_keeps_twenty_hz_with_float_timestamp_truncation():
    path = Path(__file__).resolve().parents[1] / 'third_party/push_anything/online_bridge_relay.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PlannerClockGate')
    scope = {'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    gate = scope['PlannerClockGate'](50)
    timestamps = [int((.1 + i * .001) * 1e6) for i in range(2000)]
    publication_steps = [i for i, stamp in enumerate(timestamps) if gate.due(stamp)]
    assert publication_steps == list(range(0, 2000, 50))
    with pytest.raises(ValueError, match='backwards'):
        gate.due(1)
    for period in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            scope['PlannerClockGate'](period)
