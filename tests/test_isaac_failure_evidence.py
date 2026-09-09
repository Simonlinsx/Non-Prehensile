"""Exercise the real executor error/finally blocks without launching Isaac."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

from scripts.isaac_failure_evidence import execution_failure_evidence


def inject_timeout(tmp_path, **state):
    source = Path(__file__).resolve().parents[1] / "scripts/run_c3_online_isaaclab.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    block = copy.deepcopy(next(node for node in main.body if isinstance(node, ast.Try) and node.handlers))
    # Replace the simulator body with the same exception observed in scene001;
    # execute the repository's actual handler and artifact finalization.
    block.body = ast.parse("raise TimeoutError('timed out')").body
    arguments = {
        "result": {}, "env": None, "video_recorder": None, "output": tmp_path / "result.json",
        "args_cli": SimpleNamespace(physics_dt_s=.001, control_decimation=1, audit_stride=1, trace_stride=10),
        **state,
    }
    function = ast.FunctionDef(
        name="run", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=k) for k in arguments],
                                       kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[block], decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    closed = []
    namespace = dict(execution_failure_evidence=execution_failure_evidence,
                     traceback=traceback, hashlib=hashlib, json=json,
                     simulation_app=SimpleNamespace(close=lambda: closed.append(True)))
    exec(compile(module, str(source), "exec"), namespace)
    assert namespace["run"](**arguments) == 1
    assert closed == [True]
    return json.loads(arguments["output"].read_text())


def test_timeout_preserves_safety_trace_and_hashes_closed_finger_artifact(tmp_path):
    path = tmp_path / "result.finger_state.jsonl"
    stream = path.open("w")
    content = '{"step":0,"position_m":[0,0],"target_m":[0,0]}\n'
    stream.write(content)
    trace = [dict(step=0, forbidden_robot_contact=True, legal_safe_robot_contact=False,
                  robot_target_contact_force_n_by_sensor={"target_robot_contacts:panda_link7": 2.})]
    report = inject_timeout(
        tmp_path, executed_steps=1, control_period_s=.001, trace=trace,
        counters=dict(fresh=1, ready=1), contact_audited_steps=1,
        safe_contact_ever=False, forbidden_contact_ever=True, first_forbidden_contact_step=0,
        finger_state_path=path, finger_state_stream=stream, finger_state_recorded_steps=1,
        physics_step_in_progress=False,
    )
    assert stream.closed
    assert report["trace"] == trace
    assert report["forbidden_robot_contact_ever"] is True
    assert report["first_forbidden_contact_step"] == 0
    assert report["contact_audited_steps"] == 1
    assert report["executed_steps"] == 1
    assert report["executed_sim_time_s"] == .001
    assert report["command_counts"] == dict(fresh=1, ready=1)
    assert report["finger_state_artifact"] == dict(
        path=str(path), recorded_steps=1, sha256=hashlib.sha256(content.encode()).hexdigest())
    assert report["error"] == "TimeoutError: timed out"
    assert "TimeoutError" in report["traceback"]
    assert report["stopped_reason"] == "executor_exception"
    assert report["execution_evidence_complete"] is False
    assert report["online_closed_loop_success"] is False
    assert report["acceptance_eligible"] is False


def test_startup_error_does_not_invent_measurements_or_zero_c1(tmp_path):
    report = inject_timeout(tmp_path)
    for key in ("trace", "executed_steps", "forbidden_robot_contact_ever", "finger_state_artifact"):
        assert key not in report
    assert not report["execution_evidence_complete"]
    assert report["error"] == "TimeoutError: timed out"


def test_failure_during_physics_keeps_completed_and_audited_counts_distinct(tmp_path):
    report = inject_timeout(
        tmp_path, executed_steps=9, control_period_s=.001, counters=dict(fresh=10, ready=10),
        contact_audited_steps=8, physics_step_in_progress=True, trace=[dict(step=7)],
    )
    assert report["executed_steps"] == 9
    assert report["contact_audited_steps"] == 8
    assert report["command_counts"]["fresh"] == 10
    assert report["physics_step_in_progress_at_error"] is True
    assert report["last_recorded_trace_step"] == 7
    assert "final_target_pose_wxyz" not in report
    assert report["execution_evidence_complete"] is False


def test_timeout_still_finalizes_recorded_isaac_video(tmp_path):
    closed = []
    def close():
        closed.append(True)
        return dict(path='recorded.mp4', frames=50, source='Isaac RGB')
    report = inject_timeout(tmp_path, video_recorder=SimpleNamespace(close=close))
    assert closed == [True]
    assert report['isaaclab_video']['frames'] == 50
    assert report['error'] == 'TimeoutError: timed out'
    assert not report['online_closed_loop_success']
