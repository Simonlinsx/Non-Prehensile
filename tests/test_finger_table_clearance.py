import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "clearance_audit", Path(__file__).resolve().parents[1] / "scripts/audit_finger_table_clearance.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_clearance_uses_actual_finger_pose_and_environment_origin():
    names = ("panda_leftfinger", "panda_rightfinger")
    export = {"colliders": [{"body_path": "/Robot/" + n, "cooked_convexes": [
        {"vertices_body_m": [[0, 0, .02], [0, 0, 0]]}]} for n in names]}
    # Rx(pi) points downward; the right finger has an independently measured
    # lower pose. Nonzero environment origin must not become fake clearance.
    row = {"sim_time_s": 2, "env_origin_w": [1, 2, 3], "contact_body_poses_w": {
        names[0]: [1, 2, 3.04, 0, 2, 0, 0],
        names[1]: [1, 2, 3.01, 0, -2, 0, 0]}}
    report = module.audit({"trace": [row]}, export)
    assert report["minimum_clearance_m"] == pytest.approx(-.01)
    assert report["worst_sample"]["by_body_m"][names[0]] == pytest.approx(.02)
    assert report["samples_below_table"] == 1
    row["contact_body_poses_w"]["panda_hand"] = [1, 2, 3, 1, 0, 0, 0]
    row["semantic_c1_guard_active"] = True
    row["applied_feedforward_force_n"] = [0, 0, -.5]
    row["osc_reference_tip_velocity_m_s"] = [0, 0, 0]
    report = module.audit({"trace": [row]}, export)
    assert report["semantic_guard_sample_audit"]["active_samples_with_nonzero_force_or_velocity"] == 1
    assert report["hand_orientation_from_first_trace_pose"]["worst_sample"]["error_rad"] == 0
    json.dumps(report)
    del row["contact_body_poses_w"][names[1]]
    with pytest.raises(KeyError):
        module.audit({"trace": [row]}, export)
