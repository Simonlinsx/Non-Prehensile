"""Independently audit all50 original-OSC FR3 results under one frozen controller."""
import argparse
import json
import math
from pathlib import Path

import yaml

try:
    from .fr3_frozen_batch import check_file, sha256, verify_batch, verify_scene
    from .audit_fr3_finger_closure import audit_file as audit_closure
    from .audit_fr3_native_model_alignment import audit as audit_model
    from .audit_scene_pose_contract import audit as audit_scene_pose
    from .audit_strict_pose_dwell import audit as audit_dwell, pose_errors
    from .audit_simulation_acceptance import physical_safe_contact_observed
except ImportError:
    from fr3_frozen_batch import check_file, sha256, verify_batch, verify_scene
    from audit_fr3_finger_closure import audit_file as audit_closure
    from audit_fr3_native_model_alignment import audit as audit_model
    from audit_scene_pose_contract import audit as audit_scene_pose
    from audit_strict_pose_dwell import audit as audit_dwell, pose_errors
    from audit_simulation_acceptance import physical_safe_contact_observed


def execution_checks(result, protocol):
    steps = result.get("executed_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError("Missing valid physics step count")
    if (result.get("schema") != "nonprehensile.c3_online_isaaclab.v1"
            or result.get("executor_mode") != "measured_isaac_state_to_online_c3_osc_effort"
            or result.get("robot_model") != "FR3"
            or result.get("physical_end_effector") != "stock_franka_gripper_closed"
            or result.get("error") is not None
            or result.get("control_period_s") != .001 or result.get("physics_dt_s") != .001
            or result.get("control_decimation") != 1
            or result.get("strict_pose_thresholds") != protocol["strict_pose_thresholds"]
            or not 0 < result.get("executed_sim_time_s", 0) <= 180
            or not math.isclose(result["executed_sim_time_s"], steps * .001, abs_tol=1e-9)
            or type(result.get("forbidden_robot_contact_ever")) is not bool):
        raise ValueError("Result does not implement the original-OSC FR3 execution contract")
    counts = result.get("command_counts", {})
    if (counts.get("fresh") != steps or counts.get("ready") != steps
            or any(counts.get(k) != 0 for k in ("stale", "watchdog", "planner_failure"))
            or not 0 <= result.get("maximum_command_age_s", math.inf) <= .0001):
        raise ValueError("Incomplete or stale native torque execution")
    parameters = result.get("controller_parameters", {})
    if (parameters.get("native_osc") is not True or parameters.get("force_c3_on_contact") is not False
            or parameters.get("native_goal_mode") != 2 or str(parameters.get("planner_period_ms")) != "50"):
        raise ValueError("Controller metadata differs from frozen native execution")
    final = result["final_target_pose_wxyz"]
    errors = pose_errors(final[:3], final[3:], result["goal_pose_wxyz"])
    for value, key in zip(errors, ("final_planar_error_m", "final_height_error_m", "final_rotation_error_rad")):
        if not math.isclose(value, result[key], rel_tol=0, abs_tol=2e-6):
            raise ValueError("Reported terminal pose errors differ from measured pose")
    return errors


def audit_scene(root, freeze, protocol, specification, semantic, freeze_sha):
    scene = verify_scene(root, freeze, specification, semantic)
    execution = json.loads((scene / "execution.json").read_text())
    if (execution.get("status") != "finished" or execution.get("freeze_sha256") != freeze_sha
            or execution.get("scene_id") != specification["scene_id"]):
        raise ValueError("Missing completed attribution to this frozen batch")
    check_file(scene / "result.json", execution["result_sha256"])
    result = json.loads((scene / "result.json").read_text())
    errors = execution_checks(result, protocol)
    with (scene / "effect_audit.jsonl").open() as stream:
        metadata = json.loads(next(stream))
        if (metadata.get("command_mode") != "effort" or metadata.get("diagnostic_osc_hold") is not False
                or metadata.get("diagnostic_reference_replay_start_s") is not None
                or metadata.get("planner_period_ms") != 50
                or metadata.get("osc_state_channel") != "FRANKA_STATE_OSC_SIMULATION"
                or metadata.get("planner_state_channel") != "FRANKA_STATE_SIMULATION"):
            raise ValueError("Actual relay mode differs from native OSC contract")
    artifact = result["controller_parameters"]["native_goal_artifact"]
    goal_path = scene / "runtime/examples/sampling_c3/anything/parameters/goal_params.yaml"
    if Path(artifact["path"]).resolve() != goal_path.resolve():
        raise ValueError("Result refers to another scene's native goal")
    check_file(goal_path, artifact["sha256"])
    pose = audit_scene_pose(specification, result, yaml.safe_load(goal_path.read_text()), semantic)
    if result["robot_model_contract"]["manifest_sha256"] != freeze["robot_model_manifest_sha256"]:
        raise ValueError("Robot model differs from the frozen FR3")
    closure = audit_closure(scene / "result.json", freeze["finger_closure_tolerance_m"])
    (scene / "finger_closure_audit.json").write_text(json.dumps(closure, indent=2) + "\n")
    if not closure["closure_pass"]:
        raise ValueError("Gripper did not remain nominally closed")
    model = audit_model(scene / "result.json", root / "binary/bazel-bin/examples/sampling_c3/franka_osc_controller",
                        scene / "runtime", scene / "model_alignment_audit.json")
    if not model["model_alignment_pass"]:
        raise ValueError("Actual FR3 physics/model alignment failed")
    dwell = audit_dwell(result)
    (scene / "strict_pose_dwell_audit.json").write_text(json.dumps(dwell, indent=2) + "\n")
    unsafe = result["forbidden_robot_contact_ever"] or any(r.get("forbidden_robot_contact") is True for r in result["trace"])
    contact = physical_safe_contact_observed(result)
    strict = all(e < t for e, t in zip(errors, (.02, .01, .105)))
    success = (result.get("online_closed_loop_success") is True
               and result.get("stopped_reason") == "strict_pose_dwell_success"
               and dwell["terminal_dwell_certified"] and strict and contact and not unsafe)
    if result.get("online_closed_loop_success") is True and not success:
        raise ValueError("Claimed success lacks independent pose/contact/safety evidence")
    report = dict(scene_id=specification["scene_id"], valid=True, success=success, c1_violation=unsafe,
                  final_pose_errors=errors, physical_safe_contact=contact, scene_pose_audit=pose,
                  executed_sim_time_s=result["executed_sim_time_s"], result_sha256=sha256(scene / "result.json"))
    (scene / "formal_scene_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_batch(root, freeze_sha, output):
    root = root.resolve()
    freeze, protocol, scenes, semantic = verify_batch(root, freeze_sha)
    records, missing, invalid, unsafe_observed = [], [], {}, set()
    for specification in scenes:
        scene = root / specification["scene_id"]
        if not (scene / "execution.json").exists():
            missing.append(specification["scene_id"])
            continue
        execution = json.loads((scene / "execution.json").read_text())
        if execution.get("status") in ("starting", "running"):
            missing.append(specification["scene_id"])
            continue
        if (scene / "result.json").exists():
            raw = json.loads((scene / "result.json").read_text())
            if raw.get("forbidden_robot_contact_ever") is True or any(
                    row.get("forbidden_robot_contact") is True for row in raw.get("trace", [])):
                unsafe_observed.add(specification["scene_id"])
        try:
            records.append(audit_scene(root, freeze, protocol, specification, semantic, freeze_sha))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            invalid[specification["scene_id"]] = f"{type(exc).__name__}: {exc}"
    successes = [r["scene_id"] for r in records if r["success"]]
    unsafe = sorted(unsafe_observed | {r["scene_id"] for r in records if r["c1_violation"]})
    report = dict(schema="nonprehensile.fr3_osc_batch_audit.v1", freeze_sha256=freeze_sha,
                  simulation_acceptance_pass=not missing and not invalid and not unsafe and len(successes) >= 21 and len(successes)/50 > .4,
                  total_fixed_scenes=50, successes=len(successes), success_rate=len(successes)/50,
                  successful_scene_ids=successes, missing_scene_ids=missing, invalid_scene_ids=invalid,
                  c1_violation_scene_ids=unsafe, scene_audits=records,
                  audit_source_sha256=sha256(Path(__file__)), repository_commit_and_push_completed=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("simulation_acceptance_pass", "successes", "success_rate", "missing_scene_ids", "invalid_scene_ids")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_batch(args.batch_root, args.freeze_sha256, args.output)
    raise SystemExit(0 if report["simulation_acceptance_pass"] else 2)
