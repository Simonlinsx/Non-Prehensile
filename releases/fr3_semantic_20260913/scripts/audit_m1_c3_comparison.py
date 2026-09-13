#!/usr/bin/env python3
"""Independently aggregate the three-group diagnostic; pending is not failure."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from audit_strict_pose_dwell import pose_errors


def audit_rows(rows, goal, initial, dt=.001):
    count = dwell = maximum = force_low = force_high = 0
    safe = violation = False
    fingers = displacement = 0.
    first_contact = None
    for row in rows:
        count += 1
        if row["step"] != count:
            raise ValueError("Missing, duplicate, or reordered physics step")
        p, q = row["pose_wxyz"][:3], row["pose_wxyz"][3:]
        errors = pose_errors(p, q, goal)
        strict = all(e < t for e, t in zip(errors, (.02, .01, .105)))
        dwell = dwell + 1 if strict else 0
        maximum = max(maximum, dwell)
        f = row["force_n"]
        if not math.isfinite(f) or f < 0:
            raise ValueError("Invalid force")
        if len(row["finger_m"]) != 2 or not all(math.isfinite(v) for v in row["finger_m"]):
            raise ValueError("Invalid measured gripper position")
        if type(row["safe"]) is not bool or type(row["c1"]) is not bool:
            raise ValueError("Invalid semantic predicate type")
        if row["safe"] and f <= .5:
            raise ValueError("Safe physical contact without the registered force threshold")
        force_low += int(f > .02)
        force_high += int(f > .5)
        if f > .02 and first_contact is None:
            first_contact = count * dt
        safe |= row["safe"]
        violation |= row["c1"]
        fingers = max(fingers, *map(abs, row["finger_m"]))
        displacement = max(displacement, math.dist(p[:2], initial[:2]))
    if not count:
        raise ValueError("Empty physical trace")
    return {"steps": count, "sim_s": count * dt, "final_xy_mm": errors[0] * 1000,
            "final_height_mm": errors[1] * 1000, "final_so3_rad": errors[2],
            "terminal_dwell_s": dwell * dt, "maximum_dwell_s": maximum * dt,
            "contact_steps_gt002N": force_low, "contact_steps_gt05N": force_high,
            "first_contact_s": first_contact, "safe_contact_gt05N": safe,
            "c1_violation": violation, "max_finger_mm": fingers * 1000,
            "max_object_displacement_mm": displacement * 1000,
            "strict_success": dwell >= math.ceil(.5 / dt - 1e-12) and safe and not violation and fingers < .001}


def audit_folder(folder, group):
    process_path = folder / "process.json"
    if not process_path.exists():
        return {"scene": folder.name, "status": "pending"}
    proc = json.loads(process_path.read_text())
    result_path = folder / "result.json"
    if not result_path.exists():
        return {"scene": folder.name, "status": "timeout" if proc["timeout"] else "execution_error", **proc}
    data = json.loads(result_path.read_text())
    if data.get("stopped_reason") == "executor_exception":
        return {"scene": folder.name, "status": "execution_error", "strict_success": False,
                "stop_reason": "executor_exception", "executed_steps": data.get("executed_steps"), **proc}
    if group == "B_fr3_m1":
        with (folder / "physics.jsonl").open() as stream:
            audit = audit_rows((json.loads(line) for line in stream), data["goal_pose_wxyz"], data["initial_pose_wxyz"])
        if audit["steps"] != data["steps"]:
            raise ValueError("Result/physical trace step count mismatch")
        reason = data["stop_reason"]
        wall = data["main_wall_s"]
    else:
        def converted():
            for row in data["trace"]:
                yield {"step": row["step"] + 1,
                       "pose_wxyz": row["target_position_m"] + row["target_quaternion_wxyz"],
                       "force_n": row["robot_target_contact_force_n_by_sensor"]["target_hand_contacts"],
                       "safe": row["legal_safe_robot_contact"], "c1": row["forbidden_robot_contact"],
                       "finger_m": row["finger_joint_position_m"]}
        audit = audit_rows(converted(), data["goal_pose_wxyz"], data["initial_target_pose_wxyz"])
        if audit["steps"] != data["executed_steps"]:
            raise ValueError("Result/physical trace step count mismatch")
        reason = data["stopped_reason"]
        wall = data["wall_time_s"]
        qp = []
        qp_by_clock = {}
        timings = []
        for line in (folder / "controller_online.log").open():
            for prefix in ("C3_FINAL_QP_STATUS ", "C3_NATIVE_TIMING "):
                if line.startswith(prefix):
                    try:
                        item = json.loads(line[len(prefix):])
                    except ValueError:
                        continue
                    if prefix.startswith("C3_FINAL") and item["candidate_index"] == 0:
                        qp.append(item["succeeded"])
                        qp_by_clock[int(item["clock_us"])] = item["succeeded"]
                    elif prefix.startswith("C3_NATIVE"):
                        timings.append(item["total_s"] * 1000)
        audit["current_candidate_qp_frames"] = len(qp)
        audit["current_candidate_qp_failures"] = sum(not ok for ok in qp)
        audit["planner_mean_ms_excluding_startup"] = statistics.mean(timings[1:]) if len(timings) > 1 else None
        audit["phase_wall_time_s"] = data.get("phase_wall_time_s")
        rejected_executed = [row["step"] for row in data["trace"]
            if row["command_flags"] & 16
            and qp_by_clock.get(100000 + (row["step"] // 50) * 50000) is False]
        audit["failed_current_qp_executed_steps"] = rejected_executed
        audit["strict_success"] &= not rejected_executed
    audit.update(scene=folder.name, stop_reason=reason, main_wall_s=wall, process_wall_s=proc["wall_s"],
                 result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    audit["frozen_inputs_unchanged"] = proc.get("frozen_inputs_unchanged", False)
    audit["strict_success"] &= not proc["timeout"] and proc["returncode"] == 0 and audit["frozen_inputs_unchanged"]
    audit["status"] = "success" if audit["strict_success"] else "not_success"
    if proc["timeout"]:
        audit["status"] = "timeout"
    elif reason == "diagnostic_wall_budget_exhausted":
        audit["status"] = "wall_censored"
    elif proc["returncode"] not in (0, 2):
        audit["status"] = "execution_error"
    return audit


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    result = {"scope": "Diagnostic system comparison; not independent acceptance or isolated model ablation", "groups": {}}
    a = root / "A_original"
    if (a / "result.json").exists():
        historical = json.loads((a / "result.json").read_text())
        result["groups"]["A_original"] = {"status": "completed", **historical["summary"],
            "process": json.loads((a / "process.json").read_text()) if (a / "process.json").exists() else None}
    else:
        result["groups"]["A_original"] = {"status": "pending" if not (a / "process.json").exists() else "execution_error"}
    paired = root / "A_paired_panda"
    if (paired / "process.json").exists() and (paired / "result.json").exists():
        data = json.loads((paired / "result.json").read_text())
        physical = [json.loads(line) for line in (paired / "physics.jsonl").read_text().splitlines()]
        process = json.loads((paired / "process.json").read_text())
        valid = process["returncode"] == 0 and not process["timeout"] and process.get("frozen_inputs_unchanged", False)
        if len(physical) != data["steps"]:
            raise ValueError("Paired Panda result/trace mismatch")
        paired_rows = []
        for i, scene in enumerate(("diag001", "diag002", "diag003")):
            rows = [{"step": x["step"], **{k: x[k][i] for k in ("pose_wxyz", "force_n", "safe", "c1", "finger_m")}} for x in physical]
            a = audit_rows(rows, data["goal_pose_wxyz"][i], data["initial_pose_wxyz"][i], data["physics_dt_s"])
            a["scene"] = scene
            a["safe_contact_gt002N"] = any(x["safe002"][i] for x in physical)
            a["strict_success"] &= valid
            a["success_with_002N_contact_gate"] = valid and a["terminal_dwell_s"] >= .5 and a["safe_contact_gt002N"] and not a["c1_violation"] and a["max_finger_mm"] < 1
            paired_rows.append(a)
        result["groups"]["A_paired_panda"] = {"rows": paired_rows, "process": process,
            "scope": "Original Panda controller and12.5ms physics; same three scenes,30s cap; exploratory distribution check"}
    elif paired.exists():
        result["groups"]["A_paired_panda"] = {"status": "pending" if not (paired / "process.json").exists() else "execution_error"}
    for group in ("B_fr3_m1", "C_fr3_c3"):
        rows = []
        for scene in ("diag001", "diag002", "diag003"):
            try:
                rows.append(audit_folder(root / group / scene, group))
            except Exception as exc:
                rows.append({"scene": scene, "status": "audit_error", "error": repr(exc)})
        result["groups"][group] = {"registered_scenes": 3, "pending": sum(x["status"] == "pending" for x in rows),
            "strict_successes": sum(x.get("strict_success", False) for x in rows), "rows": rows}
    result["paired_input_checks"] = []
    for scene in ("diag001", "diag002", "diag003"):
        b, c = root / "B_fr3_m1" / scene, root / "C_fr3_c3" / scene
        if not (b / "result.json").exists() or not (c / "result.json").exists():
            continue
        bd, cd = json.loads((b / "result.json").read_text()), json.loads((c / "result.json").read_text())
        if cd.get("stopped_reason") == "executor_exception":
            continue
        model = json.loads((b / "model_audit.json").read_text())
        checks = {"scene": scene, "identical_manifest": (b / "manifest.jsonl").read_bytes() == (c / "manifest.jsonl").read_bytes(),
                  "identical_initial_object_pose": bd["initial_pose_wxyz"] == cd["initial_target_pose_wxyz"],
                  "identical_goal_pose": bd["goal_pose_wxyz"] == cd["goal_pose_wxyz"],
                  "identical_initial_arm_joints": model["initial_q"][:7] == cd["initial_robot_joint_position_rad"]}
        for key in ("body_names", "body_masses_kg", "body_com_pose_xyzw", "body_inertias_kg_m2"):
            checks["identical_" + key] = model[key] == cd["initial_robot_dynamics"][key]
        checks["all_checks_passed"] = all(v for k, v in checks.items() if k != "scene")
        result["paired_input_checks"].append(checks)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
