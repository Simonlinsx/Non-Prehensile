#!/usr/bin/env python3
"""Compare observed closed-gripper contacts with the planner's sphere geometry.

This is an offline model-consistency diagnostic, not a PhysX collision audit.
The observed contact flag comes from the executor; sphere clearance is measured
against the exported full target mesh at the same recorded object/robot pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def proxy_center_in_target_frame(row, result, support):
    """Map robot-base proxy coordinates into the support-baked target mesh."""
    target_position = np.array(row["target_position_m"], dtype=float)
    tip = np.array(row["planner_tip_position_m"], dtype=float)
    if "robot_root_pose_w" in row:
        root = row["robot_root_pose_w"]
        tip = Rotation.from_quat(np.roll(root[3:7], -1)).apply(tip) + root[:3]
        target_position += np.array(row["env_origin_w"], dtype=float)
    else:
        # Legacy runs used an identity-orientation base at (0, 0, 0.029).
        target_position[2] -= result["franka_base_height_m"]
    object_rotation = Rotation.from_quat(np.roll(row["target_quaternion_wxyz"], -1)) * support.inv()
    return object_rotation.inv().apply(tip - target_position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    stage = json.loads(args.stage_manifest.read_text())
    if result.get("physical_end_effector") != "stock_franka_gripper_closed":
        raise ValueError("this diagnostic requires the stock closed gripper")
    radius = float(stage["end_effector"]["proxy_radius_m"])
    if radius <= 0:
        raise ValueError("invalid proxy radius")
    # OBJ exports may duplicate vertices at face boundaries; weld these before
    # checking inside/outside, without changing vertex coordinates or surfaces.
    mesh = trimesh.load(args.mesh, process=True, force="mesh")
    if not mesh.is_watertight:
        raise ValueError("full target mesh must be watertight for signed clearance")
    support = Rotation.from_quat(np.roll(stage["support_quaternion_wxyz"], -1))
    rows = []
    stale_contact_samples = 0
    for row in result["trace"]:
        if not row.get("legal_safe_robot_contact"):
            continue
        if ("contact_audit_measurement_utime_us" in row
                and row["contact_audit_measurement_utime_us"] != row.get("measurement_utime_us")):
            stale_contact_samples += 1
            continue
        local_center = proxy_center_in_target_frame(row, result, support)[None, :]
        _, distance, _ = trimesh.proximity.closest_point_naive(mesh, local_center)
        if mesh.contains(local_center)[0]:
            distance[0] *= -1
        rows.append({
            "sim_time_s": row["sim_time_s"],
            "measurement_utime_us": row.get("measurement_utime_us"),
            "proxy_surface_clearance_m": float(distance[0]) - radius,
            "c3_mode": row["c3_mode"],
            "physical_hand_contact_force_n": row.get(
                "robot_target_contact_force_n_by_sensor", {}
            ).get("target_hand_contacts"),
        })
    gaps = np.array([row["proxy_surface_clearance_m"] for row in rows])
    summary = {
        "schema": "nonprehensile.closed_gripper_proxy_contact_audit.v1",
        "result": str(args.result),
        "result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "proxy_radius_m": radius,
        "sampled_physical_contact_count": len(rows),
        "excluded_stale_contact_samples": stale_contact_samples,
        "samples_with_proxy_clearance_over_2mm": int(np.sum(gaps > 0.002)),
        "median_proxy_clearance_m": float(np.median(gaps)) if len(gaps) else None,
        "maximum_proxy_clearance_m": float(np.max(gaps)) if len(gaps) else None,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
