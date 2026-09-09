#!/usr/bin/env python3
"""Reconstruct sampled finger/table clearance from cooked body-frame geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid measured body quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def audit(result, export, table_local_z_m=0.0):
    if not np.isfinite(table_local_z_m):
        raise ValueError("Invalid table plane")
    vertices = {}
    for name in ("panda_leftfinger", "panda_rightfinger"):
        collider = next(c for c in export["colliders"] if c["body_path"].endswith("/" + name))
        points = np.asarray([p for convex in collider["cooked_convexes"]
                             for p in convex["vertices_body_m"]], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Invalid cooked finger vertices")
        vertices[name] = points
    rows = []
    orientation_samples = []
    initial_hand_rotation = None
    guard_samples = []
    for row in result["trace"]:
        table_z = float(row["env_origin_w"][2]) + table_local_z_m
        if not np.isfinite(table_z):
            raise ValueError("Invalid measured table origin")
        by_body = {}
        for name, points in vertices.items():
            # Each finger's measured pose includes its actual joint opening.
            pose = np.asarray(row["contact_body_poses_w"][name], dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("Missing or invalid measured finger pose")
            world = points @ rotation_wxyz(pose[3:]).T + pose[:3]
            by_body[name] = float(world[:, 2].min() - table_z)
        entry = {"sim_time_s": row["sim_time_s"],
                 "minimum_clearance_m": min(by_body.values()), "by_body_m": by_body}
        for key in ("raw_task_target_c3_m", "governed_task_target_c3_m",
                    "planner_tip_position_m", "task_reference_floor_c3_m",
                    "applied_feedforward_force_n", "osc_reference_tip_velocity_m_s",
                    "task_tracking_error_m"):
            entry[key] = row.get(key)
        rows.append(entry)
        hand_pose = row["contact_body_poses_w"].get("panda_hand")
        if hand_pose is not None:
            hand_rotation = rotation_wxyz(hand_pose[3:])
            if initial_hand_rotation is None:
                initial_hand_rotation = hand_rotation
            cosine = (np.trace(initial_hand_rotation.T @ hand_rotation) - 1) / 2
            orientation_samples.append({"sim_time_s": row["sim_time_s"],
                "error_rad": float(np.arccos(np.clip(cosine, -1, 1)))})
        if "semantic_c1_guard_active" in row:
            if row["semantic_c1_guard_active"]:
                for key in ("applied_feedforward_force_n", "osc_reference_tip_velocity_m_s"):
                    value = np.asarray(row[key], dtype=float)
                    if value.shape != (3,) or not np.isfinite(value).all():
                        raise ValueError("Invalid guarded force/velocity trace")
            guard_samples.append(row)
    if not rows:
        raise ValueError("No measured trace to audit")
    clearances = np.array([r["minimum_clearance_m"] for r in rows])
    clamped = [r for r in rows if r["raw_task_target_c3_m"] is not None
               and r["task_reference_floor_c3_m"] is not None
               and r["raw_task_target_c3_m"][2] < r["task_reference_floor_c3_m"]]
    return {
        "schema": "nonprehensile.sampled_finger_clearance.v1",
        "sample_count": len(rows), "table_local_z_m": table_local_z_m,
        "minimum_clearance_m": float(clearances.min()),
        "median_clearance_m": float(np.median(clearances)),
        "samples_below_table": int(np.sum(clearances < 0)),
        "samples_within_0_1mm": int(np.sum(clearances < .0001)),
        "height_clamped_samples": len(clamped),
        "height_clamped_with_downward_feedforward": sum(
            r["applied_feedforward_force_n"] is not None
            and r["applied_feedforward_force_n"][2] < 0 for r in clamped),
        "worst_sample": min(rows, key=lambda r: r["minimum_clearance_m"]),
        "hand_orientation_from_first_trace_pose": {
            "sample_count": len(orientation_samples),
            "worst_sample": max(orientation_samples, key=lambda r: r["error_rad"])
                if orientation_samples else None,
            "median_error_rad": float(np.median([r["error_rad"] for r in orientation_samples]))
                if orientation_samples else None,
        },
        "semantic_guard_sample_audit": {
            "available": bool(guard_samples),
            "sample_count": len(guard_samples),
            "active_samples": sum(bool(r["semantic_c1_guard_active"]) for r in guard_samples),
            "active_samples_with_nonzero_force_or_velocity": int(sum(
                bool(r["semantic_c1_guard_active"]) and (
                    np.linalg.norm(r["applied_feedforward_force_n"]) > 1e-12 or
                    np.linalg.norm(r["osc_reference_tip_velocity_m_s"]) > 1e-12)
                for r in guard_samples)),
        },
        "limitations": [
            "Only recorded trace samples; not a continuous clearance guarantee.",
            "Cooked convex surface distance excludes contact/rest offsets and does not prove a contact force.",
            "Only the two fingers are reconstructed; palm and arm are not included.",
            "Hand orientation is relative to the first recorded pose; it is not a contact-force measurement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--contact-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-local-z-m", type=float, default=0.0)
    args = parser.parse_args()
    report = audit(json.loads(args.result.read_text()),
                   json.loads(args.contact_export.read_text()), args.table_local_z_m)
    for name, path in (("result", args.result), ("contact_export", args.contact_export)):
        report[name] = str(path)
        report[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("sample_count", "minimum_clearance_m",
          "samples_below_table", "height_clamped_with_downward_feedforward")}))


if __name__ == "__main__":
    main()
