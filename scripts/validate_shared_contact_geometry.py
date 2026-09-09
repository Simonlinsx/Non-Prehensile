#!/usr/bin/env python3
"""Cross-check native Drake contact queries against PhysX contact manifolds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def compare(physx, native_log):
    native = [list(map(float, line.split()[1:])) for line in native_log.splitlines()
              if line.startswith("CLOSED_GRIPPER_GEOMETRY_AUDIT ")]
    reconstructed = physx["reconstructed_contact_poses"]
    if not native or len(native) != len(reconstructed):
        raise ValueError("Native/PhysX contact sample count differs")
    samples = []
    for i, (query, recorded) in enumerate(zip(native, reconstructed)):
        if len(query) != 9 or int(query[0]) != i or not np.all(np.isfinite(query)):
            raise ValueError("Invalid native contact query")
        normal = np.array(query[2:5])
        if abs(np.linalg.norm(normal) - 1) > 1e-6:
            raise ValueError("Native contact normal is not unit length")
        contacts = recorded["one_step_contact_probe"]["contacts"]
        if not contacts:
            raise ValueError(f"No PhysX contact manifold at sample {i}")
        # PhysX's target reporter gives the opposite geometric normal to
        # Drake's target-A/finger-B nhat_BA; its signed force scalar is negative.
        angles = [np.arccos(np.clip(np.dot(-np.array(c["normal_w"]) / np.linalg.norm(c["normal_w"]), normal), -1, 1)) for c in contacts]
        hand = recorded["body_poses_w"][physx["body_names"].index("panda_hand")]
        rotation_error = (Rotation.from_quat(np.roll(query[5:9], -1)).inv() * Rotation.from_quat(np.roll(hand[3:7], -1))).magnitude()
        samples.append({
            "index": i, "reference_result": recorded["result"], "reference_step": recorded["step"],
            "native_signed_distance_m": query[1],
            "physx_minimum_separation_m": min(c["separation_m"] for c in contacts),
            "separation_error_m": abs(query[1] - min(c["separation_m"] for c in contacts)),
            "closest_manifold_normal_error_deg": float(np.degrees(min(angles))),
            "hand_orientation_error_rad": float(rotation_error),
        })
    summary = {
        "schema": "nonprehensile.shared_contact_geometry_validation.v1",
        "sample_count": len(samples),
        "maximum_positive_native_gap_m": max(s["native_signed_distance_m"] for s in samples),
        "maximum_separation_error_m": max(s["separation_error_m"] for s in samples),
        "maximum_closest_manifold_normal_error_deg": max(s["closest_manifold_normal_error_deg"] for s in samples),
        "maximum_hand_orientation_error_rad": max(s["hand_orientation_error_rad"] for s in samples),
        "samples": samples,
        "limitation": "Zero-velocity, one-step contact probe; compares closest manifold normal, not every contact force or task success.",
    }
    summary["geometry_check_passed"] = (
        summary["maximum_separation_error_m"] < 5e-5
        and summary["maximum_closest_manifold_normal_error_deg"] < 5
        and summary["maximum_hand_orientation_error_rad"] < 2e-6
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physx-export", type=Path, required=True)
    p.add_argument("--native-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare(json.loads(args.physx_export.read_text()), args.native_log.read_text())
    report["physx_export_sha256"] = hashlib.sha256(args.physx_export.read_bytes()).hexdigest()
    report["native_log_sha256"] = hashlib.sha256(args.native_log.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    return 0 if report["geometry_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
