#!/usr/bin/env python3
"""Compare live PhysX convex geometry with C3 at recorded contact poses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh


def rotation(pose):
    return Rotation.from_quat(np.roll(pose[3:7], -1))


def mesh_from_record(record):
    faces, offset = [], 0
    for count in record["face_vertex_counts"]:
        face = record["face_vertex_indices"][offset:offset + count]
        faces.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
        offset += count
    return trimesh.Trimesh(record["vertices_body_m"], faces, process=True)


def signed_surface_distance(mesh, points):
    _, distances, _ = trimesh.proximity.closest_point(mesh, np.atleast_2d(points))
    distances[mesh.contains(np.atleast_2d(points))] *= -1
    return distances


def convex_distance(vertices_a, vertices_b):
    """Convex-set distance QP with feasibility and optimality verification."""
    origin = np.mean(vertices_a, axis=0)
    a, b = (np.asarray(vertices_a) - origin) * 100, (np.asarray(vertices_b) - origin) * 100
    ea, eb = [np.unique(np.round(ConvexHull(v).equations, 12), axis=0) for v in (a, b)]
    matrix = np.zeros((len(ea) + len(eb), 6))
    matrix[:len(ea), :3], matrix[len(ea):, 3:] = ea[:, :3], eb[:, :3]
    offset = np.r_[ea[:, 3], eb[:, 3]]
    result = minimize(lambda x: .5 * np.sum((x[:3] - x[3:]) ** 2),
                      np.r_[a.mean(0), b.mean(0)],
                      jac=lambda x: np.r_[x[:3] - x[3:], x[3:] - x[:3]],
                      constraints={"type": "ineq", "fun": lambda x: -matrix @ x - offset,
                                   "jac": lambda x: -matrix},
                      method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
    delta = result.x[:3] - result.x[3:]
    # These inequalities certify the first-order minimum over both convex
    # hulls, even if SLSQP reports a line-search precision warning.
    optimality = min(np.min((a - result.x[:3]) @ delta),
                     np.min((b - result.x[3:]) @ -delta))
    if np.max(matrix @ result.x + offset) > 1e-6 or optimality < -1e-5:
        raise RuntimeError(f"convex distance solver failed: {result.message}; feasibility={np.max(matrix @ result.x + offset)} optimality={optimality}")
    return float(np.linalg.norm(delta) / 100), result.x.reshape(2, 3) / 100 + origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    live = json.loads(args.export.read_text())
    stage = json.loads(args.stage_manifest.read_text())
    support = Rotation.from_quat(np.roll(stage["support_quaternion_wxyz"], -1))
    radius = stage["end_effector"]["proxy_radius_m"]
    colliders = {c["body_path"].split("/")[-1]: c for c in live["colliders"]}
    target_collider = colliders["Target"]
    target_source = mesh_from_record(target_collider)
    target_hulls = [mesh_from_record(c) for c in target_collider["cooked_convexes"]]
    planner_mesh = trimesh.load(args.mesh, force="mesh", process=True)
    source_in_planner = support.apply(target_source.vertices)
    # Surface distances, not nearest-vertex distances (tessellations differ).
    mesh_difference = np.abs(signed_surface_distance(planner_mesh, source_in_planner))
    samples, result_cache = [], {}
    for reconstructed in live["reconstructed_contact_poses"]:
        path = reconstructed["result"]
        if path not in result_cache:
            result_cache[path] = json.loads(Path(path).read_text())
        result = result_cache[path]
        row = next(r for r in result["trace"] if r["step"] == reconstructed["step"])
        target_pose = row["target_position_m"] + row["target_quaternion_wxyz"]
        target_rotation = rotation(target_pose)
        root_pose = live["robot_root_pose_w"]
        tip_world = rotation(root_pose).apply(row["planner_tip_position_m"]) + np.array(root_pose[:3])
        tip_body = target_rotation.inv().apply(tip_world - target_pose[:3])
        planner_gap = signed_surface_distance(planner_mesh, support.apply(tip_body))[0] - radius
        source_gap = signed_surface_distance(target_source, tip_body)[0] - radius
        cooked_gap = min(signed_surface_distance(h, tip_body)[0] for h in target_hulls) - radius
        hand_pose = reconstructed["body_poses_w"][live["body_names"].index("panda_hand")]
        reconstructed_tip = rotation(hand_pose).apply([0, 0, stage["end_effector"]["reference_offset_m"]]) + hand_pose[:3]
        best_distance, best_points, best_body = float("inf"), None, None
        for body_name in ("panda_leftfinger", "panda_rightfinger"):
            body_pose = reconstructed["body_poses_w"][live["body_names"].index(body_name)]
            for finger in colliders[body_name]["cooked_convexes"]:
                finger_world = rotation(body_pose).apply(finger["vertices_body_m"]) + body_pose[:3]
                finger_target = target_rotation.inv().apply(finger_world - target_pose[:3])
                for hull in target_hulls:
                    distance, points = convex_distance(finger_target, hull.vertices)
                    if distance < best_distance:
                        best_distance, best_points, best_body = distance, points, body_name
        samples.append({
            "result": path, "step": row["step"], "force_n": row["robot_target_contact_force_n_by_sensor"]["target_hand_contacts"],
            "planner_sphere_clearance_m": float(planner_gap),
            "usd_source_sphere_clearance_m": float(source_gap),
            "physx_convex_sphere_clearance_m": float(cooked_gap),
            "closed_fingers_to_physx_target_distance_m": best_distance,
            "closest_finger": best_body, "closest_points_target_frame_m": best_points.tolist(),
            "reconstructed_tip_error_m": float(np.linalg.norm(reconstructed_tip - tip_world)),
            "assumed_finger_positions_m": reconstructed["assumed_finger_positions_m"],
        })
    summary = {
        "schema": "nonprehensile.closed_gripper_geometry_comparison.v1",
        "export_sha256": hashlib.sha256(args.export.read_bytes()).hexdigest(),
        "target_physx_convex_count": len(target_hulls),
        "usd_source_to_planner_surface_distance_m": dict(zip(("median", "p95", "max"), np.quantile(mesh_difference, [.5, .95, 1]).tolist())),
        "maximum_reconstructed_tip_error_m": max(s["reconstructed_tip_error_m"] for s in samples),
        "sample_count": len(samples),
        "samples": samples,
        "limitation": "Reconstruction assumes zero finger displacement; historical finger states and substep contact points were not logged. Convex distance is zero for overlap, not penetration depth.",
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    for key in ("planner_sphere_clearance_m", "physx_convex_sphere_clearance_m", "closed_fingers_to_physx_target_distance_m"):
        print(key, "median/max", np.quantile([s[key] for s in samples], [.5, 1]))
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        for key, label, marker in (
            ("planner_sphere_clearance_m", "8 mm sphere vs C3 target mesh", "o"),
            ("physx_convex_sphere_clearance_m", "8 mm sphere vs PhysX target convexes", "s"),
            ("closed_fingers_to_physx_target_distance_m", "Closed finger convexes vs PhysX target convexes", "^"),
        ):
            ax.plot(np.arange(1, len(samples) + 1), [s[key] * 1000 for s in samples], marker, label=label)
        ax.axhline(1.011, color="gray", linestyle="--", linewidth=1, label="Finger + target contact offsets (1.011 mm)")
        ax.set(xlabel="Recorded contact sample (v2: 1–8; v3: 9–15; v4: 16–19)", ylabel="Surface gap / convex-set distance (mm)",
               title="Physical finger contacts remain while the sphere proxy is separated")
        ax.set_xticks(range(1, len(samples) + 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
