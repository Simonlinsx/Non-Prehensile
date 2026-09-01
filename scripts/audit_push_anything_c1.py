#!/usr/bin/env python3
"""Audit Push Anything EE contacts against DOMINO semantic surface meshes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=repo_root / "data/push_anything_semantics/020_hammer_0",
    )
    parser.add_argument("--ee-radius-m", type=float, default=0.0195)
    parser.add_argument("--contact-tolerance-m", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("trajectory contains a zero object quaternion")
    w, x, y, z = (quaternions / norms).T
    matrices = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def load_trajectory(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = [
        "ee_x_m", "ee_y_m", "ee_z_m",
        "object_qw", "object_qx", "object_qy", "object_qz",
        "object_x_m", "object_y_m", "object_z_m",
    ]
    if not rows or any(name not in rows[0] for name in required):
        raise ValueError(
            "trajectory CSV lacks object quaternion columns; rerun with the "
            "current Push Anything monitor"
        )
    valid_rows: list[dict[str, str]] = []
    world_ee: list[list[float]] = []
    world_object: list[list[float]] = []
    quaternions: list[list[float]] = []
    for row in rows:
        try:
            ee = [float(row[name]) for name in required[0:3]]
            quat = [float(row[name]) for name in required[3:7]]
            obj = [float(row[name]) for name in required[7:10]]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in ee + quat + obj):
            continue
        valid_rows.append(row)
        world_ee.append(ee)
        quaternions.append(quat)
        world_object.append(obj)
    if not valid_rows:
        raise ValueError("trajectory CSV contains no complete EE/object poses")
    rotations = quaternion_wxyz_to_matrix(np.asarray(quaternions))
    relative = np.asarray(world_ee) - np.asarray(world_object)
    object_local_ee = np.einsum("nji,nj->ni", rotations, relative)
    return valid_rows, object_local_ee


def mesh_distances(mesh: trimesh.Trimesh, points: np.ndarray, batch_size: int) -> np.ndarray:
    distances = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        _, batch_distances, _ = trimesh.proximity.closest_point_naive(
            mesh, points[start:stop]
        )
        distances.append(batch_distances)
    return np.concatenate(distances)


def main() -> None:
    args = parse_args()
    if args.ee_radius_m <= 0 or args.contact_tolerance_m < 0 or args.batch_size <= 0:
        raise ValueError("radius/batch must be positive and tolerance non-negative")
    trajectory_path = args.trajectory_csv.resolve()
    semantic_dir = args.semantic_dir.resolve()
    output_json = args.output_json or trajectory_path.with_name("c1_semantic_audit.json")
    output_csv = args.output_csv or trajectory_path.with_name("c1_semantic_audit.csv")

    manifest = json.loads(
        (semantic_dir / "semantic_mesh_manifest.json").read_text(encoding="utf-8")
    )
    rows, local_ee = load_trajectory(trajectory_path)
    distances: dict[str, np.ndarray] = {}
    for semantic_class in ("safe", "protected", "neutral"):
        mesh_path = semantic_dir / manifest["meshes"][semantic_class]
        mesh = trimesh.load_mesh(mesh_path, process=False)
        distances[semantic_class] = mesh_distances(mesh, local_ee, args.batch_size)

    contact_threshold = args.ee_radius_m + args.contact_tolerance_m
    safe_contact = distances["safe"] <= contact_threshold
    protected_contact = distances["protected"] <= contact_threshold
    neutral_contact = distances["neutral"] <= contact_threshold
    any_contact = safe_contact | protected_contact | neutral_contact
    violation = protected_contact | neutral_contact
    legal_safe_contact = safe_contact & ~violation
    c3_mode = np.asarray([int(row["is_c3_mode"]) != 0 for row in rows])

    fieldnames = list(rows[0]) + [
        "safe_surface_distance_m",
        "protected_surface_distance_m",
        "neutral_surface_distance_m",
        "semantic_contact",
        "legal_safe_contact",
        "c1_violation",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            audited = dict(row)
            audited.update({
                "safe_surface_distance_m": f"{distances['safe'][index]:.9f}",
                "protected_surface_distance_m": f"{distances['protected'][index]:.9f}",
                "neutral_surface_distance_m": f"{distances['neutral'][index]:.9f}",
                "semantic_contact": int(any_contact[index]),
                "legal_safe_contact": int(legal_safe_contact[index]),
                "c1_violation": int(violation[index]),
            })
            writer.writerow(audited)

    summary = {
        "schema": "nonprehensile.push_anything_c1_audit.v1",
        "trajectory_csv": str(trajectory_path),
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "semantic_manifest": str(
            (semantic_dir / "semantic_mesh_manifest.json").resolve()
        ),
        "ee_radius_m": args.ee_radius_m,
        "contact_tolerance_m": args.contact_tolerance_m,
        "contact_threshold_m": contact_threshold,
        "rows": len(rows),
        "c3_mode_rows": int(c3_mode.sum()),
        "contact_rows": int(any_contact.sum()),
        "legal_safe_contact_rows": int(legal_safe_contact.sum()),
        "protected_contact_rows": int(protected_contact.sum()),
        "neutral_contact_rows": int(neutral_contact.sum()),
        "c1_violation_rows": int(violation.sum()),
        "c1_pass": bool(legal_safe_contact.any() and not violation.any()),
        "minimum_surface_distance_m": {
            name: float(values.min()) for name, values in distances.items()
        },
        "output_csv": str(output_csv.resolve()),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["c1_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
