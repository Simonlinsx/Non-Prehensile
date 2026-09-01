#!/usr/bin/env python3
"""Export conservative DOMINO safe/protected/neutral triangle meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from dapl.contact_planner.semantic_mesh import FaceSemantic, partition_mesh_faces
from dapl.catalog import stable_poses_from_mesh
from dapl.domino import DominoDataPaths, load_domino_affordance_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domino-root", type=Path, required=True)
    parser.add_argument("--asset-id", default="020_hammer:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/push_anything_semantics/020_hammer_0"),
    )
    parser.add_argument("--safe-threshold", type=float, default=0.25)
    parser.add_argument("--protected-threshold", type=float, default=0.25)
    parser.add_argument(
        "--safe-boundary-clearance",
        type=float,
        default=0.04,
        help=(
            "minimum surface distance from every sampled point of a retained "
            "safe face to protected/neutral faces, in meters"
        ),
    )
    parser.add_argument(
        "--sampler-normal-offset",
        type=float,
        default=0.035,
        help="Push Anything surface-normal EE-center offset in meters",
    )
    parser.add_argument(
        "--sampler-center-clearance",
        type=float,
        default=0.065,
        help="minimum unsafe-surface distance for every offset EE-center sample",
    )
    parser.add_argument(
        "--support-pose-index",
        type=int,
        default=0,
        help=(
            "stable support pose baked into every exported mesh; index 0 is "
            "the same highest-probability support face used by the C1 task"
        ),
    )
    return parser.parse_args()


def _load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene {path} contains no geometry")
        # Apply every scene-graph transform before concatenation.  GLB assets
        # are allowed to store non-identity per-node transforms.
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path} is not a usable triangle mesh")
    return mesh


def _export_partition(
    output_path: Path,
    vertices_m: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
) -> str | None:
    if face_indices.size == 0:
        return None
    partition = trimesh.Trimesh(
        vertices=vertices_m,
        faces=faces[face_indices],
        process=False,
    )
    partition.remove_unreferenced_vertices()
    partition.export(output_path)
    return output_path.name


def _quaternion_wxyz_to_matrix(quaternion: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _mesh_distances(
    mesh: trimesh.Trimesh, points: np.ndarray, batch_size: int = 128
) -> np.ndarray:
    distances = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        _, batch_distances, _ = trimesh.proximity.closest_point_naive(
            mesh, points[start:stop]
        )
        distances.append(batch_distances)
    return np.concatenate(distances)


def _portable_asset_path(path: Path) -> str:
    """Keep provenance useful without embedding the exporting host path."""
    parts = path.resolve().parts
    if "assets" in parts:
        asset_index = parts.index("assets")
        start = max(0, asset_index - 1)
        return Path(*parts[start:]).as_posix()
    return path.name


def main() -> None:
    args = parse_args()
    if args.support_pose_index < 0:
        raise ValueError("support-pose-index must be non-negative")
    if args.safe_boundary_clearance <= 0.0:
        raise ValueError("safe-boundary-clearance must be positive")
    if args.sampler_normal_offset <= 0.0:
        raise ValueError("sampler-normal-offset must be positive")
    if args.sampler_center_clearance <= 0.0:
        raise ValueError("sampler-center-clearance must be positive")
    paths = DominoDataPaths.resolve(args.domino_root)
    asset = paths.require_source_asset(args.asset_id)
    annotation = load_domino_affordance_annotation(asset)
    mesh = _load_triangle_mesh(asset.collision_mesh)

    # Preserve small but valid triangles during geometry validation.  The raw
    # GLB contains details whose doubled area is below float32 epsilon.
    vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float64)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    source_triangles = np.asarray(mesh.vertices, dtype=np.float64)[source_faces]
    doubled_area = np.linalg.norm(
        np.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        ),
        axis=-1,
    )
    valid_face_mask = np.isfinite(doubled_area) & (doubled_area > 1.0e-12)
    valid_source_indices = np.flatnonzero(valid_face_mask)
    if valid_source_indices.size == 0:
        raise ValueError(f"{asset.collision_mesh} has no non-degenerate triangles")
    faces = torch.as_tensor(source_faces[valid_face_mask], dtype=torch.long)
    partition = partition_mesh_faces(
        vertices,
        faces,
        annotation,
        safe_threshold=args.safe_threshold,
        protected_threshold=args.protected_threshold,
    )
    counts = partition.counts()
    if counts["safe"] == 0 or counts["protected"] == 0:
        raise RuntimeError(
            "target semantic export requires at least one safe and one protected face"
        )

    support_poses = stable_poses_from_mesh(
        asset.collision_mesh,
        annotation.scale,
        max_candidates=args.support_pose_index + 1,
    )
    if args.support_pose_index >= len(support_poses):
        raise ValueError(
            f"support-pose-index {args.support_pose_index} is outside the "
            f"{len(support_poses)} available poses"
        )
    support_pose = support_poses[args.support_pose_index]
    support_rotation = _quaternion_wxyz_to_matrix(support_pose.quaternion)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vertices_m_raw = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(
        annotation.scale, dtype=np.float64
    )[None, :]
    # Bake the stable support orientation into the planner asset.  Push
    # Anything can then represent every valid task pose with planar yaw only,
    # while the physical and semantic meshes remain exactly aligned.
    vertices_m = vertices_m_raw @ support_rotation.T
    labels = partition.labels.cpu().numpy()

    full_mesh = trimesh.Trimesh(
        vertices=vertices_m,
        faces=source_faces[valid_face_mask],
        process=False,
    )
    full_mesh.export(args.output_dir / "full.obj")

    exported: dict[str, str | None] = {}
    face_indices: dict[str, list[int]] = {}
    for semantic in FaceSemantic:
        name = semantic.name.lower()
        valid_indices = np.flatnonzero(labels == int(semantic))
        indices = valid_source_indices[valid_indices]
        face_indices[name] = indices.tolist()
        exported[name] = _export_partition(
            args.output_dir / f"{name}.obj",
            vertices_m,
            source_faces,
            indices,
        )

    # C1 treats every non-safe face as forbidden.  Export a single union mesh
    # for the online execution guard so protected and conservative neutral
    # boundary faces cannot be skipped by configuring only the sampler.
    unsafe_valid_indices = np.flatnonzero(labels != int(FaceSemantic.SAFE))
    unsafe_indices = valid_source_indices[unsafe_valid_indices]
    face_indices["unsafe"] = unsafe_indices.tolist()
    exported["unsafe"] = _export_partition(
        args.output_dir / "unsafe.obj",
        vertices_m,
        source_faces,
        unsafe_indices,
    )

    # A point sampled on a safe-labelled triangle is not sufficient for C1:
    # the finite-radius EE sphere can still overlap a nearby protected or
    # neutral face.  Conservatively erode the sampler mesh by requiring all
    # seven face samples (vertices, edge midpoints, centroid) to be separated
    # from the unsafe union.  Physical geometry and the semantic audit meshes
    # remain unchanged.
    safe_valid_indices = np.flatnonzero(labels == int(FaceSemantic.SAFE))
    safe_indices = valid_source_indices[safe_valid_indices]
    safe_triangles = vertices_m[source_faces[safe_indices]]
    safe_face_samples = np.concatenate(
        (
            safe_triangles,
            0.5 * (safe_triangles[:, 0:1] + safe_triangles[:, 1:2]),
            0.5 * (safe_triangles[:, 1:2] + safe_triangles[:, 2:3]),
            0.5 * (safe_triangles[:, 2:3] + safe_triangles[:, 0:1]),
            safe_triangles.mean(axis=1, keepdims=True),
        ),
        axis=1,
    )
    unsafe_mesh = trimesh.Trimesh(
        vertices=vertices_m,
        faces=source_faces[unsafe_indices],
        process=False,
    )
    unsafe_mesh.remove_unreferenced_vertices()
    safe_sample_distances = _mesh_distances(
        unsafe_mesh, safe_face_samples.reshape(-1, 3)
    ).reshape(-1, safe_face_samples.shape[1])
    surface_clear_mask = np.all(
        safe_sample_distances >= args.safe_boundary_clearance, axis=1
    )
    safe_face_cross = np.cross(
        safe_triangles[:, 1] - safe_triangles[:, 0],
        safe_triangles[:, 2] - safe_triangles[:, 0],
    )
    safe_face_normals = safe_face_cross / np.linalg.norm(
        safe_face_cross, axis=1, keepdims=True
    )
    offset_center_samples = (
        safe_face_samples
        + args.sampler_normal_offset * safe_face_normals[:, None, :]
    )
    offset_center_distances = _mesh_distances(
        unsafe_mesh, offset_center_samples.reshape(-1, 3)
    ).reshape(-1, offset_center_samples.shape[1])
    center_clear_mask = np.all(
        offset_center_distances >= args.sampler_center_clearance, axis=1
    )
    guarded_safe_mask = surface_clear_mask & center_clear_mask
    guarded_safe_indices = safe_indices[guarded_safe_mask]
    if guarded_safe_indices.size == 0:
        raise RuntimeError(
            "safe-boundary-clearance removed every safe sampling face"
        )
    face_indices["safe_guarded"] = guarded_safe_indices.tolist()
    exported["safe_guarded"] = _export_partition(
        args.output_dir / "safe_guarded.obj",
        vertices_m,
        source_faces,
        guarded_safe_indices,
    )

    source_digest = hashlib.sha256(asset.collision_mesh.read_bytes()).hexdigest()
    manifest = {
        "schema": "nonprehensile.semantic_mesh.v1",
        "asset_id": args.asset_id,
        "source_collision_mesh": _portable_asset_path(asset.collision_mesh),
        "source_sha256": source_digest,
        "source_frame": "domino_raw_mesh",
        "export_frame": "object_local_same_support_meters",
        "scale": list(annotation.scale),
        "support_pose_index": args.support_pose_index,
        "support_quaternion_wxyz": list(support_pose.quaternion),
        "support_height_m": support_pose.support_height,
        "raw_to_export_rotation": support_rotation.tolist(),
        "safe_threshold": args.safe_threshold,
        "protected_threshold": args.protected_threshold,
        "safe_boundary_clearance_m": args.safe_boundary_clearance,
        "sampler_normal_offset_m": args.sampler_normal_offset,
        "sampler_center_clearance_m": args.sampler_center_clearance,
        "guarded_safe_face_count": int(guarded_safe_indices.size),
        "guarded_safe_min_sample_distance_m": float(
            safe_sample_distances[guarded_safe_mask].min()
        ),
        "guarded_safe_min_center_distance_m": float(
            offset_center_distances[guarded_safe_mask].min()
        ),
        "samples_per_face": partition.samples_per_face,
        "classification": {
            "protected": "any sampled point >= protected threshold",
            "safe": "all sampled points >= safe threshold and none protected",
            "neutral": "all remaining faces",
        },
        "face_count": int(source_faces.shape[0]),
        "valid_face_count": int(valid_source_indices.size),
        "dropped_degenerate_face_count": int(
            source_faces.shape[0] - valid_source_indices.size
        ),
        "counts": counts,
        "physical_mesh": "full.obj",
        "meshes": exported,
        "source_face_indices": face_indices,
    }
    manifest_path = args.output_dir / "semantic_mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
