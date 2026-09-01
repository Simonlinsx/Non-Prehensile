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


def main() -> None:
    args = parse_args()
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vertices_m = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(
        annotation.scale, dtype=np.float64
    )[None, :]
    labels = partition.labels.cpu().numpy()

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

    source_digest = hashlib.sha256(asset.collision_mesh.read_bytes()).hexdigest()
    manifest = {
        "schema": "nonprehensile.semantic_mesh.v1",
        "asset_id": args.asset_id,
        "source_collision_mesh": str(asset.collision_mesh),
        "source_sha256": source_digest,
        "source_frame": "domino_raw_mesh",
        "export_frame": "object_local_meters",
        "scale": list(annotation.scale),
        "safe_threshold": args.safe_threshold,
        "protected_threshold": args.protected_threshold,
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
        "meshes": exported,
        "source_face_indices": face_indices,
    }
    manifest_path = args.output_dir / "semantic_mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
