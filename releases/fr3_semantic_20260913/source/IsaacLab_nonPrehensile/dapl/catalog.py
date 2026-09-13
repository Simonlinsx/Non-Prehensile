"""Asset-catalog adapters used by the Clutter6D manifest generator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Sequence

from .generation import ClutterAsset, StablePose


def _rotation_matrix_to_quaternion(matrix: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""

    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    norm = math.sqrt(sum(item * item for item in quaternion))
    return tuple(float(item / norm) for item in quaternion)


def stable_poses_from_mesh(
    mesh_path: str | Path,
    scale: Sequence[float],
    *,
    max_candidates: int = 64,
    minimum_probability: float = 1.0e-6,
) -> tuple[StablePose, ...]:
    """Compute stable orientations and support footprints with trimesh.

    The calculation is an offline manifest-generation operation.  Importing
    :mod:`dapl` itself therefore remains independent of trimesh.
    """

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    import numpy as np
    import trimesh

    mesh = trimesh.load(Path(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
        raise ValueError(f"{mesh_path} is not a usable triangle mesh")
    scale_array = np.asarray(tuple(float(item) for item in scale), dtype=np.float64)
    if scale_array.shape != (3,) or np.any(scale_array <= 0.0):
        raise ValueError("scale must contain three positive values")
    scaled_vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale_array[None, :]
    scaled_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=mesh.faces, process=False)
    transforms, probabilities = scaled_mesh.compute_stable_poses()
    order = np.argsort(-np.asarray(probabilities, dtype=np.float64), kind="stable")

    result: list[StablePose] = []
    seen: set[tuple[float, ...]] = set()
    for index in order:
        if len(result) == max_candidates:
            break
        if float(probabilities[index]) < minimum_probability:
            continue
        rotation = np.asarray(transforms[index], dtype=np.float64)[:3, :3]
        rotated = scaled_vertices @ rotation.T
        minimum = rotated.min(axis=0)
        maximum = rotated.max(axis=0)
        quaternion = _rotation_matrix_to_quaternion(rotation.tolist())
        signature = tuple(round(item, 6) for item in quaternion)
        # q and -q describe the same rotation.
        inverse_signature = tuple(round(-item, 6) for item in quaternion)
        if signature in seen or inverse_signature in seen:
            continue
        seen.add(signature)
        result.append(
            StablePose(
                quaternion=quaternion,
                support_height=max(0.0, float(-minimum[2])),
                footprint=(
                    float(minimum[0]),
                    float(minimum[1]),
                    float(maximum[0]),
                    float(maximum[1]),
                ),
            )
        )
    if not result:
        minimum = scaled_vertices.min(axis=0)
        maximum = scaled_vertices.max(axis=0)
        result.append(
            StablePose(
                quaternion=(1.0, 0.0, 0.0, 0.0),
                support_height=max(0.0, float(-minimum[2])),
                footprint=(
                    float(minimum[0]),
                    float(minimum[1]),
                    float(maximum[0]),
                    float(maximum[1]),
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class DGNAssetRecord:
    """One scaled DGN object used for the local Clutter6D integration path."""

    asset_id: str
    base_name: str
    scale: float
    mesh_path: Path
    usd_path: Path
    volume_m3: float
    maximum_extent_m: float


def load_dgn_asset_records(root: str | Path) -> tuple[DGNAssetRecord, ...]:
    """Load the released DGN index without importing Isaac Sim."""

    root = Path(root).expanduser().resolve()
    with (root / "yes.json").open("r", encoding="utf-8") as stream:
        entries = json.load(stream)
    with (root / "meta-v8" / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    records: list[DGNAssetRecord] = []
    for entry in entries:
        if not isinstance(entry, str) or "-" not in entry:
            raise ValueError(f"invalid DGN asset entry: {entry!r}")
        base_name, scale_text = entry.rsplit("-", 1)
        scale = float(scale_text)
        item = metadata.get(entry)
        if item is None:
            raise KeyError(f"DGN metadata is missing {entry!r}")
        bounds = item["aabb"]
        extents = [float(bounds[1][axis]) - float(bounds[0][axis]) for axis in range(3)]
        mesh_path = root / "coacd_normalized" / f"{base_name}.obj"
        usd_path = root / "coacd_usd_convexhull" / base_name / f"{base_name}.usd"
        if not mesh_path.is_file() or not usd_path.is_file():
            raise FileNotFoundError(f"DGN files are incomplete for {entry!r}")
        records.append(
            DGNAssetRecord(
                asset_id=entry,
                base_name=base_name,
                scale=scale,
                mesh_path=mesh_path,
                usd_path=usd_path,
                volume_m3=float(item["volume"]),
                maximum_extent_m=max(extents),
            )
        )
    return tuple(records)


def build_dgn_clutter_catalog(
    root: str | Path,
    *,
    seed: int,
    assets_per_cohort: int = 12,
    large_extent_threshold_m: float = 0.18,
    density_kg_m3: float = 500.0,
    maximum_stable_poses: int = 64,
) -> tuple[ClutterAsset, ...]:
    """Build a small DGN-backed development catalog.

    DAPL classifies its normalized 10K assets by a source scale threshold of
    0.25. DGN uses a different normalization, so this adapter explicitly uses
    the physical maximum extent only for integration testing. It is not a
    substitute for the released Clutter6D object metadata.
    """

    if assets_per_cohort <= 0 or density_kg_m3 <= 0.0:
        raise ValueError("catalog size and density must be positive")
    records = load_dgn_asset_records(root)
    large = [item for item in records if item.maximum_extent_m >= large_extent_threshold_m]
    small = [item for item in records if item.maximum_extent_m < large_extent_threshold_m]
    if len(large) < assets_per_cohort or len(small) < assets_per_cohort:
        raise ValueError(
            f"DGN cohort threshold yields {len(large)} large and {len(small)} small records; "
            f"need {assets_per_cohort} each"
        )
    rng = random.Random(int(seed))
    selected = rng.sample(large, assets_per_cohort) + rng.sample(small, assets_per_cohort)
    catalog: list[ClutterAsset] = []
    for record in selected:
        scale = (record.scale, record.scale, record.scale)
        mass = min(1.5, max(0.05, record.volume_m3 * density_kg_m3))
        catalog.append(
            ClutterAsset(
                asset_id=record.asset_id,
                scale=scale,
                mass_kg=mass,
                stable_poses=stable_poses_from_mesh(
                    record.mesh_path, scale, max_candidates=maximum_stable_poses
                ),
                is_large=record.maximum_extent_m >= large_extent_threshold_m,
            )
        )
    return tuple(catalog)
