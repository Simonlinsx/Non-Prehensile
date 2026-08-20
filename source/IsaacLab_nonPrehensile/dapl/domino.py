"""DOMINO/RoboTwin rigid assets and sparse affordance annotations.

DOMINO stores rigid-object annotations in ``model_data<ID>.json`` files.  A
``contact_points_pose`` entry describes a manipulation/contact anchor, while a
``functional_matrix`` entry describes the part that performs the object's
function (for example, the head of a hammer).  Both are 4x4 transforms in the
unscaled object mesh frame.

The helpers in this module deliberately remain independent of Isaac Sim.  They
resolve source/converted assets, validate annotations, and turn sparse anchors
into point-wise safe/protected scores that can be consumed by Isaac Lab or
offline dataset tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch


DOMINO_ROOT_ENV = "DOMINO_ROOT"
DOMINO_USD_ROOT_ENV = "DOMINO_USD_ROOT"

_ASSET_ID_PATTERN = re.compile(
    r"^(?P<category>[0-9]{3}_[A-Za-z0-9][A-Za-z0-9_-]*):(?P<model_id>[0-9]+)$"
)


def parse_domino_asset_id(asset_id: str) -> tuple[str, int]:
    """Parse a portable DOMINO id such as ``020_hammer:0``.

    Restricting the id to a category basename and an integer model id prevents
    a manifest from escaping the configured asset root.
    """

    match = _ASSET_ID_PATTERN.fullmatch(str(asset_id))
    if match is None:
        raise ValueError(
            "DOMINO asset_id must use '<NNN_category>:<model_id>', "
            f"got {asset_id!r}"
        )
    return match.group("category"), int(match.group("model_id"))


def make_domino_asset_id(category: str, model_id: int) -> str:
    """Create and validate a portable DOMINO asset id."""

    asset_id = f"{category}:{int(model_id)}"
    parse_domino_asset_id(asset_id)
    return asset_id


def _resolve_objects_root(root: Path) -> Path:
    if (root / "assets" / "objects").is_dir():
        return (root / "assets" / "objects").resolve()
    if root.name == "objects" and root.is_dir():
        return root.resolve()
    raise FileNotFoundError(
        f"DOMINO root {root} does not contain assets/objects and is not an objects directory"
    )


def _first_existing(directory: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path.resolve()
    return (directory / names[0]).resolve()


@dataclass(frozen=True)
class DominoAssetPaths:
    """Resolved source annotation/meshes and one converted Isaac Lab USD."""

    asset_id: str
    category: str
    model_id: int
    annotation: Path
    points_info: Path
    collision_mesh: Path
    visual_mesh: Path
    usd: Path

    def missing_source_files(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (self.annotation, self.collision_mesh, self.visual_mesh)
            if not path.is_file()
        )

    def missing_sim_files(self) -> tuple[Path, ...]:
        return (*self.missing_source_files(), *((self.usd,) if not self.usd.is_file() else ()))


@dataclass(frozen=True)
class DominoDataPaths:
    """Portable resolver for a DOMINO checkout and its converted USD cache."""

    root: Path
    objects_root: Path
    usd_root: Path

    @classmethod
    def resolve(
        cls,
        root: str | Path | None = None,
        usd_root: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "DominoDataPaths":
        environ = os.environ if environ is None else environ
        if root is None:
            root = environ.get(DOMINO_ROOT_ENV)
        if root is None or not str(root).strip():
            raise ValueError(f"set {DOMINO_ROOT_ENV} or pass a DOMINO root")
        root_path = Path(root).expanduser().resolve()
        objects_root = _resolve_objects_root(root_path)

        if usd_root is None:
            usd_root = environ.get(DOMINO_USD_ROOT_ENV)
        if usd_root is None or not str(usd_root).strip():
            # Source-only tools can still resolve annotations.  Simulation
            # resolution will report the exact missing converted path.
            usd_path = root_path / ".isaaclab_nonprehensile_usd"
        else:
            usd_path = Path(usd_root).expanduser().resolve()
        return cls(root=root_path, objects_root=objects_root, usd_root=usd_path)

    def asset(self, asset_id: str) -> DominoAssetPaths:
        category, model_id = parse_domino_asset_id(asset_id)
        model_dir = self.objects_root / category
        suffix = str(model_id)
        collision_dir = model_dir / "collision" if (model_dir / "collision").is_dir() else model_dir
        visual_dir = model_dir / "visual" if (model_dir / "visual").is_dir() else model_dir
        mesh_names = (f"base{suffix}.glb", f"textured{suffix}.obj")
        return DominoAssetPaths(
            asset_id=asset_id,
            category=category,
            model_id=model_id,
            annotation=(model_dir / f"model_data{suffix}.json").resolve(),
            points_info=(model_dir / "points_info.json").resolve(),
            collision_mesh=_first_existing(collision_dir, mesh_names),
            visual_mesh=_first_existing(visual_dir, mesh_names),
            usd=(self.usd_root / category / f"base{suffix}.usd").resolve(),
        )

    def require_source_asset(self, asset_id: str) -> DominoAssetPaths:
        paths = self.asset(asset_id)
        missing = paths.missing_source_files()
        if missing:
            formatted = "\n  - ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"DOMINO source asset {asset_id!r} is incomplete; missing:\n  - {formatted}"
            )
        return paths

    def require_sim_asset(self, asset_id: str) -> DominoAssetPaths:
        paths = self.asset(asset_id)
        missing = paths.missing_sim_files()
        if missing:
            formatted = "\n  - ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"DOMINO simulation asset {asset_id!r} is incomplete; missing:\n  - {formatted}\n"
                "Run scripts/prepare_domino_affordance_assets.py first."
            )
        return paths


def _finite_vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _matrix4(value: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, ...], ...]:
    if len(value) != 4:
        raise ValueError(f"{name} must be a 4x4 matrix")
    matrix = tuple(_finite_vector(row, 4, name) for row in value)
    if any(abs(matrix[3][index]) > 1.0e-5 for index in range(3)) or not math.isclose(
        matrix[3][3], 1.0, abs_tol=1.0e-5
    ):
        raise ValueError(f"{name} must be a homogeneous transform")
    return matrix


def _descriptions(data: Mapping[str, object], stem: str, count: int) -> tuple[str, ...]:
    # Several released RoboTwin files use the historical "discription" typo.
    candidates = (
        f"{stem}_description",
        f"{stem}_descriptions",
        f"{stem}_discription",
        f"{stem}_discriptions",
    )
    values: Sequence[object] = ()
    for key in candidates:
        item = data.get(key)
        if isinstance(item, list):
            values = item
            break
    return tuple(str(values[index]) if index < len(values) else "" for index in range(count))


@dataclass(frozen=True)
class DominoAffordanceAnchor:
    """One object-local affordance pose and its free-form description."""

    matrix: tuple[tuple[float, ...], ...]
    description: str = ""

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.matrix[0][3], self.matrix[1][3], self.matrix[2][3])

    def scaled_position(self, scale: Sequence[float]) -> tuple[float, float, float]:
        scale = _finite_vector(scale, 3, "scale")
        return tuple(self.position[index] * scale[index] for index in range(3))


@dataclass(frozen=True)
class DominoAffordanceAnnotation:
    """Validated rigid-object metadata relevant to semantic contact."""

    asset_id: str
    scale: tuple[float, float, float]
    center: tuple[float, float, float]
    extents: tuple[float, float, float]
    contact_anchors: tuple[DominoAffordanceAnchor, ...]
    functional_anchors: tuple[DominoAffordanceAnchor, ...]

    @property
    def maximum_extent_m(self) -> float:
        return max(self.extents[index] * self.scale[index] for index in range(3))

    def anchor_positions(
        self,
        kind: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if kind == "contact":
            anchors = self.contact_anchors
        elif kind == "functional":
            anchors = self.functional_anchors
        else:
            raise ValueError("kind must be 'contact' or 'functional'")
        positions = [anchor.scaled_position(self.scale) for anchor in anchors]
        if not positions:
            return torch.empty((0, 3), device=device, dtype=dtype)
        return torch.tensor(positions, device=device, dtype=dtype)


def load_domino_affordance_annotation(
    path_or_asset: str | Path | DominoAssetPaths,
    *,
    asset_id: str | None = None,
) -> DominoAffordanceAnnotation:
    """Load one DOMINO ``model_data`` annotation."""

    if isinstance(path_or_asset, DominoAssetPaths):
        path = path_or_asset.annotation
        asset_id = path_or_asset.asset_id
    else:
        path = Path(path_or_asset)
    if asset_id is None:
        asset_id = path.stem
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    scale = _finite_vector(data.get("scale", (1.0, 1.0, 1.0)), 3, "scale")
    if any(item <= 0.0 for item in scale):
        raise ValueError("DOMINO annotation scale components must be positive")
    center = _finite_vector(data.get("center", (0.0, 0.0, 0.0)), 3, "center")
    extents = _finite_vector(data.get("extents", (0.0, 0.0, 0.0)), 3, "extents")
    if any(item < 0.0 for item in extents):
        raise ValueError("DOMINO annotation extents must be non-negative")

    contact_values = data.get("contact_points_pose", ())
    functional_values = data.get("functional_matrix", ())
    if not isinstance(contact_values, list) or not isinstance(functional_values, list):
        raise ValueError("DOMINO contact_points_pose and functional_matrix must be lists")
    contact_descriptions = _descriptions(data, "contact_points", len(contact_values))
    functional_descriptions = _descriptions(data, "functional_point", len(functional_values))
    contacts = tuple(
        DominoAffordanceAnchor(
            matrix=_matrix4(value, f"contact_points_pose[{index}]"),
            description=contact_descriptions[index],
        )
        for index, value in enumerate(contact_values)
    )
    functional = tuple(
        DominoAffordanceAnchor(
            matrix=_matrix4(value, f"functional_matrix[{index}]"),
            description=functional_descriptions[index],
        )
        for index, value in enumerate(functional_values)
    )
    return DominoAffordanceAnnotation(
        asset_id=str(asset_id),
        scale=scale,
        center=center,
        extents=extents,
        contact_anchors=contacts,
        functional_anchors=functional,
    )


def default_affordance_radius(annotation: DominoAffordanceAnnotation) -> float:
    """Choose a conservative metric radius for a sparse DOMINO anchor."""

    return max(0.015, 0.10 * annotation.maximum_extent_m)


def domino_point_affordance_features(
    canonical_points: torch.Tensor,
    annotation: DominoAffordanceAnnotation,
    *,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
) -> torch.Tensor:
    """Convert sparse anchors into aligned ``[safe, protected]`` point scores.

    ``canonical_points`` are in the raw DOMINO mesh frame.  The returned
    tensor follows the same leading shape and contains Gaussian proximity
    scores in ``[0, 1]``.  Protected semantics take precedence in overlap
    regions by suppressing the safe score.
    """

    if canonical_points.ndim < 2 or canonical_points.shape[-1] != 3:
        raise ValueError("canonical_points must have shape [..., points, 3]")
    if not canonical_points.is_floating_point():
        raise ValueError("canonical_points must use a floating-point dtype")
    default_radius = default_affordance_radius(annotation)
    safe_radius = default_radius if safe_radius_m is None else float(safe_radius_m)
    protected_radius = default_radius if protected_radius_m is None else float(protected_radius_m)
    if safe_radius <= 0.0 or protected_radius <= 0.0:
        raise ValueError("affordance radii must be positive")

    scale = torch.tensor(annotation.scale, device=canonical_points.device, dtype=canonical_points.dtype)
    scaled_points = canonical_points * scale

    def score(kind: str, radius: float) -> torch.Tensor:
        anchors = annotation.anchor_positions(
            kind, device=canonical_points.device, dtype=canonical_points.dtype
        )
        if anchors.numel() == 0:
            return torch.zeros(canonical_points.shape[:-1], device=canonical_points.device, dtype=canonical_points.dtype)
        distances = torch.cdist(scaled_points.reshape(-1, scaled_points.shape[-2], 3), anchors.unsqueeze(0))
        minimum = distances.amin(dim=-1).reshape(canonical_points.shape[:-1])
        return torch.exp(-0.5 * (minimum / radius).square())

    safe = score("contact", safe_radius)
    protected = score("functional", protected_radius)
    # A point close to both anchors is conservatively treated as protected.
    safe = safe * (1.0 - protected)
    return torch.stack((safe, protected), dim=-1)


def build_domino_clutter_catalog(
    root: str | Path,
    asset_ids: Sequence[str],
    *,
    large_extent_threshold_m: float = 0.16,
    density_kg_m3: float = 500.0,
    maximum_stable_poses: int = 64,
    require_affordance: bool = False,
):
    """Build :class:`dapl.generation.ClutterAsset` records from DOMINO meshes."""

    if not asset_ids:
        raise ValueError("asset_ids must be non-empty")
    if large_extent_threshold_m <= 0.0 or density_kg_m3 <= 0.0:
        raise ValueError("extent threshold and density must be positive")

    import numpy as np
    import trimesh

    from .catalog import stable_poses_from_mesh
    from .generation import ClutterAsset

    paths = DominoDataPaths.resolve(root)
    result = []
    for asset_id in asset_ids:
        asset = paths.require_source_asset(asset_id)
        annotation = load_domino_affordance_annotation(asset)
        if require_affordance and (
            not annotation.contact_anchors or not annotation.functional_anchors
        ):
            raise ValueError(
                f"DOMINO asset {asset_id!r} lacks contact or functional anchors"
            )
        mesh = trimesh.load(asset.collision_mesh, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
            raise ValueError(f"{asset.collision_mesh} is not a usable triangle mesh")
        scale_array = np.asarray(annotation.scale, dtype=np.float64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale_array[None, :]
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        maximum_extent = float(extents.max())
        volume = abs(float(mesh.volume)) * float(np.prod(scale_array))
        if not math.isfinite(volume) or volume <= 1.0e-9:
            volume = float(np.prod(extents)) * 0.25
        mass = min(1.5, max(0.05, volume * density_kg_m3))
        result.append(
            ClutterAsset(
                asset_id=asset_id,
                scale=annotation.scale,
                mass_kg=mass,
                stable_poses=stable_poses_from_mesh(
                    asset.collision_mesh,
                    annotation.scale,
                    max_candidates=maximum_stable_poses,
                ),
                is_large=maximum_extent >= large_extent_threshold_m,
            )
        )
    return tuple(result)
