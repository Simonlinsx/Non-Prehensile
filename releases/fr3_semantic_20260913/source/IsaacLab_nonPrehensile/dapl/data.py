"""Portable paths for the public DAPL asset dataset."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


DAPL_DATA_ROOT_ENV = "DAPL_DATA_ROOT"


def _validate_asset_id(asset_id: str) -> str:
    """Reject path-like asset identifiers before constructing local paths."""

    if not asset_id or asset_id in {".", ".."} or Path(asset_id).name != asset_id:
        raise ValueError(f"asset_id must be a non-empty basename, got {asset_id!r}")
    return asset_id


@dataclass(frozen=True)
class DAPLAssetPaths:
    """Resolved files for one Objaverse asset in ``Steve3zz/DAPL-dataset``."""

    asset_id: str
    flattened_usd: Path
    collision_mesh: Path
    visual_mesh: Path

    def missing(self) -> tuple[Path, ...]:
        """Return expected files that are absent locally."""

        return tuple(path for path in (self.flattened_usd, self.collision_mesh) if not path.is_file())


@dataclass(frozen=True)
class DAPLDataPaths:
    """Dataset layout without machine-specific paths in environment configs.

    The public dataset currently contains assets but no Clutter6D train/eval
    scene manifests.  Generated manifests should be stored outside these
    asset directories (``manifests/`` is used by convention).
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @classmethod
    def resolve(
        cls,
        root: str | os.PathLike[str] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "DAPLDataPaths":
        """Resolve an explicit root or ``DAPL_DATA_ROOT``.

        No developer-home fallback is provided: a missing configuration is
        reported immediately instead of failing later during stage creation.
        """

        if root is None:
            env = os.environ if environ is None else environ
            root = env.get(DAPL_DATA_ROOT_ENV)
        if root is None or not str(root).strip():
            raise ValueError(
                f"DAPL dataset root is unset; pass root=... or export {DAPL_DATA_ROOT_ENV}"
            )
        return cls(Path(root))

    @property
    def flattened_usds(self) -> Path:
        return self.root / "flattened_usds"

    @property
    def source_usds(self) -> Path:
        return self.root / "usds"

    @property
    def embodiments(self) -> Path:
        return self.root / "embodiments"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def hand_mesh(self) -> Path:
        return self.embodiments / "hand_merged.obj"

    @property
    def hand_points(self) -> Path:
        return self.embodiments / "pc_npy_cache" / "hand_merged.npy"

    def asset(self, asset_id: str) -> DAPLAssetPaths:
        """Construct paths using the layout published on Hugging Face."""

        asset_id = _validate_asset_id(asset_id)
        source_dir = self.source_usds / asset_id
        return DAPLAssetPaths(
            asset_id=asset_id,
            flattened_usd=self.flattened_usds / asset_id / f"_{asset_id}.usd",
            collision_mesh=source_dir / f"{asset_id}_geometry.obj",
            visual_mesh=source_dir / f"{asset_id}_geometry_wo_coacd.obj",
        )

    def require_root(self, *, require_hand: bool = False) -> None:
        """Fail with one actionable error if required dataset paths are absent."""

        required = [self.flattened_usds, self.source_usds, self.embodiments]
        if require_hand:
            required.extend((self.hand_mesh, self.hand_points))
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n  - ".join(str(path) for path in missing)
            raise FileNotFoundError(f"DAPL dataset is incomplete; missing:\n  - {formatted}")

    def require_asset(self, asset_id: str) -> DAPLAssetPaths:
        """Resolve an asset and require its simulation and collision files."""

        paths = self.asset(asset_id)
        missing = paths.missing()
        if missing:
            formatted = "\n  - ".join(str(path) for path in missing)
            raise FileNotFoundError(f"DAPL asset {asset_id!r} is incomplete; missing:\n  - {formatted}")
        return paths
