"""Isaac Lab adapters for versioned Clutter6D manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

from dapl.data import DAPLDataPaths
from dapl.domino import DominoDataPaths
from dapl.scene import ClutterScene, SceneObject


AssetResolver = Callable[[str], tuple[Path, Path]]


def resolve_dgn_asset(asset_id: str, root: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve one ``<base>-<scale>`` DGN development asset."""

    if root is None:
        root = os.environ.get("DGN_DATA_ROOT")
    if root is None:
        raise ValueError("DGN_DATA_ROOT is required for a DGN-backed clutter manifest")
    if "-" not in asset_id:
        raise ValueError(f"invalid DGN manifest asset_id: {asset_id!r}")
    base_name, scale_text = asset_id.rsplit("-", 1)
    float(scale_text)
    if not base_name or Path(base_name).name != base_name:
        raise ValueError(f"invalid DGN manifest asset_id: {asset_id!r}")
    root = Path(root).expanduser().resolve()
    usd_path = root / "coacd_usd_convexhull" / base_name / f"{base_name}.usd"
    mesh_path = root / "coacd_normalized" / f"{base_name}.obj"
    if not usd_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError(f"DGN asset {asset_id!r} is incomplete")
    return usd_path, mesh_path


def resolve_dapl_asset(asset_id: str, root: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve one public DAPL object asset."""

    paths = DAPLDataPaths.resolve(root).require_asset(asset_id)
    return paths.flattened_usd, paths.collision_mesh


def resolve_domino_asset(asset_id: str, root: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve a DOMINO source annotation/mesh and its prepared Isaac Lab USD."""

    paths = DominoDataPaths.resolve(root).require_sim_asset(asset_id)
    return paths.usd, paths.collision_mesh


def manifest_asset_resolver(
    source: str,
    *,
    root: str | Path | None = None,
) -> AssetResolver:
    """Select an explicit asset namespace for a manifest."""

    if source == "dgn":
        return lambda asset_id: resolve_dgn_asset(asset_id, root)
    if source == "dapl":
        return lambda asset_id: resolve_dapl_asset(asset_id, root)
    if source == "domino":
        return lambda asset_id: resolve_domino_asset(asset_id, root)
    raise ValueError("clutter asset source must be 'dgn', 'dapl', or 'domino'")


def _usd_cfg(scene_object: SceneObject, resolver: AssetResolver) -> sim_utils.UsdFileCfg:
    usd_path, mesh_path = resolver(scene_object.asset_id)
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        scale=scene_object.scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=scene_object.mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
    )
    # Point-cloud extraction needs the collision mesh paired with this USD.
    cfg.obj_path = str(mesh_path)
    cfg.dapl_asset_id = scene_object.asset_id
    return cfg


def build_clutter_rigid_assets(
    scenes: Sequence[ClutterScene],
    *,
    resolver: AssetResolver,
) -> tuple[RigidObjectCfg, RigidObjectCollectionCfg]:
    """Create target and obstacle configs aligned with ``env_id % scenes``."""

    scenes = tuple(scenes)
    if not scenes:
        raise ValueError("at least one clutter scene is required")
    tracks = {scene.track for scene in scenes}
    if len(tracks) != 1:
        raise ValueError("one vectorized environment cannot mix Clutter6D tracks")
    obstacle_count = len(scenes[0].obstacle_objects)
    if any(len(scene.obstacle_objects) != obstacle_count for scene in scenes):
        raise ValueError("all manifest scenes must contain the same obstacle count")

    target_assets = [_usd_cfg(scene.target_object, resolver) for scene in scenes]
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=target_assets,
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )

    obstacle_cfgs: dict[str, RigidObjectCfg] = {}
    for obstacle_index in range(obstacle_count):
        slot_assets = [
            _usd_cfg(scene.obstacle_objects[obstacle_index], resolver) for scene in scenes
        ]
        obstacle_cfgs[f"obstacle_{obstacle_index:02d}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{obstacle_index:02d}",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=slot_assets,
                random_choice=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.7 + obstacle_index * 0.05)),
        )
    return target, RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)
