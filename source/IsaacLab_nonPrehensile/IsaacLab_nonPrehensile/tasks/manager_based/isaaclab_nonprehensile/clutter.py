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


def _usd_cfg(
    scene_object: SceneObject,
    resolver: AssetResolver,
    *,
    enabled: bool = True,
    kinematic: bool = False,
) -> sim_utils.UsdFileCfg:
    usd_path, mesh_path = resolver(scene_object.asset_id)
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        scale=scene_object.scale,
        # Early curriculum stages retain obstacle slots for checkpoint-compatible
        # observations, but inactive objects must not exist in the rendered or
        # physical task.  Parking a dynamic collider below an infinite ground
        # plane lets PhysX depenetrate it back into the scene.
        visible=enabled,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            # A typed safety blocker may be deliberately fixed: C2 asks
            # whether the moving target's protected part hits surrounding
            # clutter, not whether an imperfect support pose lets the blocker
            # roll before the robot acts. It remains visible, collidable and
            # contact-sensed. Targets always use ``kinematic=False``.
            disable_gravity=(not enabled) or kinematic,
            kinematic_enabled=(not enabled) or kinematic,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=scene_object.mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=enabled),
        activate_contact_sensors=enabled,
    )
    # Point-cloud extraction needs the collision mesh paired with this USD.
    cfg.obj_path = str(mesh_path)
    cfg.dapl_asset_id = scene_object.asset_id
    return cfg


def inactive_obstacle_parking_pose(
    scene_object: SceneObject,
    obstacle_index: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Place an inactive slot on its support face outside the robot workspace.

    The original implementation used ``z=-10``.  An infinite ground plane can
    treat a body below the plane as deeply penetrating and eject it upward.
    Keeping the annotated support height and orientation avoids that failure
    even if an inactive slot is accidentally re-enabled later.
    """

    if obstacle_index < 0:
        raise ValueError("obstacle_index must be non-negative")
    pose = scene_object.pose
    parking_x = -1.5 + 0.4 * float(obstacle_index % 4)
    parking_y = 1.5 - 0.4 * float(obstacle_index // 4)
    return (parking_x, parking_y, pose[2], *pose[3:])


def _periodic_scene_objects(objects: Sequence[SceneObject]) -> tuple[SceneObject, ...]:
    """Return the shortest prefix that exactly repeats across environment ids.

    ``MultiAssetSpawnerCfg(random_choice=False)`` selects by environment index
    modulo the number of configs. Manifests often contain hundreds of spatial
    layouts but only one physical asset per slot; spawning hundreds of duplicate
    USD prototypes is both slow and unstable. A periodic prefix preserves the
    exact original env-to-asset mapping.
    """

    objects = tuple(objects)
    signatures = tuple(
        (
            item.asset_id,
            item.scale,
            item.mass_kg,
        )
        for item in objects
    )
    for period in range(1, len(objects) + 1):
        if len(objects) % period != 0:
            continue
        if all(signatures[index] == signatures[index % period] for index in range(len(objects))):
            return objects[:period]
    return objects


def build_clutter_rigid_assets(
    scenes: Sequence[ClutterScene],
    *,
    resolver: AssetResolver,
    active_obstacle_count: int | None = None,
    kinematic_active_obstacles: bool = False,
    activate_contact_sensors: bool = False,
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
    enabled_obstacle_count = (
        obstacle_count
        if active_obstacle_count is None
        else int(active_obstacle_count)
    )
    if not 0 <= enabled_obstacle_count <= obstacle_count:
        raise ValueError(
            "active_obstacle_count must be between zero and the manifest obstacle count"
        )

    target_objects = _periodic_scene_objects([scene.target_object for scene in scenes])
    target_assets = [_usd_cfg(item, resolver) for item in target_objects]
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=target_assets,
            random_choice=False,
            # MultiAssetSpawnerCfg otherwise overwrites the per-USD flag with
            # its own False default.  Teacher tasks need this reporter on the
            # selected target root for filtered C1/C2 PhysX auditing.
            activate_contact_sensors=activate_contact_sensors,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )

    obstacle_cfgs: dict[str, RigidObjectCfg] = {}
    for obstacle_index in range(obstacle_count):
        enabled = obstacle_index < enabled_obstacle_count
        slot_objects = _periodic_scene_objects(
            [scene.obstacle_objects[obstacle_index] for scene in scenes]
        )
        slot_assets = [
            _usd_cfg(
                item,
                resolver,
                enabled=enabled,
                kinematic=enabled and kinematic_active_obstacles,
            )
            for item in slot_objects
        ]
        parking_pose = inactive_obstacle_parking_pose(
            scenes[0].obstacle_objects[obstacle_index], obstacle_index
        )
        obstacle_cfgs[f"obstacle_{obstacle_index:02d}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{obstacle_index:02d}",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=slot_assets,
                random_choice=False,
                activate_contact_sensors=activate_contact_sensors,
            ),
            # Never create multiple dynamic obstacles in an overlapping stack
            # before the first reset event.  Active slots are moved to their
            # manifest poses on reset; inactive slots remain safely parked.
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=parking_pose[:3],
                rot=parking_pose[3:],
            ),
        )
    return target, RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)
