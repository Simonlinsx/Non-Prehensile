"""Simulator-independent building blocks for the DAPL reproduction.

This package deliberately does not import Isaac Sim or Isaac Lab.  Dataset
tools, scene manifests, and physical scene tensor construction can therefore
be tested with a regular PyTorch installation before launching the simulator.
"""

from .data import DAPLAssetPaths, DAPLDataPaths
from .embodiment import (
    DAPL_HAND_POINT_COUNT,
    DAPL_HAND_POINTS_ENV,
    load_dapl_hand_points,
)
from .domino import (
    DOMINO_ROOT_ENV,
    DOMINO_USD_ROOT_ENV,
    DominoAffordanceAnchor,
    DominoAffordanceAnnotation,
    DominoAssetPaths,
    DominoDataPaths,
    build_domino_clutter_catalog,
    default_affordance_radius,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
    make_domino_asset_id,
    parse_domino_asset_id,
)
from .catalog import (
    DGNAssetRecord,
    build_dgn_clutter_catalog,
    load_dgn_asset_records,
    stable_poses_from_mesh,
)
from .generation import (
    ClutterAsset,
    ClutterGenerationConfig,
    StablePose,
    generate_clutter_scenes,
)
from .representation import (
    DAPLSceneTensorBuilder,
    DAPLSceneTensorConfig,
    PhysicalSceneBatch,
    PhysicalSceneTensor,
    SceneComponent,
)
from .scene import (
    ClutterScene,
    ClutterTrack,
    ManipulationTask,
    SceneObject,
    load_scene_manifest,
    write_scene_manifest,
)

__all__ = [
    "ClutterScene",
    "ClutterAsset",
    "ClutterGenerationConfig",
    "ClutterTrack",
    "DAPLAssetPaths",
    "DAPLDataPaths",
    "DAPL_HAND_POINT_COUNT",
    "DAPL_HAND_POINTS_ENV",
    "DOMINO_ROOT_ENV",
    "DOMINO_USD_ROOT_ENV",
    "DAPLSceneTensorBuilder",
    "DAPLSceneTensorConfig",
    "DGNAssetRecord",
    "DominoAffordanceAnchor",
    "DominoAffordanceAnnotation",
    "DominoAssetPaths",
    "DominoDataPaths",
    "ManipulationTask",
    "PhysicalSceneBatch",
    "PhysicalSceneTensor",
    "SceneComponent",
    "SceneObject",
    "StablePose",
    "build_dgn_clutter_catalog",
    "build_domino_clutter_catalog",
    "default_affordance_radius",
    "domino_point_affordance_features",
    "generate_clutter_scenes",
    "load_dgn_asset_records",
    "load_dapl_hand_points",
    "load_domino_affordance_annotation",
    "load_scene_manifest",
    "write_scene_manifest",
    "stable_poses_from_mesh",
    "make_domino_asset_id",
    "parse_domino_asset_id",
]
