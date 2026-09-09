#!/usr/bin/env python3
"""Stage one target-first DOMINO clutter scene for online Push Anything C3+.

The target keeps its conservative safe sampling mesh and protected/neutral
execution guard.  Each active clutter object uses its full supported mesh for
physics and as an unsafe EE surface, so the C3 planner and high-rate shield see
the same C3 obstacle geometry that Isaac publishes online.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Sequence

import numpy as np
import trimesh

from dapl.catalog import stable_poses_from_mesh
from dapl.domino import DominoDataPaths
from dapl.scene import load_scene_manifest, write_scene_manifest


TABLE_HEIGHT_M = -0.029
TARGET_C3_NAME = "DOMINO_020_hammer_safe"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="optional one-scene Isaac manifest matching the staged C3 goal",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--domino-root", type=Path, required=True)
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=repo_root / "data/push_anything_semantics/020_hammer_0",
    )
    parser.add_argument(
        "--generator-python",
        type=Path,
        default=Path("/data1/linsixu/miniconda3/envs/domino/bin/python"),
    )
    parser.add_argument(
        "--ruamel-root",
        type=Path,
        default=Path("/data1/linsixu/.local/share/push-anything-python"),
    )
    parser.add_argument(
        "--support-pose-indices",
        type=int,
        nargs="+",
        required=True,
        help="target-first stable-pose index for every staged object",
    )
    parser.add_argument("--sampling-seed", type=int, default=17)
    parser.add_argument(
        "--num-additional-samples-reposition", type=int, default=4
    )
    parser.add_argument(
        "--num-additional-samples-c3",
        type=int,
        default=5,
        help=(
            "fresh target-surface candidates evaluated at each contact-rich "
            "planning cycle; increase this to test sampling coverage without "
            "changing the controller or safety contract"
        ),
    )
    parser.add_argument(
        "--maximum-clutter-faces",
        type=int,
        default=8000,
        help="pre-simplify clutter below upstream's version-sensitive 10k cutoff",
    )
    parser.add_argument("--realtime-rate", type=float, default=0.5)
    parser.add_argument("--semantic-guard-clearance", type=float, default=0.025)
    parser.add_argument("--semantic-guard-stop-distance", type=float, default=0.055)
    parser.add_argument("--position-success-threshold", type=float, default=0.015)
    parser.add_argument("--orientation-success-threshold", type=float, default=0.075)
    parser.add_argument(
        "--quaternion-cost-weight",
        type=float,
        default=1000.0,
        help=(
            "C3+ state-dependent quaternion cost; 1000 is the verified "
            "stable value for the staged DOMINO hammer scene"
        ),
    )
    parser.add_argument("--neutral-yaw-contact-max-moment-arm-m", type=float)
    parser.add_argument("--neutral-yaw-contact-activation-rad", type=float)
    parser.add_argument(
        "--pose-effect-max-planar-regression-m", type=float, default=0.010
    )
    parser.add_argument(
        "--pose-effect-max-rotation-regression-rad", type=float, default=0.020
    )
    parser.add_argument(
        "--pose-effect-minimum-normalized-progress",
        type=float,
        default=0.05,
        help=(
            "joint normalized XY/yaw progress required from a candidate; "
            "0.05 rejects the observed yaw-only false-progress deadlock. "
            "Zero is retained only as an explicit diagnostic ablation"
        ),
    )
    parser.add_argument(
        "--pose-effect-max-horizon-rotation-error-rad",
        type=float,
        default=0.075,
    )
    parser.add_argument(
        "--pose-effect-rotation-velocity-lookahead-s",
        type=float,
        default=0.200,
    )
    parser.add_argument(
        "--pose-effect-yaw-change-scale",
        type=float,
        default=15.0,
        help=(
            "robust scale applied to C3's predicted signed-yaw change; the "
            "default covers the measured Drake-to-Isaac contact-effect "
            "underprediction while remaining directly calibratable online"
        ),
    )
    parser.add_argument(
        "--contact-translation-position-effect-scale",
        type=float,
        default=0.30,
        help=(
            "quasi-static fraction of an action-conditioned pusher motion "
            "assigned to target translation for candidate ranking"
        ),
    )
    parser.add_argument(
        "--contact-translation-yaw-wrench-gain-rad-per-m2",
        type=float,
        default=35.0,
        help=(
            "quasi-static yaw gain multiplying cross(contact offset, "
            "commanded pusher motion); calibratable from observed short pushes"
        ),
    )
    parser.add_argument(
        "--target-com-offset-c3-xy-m",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "target COM offset in the supported C3 link frame; when omitted, "
            "estimate it from the staged physical mesh's uniform-density "
            "mass properties"
        ),
    )
    parser.add_argument(
        "--contact-translation-effect-uncertainty-scale",
        type=float,
        default=1.50,
        help=(
            "multiplicative under/over-response interval used by the robust "
            "yaw guard for translation contacts"
        ),
    )
    parser.add_argument(
        "--torque-stratified-oversample-factor",
        type=int,
        default=4,
        help=(
            "draw this many safe-surface points per retained candidate, then "
            "keep a balanced near-zero/positive/negative moment-arm set "
            "before C3; 1 disables stratification without changing C3 count"
        ),
    )
    parser.add_argument(
        "--contact-twist-proposal-yaw-scale",
        type=float,
        default=2.0,
        help=(
            "yaw component scale for the three rigid-body contact-twist "
            "action proposals; positive values keep the feature enabled"
        ),
    )
    parser.add_argument(
        "--contact-twist-proposal-preload-distance-m",
        type=float,
        default=0.004,
        help=(
            "bounded inward contact displacement used with yaw-recovery "
            "twists to maintain normal force; set to zero for ablation"
        ),
    )
    parser.add_argument(
        "--contact-reposition-capture-distance-m",
        type=float,
        default=0.006,
        help=(
            "3D distance from a sampled contact at which C3 execution may "
            "begin; a tight value preserves the planned contact moment arm"
        ),
    )
    parser.add_argument(
        "--contact-twist-proposal-ee-tracking-scale",
        type=float,
        default=1000.0,
        help=(
            "temporary EE state-cost multiplier used to condition each C3 "
            "contact-twist proposal; final ranking keeps the base task cost"
        ),
    )
    parser.add_argument(
        "--controller-dynamics-mode",
        choices=("push-anything-default", "manifest-matched"),
        default="push-anything-default",
        help=(
            "planner-side object inertia; keep Push Anything's normalized "
            "model by default because its C3 costs are tuned for that scale. "
            "manifest-matched is retained only as an explicit ablation"
        ),
    )
    parser.add_argument(
        "--pose-cost-switching-distance-m",
        type=float,
        default=0.30,
        help=(
            "target XY distance below which C3+ uses full-pose rather than "
            "position-only cost; 0.30 m keeps pose tracking active throughout "
            "the repository's supported task range"
        ),
    )
    parser.add_argument("--output-scene-spec", type=Path, required=True)
    return parser.parse_args()


def replace_yaml_line(path: Path, key: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(?!\s*#)\s*{re.escape(key)}\s*:.*$", re.MULTILINE)
    replacement = f"{key}: {value}"
    if pattern.search(text):
        text = pattern.sub(replacement, text, count=1)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += replacement + "\n"
    path.write_text(text, encoding="utf-8")


def remove_yaml_key(path: Path, key: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(?!\s*#)\s*{re.escape(key)}\s*:.*\n?", re.MULTILINE)
    path.write_text(pattern.sub("", text), encoding="utf-8")


def quaternion_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def quaternion_multiply_wxyz(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, ...]:
    w1, x1, y1, z1 = (float(value) for value in left)
    w2, x2, y2, z2 = (float(value) for value in right)
    result = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1.0e-12:
        raise ValueError("quaternion product is not normalizable")
    return tuple(value / norm for value in result)


def remove_support_quaternion(
    world_quaternion: Sequence[float], support_quaternion: Sequence[float]
) -> tuple[float, ...]:
    w, x, y, z = (float(value) for value in support_quaternion)
    planar = quaternion_multiply_wxyz(world_quaternion, (w, -x, -y, -z))
    # The staged mesh already contains roll/pitch.  A valid same-support scene
    # therefore leaves only a world-Z yaw in the C3 object state.
    if math.hypot(planar[1], planar[2]) > 2.0e-3:
        raise ValueError(
            "manifest pose does not match the requested stable support pose")
    planar = (planar[0], 0.0, 0.0, planar[3])
    norm = math.hypot(planar[0], planar[3])
    return tuple(value / norm for value in planar)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            # trimesh<4 exposes the same transformed concatenation through
            # Scene.dump; DOMINO generation currently uses that older build.
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path} is not a usable triangle mesh")
    return mesh


def vertex_cluster_simplify(
    mesh: trimesh.Trimesh, maximum_faces: int
) -> trimesh.Trimesh:
    """Deterministically coarsen a mesh without an optional native backend."""
    working = mesh.copy()
    working.merge_vertices()
    vertices = np.asarray(working.vertices, dtype=np.float64)
    faces = np.asarray(working.faces, dtype=np.int64)
    origin = vertices.min(axis=0)
    maximum_pitch = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if maximum_pitch <= 0.0:
        raise RuntimeError("cannot simplify a zero-extent mesh")

    def cluster(pitch: float) -> trimesh.Trimesh:
        keys = np.floor((vertices - origin) / pitch + 0.5).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        clustered_vertices = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(clustered_vertices, inverse, vertices)
        clustered_vertices /= counts[:, None]
        clustered_faces = inverse[faces]
        nondegenerate = (
            (clustered_faces[:, 0] != clustered_faces[:, 1])
            & (clustered_faces[:, 1] != clustered_faces[:, 2])
            & (clustered_faces[:, 2] != clustered_faces[:, 0])
        )
        clustered_faces = clustered_faces[nondegenerate]
        if len(clustered_faces):
            canonical_faces = np.sort(clustered_faces, axis=1)
            _, unique_indices = np.unique(
                canonical_faces, axis=0, return_index=True
            )
            clustered_faces = clustered_faces[np.sort(unique_indices)]
        simplified = trimesh.Trimesh(
            vertices=clustered_vertices,
            faces=clustered_faces,
            process=False,
        )
        simplified.remove_unreferenced_vertices()
        return simplified

    lower_pitch = 0.0
    upper_pitch = maximum_pitch
    best: trimesh.Trimesh | None = None
    for _ in range(48):
        pitch = 0.5 * (lower_pitch + upper_pitch)
        candidate = cluster(pitch)
        if len(candidate.faces) > maximum_faces:
            lower_pitch = pitch
        else:
            upper_pitch = pitch
            if len(candidate.faces) > 0:
                best = candidate
    if best is None or len(best.faces) == 0:
        raise RuntimeError("deterministic vertex-cluster simplification failed")
    return best


def export_supported_mesh(
    source: Path,
    scale: Sequence[float],
    support_quaternion: Sequence[float],
    destination: Path,
    maximum_faces: int,
) -> tuple[float, float]:
    mesh = load_mesh(source)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices *= np.asarray(tuple(float(value) for value in scale))[None, :]
    vertices = vertices @ quaternion_matrix_wxyz(support_quaternion).T
    doubled_area = np.linalg.norm(
        np.cross(
            vertices[np.asarray(mesh.faces)[:, 1]] - vertices[np.asarray(mesh.faces)[:, 0]],
            vertices[np.asarray(mesh.faces)[:, 2]] - vertices[np.asarray(mesh.faces)[:, 0]],
        ),
        axis=1,
    )
    valid = np.isfinite(doubled_area) & (doubled_area > 1.0e-12)
    supported = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64)[valid],
        process=False,
    )
    supported.remove_unreferenced_vertices()
    if len(supported.faces) > maximum_faces:
        try:
            supported = supported.simplify_quadric_decimation(maximum_faces)
        except ModuleNotFoundError as error:
            if error.name != "fast_simplification":
                raise
            supported = vertex_cluster_simplify(supported, maximum_faces)
    if not isinstance(supported, trimesh.Trimesh) or len(supported.faces) == 0:
        raise RuntimeError(f"mesh simplification failed for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    supported.export(destination)
    return float(vertices[:, 2].min()), float(vertices[:, 2].max())


def align_controller_inertial_parameters(
    controller_sdf: Path,
    simulation_sdf: Path,
    mass_kg: float,
) -> dict[str, object]:
    """Replace Push Anything's generic 1 kg inertia with scene physics."""
    if mass_kg <= 0.0 or not math.isfinite(mass_kg):
        raise ValueError("controller object mass must be positive and finite")
    controller_text = controller_sdf.read_text(encoding="utf-8")
    simulation_text = simulation_sdf.read_text(encoding="utf-8")
    inertial_pattern = re.compile(r"<inertial>.*?</inertial>", re.DOTALL)
    controller_match = inertial_pattern.search(controller_text)
    simulation_match = inertial_pattern.search(simulation_text)
    if controller_match is None or simulation_match is None:
        raise RuntimeError("generated SDF is missing an inertial block")
    controller_inertial = controller_match.group(0)
    simulation_inertial = simulation_match.group(0)

    def value(block: str, name: str) -> float:
        match = re.search(rf"<{name}>\s*([^<]+?)\s*</{name}>", block)
        if match is None:
            raise RuntimeError(f"generated SDF is missing inertial {name}")
        return float(match.group(1))

    def replace_value(block: str, name: str, replacement: float) -> str:
        pattern = re.compile(rf"(<{name}>)[^<]+(</{name}>)")
        updated, count = pattern.subn(
            rf"\g<1>{replacement:.12g}\g<2>", block, count=1
        )
        if count != 1:
            raise RuntimeError(f"generated SDF is missing inertial {name}")
        return updated

    reference_mass = value(simulation_inertial, "mass")
    if reference_mass <= 0.0:
        raise RuntimeError("generated simulation SDF has invalid mass")
    scale = mass_kg / reference_mass
    controller_inertial = replace_value(controller_inertial, "mass", mass_kg)
    inertia_values: dict[str, float] = {}
    for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        scaled_value = value(simulation_inertial, name) * scale
        controller_inertial = replace_value(
            controller_inertial, name, scaled_value
        )
        inertia_values[name] = scaled_value
    controller_text = (
        controller_text[: controller_match.start()]
        + controller_inertial
        + controller_text[controller_match.end() :]
    )
    controller_sdf.write_text(controller_text, encoding="utf-8")
    return {
        "mass_kg": mass_kg,
        "inertia_kg_m2": inertia_values,
        "reference_sdf_mass_kg": reference_mass,
        "source": "manifest_mass_and_supported_mesh_uniform_density_inertia",
    }


def read_controller_inertial_parameters(controller_sdf: Path) -> dict[str, object]:
    """Record the generated planner inertia without changing it."""
    text = controller_sdf.read_text(encoding="utf-8")
    match = re.search(r"<inertial>.*?</inertial>", text, re.DOTALL)
    if match is None:
        raise RuntimeError("generated controller SDF is missing an inertial block")
    block = match.group(0)

    def value(name: str) -> float:
        item = re.search(rf"<{name}>\s*([^<]+?)\s*</{name}>", block)
        if item is None:
            raise RuntimeError(f"generated controller SDF is missing inertial {name}")
        return float(item.group(1))

    return {
        "mass_kg": value("mass"),
        "inertia_kg_m2": {
            name: value(name) for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        },
        "source": "push_anything_normalized_controller_default",
    }


def yaml_list(values: object) -> str:
    return json.dumps(values, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    if any(index < 0 for index in args.support_pose_indices):
        raise ValueError("support pose indices must be non-negative")
    if not 4 <= args.maximum_clutter_faces < 10000:
        raise ValueError("maximum clutter faces must be in [4, 10000)")
    if not 0.0 < args.realtime_rate <= 1.0:
        raise ValueError("realtime rate must be in (0, 1]")
    if args.semantic_guard_clearance <= 0.0:
        raise ValueError("semantic guard clearance must be positive")
    if args.semantic_guard_stop_distance < args.semantic_guard_clearance:
        raise ValueError("semantic guard stop distance must cover planner clearance")
    if args.pose_cost_switching_distance_m <= 0.0:
        raise ValueError("pose cost switching distance must be positive")
    if args.quaternion_cost_weight <= 0.0:
        raise ValueError("quaternion cost weight must be positive")
    legacy_neutral_values = (
        args.neutral_yaw_contact_max_moment_arm_m,
        args.neutral_yaw_contact_activation_rad,
    )
    if (legacy_neutral_values[0] is None) != (legacy_neutral_values[1] is None):
        raise ValueError("both legacy neutral-yaw arguments must be set together")
    if any(value is not None and value <= 0.0 for value in legacy_neutral_values):
        raise ValueError("legacy neutral-yaw arguments must be positive")
    if args.pose_effect_max_planar_regression_m < 0.0:
        raise ValueError("pose-effect planar regression must be non-negative")
    if args.pose_effect_max_rotation_regression_rad < 0.0:
        raise ValueError("pose-effect rotation regression must be non-negative")
    if args.pose_effect_minimum_normalized_progress < 0.0:
        raise ValueError("pose-effect minimum progress must be non-negative")
    if args.pose_effect_max_horizon_rotation_error_rad <= 0.0:
        raise ValueError("pose-effect horizon rotation error must be positive")
    if args.pose_effect_rotation_velocity_lookahead_s < 0.0:
        raise ValueError("pose-effect rotation lookahead must be non-negative")
    if args.pose_effect_yaw_change_scale < 1.0:
        raise ValueError("pose-effect yaw change scale must be at least one")
    if not 0.0 < args.contact_translation_position_effect_scale <= 1.0:
        raise ValueError(
            "contact translation position-effect scale must be in (0, 1]"
        )
    if args.contact_translation_yaw_wrench_gain_rad_per_m2 <= 0.0:
        raise ValueError("contact translation yaw-wrench gain must be positive")
    if args.target_com_offset_c3_xy_m is not None and not all(
        math.isfinite(value) for value in args.target_com_offset_c3_xy_m
    ):
        raise ValueError("target COM offset must contain two finite values")
    if args.contact_translation_effect_uncertainty_scale < 1.0:
        raise ValueError(
            "contact translation effect uncertainty scale must be at least one"
        )
    if args.torque_stratified_oversample_factor < 1:
        raise ValueError("torque-stratified oversample factor must be at least one")
    if args.contact_twist_proposal_yaw_scale <= 0.0:
        raise ValueError("contact-twist proposal yaw scale must be positive")
    if args.contact_twist_proposal_preload_distance_m < 0.0:
        raise ValueError("contact-twist proposal preload must be non-negative")
    if not 0.0 < args.contact_reposition_capture_distance_m <= 0.05:
        raise ValueError(
            "contact reposition capture distance must be in (0, 0.05]"
        )
    if args.contact_twist_proposal_ee_tracking_scale < 1.0:
        raise ValueError("contact-twist EE tracking scale must be at least one")
    if min(
        args.num_additional_samples_reposition,
        args.num_additional_samples_c3,
    ) <= 0:
        raise ValueError("additional sample counts must be positive")

    manifest = args.manifest.expanduser().resolve()
    runtime = args.runtime_root.expanduser().resolve()
    params = runtime / "examples/sampling_c3/anything/parameters"
    rewrite = runtime / "examples/sampling_c3/multiyaml_rewrite.py"
    if not manifest.is_file() or not params.is_dir() or not rewrite.is_file():
        raise FileNotFoundError("manifest or Push Anything runtime template is missing")
    scenes = tuple(load_scene_manifest(manifest))
    if not scenes:
        raise ValueError("manifest contains no scenes")
    if not 0 <= args.scene_index < len(scenes):
        raise ValueError("scene index is outside the manifest")
    active_count = len(args.support_pose_indices)
    first = scenes[args.scene_index]
    if active_count > len(first.objects):
        raise ValueError("support pose list exceeds manifest object count")
    expected_asset_ids = [item.asset_id for item in first.objects[:active_count]]
    for scene in scenes:
        actual = [item.asset_id for item in scene.objects[:active_count]]
        if actual != expected_asset_ids:
            raise ValueError("all manifest scenes must keep the staged object order")
    if first.target_object.asset_id != "020_hammer:0":
        raise ValueError("the staged semantic target must be DOMINO 020_hammer:0")
    execution_manifest = manifest
    if args.output_manifest is not None:
        execution_manifest = args.output_manifest.expanduser().resolve()
        execution_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_scene_manifest(execution_manifest, (first,))
    elif args.scene_index != 0:
        raise ValueError("non-zero scene index requires --output-manifest")

    semantic_manifest_path = args.semantic_dir / "semantic_mesh_manifest.json"
    semantic_manifest = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
    paths = DominoDataPaths.resolve(args.domino_root)
    staged_objects = []
    sampling_meshes = []
    unsafe_meshes = []
    z_extents = []
    target_com_offset_c3_xy: list[float] | None = None
    target_com_offset_source: str | None = None
    for object_index, (item, support_index) in enumerate(
        zip(first.objects[:active_count], args.support_pose_indices)
    ):
        source_asset = paths.require_source_asset(item.asset_id)
        supports = stable_poses_from_mesh(
            source_asset.collision_mesh,
            item.scale,
            max_candidates=support_index + 1,
        )
        if support_index >= len(supports):
            raise ValueError(f"{item.asset_id} has no stable pose {support_index}")
        support = supports[support_index]
        if object_index == 0:
            c3_name = TARGET_C3_NAME
        else:
            asset_stem = item.asset_id.split(":", 1)[0].replace("-", "_")
            c3_name = f"DOMINO_{asset_stem}_full"
        asset_dir = runtime / "examples/sampling_c3/urdf" / c3_name
        full_name = f"{c3_name}.obj"
        full_path = asset_dir / full_name
        if object_index == 0:
            asset_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                args.semantic_dir / semantic_manifest["physical_mesh"], full_path
            )
            safe_name = f"{c3_name}_safe_guarded.obj"
            unsafe_name = f"{c3_name}_unsafe.obj"
            shutil.copyfile(
                args.semantic_dir / semantic_manifest["meshes"]["safe_guarded"],
                asset_dir / safe_name,
            )
            shutil.copyfile(
                args.semantic_dir / semantic_manifest["meshes"]["unsafe"],
                asset_dir / unsafe_name,
            )
            sampling_path = asset_dir / safe_name
            unsafe_path = asset_dir / unsafe_name
            full_mesh = load_mesh(full_path)
            z_min = float(np.asarray(full_mesh.vertices)[:, 2].min())
            z_max = float(np.asarray(full_mesh.vertices)[:, 2].max())
            if args.target_com_offset_c3_xy_m is None:
                mesh_com = np.asarray(full_mesh.center_mass, dtype=np.float64)
                if mesh_com.shape != (3,) or not np.all(np.isfinite(mesh_com)):
                    mesh_com = np.asarray(full_mesh.centroid, dtype=np.float64)
                target_com_offset_c3_xy = [
                    float(mesh_com[0]), float(mesh_com[1])
                ]
                target_com_offset_source = "uniform_density_supported_mesh"
            else:
                target_com_offset_c3_xy = [
                    float(value) for value in args.target_com_offset_c3_xy_m
                ]
                target_com_offset_source = "explicit_asset_physics_metadata"
        else:
            z_min, z_max = export_supported_mesh(
                source_asset.collision_mesh,
                item.scale,
                support.quaternion,
                full_path,
                args.maximum_clutter_faces,
            )
            # Keep the full clutter mesh in C3 dynamics and collision checking.
            # ``sampleable_objects`` below prevents it from becoming a robot
            # contact target even if it drifts away from its nominal pose.
            # A later C2 planner patch can add target-protected/object pairwise
            # clearance without changing this state contract.
            sampling_path = full_path
            unsafe_path = full_path
        relative = lambda path: path.relative_to(runtime).as_posix()
        sampling_meshes.append(relative(sampling_path))
        unsafe_meshes.append(relative(unsafe_path))
        z_extents.append((z_min, z_max))
        staged_objects.append(
            {
                "role": "target" if object_index == 0 else "clutter",
                "asset_id": item.asset_id,
                "c3_body_name": c3_name,
                "state_channel": f"OBJECT_{c3_name}_STATE_SIMULATION",
                "support_pose_index": support_index,
                "support_quaternion_wxyz": list(support.quaternion),
                **(
                    {
                        "center_of_mass_offset_c3_xy_m": (
                            target_com_offset_c3_xy
                        ),
                        "center_of_mass_offset_source": (
                            target_com_offset_source
                        ),
                    }
                    if object_index == 0
                    else {}
                ),
                **(
                    {}
                    if object_index == 0
                    else {"isaac_obstacle_index": object_index - 1}
                ),
            }
        )

    controller = params / "sampling_c3_controller_params.yaml"
    replace_yaml_line(
        controller,
        "base_names",
        yaml_list([item["c3_body_name"] for item in staged_objects]),
    )
    replace_yaml_line(controller, "sampling_meshes", yaml_list(sampling_meshes))
    replace_yaml_line(
        controller,
        "sampleable_objects",
        yaml_list([int(index == 0) for index in range(len(staged_objects))]),
    )
    replace_yaml_line(controller, "unsafe_meshes", yaml_list(unsafe_meshes))
    replace_yaml_line(
        controller,
        "semantic_guard_clearance",
        f"{args.semantic_guard_clearance:.12g}",
    )
    replace_yaml_line(
        controller,
        "semantic_guard_stop_distance",
        f"{args.semantic_guard_stop_distance:.12g}",
    )
    if args.neutral_yaw_contact_max_moment_arm_m is None:
        remove_yaml_key(controller, "neutral_yaw_contact_max_moment_arm")
        remove_yaml_key(controller, "neutral_yaw_contact_activation_threshold")
    else:
        replace_yaml_line(
            controller,
            "neutral_yaw_contact_max_moment_arm",
            f"{args.neutral_yaw_contact_max_moment_arm_m:.12g}",
        )
        replace_yaml_line(
            controller,
            "neutral_yaw_contact_activation_threshold",
            f"{args.neutral_yaw_contact_activation_rad:.12g}",
        )
    replace_yaml_line(
        controller,
        "pose_effect_max_planar_regression",
        f"{args.pose_effect_max_planar_regression_m:.12g}",
    )
    replace_yaml_line(
        controller,
        "pose_effect_max_rotation_regression",
        f"{args.pose_effect_max_rotation_regression_rad:.12g}",
    )
    replace_yaml_line(
        controller,
        "pose_effect_minimum_normalized_progress",
        f"{args.pose_effect_minimum_normalized_progress:.12g}",
    )
    replace_yaml_line(
        controller,
        "pose_effect_max_horizon_rotation_error",
        f"{args.pose_effect_max_horizon_rotation_error_rad:.12g}",
    )
    replace_yaml_line(
        controller,
        "pose_effect_rotation_velocity_lookahead_s",
        f"{args.pose_effect_rotation_velocity_lookahead_s:.12g}",
    )
    replace_yaml_line(
        controller,
        "pose_effect_yaw_change_scale",
        f"{args.pose_effect_yaw_change_scale:.12g}",
    )
    replace_yaml_line(
        controller,
        "contact_translation_position_effect_scale",
        f"{args.contact_translation_position_effect_scale:.12g}",
    )
    replace_yaml_line(
        controller,
        "contact_translation_yaw_wrench_gain_rad_per_m2",
        f"{args.contact_translation_yaw_wrench_gain_rad_per_m2:.12g}",
    )
    if target_com_offset_c3_xy is None:
        raise RuntimeError("target COM offset was not resolved")
    replace_yaml_line(
        controller,
        "contact_translation_com_offset_xy",
        yaml_list(target_com_offset_c3_xy),
    )
    replace_yaml_line(
        controller,
        "contact_translation_effect_uncertainty_scale",
        f"{args.contact_translation_effect_uncertainty_scale:.12g}",
    )
    replace_yaml_line(
        controller,
        "torque_stratified_oversample_factor",
        str(args.torque_stratified_oversample_factor),
    )
    replace_yaml_line(
        controller,
        "contact_twist_proposal_yaw_scale",
        f"{args.contact_twist_proposal_yaw_scale:.12g}",
    )
    replace_yaml_line(
        controller,
        "contact_twist_proposal_preload_distance",
        f"{args.contact_twist_proposal_preload_distance_m:.12g}",
    )
    replace_yaml_line(
        controller,
        "contact_reposition_capture_distance",
        f"{args.contact_reposition_capture_distance_m:.12g}",
    )
    replace_yaml_line(
        controller,
        "contact_twist_proposal_ee_tracking_scale",
        f"{args.contact_twist_proposal_ee_tracking_scale:.12g}",
    )

    environment = os.environ.copy()
    python_paths = [str(runtime)]
    if args.ruamel_root.is_dir():
        python_paths.append(str(args.ruamel_root.resolve()))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(
        [str(args.generator_python.resolve()), str(rewrite), "--recreate-sdf"],
        cwd=runtime,
        env=environment,
        check=True,
    )
    planner_dynamics = []
    for staged, item in zip(staged_objects, first.objects[:active_count]):
        asset_dir = runtime / "examples/sampling_c3/urdf" / staged["c3_body_name"]
        controller_sdf = asset_dir / f"{staged['c3_body_name']}_controller.sdf"
        if args.controller_dynamics_mode == "manifest-matched":
            planner_dynamics.append(
                align_controller_inertial_parameters(
                    controller_sdf,
                    asset_dir / f"{staged['c3_body_name']}.sdf",
                    item.mass_kg,
                )
            )
        else:
            planner_dynamics.append(
                read_controller_inertial_parameters(controller_sdf)
            )

    initial_poses = [first.tasks[0].initial_pose]
    initial_poses.extend(item.pose for item in first.objects[1:active_count])
    goal_poses = [first.tasks[0].goal_pose, *initial_poses[1:]]
    planner_initial = []
    planner_goals = []
    for initial, goal, staged in zip(initial_poses, goal_poses, staged_objects):
        support_quaternion = staged["support_quaternion_wxyz"]
        initial_quaternion = remove_support_quaternion(
            initial[3:7], support_quaternion
        )
        goal_quaternion = remove_support_quaternion(goal[3:7], support_quaternion)
        planner_initial.append(
            [*initial_quaternion, initial[0], initial[1], initial[2] + TABLE_HEIGHT_M]
        )
        planner_goals.append(
            [goal[0], goal[1], goal[2] + TABLE_HEIGHT_M, *goal_quaternion]
        )

    sim = params / "sim_params.yaml"
    goal = params / "goal_params.yaml"
    sampling = params / "sampling_params.yaml"
    reposition = params / "reposition_params.yaml"
    progress = params / "progress_params_c3plus.yaml"
    c3_options = params / "sampling_c3plus_options.yaml"
    replace_yaml_line(sim, "q_init_objects", yaml_list(planner_initial))
    replace_yaml_line(sim, "realtime_rate", f"{args.realtime_rate:.12g}")
    replace_yaml_line(
        goal,
        "resting_object_heights",
        yaml_list([pose[6] for pose in planner_initial]),
    )
    replace_yaml_line(
        goal,
        "fixed_target_positions",
        yaml_list([pose[:3] for pose in planner_goals]),
    )
    replace_yaml_line(
        goal,
        "fixed_target_orientations",
        yaml_list([pose[3:7] for pose in planner_goals]),
    )
    replace_yaml_line(
        goal,
        "position_success_threshold",
        f"{args.position_success_threshold:.12g}",
    )
    replace_yaml_line(
        goal,
        "orientation_success_threshold",
        f"{args.orientation_success_threshold:.12g}",
    )
    target_height = z_extents[0][1] - z_extents[0][0]
    contact_height = max(0.002, TABLE_HEIGHT_M + 0.5 * target_height + 0.010)
    reposition_height = max(0.073, TABLE_HEIGHT_M + target_height + 0.050)
    replace_yaml_line(sampling, "z_height", f"{contact_height:.12g}")
    replace_yaml_line(sampling, "random_seed", str(args.sampling_seed))
    replace_yaml_line(
        sampling,
        "num_additional_samples_repos",
        str(args.num_additional_samples_reposition),
    )
    replace_yaml_line(
        sampling,
        "num_additional_samples_c3",
        str(args.num_additional_samples_c3),
    )
    for key in (
        "terminal_orientation_rescore_weight",
        "two_contact_lookahead_weight",
        "two_contact_lookahead_beam_width",
        "two_contact_lookahead_samples",
        "goal_conditioned_mesh_normal_fraction",
    ):
        remove_yaml_key(sampling, key)
    replace_yaml_line(reposition, "pwl_waypoint_height", f"{reposition_height:.12g}")
    replace_yaml_line(c3_options, "use_predicted_x0_repos", "false")
    replace_yaml_line(c3_options, "use_quaternion_dependent_cost", "true")
    replace_yaml_line(
        c3_options,
        "q_quaternion_dependent_weight",
        f"{args.quaternion_cost_weight:.12g}",
    )
    # Upstream compares the sum of every object's XY error with
    # ``threshold * num_objects``.  In this target-first contract the clutter
    # goals are deliberately their initial poses, so their errors are zero and
    # an unmodified threshold would be loosened by the number of static
    # obstacles (5 cm becomes 10 cm with one blocker).  Normalize the upstream
    # value so the CLI remains the actual target-object switching distance.
    upstream_pose_cost_switching_distance = (
        args.pose_cost_switching_distance_m / active_count
    )
    replace_yaml_line(
        progress,
        "cost_switching_threshold_distance",
        f"{upstream_pose_cost_switching_distance:.12g}",
    )
    replace_yaml_line(progress, "progress_enforced_cost_drop", "0.5")
    replace_yaml_line(progress, "progress_enforced_over_n_loops", "35")

    output = args.output_scene_spec.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene_spec = {
        "schema": "nonprehensile.c3_online_scene.v1",
        "source_manifest": str(execution_manifest),
        "source_manifest_set": str(manifest),
        "source_scene_index": args.scene_index,
        "source_scene_id": first.scene_id,
        "runtime_root": str(runtime),
        "table_height_offset_m": -TABLE_HEIGHT_M,
        "objects": staged_objects,
        "planner_initial_qwxyz_xyz": planner_initial,
        "planner_goal_xyz_qwxyz": planner_goals,
        "controller": {
            "target_pose_cost_switching_distance_m": (
                args.pose_cost_switching_distance_m
            ),
            "upstream_mean_pose_cost_switching_distance_m": (
                upstream_pose_cost_switching_distance
            ),
            "position_success_threshold_m": args.position_success_threshold,
            "orientation_success_threshold_rad": args.orientation_success_threshold,
            "quaternion_cost_weight": args.quaternion_cost_weight,
            "num_additional_samples_reposition": (
                args.num_additional_samples_reposition
            ),
            "num_additional_samples_c3": args.num_additional_samples_c3,
            "legacy_neutral_yaw_contact_max_moment_arm_m": (
                args.neutral_yaw_contact_max_moment_arm_m
            ),
            "legacy_neutral_yaw_contact_activation_rad": (
                args.neutral_yaw_contact_activation_rad
            ),
            "pose_effect_max_planar_regression_m": (
                args.pose_effect_max_planar_regression_m
            ),
            "pose_effect_max_rotation_regression_rad": (
                args.pose_effect_max_rotation_regression_rad
            ),
            "pose_effect_minimum_normalized_progress": (
                args.pose_effect_minimum_normalized_progress
            ),
            "pose_effect_max_horizon_rotation_error_rad": (
                args.pose_effect_max_horizon_rotation_error_rad
            ),
            "pose_effect_rotation_velocity_lookahead_s": (
                args.pose_effect_rotation_velocity_lookahead_s
            ),
            "pose_effect_yaw_change_scale": args.pose_effect_yaw_change_scale,
            "contact_translation_position_effect_scale": (
                args.contact_translation_position_effect_scale
            ),
            "contact_translation_yaw_wrench_gain_rad_per_m2": (
                args.contact_translation_yaw_wrench_gain_rad_per_m2
            ),
            "target_com_offset_c3_xy_m": target_com_offset_c3_xy,
            "target_com_offset_source": target_com_offset_source,
            "contact_translation_effect_uncertainty_scale": (
                args.contact_translation_effect_uncertainty_scale
            ),
            "torque_stratified_oversample_factor": (
                args.torque_stratified_oversample_factor
            ),
            "contact_twist_proposal_yaw_scale": (
                args.contact_twist_proposal_yaw_scale
            ),
            "contact_twist_proposal_preload_distance_m": (
                args.contact_twist_proposal_preload_distance_m
            ),
            "contact_reposition_capture_distance_m": (
                args.contact_reposition_capture_distance_m
            ),
            "contact_twist_proposal_ee_tracking_scale": (
                args.contact_twist_proposal_ee_tracking_scale
            ),
            "semantic_guard_clearance_m": args.semantic_guard_clearance,
            "semantic_guard_stop_distance_m": args.semantic_guard_stop_distance,
            "controller_dynamics_mode": args.controller_dynamics_mode,
            "object_dynamics": planner_dynamics,
        },
        "semantic_contract": {
            "target_sampling": "safe_guarded_only",
            "sampleable_objects": [
                index == 0 for index in range(len(staged_objects))
            ],
            "target_unsafe": "protected_plus_neutral",
            "clutter_unsafe": "full_surface",
            "prospective_c2_pairwise_guard": False,
            "isaac_fail_closed_c1_c2_c3_audit": True,
        },
    }
    output.write_text(json.dumps(scene_spec, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(scene_spec, indent=2))


if __name__ == "__main__":
    main()
