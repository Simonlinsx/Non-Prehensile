#!/usr/bin/env python3
"""Convert staged Push Anything scene JSON/JSONL into an Isaac Lab manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dapl.contact_planner import (
    PUSH_ANYTHING_TABLE_TO_ISAAC_Z_M,
    stage_to_isaac_scene,
)
from dapl.scene import load_scene_manifest, write_scene_manifest


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument(
        "--template-manifest",
        type=Path,
        default=repo_root / "data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl",
    )
    parser.add_argument(
        "--semantic-manifest",
        type=Path,
        default=(
            repo_root
            / "data/push_anything_semantics/020_hammer_0/semantic_mesh_manifest.json"
        ),
        help=(
            "canonical support-frame metadata; also upgrades staged manifests "
            "that predate the explicit support quaternion"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of JSONL scenes to convert; zero keeps all scenes.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based first staged scene to convert.",
    )
    parser.add_argument(
        "--table-height-offset-m",
        type=float,
        default=PUSH_ANYTHING_TABLE_TO_ISAAC_Z_M,
    )
    parser.add_argument("--target-mass-kg", type=float, default=0.05)
    parser.add_argument("--target-static-friction", type=float, default=0.3)
    parser.add_argument("--target-dynamic-friction", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_path = args.stage_manifest.expanduser().resolve()
    stage_text = stage_path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(stage_text)
    except json.JSONDecodeError:
        decoded = [
            json.loads(line)
            for line in stage_text.splitlines()
            if line.strip()
        ]
    stages = decoded if isinstance(decoded, list) else [decoded]
    if args.limit < 0 or args.start_index < 0:
        raise ValueError("limit and start index must be non-negative")
    stages = stages[args.start_index :]
    if args.limit:
        stages = stages[: args.limit]
    if not stages:
        raise ValueError("stage manifest is empty")
    if args.scene_id is not None and len(stages) != 1:
        raise ValueError("--scene-id is valid only for a single staged scene")
    templates = list(load_scene_manifest(args.template_manifest.expanduser().resolve()))
    if not templates:
        raise ValueError("template manifest is empty")
    semantic_manifest = json.loads(
        args.semantic_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    if semantic_manifest.get("asset_id") != "020_hammer:0":
        raise ValueError("semantic manifest must describe DOMINO hammer 020")
    support_quaternion = semantic_manifest.get("support_quaternion_wxyz")
    if not isinstance(support_quaternion, list) or len(support_quaternion) != 4:
        raise ValueError("semantic manifest is missing its support quaternion")
    scenes = []
    for index, stage in enumerate(stages):
        source_scene_id = str(stage.get("scene_id", f"scene{index:03d}"))
        scene_id = args.scene_id or f"push-anything-{source_scene_id}"
        template = templates[index % len(templates)]
        if stage.get("schema") in (
            "nonprehensile.push_anything_c1_scene.v1",
            "nonprehensile.push_anything_c1_scene.v2",
        ):
            if stage.get("asset_id") != "020_hammer:0":
                raise ValueError(
                    "the current Isaac bridge only supports DOMINO hammer 020"
                )
            stage = {
                **stage,
                "schema": "nonprehensile.push_anything_stage.v1",
                "asset_name": "DOMINO_020_hammer_safe",
                "root_height_m": (
                    float(template.target_object.pose[2])
                    - args.table_height_offset_m
                ),
                "support_quaternion_wxyz": support_quaternion,
            }
        elif "support_quaternion_wxyz" not in stage:
            # Upgrade old stage artifacts deterministically instead of using
            # the template target's potentially randomized planar yaw.
            stage = {**stage, "support_quaternion_wxyz": support_quaternion}
        scenes.append(
            stage_to_isaac_scene(
                stage,
                template,
                scene_id=scene_id,
                table_height_offset_m=args.table_height_offset_m,
                target_mass_kg=args.target_mass_kg,
                target_static_friction=args.target_static_friction,
                target_dynamic_friction=args.target_dynamic_friction,
            )
        )
    output = args.output.expanduser().resolve()
    write_scene_manifest(output, scenes)
    print(
        json.dumps(
            {
                "output": str(output),
                "scenes": len(scenes),
                "scene_ids": [scene.scene_id for scene in scenes],
                "target_mass_kg": scenes[0].target_object.mass_kg,
                "target_static_friction": scenes[0].target_object.static_friction,
                "target_dynamic_friction": scenes[0].target_object.dynamic_friction,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
