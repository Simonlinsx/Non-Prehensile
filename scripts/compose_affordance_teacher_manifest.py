#!/usr/bin/env python3
"""Compose randomized-pose and audited safety scenes for teacher training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

from dapl.scene import load_scene_manifest, write_scene_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--safety-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-repeats", type=int, default=1)
    parser.add_argument("--safety-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pose_repeats <= 0 or args.safety_repeats <= 0:
        raise ValueError("manifest repeat counts must be positive")
    pose_scenes = tuple(load_scene_manifest(args.pose_manifest))
    safety_scenes = tuple(load_scene_manifest(args.safety_manifest))
    if not pose_scenes or not safety_scenes:
        raise ValueError("both source manifests must contain scenes")
    target_assets = {
        scene.target_object.asset_id for scene in (*pose_scenes, *safety_scenes)
    }
    if target_assets != {"020_hammer:0"}:
        raise ValueError(f"teacher mixture requires only DOMINO hammer targets: {target_assets}")

    mixed = []
    for source_name, scenes, repeats in (
        ("pose", pose_scenes, args.pose_repeats),
        ("safety", safety_scenes, args.safety_repeats),
    ):
        for repeat_index in range(repeats):
            for scene_index, scene in enumerate(scenes):
                mixed.append(
                    replace(
                        scene,
                        scene_id=(
                            f"teacher-mixed-{source_name}-r{repeat_index:02d}-"
                            f"{scene_index:04d}"
                        ),
                        split="train",
                    )
                )
    random.Random(args.seed).shuffle(mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_scene_manifest(args.output, tuple(mixed))
    summary = {
        "schema_version": 1,
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "scene_count": len(mixed),
        "pose_source": str(args.pose_manifest.resolve()),
        "pose_source_scenes": len(pose_scenes),
        "pose_repeats": args.pose_repeats,
        "safety_source": str(args.safety_manifest.resolve()),
        "safety_source_scenes": len(safety_scenes),
        "safety_repeats": args.safety_repeats,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
