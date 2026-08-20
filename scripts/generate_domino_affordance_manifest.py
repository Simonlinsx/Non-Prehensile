#!/usr/bin/env python3
"""Generate DOMINO-backed manifests for affordance-aware pushing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dapl.domino import build_domino_clutter_catalog
from dapl.generation import ClutterGenerationConfig, generate_clutter_scenes
from dapl.scene import ClutterTrack, write_scene_manifest


DEFAULT_ASSET_IDS = (
    "020_hammer:0",
    "032_screwdriver:0",
    "034_knife:0",
    "082_smallshovel:0",
    "084_woodenmallet:0",
    "001_bottle:0",
    "002_bowl:1",
    "021_cup:0",
    "039_mug:0",
    "041_shoe:0",
    "062_plasticbox:0",
    "077_phone:0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domino-root",
        type=Path,
        default=os.environ.get("DOMINO_ROOT"),
        help="DOMINO checkout or objects directory (defaults to DOMINO_ROOT)",
    )
    parser.add_argument(
        "--asset-id",
        action="append",
        dest="asset_ids",
        help="catalog asset '<NNN_category>:<id>'; repeat to override defaults",
    )
    parser.add_argument(
        "--target-asset-id",
        action="append",
        dest="target_asset_ids",
        default=None,
        help="allowed annotated target; repeat for a target set (default: 020_hammer:0)",
    )
    parser.add_argument(
        "--track", choices=[item.value for item in ClutterTrack], default="sparse"
    )
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--scene-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--large-extent-threshold", type=float, default=0.16)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--maximum-stable-poses", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.domino_root is None:
        raise SystemExit("--domino-root or DOMINO_ROOT is required")
    asset_ids = tuple(args.asset_ids or DEFAULT_ASSET_IDS)
    target_asset_ids = tuple(args.target_asset_ids or ("020_hammer:0",))
    missing_targets = set(target_asset_ids) - set(asset_ids)
    if missing_targets:
        raise SystemExit(
            f"target assets must also be included by --asset-id: {sorted(missing_targets)}"
        )
    catalog = build_domino_clutter_catalog(
        args.domino_root,
        asset_ids,
        large_extent_threshold_m=args.large_extent_threshold,
        density_kg_m3=args.density,
        maximum_stable_poses=args.maximum_stable_poses,
    )
    annotated_targets = build_domino_clutter_catalog(
        args.domino_root,
        target_asset_ids,
        large_extent_threshold_m=args.large_extent_threshold,
        density_kg_m3=args.density,
        maximum_stable_poses=1,
        require_affordance=True,
    )
    del annotated_targets  # Validation only; catalog already contains these records.
    scenes = generate_clutter_scenes(
        catalog,
        track=args.track,
        split=args.split,
        scene_count=args.scene_count,
        seed=args.seed,
        config=ClutterGenerationConfig(preserve_target_support_pose=True),
        target_asset_ids=target_asset_ids,
    )
    write_scene_manifest(args.output, scenes)
    used = sorted({item.asset_id for scene in scenes for item in scene.objects})
    print(
        f"wrote {len(scenes)} DOMINO {args.track} scenes to {args.output}\n"
        f"targets={','.join(target_asset_ids)}\n"
        f"used_assets={','.join(used)}"
    )


if __name__ == "__main__":
    main()
