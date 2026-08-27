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
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        choices=range(4),
        default=3,
        help="0=XY, 1=+yaw, 2=+protected avoidance, 3=+clutter",
    )
    parser.add_argument(
        "--settings",
        choices=(
            "dapl-paper",
            "dapl-planar-push",
            "dywa-arm-div-planar-push",
            "legacy-curriculum",
        ),
        default="dapl-paper",
        help=(
            "scene/task randomization contract; dapl-paper preserves the "
            "reported independent stable-pose sampling, dapl-planar-push "
            "keeps the same paper ranges but reuses the initial support face "
            "at the goal, dywa-arm-div-planar-push is the released DyWA "
            "centre-ray XY diagnostic with the same support face, and "
            "legacy-curriculum preserves old ablations"
        ),
    )
    parser.add_argument(
        "--scene-count",
        type=int,
        default=None,
        help=(
            "defaults to 1024 train / 128 eval for dapl-paper, or 1 for "
            "legacy stage 0 and 128 for later legacy stages"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--large-extent-threshold", type=float, default=0.16)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--maximum-stable-poses", type=int, default=64)
    return parser.parse_args()


def generation_config_for_settings(
    settings: str, curriculum_stage: int
) -> ClutterGenerationConfig:
    """Return one explicitly named scene/task randomization contract."""

    legacy_stage_configs = {
        0: ClutterGenerationConfig(
            tasks_per_scene=1,
            minimum_planar_displacement=0.08,
            target_obstacle_clearance=0.12,
            maximum_goal_yaw_delta=0.0,
            preserve_target_support_pose=True,
        ),
        1: ClutterGenerationConfig(
            tasks_per_scene=8,
            minimum_planar_displacement=0.10,
            target_obstacle_clearance=0.06,
            maximum_goal_yaw_delta=0.50,
            preserve_target_support_pose=True,
        ),
        2: ClutterGenerationConfig(
            tasks_per_scene=8,
            minimum_planar_displacement=0.10,
            target_obstacle_clearance=0.04,
            maximum_goal_yaw_delta=0.75,
            preserve_target_support_pose=True,
        ),
        3: ClutterGenerationConfig(
            tasks_per_scene=16,
            minimum_planar_displacement=0.12,
            target_obstacle_clearance=0.008,
            maximum_goal_yaw_delta=3.141592653589793,
            preserve_target_support_pose=True,
        ),
    }
    if settings == "dapl-paper":
        return ClutterGenerationConfig()
    if settings == "dapl-planar-push":
        # Preserve every reported DAPL spatial/count setting while keeping the
        # manipulation target inside the physically reachable tabletop-push
        # manifold.  The initial support face remains random; only the goal
        # reuses it, with an independently sampled full-range yaw.
        return ClutterGenerationConfig(preserve_target_support_pose=True)
    if settings == "dywa-arm-div-planar-push":
        # Released DyWA arm_div_base: a 0.4 x 0.5 m tabletop with scene margin
        # scale 0.95, a 5 cm goal radius, min_separation_scale 1.1, and task
        # margin_scale 0 (goal sampled on the object-to-table-centre ray).
        # Keeping one support face is the named planar-push compatibility
        # change required by the strict Z/full-SO(3) benchmark.
        return ClutterGenerationConfig(
            target_x_offset_range=(-0.19, 0.19),
            target_y_offset_range=(-0.2375, 0.2375),
            minimum_planar_displacement=0.055,
            goal_xy_sampling="center_ray",
            preserve_target_support_pose=True,
        )
    if settings == "legacy-curriculum":
        return legacy_stage_configs[curriculum_stage]
    raise ValueError(f"unknown generation settings: {settings}")


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
    generation_config = generation_config_for_settings(
        args.settings, args.curriculum_stage
    )
    if args.settings in (
        "dapl-paper",
        "dapl-planar-push",
        "dywa-arm-div-planar-push",
    ):
        # Exact task sampling values reported in the DAPL appendix.  In
        # particular, do not carry over the old directional 6--10 cm proof
        # distribution or its bounded yaw deltas.
        scene_count = args.scene_count
        if scene_count is None:
            scene_count = 1024 if args.split == "train" else 128
    else:
        scene_count = args.scene_count
        if scene_count is None:
            scene_count = 1 if args.curriculum_stage == 0 else 128
    scenes = generate_clutter_scenes(
        catalog,
        track=args.track,
        split=args.split,
        scene_count=scene_count,
        seed=args.seed,
        config=generation_config,
        target_asset_ids=target_asset_ids,
    )
    write_scene_manifest(args.output, scenes)
    used = sorted({item.asset_id for scene in scenes for item in scene.objects})
    print(
        f"wrote {len(scenes)} DOMINO {args.track} scenes using "
        f"settings={args.settings} to {args.output}\n"
        f"targets={','.join(target_asset_ids)}\n"
        f"used_assets={','.join(used)}"
    )


if __name__ == "__main__":
    main()
