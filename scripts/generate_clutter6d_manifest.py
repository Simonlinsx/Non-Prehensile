#!/usr/bin/env python3
"""Generate deterministic Clutter6D JSONL manifests.

The DGN adapter is an immediately runnable Isaac Lab integration fixture.  It
uses the locally released DGN assets and must not be reported as the paper's
10K Objaverse asset benchmark.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dapl.catalog import build_dgn_clutter_catalog
from dapl.generation import generate_clutter_scenes
from dapl.scene import ClutterTrack, write_scene_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", choices=[item.value for item in ClutterTrack], default="sparse")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--scene-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dgn-root",
        type=Path,
        default=os.environ.get("DGN_DATA_ROOT"),
        help="DGN release root (defaults to DGN_DATA_ROOT)",
    )
    parser.add_argument("--assets-per-cohort", type=int, default=12)
    parser.add_argument("--large-extent-threshold", type=float, default=0.18)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--maximum-stable-poses", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dgn_root is None:
        raise SystemExit("--dgn-root or DGN_DATA_ROOT is required")
    catalog = build_dgn_clutter_catalog(
        args.dgn_root,
        seed=args.seed,
        assets_per_cohort=args.assets_per_cohort,
        large_extent_threshold_m=args.large_extent_threshold,
        density_kg_m3=args.density,
        maximum_stable_poses=args.maximum_stable_poses,
    )
    scenes = generate_clutter_scenes(
        catalog,
        track=args.track,
        split=args.split,
        scene_count=args.scene_count,
        seed=args.seed,
    )
    write_scene_manifest(args.output, scenes)
    print(
        f"wrote {len(scenes)} deterministic {args.track} scenes "
        f"({len(scenes[0].objects)} objects, {len(scenes[0].tasks)} tasks each) to {args.output}"
    )


if __name__ == "__main__":
    main()
