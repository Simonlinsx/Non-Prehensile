#!/usr/bin/env python3
"""Convert selected DOMINO GLB/OBJ meshes into Isaac Lab USD assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--domino-root",
    type=Path,
    default=os.environ.get("DOMINO_ROOT"),
    help="DOMINO checkout or objects directory (defaults to DOMINO_ROOT)",
)
parser.add_argument(
    "--usd-root",
    type=Path,
    default=os.environ.get("DOMINO_USD_ROOT"),
    help="converted USD directory (defaults to DOMINO_USD_ROOT)",
)
parser.add_argument(
    "--asset-id",
    action="append",
    dest="asset_ids",
    help="asset '<NNN_category>:<id>'; repeat for multiple assets",
)
parser.add_argument(
    "--manifest",
    type=Path,
    help="also convert every asset referenced by this Clutter6D manifest",
)
parser.add_argument(
    "--collision-approximation",
    choices=(
        "convexDecomposition",
        "convexHull",
        "boundingCube",
        "boundingSphere",
        "meshSimplification",
    ),
    default="convexDecomposition",
)
parser.add_argument("--force", action="store_true", help="force USD regeneration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

from dapl.domino import DominoDataPaths
from dapl.scene import load_scene_manifest


def requested_asset_ids() -> tuple[str, ...]:
    values = set(args_cli.asset_ids or ())
    if args_cli.manifest is not None:
        for scene in load_scene_manifest(args_cli.manifest):
            values.update(item.asset_id for item in scene.objects)
    if not values:
        raise ValueError("pass at least one --asset-id or --manifest")
    return tuple(sorted(values))


def main() -> None:
    if args_cli.domino_root is None:
        raise ValueError("--domino-root or DOMINO_ROOT is required")
    if args_cli.usd_root is None:
        raise ValueError("--usd-root or DOMINO_USD_ROOT is required")
    paths = DominoDataPaths.resolve(args_cli.domino_root, args_cli.usd_root)
    converted = []
    for asset_id in requested_asset_ids():
        source = paths.require_source_asset(asset_id)
        source.usd.parent.mkdir(parents=True, exist_ok=True)
        cfg = MeshConverterCfg(
            asset_path=str(source.visual_mesh),
            force_usd_conversion=bool(args_cli.force),
            usd_dir=str(source.usd.parent),
            usd_file_name=source.usd.name,
            make_instanceable=False,
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            # Seed MassAPI so Isaac Lab can override each scene instance with
            # the manifest mass at spawn time.
            mass_props=schemas_cfg.MassPropertiesCfg(mass=1.0),
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            collision_approximation=args_cli.collision_approximation,
        )
        converter = MeshConverter(cfg)
        output = Path(converter.usd_path).resolve()
        if output != source.usd or not output.is_file():
            raise RuntimeError(
                f"converter produced {output}, expected prepared asset {source.usd}"
            )
        converted.append((asset_id, output))
    for asset_id, output in converted:
        print(f"{asset_id}\t{output}")
    print(f"prepared {len(converted)} DOMINO USD assets under {paths.usd_root}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
