"""Prepare fixed scenes from one validated FR3 runtime without retuning its controller."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

try:
    from .audit_scene_pose_contract import audit as audit_scene_pose, yaw_quaternion
    from .stage_shared_physx_contact_model import materialize_directory
except ImportError:
    from audit_scene_pose_contract import audit as audit_scene_pose, yaw_quaternion
    from stage_shared_physx_contact_model import materialize_directory


PARAMETERS = Path("examples/sampling_c3/anything/parameters")
SCENE_KEYS = {"sim_params.yaml": {"q_init_objects"},
              "goal_params.yaml": {"fixed_target_positions", "fixed_target_orientations"},
              "sampling_params.yaml": {"random_seed"}}


def scene_parameters(template, scene, semantic, base_height):
    """Only scene pose and prescribed sampling seed may vary across a batch."""
    data = {name: dict(value) for name, value in template.items()}
    height = semantic["support_height_m"] - base_height
    if data["goal_params.yaml"]["goal_mode"] != 2:
        raise ValueError("Runtime must already use a fixed goal")
    data["sim_params.yaml"]["q_init_objects"] = [[*yaw_quaternion(scene["initial_yaw_deg"]),
                                                  *scene["initial_xy_m"], height]]
    data["goal_params.yaml"]["fixed_target_positions"] = [[*scene["goal_xy_m"], height]]
    data["goal_params.yaml"]["fixed_target_orientations"] = [yaw_quaternion(scene["goal_yaw_deg"])]
    data["sampling_params.yaml"]["random_seed"] = scene["sampling_seed"]
    for name, baseline in template.items():
        fixed_keys = set(baseline) - SCENE_KEYS[name]
        if any(data[name][key] != baseline[key] for key in fixed_keys):
            raise ValueError("Scene preparation changed controller configuration")
    return data


def prepare(manifest, runtime_template, isaac_template, semantic_path, output_root, base_height=.029):
    from dapl.contact_planner import stage_to_isaac_scene
    from dapl.scene import load_scene_manifest, write_scene_manifest
    scenes = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = [scene["scene_id"] for scene in scenes]
    if not scenes or len(set(ids)) != len(ids) or any(Path(s).name != s or s in (".", "..") for s in ids):
        raise ValueError("Missing, duplicate, or invalid scene IDs")
    semantic = json.loads(semantic_path.read_text())
    template = next(iter(load_scene_manifest(isaac_template)))
    native_template = {name: yaml.safe_load((runtime_template / PARAMETERS / name).read_text()) for name in SCENE_KEYS}
    output_root.mkdir(parents=True, exist_ok=False)
    records = []
    for scene in scenes:
        if scene["asset_id"] != "020_hammer:0" or scene.get("clutter_count") != 0:
            raise ValueError("This batch is target-only hammer C1")
        root = output_root / scene["scene_id"]
        root.mkdir()
        runtime = root / "runtime"
        # Preserve symlinks to large native runfiles; never dereference Bazel caches.
        shutil.copytree(runtime_template, runtime, symlinks=True)
        for parent in reversed(PARAMETERS.parents):
            materialize_directory(runtime / parent)
        materialize_directory(runtime / PARAMETERS)
        parameters = scene_parameters(native_template, scene, semantic, base_height)
        for name, value in parameters.items():
            path = runtime / PARAMETERS / name
            if path.is_symlink():
                path.unlink()
            path.write_text(yaml.safe_dump(value, sort_keys=False))
        stage = dict(scene, schema="nonprehensile.push_anything_stage.v1",
                     asset_name="DOMINO_020_hammer_safe",
                     root_height_m=semantic["support_height_m"] - base_height,
                     support_quaternion_wxyz=semantic["support_quaternion_wxyz"])
        isaac = stage_to_isaac_scene(stage, template, scene_id=scene["scene_id"],
                                    table_height_offset_m=base_height,
                                    target_mass_kg=.05, target_static_friction=.3, target_dynamic_friction=.3)
        write_scene_manifest(root / "isaac_manifest.jsonl", [isaac])
        # Check the actual generated pose representation before launching any GPU work.
        stub = dict(franka_base_height_m=base_height,
                    initial_target_pose_wxyz=list(isaac.target_object.pose),
                    goal_pose_wxyz=list(isaac.tasks[0].goal_pose))
        pose_audit = audit_scene_pose(scene, stub, parameters["goal_params.yaml"], semantic, base_height)
        (root / "scene_spec.json").write_text(json.dumps(scene, indent=2) + "\n")
        (root / "prepared_scene_pose_audit.json").write_text(json.dumps(pose_audit, indent=2) + "\n")
        records.append(dict(scene_id=scene["scene_id"], preparation_pose_pass=True,
                            isaac_manifest_sha256=hashlib.sha256((root / "isaac_manifest.jsonl").read_bytes()).hexdigest()))
    report = dict(schema="nonprehensile.fr3_osc_scene_preparation.v1", count=len(records),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  runtime_template=str(runtime_template.resolve()), scene_variable_keys={k: sorted(v) for k, v in SCENE_KEYS.items()},
                  scenes=records, formal_execution_started=False)
    (output_root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("count", "manifest_sha256", "formal_execution_started")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--isaac-template", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.runtime_template, args.isaac_template, args.semantic_manifest, args.output_root)
