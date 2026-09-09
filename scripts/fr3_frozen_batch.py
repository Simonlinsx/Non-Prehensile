"""Verify the frozen FR3 controller and per-scene inputs before execution/audit."""
import hashlib
import json
from pathlib import Path

import yaml

try:
    from .audit_scene_pose_contract import audit as audit_pose
    from .fr3_runtime_inputs import inventory
except ImportError:
    from audit_scene_pose_contract import audit as audit_pose
    from fr3_runtime_inputs import inventory


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_file(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"Frozen input changed: {path}")


def verify_batch(root, expected_sha256):
    root = Path(root).resolve()
    check_file(root / "controller_freeze.json", expected_sha256)
    freeze = json.loads((root / "controller_freeze.json").read_text())
    if freeze["schema"] != "nonprehensile.fr3_osc_controller_freeze.v1":
        raise ValueError("Unexpected controller freeze schema")
    check_file(root / "protocol.json", freeze["protocol_sha256"])
    check_file(root / "input_manifest.jsonl", freeze["manifest_sha256"])
    protocol = json.loads((root / "protocol.json").read_text())
    scenes = [json.loads(line) for line in (root / "input_manifest.jsonl").read_text().splitlines() if line.strip()]
    ids = [s["scene_id"] for s in scenes]
    if (len(ids) != 50 or len(set(ids)) != 50 or set(ids) != set(freeze["scene_inputs"])
            or protocol["count"] != 50 or protocol["minimum_successes"] != 21
            or protocol["success_rate_strictly_greater_than"] != .4
            or protocol["max_sim_time_s"] != 180 or protocol["maximum_c1_violations"] != 0
            or protocol["strict_pose_thresholds"] != dict(planar_m=.02, height_m=.01, rotation_rad=.105, dwell_time_s=.5)
            or protocol["manifest_sha256"] != freeze["manifest_sha256"]):
        raise ValueError("Frozen batch does not implement the agreed50-scene protocol")
    required = dict(physics_dt_s=.001, control_decimation=1, planner_period_ms=50,
                    max_sim_time_s=180, dwell_time_s=.5, force_c3_on_contact=False,
                    diagnostic_osc_hold=False, executor_mode="effort")
    if freeze["execution"] != required:
        raise ValueError("Frozen native OSC execution contract differs")
    for relative, digest in freeze["execution_source_sha256"].items():
        check_file(root / "execution_source" / relative, digest)
    for name, digest in freeze["native_binary_sha256"].items():
        check_file(root / "binary/bazel-bin/examples/sampling_c3" / name, digest)
    for name, digest in freeze["external_input_sha256"].items():
        check_file(Path(name), digest)
        check_file(root / "input_blobs" / digest, digest)
    check_file(freeze["robot_model_manifest"], freeze["robot_model_manifest_sha256"])
    semantic_path = next(Path(name) for name, digest in freeze["external_input_sha256"].items()
                         if digest == protocol["semantic_manifest_sha256"])
    semantic = json.loads(semantic_path.read_text())
    return freeze, protocol, scenes, semantic


def verify_scene(root, freeze, scene, semantic):
    directory = Path(root).resolve() / scene["scene_id"]
    record = freeze["scene_inputs"][scene["scene_id"]]
    check_file(directory / "scene_spec.json", record["scene_spec_sha256"])
    if json.loads((directory / "scene_spec.json").read_text()) != scene:
        raise ValueError("Per-scene specification differs from the fixed manifest")
    check_file(directory / "isaac_manifest.jsonl", record["isaac_manifest_sha256"])
    runtime = directory / "runtime"
    actual = inventory(runtime)
    if (actual["files"] != record["files"] or
            actual["controller_signature_sha256"] != freeze["controller_signature_sha256"]):
        raise ValueError("Scene controller/geometry differs from the frozen inputs")
    parameters = runtime / "examples/sampling_c3/anything/parameters"
    sampling = yaml.safe_load((parameters / "sampling_params.yaml").read_text())
    if sampling["random_seed"] != scene["sampling_seed"]:
        raise ValueError("Scene sampling seed differs from the manifest")
    manifest = json.loads((directory / "isaac_manifest.jsonl").read_text())
    task = manifest["tasks"][0]
    target = next(o for o in manifest["objects"] if o["instance_id"] == task["target_instance_id"])
    stub = dict(franka_base_height_m=.029, initial_target_pose_wxyz=target["pose"], goal_pose_wxyz=task["goal_pose"])
    goal = yaml.safe_load((parameters / "goal_params.yaml").read_text())
    audit_pose(scene, stub, goal, semantic)
    return directory
