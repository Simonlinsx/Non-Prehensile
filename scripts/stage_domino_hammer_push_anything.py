#!/usr/bin/env python3
"""Stage the same-support DOMINO hammer into a Push Anything checkout.

The physical mesh stays complete.  ``sampling_meshes`` points at the
conservative safe-handle partition, while ``unsafe_meshes`` supplies the
protected-plus-neutral union to an execution-time C1 trajectory guard.
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


EXPECTED_UPSTREAM_COMMIT = "9d988c835d6e99330397701487fce5ce4ceafa3c"
ASSET_NAME = "DOMINO_020_hammer_safe"
TABLE_HEIGHT_M = -0.029


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/data1/linsixu/dairlib-push-anything"),
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=repo_root / "data/push_anything_semantics/020_hammer_0",
    )
    parser.add_argument(
        "--generator-python",
        type=Path,
        default=Path("/data1/linsixu/miniconda3/envs/domino/bin/python"),
        help="Python containing trimesh and vhacdx",
    )
    parser.add_argument(
        "--ruamel-root",
        type=Path,
        default=Path("/data1/linsixu/.local/share/push-anything-python"),
    )
    parser.add_argument("--goal-distance", type=float, default=0.10)
    parser.add_argument(
        "--goal-direction-deg",
        type=float,
        default=0.0,
        help="planar goal direction measured counter-clockwise from world +X",
    )
    parser.add_argument("--goal-yaw-deg", type=float, default=30.0)
    parser.add_argument("--initial-x", type=float, default=0.40)
    parser.add_argument("--initial-y", type=float, default=0.20)
    parser.add_argument(
        "--reposition-speed",
        type=float,
        default=0.06,
        help="collision-free EE repositioning speed in m/s",
    )
    parser.add_argument(
        "--quaternion-weight",
        type=float,
        default=2.0,
        help="fixed per-quaternion-state weight in the joint XY+yaw cost",
    )
    parser.add_argument(
        "--realtime-rate",
        type=float,
        default=0.5,
        help="Drake simulation real-time rate for stable native multi-process execution",
    )
    parser.add_argument("--sampling-seed", type=int, default=17)
    parser.add_argument(
        "--semantic-guard-clearance",
        type=float,
        default=0.025,
        help=(
            "minimum EE-center distance to protected/neutral surfaces in meters; "
            "25 mm includes the 19.5 mm C3 EE sphere plus braking margin"
        ),
    )
    parser.add_argument(
        "--semantic-guard-stop-distance",
        type=float,
        default=0.055,
        help="high-rate OSC hold boundary in meters",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="optional per-run stage manifest (default: semantic-dir)",
    )
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


def portable_repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    upstream = args.upstream_root.resolve()
    if not (upstream / ".git").is_dir():
        raise FileNotFoundError(f"Push Anything checkout not found: {upstream}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"expected Push Anything {EXPECTED_UPSTREAM_COMMIT}, found {actual_commit}"
        )
    if not args.generator_python.is_file():
        raise FileNotFoundError(f"generator Python not found: {args.generator_python}")
    manifest_path = args.semantic_dir / "semantic_mesh_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("asset_id") != "020_hammer:0":
        raise ValueError("semantic manifest must describe DOMINO 020_hammer:0")
    if manifest.get("export_frame") != "object_local_same_support_meters":
        raise ValueError("semantic meshes must have the stable support pose baked in")
    counts = manifest.get("counts", {})
    if int(counts.get("safe", 0)) <= 0 or int(counts.get("protected", 0)) <= 0:
        raise ValueError("semantic manifest must contain safe and protected faces")
    for filename in (
        manifest.get("physical_mesh"),
        manifest["meshes"].get("safe"),
        manifest["meshes"].get("safe_guarded"),
        manifest["meshes"].get("unsafe"),
    ):
        if not filename or not (args.semantic_dir / filename).is_file():
            raise FileNotFoundError(f"semantic export is missing {filename!r}")
    return manifest


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.goal_distance <= 0.0:
        raise ValueError("goal-distance must be positive")
    if args.reposition_speed <= 0.0:
        raise ValueError("reposition-speed must be positive")
    if args.quaternion_weight <= 0.0:
        raise ValueError("quaternion-weight must be positive")
    if not 0.0 < args.realtime_rate <= 1.0:
        raise ValueError("realtime-rate must be in (0, 1]")
    if not -180.0 <= args.goal_yaw_deg <= 180.0:
        raise ValueError("goal-yaw-deg must be in [-180, 180]")
    if not -180.0 <= args.goal_direction_deg <= 180.0:
        raise ValueError("goal-direction-deg must be in [-180, 180]")
    if args.semantic_guard_clearance <= 0.0:
        raise ValueError("semantic-guard-clearance must be positive")
    if args.semantic_guard_stop_distance < args.semantic_guard_clearance:
        raise ValueError(
            "semantic-guard-stop-distance must be at least semantic-guard-clearance"
        )
    manifest = validate_inputs(args)
    upstream = args.upstream_root.resolve()
    asset_dir = upstream / "examples/sampling_c3/urdf" / ASSET_NAME
    asset_dir.mkdir(parents=True, exist_ok=True)
    physical_name = f"{ASSET_NAME}.obj"
    safe_name = f"{ASSET_NAME}_safe_guarded.obj"
    unsafe_name = f"{ASSET_NAME}_unsafe.obj"
    shutil.copyfile(args.semantic_dir / manifest["physical_mesh"], asset_dir / physical_name)
    shutil.copyfile(
        args.semantic_dir / manifest["meshes"]["safe_guarded"],
        asset_dir / safe_name,
    )
    shutil.copyfile(
        args.semantic_dir / manifest["meshes"]["unsafe"], asset_dir / unsafe_name
    )

    params = upstream / "examples/sampling_c3/anything/parameters"
    controller = params / "sampling_c3_controller_params.yaml"
    replace_yaml_line(controller, "base_names", f"[{ASSET_NAME}]")
    replace_yaml_line(
        controller,
        "sampling_meshes",
        f"[examples/sampling_c3/urdf/{ASSET_NAME}/{safe_name}]",
    )
    replace_yaml_line(
        controller,
        "unsafe_meshes",
        f"[examples/sampling_c3/urdf/{ASSET_NAME}/{unsafe_name}]",
    )
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

    environment = os.environ.copy()
    python_paths = [str(upstream)]
    if args.ruamel_root.is_dir():
        python_paths.append(str(args.ruamel_root.resolve()))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(
        [
            str(args.generator_python.resolve()),
            "examples/sampling_c3/multiyaml_rewrite.py",
            "--recreate-sdf",
        ],
        cwd=upstream,
        env=environment,
        check=True,
    )

    support_height = float(manifest["support_height_m"])
    root_height = TABLE_HEIGHT_M + support_height
    vertices_z = []
    with (asset_dir / physical_name).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices_z.append(float(line.split()[3]))
    if not vertices_z:
        raise RuntimeError("staged physical OBJ contains no vertices")
    full_height = max(vertices_z) - min(vertices_z)
    # Do not lower the Franka execution plane just because this supported
    # hammer is thinner than the scanned letter baseline.  The official
    # controller's 2 mm contact plane and 73 mm collision-free waypoint are
    # already calibrated to its spherical EE and tabletop; lower values put
    # the OSC close to its table/singularity boundary.
    contact_height = max(0.002, TABLE_HEIGHT_M + 0.5 * full_height + 0.010)
    reposition_height = max(0.073, TABLE_HEIGHT_M + full_height + 0.050)

    half_yaw = math.radians(args.goal_yaw_deg) * 0.5
    goal_quaternion = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    goal_direction_rad = math.radians(args.goal_direction_deg)
    goal_x = args.initial_x + args.goal_distance * math.cos(goal_direction_rad)
    goal_y = args.initial_y + args.goal_distance * math.sin(goal_direction_rad)
    sim = params / "sim_params.yaml"
    goal = params / "goal_params.yaml"
    sampling = params / "sampling_params.yaml"
    reposition = params / "reposition_params.yaml"
    progress = params / "progress_params_c3plus.yaml"
    c3plus_options = params / "sampling_c3plus_options.yaml"
    replace_yaml_line(
        sim,
        "q_init_objects",
        "[[1.0, 0.0, 0.0, 0.0, "
        f"{args.initial_x:.12g}, {args.initial_y:.12g}, {root_height:.12g}]]",
    )
    replace_yaml_line(sim, "realtime_rate", f"{args.realtime_rate:.12g}")
    replace_yaml_line(
        goal,
        "resting_object_heights",
        f"[{root_height:.12g}]",
    )
    replace_yaml_line(
        goal,
        "fixed_target_positions",
        f"[[{goal_x:.12g}, {goal_y:.12g}, {root_height:.12g}]]",
    )
    replace_yaml_line(
        goal,
        "fixed_target_orientations",
        "[[" + ", ".join(f"{item:.12g}" for item in goal_quaternion) + "]]",
    )
    replace_yaml_line(sampling, "z_height", f"{contact_height:.12g}")
    replace_yaml_line(sampling, "random_seed", str(args.sampling_seed))
    replace_yaml_line(reposition, "pwl_waypoint_height", f"{reposition_height:.12g}")
    replace_yaml_line(reposition, "speed", f"{args.reposition_speed:.12g}")
    # Reposition is a cheap geometric trajectory and should start from the
    # measured EE state.  Feeding its own predicted state back as the next
    # measured state can create an unstable positive-feedback loop when the
    # OSC has even a small tracking lag.  Keep prediction only for C3 rollout.
    replace_yaml_line(c3plus_options, "use_predicted_x0_repos", "false")
    # The upstream pose mode adds a state-dependent quaternion Hessian with a
    # weight of 1000.  For this planar hammer task it is unnecessarily stiff
    # and made the online solve/OSC chain numerically unstable.  A moderate,
    # fixed quaternion diagonal keeps yaw in the objective from the first
    # control step without changing solver structure mid-episode.
    replace_yaml_line(c3plus_options, "use_quaternion_dependent_cost", "false")
    quaternion_weight = f"{args.quaternion_weight:.12g}"
    joint_q_vector = (
        "[0.01, 0.01, 0.01, "
        + ", ".join([quaternion_weight] * 4)
        + ", 150, 150, 120, 15, 15, 10, "
        "0.05, 0.05, 0.05, 0.05, 0.05, 0.05]"
    )
    replace_yaml_line(
        c3plus_options,
        "q_vector",
        joint_q_vector,
    )
    joint_q_vector_position = joint_q_vector.replace(
        ", 150, 150, 120,", ", 200, 200, 120,", 1
    )
    replace_yaml_line(
        c3plus_options,
        "q_vector_position",
        joint_q_vector_position,
    )
    # Do not use the paper's 5 cm XY -> full-pose switch.  Its "position"
    # configuration now already contains a finite quaternion/yaw cost, so it
    # is a joint-pose objective from the first step.  Keeping one solver
    # configuration throughout also avoids a discontinuous online mode change.
    replace_yaml_line(progress, "cost_switching_threshold_distance", "0.0")

    staged_manifest = {
        "schema": "nonprehensile.push_anything_stage.v1",
        "asset_name": ASSET_NAME,
        "source_manifest": portable_repo_path(
            args.semantic_dir / "semantic_mesh_manifest.json", repo_root
        ),
        "physical_model": str((asset_dir / f"{ASSET_NAME}.sdf").relative_to(upstream)),
        "controller_model": str(
            (asset_dir / f"{ASSET_NAME}_controller.sdf").relative_to(upstream)
        ),
        "sampling_mesh": str((asset_dir / safe_name).relative_to(upstream)),
        "unsafe_mesh": str((asset_dir / unsafe_name).relative_to(upstream)),
        "semantic_guard_clearance_m": args.semantic_guard_clearance,
        "semantic_guard_stop_distance_m": args.semantic_guard_stop_distance,
        "root_height_m": root_height,
        "object_height_m": full_height,
        "contact_height_m": contact_height,
        "initial_xy_m": [args.initial_x, args.initial_y],
        "goal_xy_m": [goal_x, goal_y],
        "goal_distance_m": args.goal_distance,
        "goal_direction_deg": args.goal_direction_deg,
        "goal_yaw_deg": args.goal_yaw_deg,
        "reposition_speed_m_s": args.reposition_speed,
        "simulation_realtime_rate": args.realtime_rate,
        "sampling_seed": args.sampling_seed,
        "use_predicted_x0_repos": False,
        "joint_pose_from_start": True,
        "upstream_cost_mode_switch_disabled": True,
        "solver_configuration": "position_dynamics_with_joint_pose_cost",
        "quaternion_cost": {
            "type": "fixed_diagonal",
            "q_vector_weight": args.quaternion_weight,
            "state_dependent_hessian": False,
        },
    }
    output = args.output_manifest or (
        args.semantic_dir / "push_anything_stage_manifest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(staged_manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(staged_manifest, indent=2))


if __name__ == "__main__":
    main()
