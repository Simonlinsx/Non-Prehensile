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
# The closed Panda fingers span z=[58.4, 112.279] mm in panda_hand.  Near the
# distal face they span about 17.5 mm in x and at least 28 mm in y. The 8 mm
# sphere is a provisional translation-only surrogate whose leading tangent
# meets the distal face, not a certified inscribed or contact-equivalent
# shape. The live collision audit (2026-09-08) shows lateral finger contacts
# while this sphere is up to 9.92 mm clear of the PhysX target convexes.
# Keep this baseline frozen pending a validated contact model. The wider
# 27 mm radius is only the full-model repositioning collision envelope.
CLOSED_GRIPPER_PROXY_RADIUS_M = 0.008
CLOSED_GRIPPER_COLLISION_ENVELOPE_RADIUS_M = 0.027
CLOSED_GRIPPER_PROXY_FROM_HAND_M = 0.11227911200523376 - CLOSED_GRIPPER_PROXY_RADIUS_M
CLOSED_GRIPPER_FINGER_LENGTH_M = 0.053879112005233765
CLOSED_GRIPPER_FINGER_CENTER_FROM_HAND_M = (
    0.0584 + 0.11227911200523376
) / 2.0
# Push Anything's mesh sampler was tuned together with its 19.5 mm spherical
# contact body. Keep the same proxy-to-surface offsets when substituting the
# smaller closed-fingertip proxy. Leaving the center clearances at
# 35/27 mm makes the 8 mm proxy start every C3 segment 11.5 mm farther from
# contact than the upstream model, so the optimizer advances while the real
# gripper is still in free space.
UPSTREAM_PROXY_RADIUS_M = 0.0195
UPSTREAM_SAMPLE_BUFFER_DISTANCE_M = 0.035
UPSTREAM_SAMPLE_PROJECTION_CLEARANCE_M = 0.027


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
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
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
        "--objective-mode",
        choices=("push-anything", "full-pose", "yaw-regularized", "joint-pose"),
        default="joint-pose",
        help=(
            "push-anything retains the upstream 5 cm XY-to-pose objective "
            "schedule; full-pose uses the upstream pose cost from the first "
            "control step; yaw-regularized adds only a weak pre-switch orientation "
            "regularizer; joint-pose applies a custom fixed XY+yaw cost. All "
            "modes keep the same strict terminal pose gate."
        ),
    )
    parser.add_argument(
        "--realtime-rate",
        type=float,
        default=0.5,
        help="Drake simulation real-time rate for stable native multi-process execution",
    )
    parser.add_argument("--sampling-seed", type=int, default=17)
    parser.add_argument(
        "--controller-inertial-mode",
        choices=("matched", "upstream-generic"),
        default="matched",
        help=(
            "matched copies the physical hammer mass/inertia into C3's controller "
            "SDF; upstream-generic preserves Push Anything's 1 kg template"
        ),
    )
    parser.add_argument(
        "--end-effector-mode",
        choices=("push-anything-sphere", "closed-gripper-proxy"),
        default="closed-gripper-proxy",
        help=(
            "physical end-effector contract used downstream.  The closed-gripper "
            "mode replaces the attached pusher in C3 with a conservative proxy "
            "whose front face matches the stock Panda's closed fingertips"
        ),
    )
    parser.add_argument(
        "--num-additional-samples-repos",
        type=int,
        default=4,
        help="new safe-surface contact candidates evaluated while repositioning",
    )
    parser.add_argument(
        "--num-additional-samples-c3",
        type=int,
        default=5,
        help="new safe-surface contact candidates evaluated while pushing",
    )
    parser.add_argument(
        "--planning-horizon",
        type=int,
        default=10,
        help="number of C3 MPC knots (upstream default: 10)",
    )
    parser.add_argument(
        "--progress-enforced-cost-drop",
        type=float,
        default=0.5,
        help=(
            "minimum fractional configuration-cost reduction required over the "
            "C3 progress window (upstream default: 0.5)"
        ),
    )
    parser.add_argument(
        "--progress-enforced-over-n-loops",
        type=int,
        default=35,
        help="C3 progress-measurement window in controller loops (upstream: 35)",
    )
    parser.add_argument(
        "--controller-position-success-threshold",
        type=float,
        default=0.02,
        help="internal controller completion threshold in meters (upstream: 0.02)",
    )
    parser.add_argument(
        "--controller-orientation-success-threshold",
        type=float,
        default=0.1,
        help="internal controller completion threshold in radians (upstream: 0.1)",
    )
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


def configure_fixed_task_goal(path: Path, position, quaternion_wxyz) -> None:
    # Native GoalMode: random=0, orientation sequence=1, fixed=2. A random
    # goal advances immediately on native pose success, before Isaac's dwell.
    replace_yaml_line(path, "goal_mode", "2")
    replace_yaml_line(path, "resting_object_heights", f"[{position[2]:.12g}]")
    replace_yaml_line(path, "fixed_target_positions",
                      "[[" + ", ".join(f"{v:.12g}" for v in position) + "]]")
    replace_yaml_line(path, "fixed_target_orientations",
                      "[[" + ", ".join(f"{v:.12g}" for v in quaternion_wxyz) + "]]")


def remove_yaml_key(path: Path, key: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(?!\s*#)\s*{re.escape(key)}\s*:.*\n?", re.MULTILINE)
    path.write_text(pattern.sub("", text), encoding="utf-8")


def portable_repo_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def inertial_block(path: Path) -> str:
    match = re.search(r"<inertial>.*?</inertial>", path.read_text(encoding="utf-8"), re.DOTALL)
    if match is None:
        raise RuntimeError(f"SDF has no inertial block: {path}")
    return match.group(0)


def synchronize_controller_inertial(physical_sdf: Path, controller_sdf: Path) -> None:
    """Use one mass/inertia model on both sides of the C3--Isaac bridge."""

    controller_text = controller_sdf.read_text(encoding="utf-8")
    synchronized, count = re.subn(
        r"<inertial>.*?</inertial>",
        inertial_block(physical_sdf),
        controller_text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"expected one controller inertial block: {controller_sdf}")
    controller_sdf.write_text(synchronized, encoding="utf-8")


def inertial_summary(path: Path) -> dict[str, float]:
    block = inertial_block(path)
    values: dict[str, float] = {}
    for name in ("mass", "ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        match = re.search(rf"<{name}>\s*([^<]+?)\s*</{name}>", block)
        if match is None:
            raise RuntimeError(f"inertial block is missing {name}: {path}")
        values[name] = float(match.group(1))
    return values


def configure_c3_end_effector(upstream: Path, mode: str) -> dict[str, object]:
    """Restore the pinned EE model, then stage the selected contact proxy."""

    relative_full = "examples/sampling_c3/urdf/end_effector_full.urdf"
    relative_simple = "examples/sampling_c3/urdf/end_effector_simple_model.urdf"
    full = subprocess.check_output(
        ["git", "-C", str(upstream), "show", f"HEAD:{relative_full}"], text=True
    )
    simple = subprocess.check_output(
        ["git", "-C", str(upstream), "show", f"HEAD:{relative_simple}"], text=True
    )
    if mode == "closed-gripper-proxy":
        proxy_radius = f"{CLOSED_GRIPPER_PROXY_RADIUS_M:.12g}"
        envelope_radius = f"{CLOSED_GRIPPER_COLLISION_ENVELOPE_RADIUS_M:.12g}"
        finger_length = f"{CLOSED_GRIPPER_FINGER_LENGTH_M:.12g}"
        # ``end_effector_peg`` starts 9.6 mm from the hand frame in the
        # upstream model.  Move/resize its cylinder to conservatively cover
        # the two closed fingers.
        peg_center_local = -(
            CLOSED_GRIPPER_FINGER_CENTER_FROM_HAND_M - 0.0096
        )
        tip_center_local = -(
            CLOSED_GRIPPER_PROXY_FROM_HAND_M - 0.0096
        )
        full = full.replace(
            '<origin rpy="0 0 0" xyz="0 0 -0.0508"/>',
            f'<origin rpy="0 0 0" xyz="0 0 {peg_center_local:.12g}"/>',
        )
        full = full.replace(
            '<cylinder radius="0.0127" length="0.1016"/>',
            f'<cylinder radius="{envelope_radius}" length="{finger_length}"/>',
        )
        full = full.replace(
            '<origin rpy="0 0 0" xyz="0 0 -0.1169"/>',
            f'<origin rpy="0 0 0" xyz="0 0 {tip_center_local:.12g}"/>',
        )
        full = full.replace(
            '<sphere radius="0.0195"/>',
            f'<sphere radius="{proxy_radius}"/>',
        )
        simple = simple.replace(
            '<sphere radius="0.0195"/>',
            f'<sphere radius="{proxy_radius}"/>',
        )
    (upstream / relative_full).write_text(full, encoding="utf-8")
    (upstream / relative_simple).write_text(simple, encoding="utf-8")
    proxy_radius_m = (
        CLOSED_GRIPPER_PROXY_RADIUS_M
        if mode == "closed-gripper-proxy" else UPSTREAM_PROXY_RADIUS_M
    )
    radius_delta_m = UPSTREAM_PROXY_RADIUS_M - proxy_radius_m
    return {
        "mode": mode,
        "reference_frame": "panda_hand",
        "reference_offset_m": (
            CLOSED_GRIPPER_PROXY_FROM_HAND_M
            if mode == "closed-gripper-proxy" else 0.1265
        ),
        "proxy_radius_m": proxy_radius_m,
        "collision_envelope_radius_m": (
            CLOSED_GRIPPER_COLLISION_ENVELOPE_RADIUS_M
            if mode == "closed-gripper-proxy" else 0.0195
        ),
        "physical_contact_surface": (
            "stock_franka_closed_fingertips"
            if mode == "closed-gripper-proxy" else "attached_spherical_pusher"
        ),
        "sample_buffer_distance_m": (
            UPSTREAM_SAMPLE_BUFFER_DISTANCE_M - radius_delta_m
        ),
        "sample_projection_clearance_m": (
            UPSTREAM_SAMPLE_PROJECTION_CLEARANCE_M - radius_delta_m
        ),
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    upstream = args.upstream_root.resolve()
    # A linked Git worktree stores ``.git`` as a pointer file rather than a
    # directory.  Both layouts are valid audited checkouts.
    if not (upstream / ".git").exists():
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
    if args.num_additional_samples_repos <= 0:
        raise ValueError("num-additional-samples-repos must be positive")
    if args.num_additional_samples_c3 <= 0:
        raise ValueError("num-additional-samples-c3 must be positive")
    if args.planning_horizon <= 0:
        raise ValueError("planning-horizon must be positive")
    if not 0.0 <= args.progress_enforced_cost_drop <= 1.0:
        raise ValueError("progress-enforced-cost-drop must be in [0, 1]")
    if args.progress_enforced_over_n_loops <= 1:
        raise ValueError("progress-enforced-over-n-loops must be at least 2")
    if args.controller_position_success_threshold <= 0.0:
        raise ValueError("controller-position-success-threshold must be positive")
    if args.controller_orientation_success_threshold <= 0.0:
        raise ValueError(
            "controller-orientation-success-threshold must be positive"
        )
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
    end_effector = configure_c3_end_effector(upstream, args.end_effector_mode)
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

    physical_sdf = asset_dir / f"{ASSET_NAME}.sdf"
    controller_sdf = asset_dir / f"{ASSET_NAME}_controller.sdf"
    if args.controller_inertial_mode == "matched":
        synchronize_controller_inertial(physical_sdf, controller_sdf)
    controller_inertial = inertial_summary(controller_sdf)
    physical_inertial = inertial_summary(physical_sdf)

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

    if not math.isfinite(args.initial_yaw_deg):
        raise ValueError("initial yaw must be finite")
    initial_half_yaw = math.radians(args.initial_yaw_deg) * 0.5
    initial_quaternion = [math.cos(initial_half_yaw), 0.0, 0.0, math.sin(initial_half_yaw)]
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
        "[[" + ", ".join(f"{v:.12g}" for v in initial_quaternion) + ", "
        + f"{args.initial_x:.12g}, {args.initial_y:.12g}, {root_height:.12g}]]",
    )
    replace_yaml_line(sim, "realtime_rate", f"{args.realtime_rate:.12g}")
    configure_fixed_task_goal(goal, [goal_x, goal_y, root_height], goal_quaternion)
    replace_yaml_line(sampling, "z_height", f"{contact_height:.12g}")
    # The sampler stores EE-center distances, while contact is determined by
    # the surface of the configured C3 proxy. Preserve the upstream
    # proxy-to-object gaps after changing proxy radius.
    replace_yaml_line(
        sampling,
        "buffer_distance",
        f"{float(end_effector['sample_buffer_distance_m']):.12g}",
    )
    replace_yaml_line(
        sampling,
        "sample_projection_clearance",
        f"{float(end_effector['sample_projection_clearance_m']):.12g}",
    )
    replace_yaml_line(sampling, "random_seed", str(args.sampling_seed))
    replace_yaml_line(
        sampling,
        "num_additional_samples_repos",
        str(args.num_additional_samples_repos),
    )
    replace_yaml_line(
        sampling,
        "num_additional_samples_c3",
        str(args.num_additional_samples_c3),
    )
    replace_yaml_line(c3plus_options, "N", str(args.planning_horizon))
    replace_yaml_line(
        progress,
        "progress_enforced_cost_drop",
        f"{args.progress_enforced_cost_drop:.12g}",
    )
    replace_yaml_line(
        progress,
        "progress_enforced_over_n_loops",
        str(args.progress_enforced_over_n_loops),
    )
    replace_yaml_line(
        goal,
        "position_success_threshold",
        f"{args.controller_position_success_threshold:.12g}",
    )
    replace_yaml_line(
        goal,
        "orientation_success_threshold",
        f"{args.controller_orientation_success_threshold:.12g}",
    )
    remove_yaml_key(sampling, "terminal_orientation_rescore_weight")
    remove_yaml_key(sampling, "two_contact_lookahead_weight")
    remove_yaml_key(sampling, "two_contact_lookahead_beam_width")
    remove_yaml_key(sampling, "two_contact_lookahead_samples")
    remove_yaml_key(sampling, "goal_conditioned_mesh_normal_fraction")
    replace_yaml_line(reposition, "pwl_waypoint_height", f"{reposition_height:.12g}")
    replace_yaml_line(reposition, "speed", f"{args.reposition_speed:.12g}")
    # Reposition is a cheap geometric trajectory and should start from the
    # measured EE state.  Feeding its own predicted state back as the next
    # measured state can create an unstable positive-feedback loop when the
    # OSC has even a small tracking lag.  Keep prediction only for C3 rollout.
    replace_yaml_line(c3plus_options, "use_predicted_x0_repos", "false")
    if args.objective_mode in ("push-anything", "full-pose", "yaw-regularized"):
        # Preserve the upstream controller's optimization schedule: position
        # is optimized outside 5 cm and full pose inside 5 cm.  This does not
        # relax acceptance -- the monitor still requires XY and SO(3) jointly.
        # multiyaml_rewrite.py has already regenerated the upstream one-object
        # q_vector and q_vector_position before this block.
        replace_yaml_line(c3plus_options, "use_quaternion_dependent_cost", "true")
        replace_yaml_line(
            progress,
            "cost_switching_threshold_distance",
            "1000.0" if args.objective_mode == "full-pose" else "0.05",
        )
        if args.objective_mode == "yaw-regularized":
            quaternion_weight = f"{args.quaternion_weight:.12g}"
            q_vector_position = (
                "[0.01, 0.01, 0.01, "
                + ", ".join([quaternion_weight] * 4)
                + ", 200, 200, 120, 15, 15, 10, "
                "0.05, 0.05, 0.05, 0.05, 0.05, 0.05]"
            )
            replace_yaml_line(
                c3plus_options, "q_vector_position", q_vector_position
            )
        joint_pose_from_start = args.objective_mode == "full-pose"
        quaternion_cost = {
            "type": (
                "weak_fixed_then_upstream_state_dependent"
                if args.objective_mode == "yaw-regularized"
                else (
                    "upstream_state_dependent_from_start"
                    if args.objective_mode == "full-pose"
                    else "upstream_state_dependent"
                )
            ),
            "q_vector_weight": (
                args.quaternion_weight
                if args.objective_mode == "yaw-regularized"
                else None
            ),
            "state_dependent_hessian": True,
        }
        if args.objective_mode == "yaw-regularized":
            solver_configuration = "upstream_xy_with_yaw_regularizer_then_full_pose"
        elif args.objective_mode == "full-pose":
            solver_configuration = "upstream_full_pose_from_start"
        else:
            solver_configuration = "upstream_xy_then_full_pose"
    else:
        # The simultaneous ablation applies a moderate fixed quaternion cost
        # from the first control step and disables the upstream mode switch.
        replace_yaml_line(c3plus_options, "use_quaternion_dependent_cost", "false")
        quaternion_weight = f"{args.quaternion_weight:.12g}"
        joint_q_vector = (
            "[0.01, 0.01, 0.01, "
            + ", ".join([quaternion_weight] * 4)
            + ", 150, 150, 120, 15, 15, 10, "
            "0.05, 0.05, 0.05, 0.05, 0.05, 0.05]"
        )
        replace_yaml_line(c3plus_options, "q_vector", joint_q_vector)
        joint_q_vector_position = joint_q_vector.replace(
            ", 150, 150, 120,", ", 200, 200, 120,", 1
        )
        replace_yaml_line(
            c3plus_options,
            "q_vector_position",
            joint_q_vector_position,
        )
        replace_yaml_line(progress, "cost_switching_threshold_distance", "0.0")
        joint_pose_from_start = True
        quaternion_cost = {
            "type": "fixed_diagonal",
            "q_vector_weight": args.quaternion_weight,
            "state_dependent_hessian": False,
        }
        solver_configuration = "position_dynamics_with_joint_pose_cost"

    staged_manifest = {
        "schema": "nonprehensile.push_anything_stage.v1",
        "native_goal_mode": 2,
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
        "controller_inertial_mode": args.controller_inertial_mode,
        "controller_inertial": controller_inertial,
        "physical_inertial": physical_inertial,
        "end_effector": end_effector,
        "semantic_guard_clearance_m": args.semantic_guard_clearance,
        "semantic_guard_stop_distance_m": args.semantic_guard_stop_distance,
        "root_height_m": root_height,
        "object_height_m": full_height,
        "contact_height_m": contact_height,
        # The planner mesh has this support rotation baked into its vertices.
        # Carry it explicitly so a downstream simulator does not inherit an
        # unrelated planar yaw from its scene template.
        "support_quaternion_wxyz": manifest["support_quaternion_wxyz"],
        "initial_xy_m": [args.initial_x, args.initial_y],
        "initial_yaw_deg": args.initial_yaw_deg,
        "goal_xy_m": [goal_x, goal_y],
        "goal_distance_m": args.goal_distance,
        "goal_direction_deg": args.goal_direction_deg,
        "goal_yaw_deg": args.goal_yaw_deg,
        "reposition_speed_m_s": args.reposition_speed,
        "simulation_realtime_rate": args.realtime_rate,
        "sampling_seed": args.sampling_seed,
        "num_additional_samples_repos": args.num_additional_samples_repos,
        "num_additional_samples_c3": args.num_additional_samples_c3,
        "planning_horizon": args.planning_horizon,
        "progress_enforced_cost_drop": args.progress_enforced_cost_drop,
        "progress_enforced_over_n_loops": args.progress_enforced_over_n_loops,
        "controller_position_success_threshold_m": (
            args.controller_position_success_threshold
        ),
        "controller_orientation_success_threshold_rad": (
            args.controller_orientation_success_threshold
        ),
        "use_predicted_x0_repos": False,
        "objective_mode": args.objective_mode,
        "joint_pose_from_start": joint_pose_from_start,
        "upstream_cost_mode_switch_disabled": joint_pose_from_start,
        "solver_configuration": solver_configuration,
        "quaternion_cost": quaternion_cost,
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
