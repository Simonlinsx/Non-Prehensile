#!/usr/bin/env python3
"""Resumable IsaacLab screen for the canonical Push Anything C1 route."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


CLOSED_GRIPPER_PROXY_FROM_HAND_M = 0.10427911200523377


def snapshot_sources(output_root, config, repo_root, native_binary):
    """Keep exact source bytes, so later edits do not erase trial provenance."""
    snapshot = output_root / "source_snapshot"
    items = [(repo_root / name, snapshot / name, digest)
             for name, digest in config["execution_source_sha256"].items()]
    items += [(native_binary, snapshot / "native/franka_sampling_c3_controller", config["native_binary_sha256"]),
              (Path(config["manifest"]), snapshot / "input_manifest.jsonl", config["manifest_sha256"])]
    for source, destination, digest in items:
        if digest is None:
            continue
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Source changed during snapshot: {source}")
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f"Existing source snapshot differs: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/data1/linsixu/dairlib-push-anything-canonical"),
    )
    parser.add_argument("--binary-root", type=Path)
    parser.add_argument(
        "--shared-physx-contact-export", type=Path,
        default=root / "data/contact_models/closed_franka_hammer_physx_v4.json",
        help="live collision export used to stage identical target/finger convexes in an isolated per-scene runtime",
    )
    parser.add_argument("--spatial-safe-sampling", action="store_true")
    parser.add_argument("--c3-admm-iterations", type=int, default=3)
    parser.add_argument("--c3-qp-max-iterations", type=int, default=200)
    parser.add_argument("--c3-nonnegative-contact-forces", action="store_true")
    parser.add_argument("--c3-quaternion-cost-weight", type=float, default=1000.)
    parser.add_argument("--c3-pd-rollout-interpolation", choices=("zoh", "foh"), default="zoh")
    parser.add_argument("--c3-relinearized-pd-cost", action="store_true")
    parser.add_argument("--c3-osc-matched-coarse-model", action="store_true")
    parser.add_argument("--enforce-planner-finger-floor", action="store_true")
    parser.add_argument("--planning-horizon", type=int, default=10)
    parser.add_argument("--c3-pd-rollout-kp", type=float, nargs=3)
    parser.add_argument("--c3-pd-rollout-kd", type=float, nargs=3)
    parser.add_argument("--c3-progress-window-loops", type=int)
    parser.add_argument("--c3-progress-cost-drop", type=float)
    parser.add_argument("--c3-final-contact-scaling-mode", choices=("ee", "all"), default="ee")
    parser.add_argument("--task-height-floor-mode", choices=("fixed", "finger-geometry"), default="fixed")
    parser.add_argument("--task-height-clearance-m", type=float, default=.002)
    parser.add_argument("--semantic-c1-guard-mode", choices=("disabled", "native-equivalent"), default="native-equivalent")
    parser.add_argument("--c3-force-action-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--c3-end-on-qp-step", action="store_true")
    parser.add_argument("--enforce-actor-workspace", action="store_true")
    parser.add_argument("--execution-height-mode", choices=("planar", "optimized"), default="planar")
    parser.add_argument("--legacy-sphere-contact-model", action="store_true", help="explicitly reproduce the historical 8 mm sphere approximation")
    parser.add_argument("--planner-clock-mode", choices=("wall", "simulation"), default="simulation",
                        help="simulation holds planner latency compensation at zero for the synchronous Isaac bridge")
    parser.add_argument(
        "--template-manifest",
        type=Path,
        default=root / "data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl",
    )
    parser.add_argument("--stage-python", type=Path, default=Path(
        "/data1/linsixu/miniconda3/envs/domino/bin/python"))
    parser.add_argument("--isaac-python", type=Path, default=Path(
        "/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python"))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--base-port", type=int, default=8500)
    parser.add_argument(
        "--max-sim-time-s",
        type=float,
        default=180.0,
        help=(
            "per-scene simulation timeout; 180 s matches the canonical Push "
            "Anything evaluation protocol"
        ),
    )
    parser.add_argument("--reposition-speed-m-s", type=float, default=0.18)
    parser.add_argument(
        "--objective-mode",
        choices=("push-anything", "full-pose"),
        default="push-anything",
    )
    parser.add_argument(
        "--physical-end-effector",
        choices=("closed_gripper", "push_anything_sphere"),
        default="closed_gripper",
    )
    parser.add_argument(
        "--local-controller",
        choices=("cartesian_impedance", "position_ik"),
        default="cartesian_impedance",
    )
    parser.add_argument("--osc-translation-stiffness", type=float, default=150.0)
    parser.add_argument("--osc-rotation-stiffness", type=float, default=25.0)
    parser.add_argument("--osc-damping-ratio", type=float, default=1.0)
    parser.add_argument("--osc-reference-velocity-mode", choices=("trajectory", "governed"), default="trajectory")
    parser.add_argument("--osc-dynamics-audit", action="store_true")
    parser.add_argument("--osc-control-point", choices=("hand", "tip"), default="tip")
    parser.add_argument(
        "--osc-track-trajectory-velocity", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--task-lookahead-ms",
        type=float,
        default=0.0,
        help=(
            "sample the fresh C3 task trajectory this far ahead of the "
            "measured-state timestamp; zero matches the native OSC clock"
        ),
    )
    parser.add_argument(
        "--force-c3-on-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "latch Push Anything's native FORCE_C3_MODE while measured legal "
            "safe-region contact persists"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--diagnostic-reference-replay-start-s", type=float,
      help="Planner clock seconds; Isaac timestamps include a 0.1 s origin offset")
    parser.add_argument("--diagnostic-reference-replay-duration-s", type=float, default=.675)
    parser.add_argument("--diagnostic-compare-pd-references", action="store_true")
    parser.add_argument("--diagnostic-pd-contact-relinearization", action="store_true")
    parser.add_argument("--diagnostic-pd-inertia-calibration", action="store_true", help="Read-only 8.2..8.8s scalar inertia observers; requires OSC 200/800 gains")
    parser.add_argument("--scene-id", action="append", dest="scene_ids")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--video", action="store_true")
    return parser.parse_args()


def load_scenes(path: Path) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            scene = json.loads(line)
            if scene.get("schema") not in {
                "nonprehensile.push_anything_c1_scene.v1",
                "nonprehensile.push_anything_c1_scene.v2",
            }:
                raise ValueError(f"invalid C1 scene at {path}:{line_number}")
            if scene.get("asset_id") != "020_hammer:0" or scene.get("clutter_count") != 0:
                raise ValueError(f"scene is outside the canonical C1 screen: {scene}")
            scenes.append(scene)
    if not scenes:
        raise ValueError(f"empty manifest: {path}")
    return scenes


def run_logged(command: list[str], log: Path, *, cwd: Path, env=None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def read_result(scene: dict[str, Any], run_dir: Path, status: int) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else {}
    )
    infrastructure_failure = not result_path.is_file() or result.get("error") is not None
    success = bool(result.get("online_closed_loop_success", False))
    return {
        "scene_id": scene["scene_id"],
        "goal_direction_deg": scene["goal_direction_deg"],
        "goal_direction_relative_deg": scene.get("goal_direction_relative_deg"),
        "goal_distance_m": scene["goal_distance_m"],
        "goal_yaw_deg": scene["goal_yaw_deg"],
        "sampling_seed": scene["sampling_seed"],
        "returncode": status,
        "infrastructure_failure": infrastructure_failure,
        "success": success,
        "c1_pass": not bool(result.get("forbidden_robot_contact_ever", True)),
        "final_planar_error_m": result.get("final_planar_error_m"),
        "final_rotation_error_rad": result.get("final_rotation_error_rad"),
        "executed_sim_time_s": result.get("executed_sim_time_s"),
        "stopped_reason": result.get("stopped_reason", result.get("error")),
        "run_dir": run_dir.name,
    }


def write_summary(output_root: Path, results: list[dict[str, Any]], total: int) -> None:
    evaluable = [item for item in results if not item["infrastructure_failure"]]
    successes = sum(int(item["success"]) for item in evaluable)
    c1_passes = sum(int(item["c1_pass"]) for item in evaluable)
    bins: dict[str, dict[str, int]] = {}
    for item in evaluable:
        angle = float(item["goal_direction_relative_deg"])
        name = "[-90,-45)" if angle < -45 else "[-45,0)" if angle < 0 else (
            "[0,45)" if angle < 45 else "[45,90]")
        bucket = bins.setdefault(name, {"attempted": 0, "successes": 0, "c1_passes": 0})
        bucket["attempted"] += 1
        bucket["successes"] += int(item["success"])
        bucket["c1_passes"] += int(item["c1_pass"])
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    summary = {
        "schema": "nonprehensile.push_anything_isaaclab_c1_summary.v1",
        "total_scenes": total,
        "attempted": len(results),
        "evaluable": len(evaluable),
        "infrastructure_failures": len(results) - len(evaluable),
        "successes": successes,
        "success_rate": successes / len(evaluable) if evaluable else None,
        "c1_passes": c1_passes,
        "c1_pass_rate": c1_passes / len(evaluable) if evaluable else None,
        "by_relative_direction_deg": bins,
        "failed_scene_ids": [item["scene_id"] for item in evaluable if not item["success"]],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.enforce_planner_finger_floor and (args.task_height_floor_mode != "finger-geometry" or args.execution_height_mode != "optimized"):
        raise ValueError("Planner finger floor requires matching geometric execution floor and optimized height")
    if args.c3_osc_matched_coarse_model and not args.c3_relinearized_pd_cost:
        raise ValueError("OSC matched coarse model requires relinearized PD cost")
    if args.planning_horizon < 2:
        raise ValueError("Planning horizon must provide at least two native reference knots")
    if args.c3_relinearized_pd_cost:
        if (args.c3_pd_rollout_interpolation != 'foh' or args.local_controller != 'cartesian_impedance' or
                args.osc_control_point != 'tip' or not args.osc_track_trajectory_velocity or
                args.osc_reference_velocity_mode != 'trajectory' or args.c3_force_action_sign != -1):
            raise ValueError('Relinearized PD cost requires FOH, tip OSC with trajectory velocity, and force sign -1')
        if args.diagnostic_compare_pd_references or args.diagnostic_pd_inertia_calibration:
            raise ValueError('Passive native-PD comparison assumes unchanged active cost; use a separate run')
    if args.diagnostic_pd_contact_relinearization and not args.diagnostic_pd_inertia_calibration:
        raise ValueError("Contact relinearization requires --diagnostic-pd-inertia-calibration")
    if (args.diagnostic_pd_inertia_calibration or args.c3_relinearized_pd_cost) and (args.osc_translation_stiffness != 200 or args.osc_rotation_stiffness != 800 or abs(args.osc_damping_ratio - 0.7071067811865476) > 1e-10):
        raise ValueError("PD inertia diagnostic requires OSC translation=200, rotation=800, damping ratio=sqrt(0.5)")
    if args.legacy_sphere_contact_model or args.physical_end_effector == "push_anything_sphere":
        args.shared_physx_contact_export = None
    repo_root = Path(__file__).resolve().parents[1]
    scenes = load_scenes(args.manifest.resolve())
    if args.scene_ids:
        wanted = set(args.scene_ids)
        scenes = [scene for scene in scenes if scene["scene_id"] in wanted]
        missing = wanted - {scene["scene_id"] for scene in scenes}
        if missing:
            raise ValueError(f"unknown scene ids: {sorted(missing)}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        scenes = scenes[: args.limit]
    if args.base_port < 1024 or args.base_port + 2 * len(scenes) >= 65535:
        raise ValueError("base port is invalid for this scene count")
    if args.reposition_speed_m_s <= 0 or args.max_sim_time_s <= 0:
        raise ValueError("speed and max sim time must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (
            args.osc_translation_stiffness, args.osc_rotation_stiffness,
            args.osc_damping_ratio)):
        raise ValueError("OSC stiffness and damping ratio must be finite and positive")
    for gains in (args.c3_pd_rollout_kp, args.c3_pd_rollout_kd):
        if gains is not None and any(not math.isfinite(value) or value <= 0 for value in gains):
            raise ValueError("PD rollout gains must be finite and positive")
    if args.c3_pd_rollout_interpolation != "zoh" and not args.shared_physx_contact_export:
        raise ValueError("FOH PD rollout requires shared contact staging")
    if (args.c3_pd_rollout_kp is not None or args.c3_pd_rollout_kd is not None) and not args.shared_physx_contact_export:
        raise ValueError("PD rollout gain overrides require shared contact staging")
    if args.c3_progress_window_loops is not None and not 2 <= args.c3_progress_window_loops <= 1000:
        raise ValueError("Progress window must contain 2--1000 planner updates")
    if args.c3_progress_cost_drop is not None and (
            not math.isfinite(args.c3_progress_cost_drop) or not 0 <= args.c3_progress_cost_drop < 1):
        raise ValueError("Progress cost drop must be finite in [0, 1)")
    if (args.c3_progress_window_loops is not None or args.c3_progress_cost_drop is not None) and not args.shared_physx_contact_export:
        raise ValueError("Progress overrides require shared contact staging")
    if args.diagnostic_reference_replay_start_s is not None:
        if (not math.isfinite(args.diagnostic_reference_replay_start_s)
                or args.diagnostic_reference_replay_start_s < 0
                or args.task_lookahead_ms != 0):
            raise ValueError("Diagnostic reference replay needs a finite nonnegative start and zero lookahead")
        if not (math.isfinite(args.diagnostic_reference_replay_duration_s)
                and 0 < args.diagnostic_reference_replay_duration_s <= min(.75, .075 * (args.planning_horizon - 1) + 1e-12)
                and args.diagnostic_reference_replay_start_s + args.diagnostic_reference_replay_duration_s < args.max_sim_time_s):
            raise ValueError("Diagnostic replay must fit the run and the published native reference horizon")
    if args.osc_dynamics_audit and (args.local_controller != "cartesian_impedance" or args.osc_control_point != "tip"):
        raise ValueError("OSC dynamics audit requires the task-point Cartesian impedance action")
    if args.task_lookahead_ms < 0:
        raise ValueError("task lookahead must be non-negative")
    if args.shared_physx_contact_export:
        if args.physical_end_effector != "closed_gripper" or args.local_controller != "cartesian_impedance":
            raise ValueError("shared finger geometry requires the stock closed gripper and Cartesian impedance")
        if not args.shared_physx_contact_export.is_file():
            raise FileNotFoundError(args.shared_physx_contact_export)
    elif args.planner_clock_mode != "wall":
        raise ValueError("legacy contact baselines require explicit --planner-clock-mode wall")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    binary_root = (args.binary_root or args.upstream_root).resolve()
    contact_acquisition_speed = (
        0.0 if args.local_controller == "cartesian_impedance" else 0.03
    )
    native_binary = binary_root / "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"
    source_paths = [repo_root / name for name in [
        "scripts/evaluate_push_anything_isaaclab_c1.py",
        "scripts/stage_domino_hammer_push_anything.py",
        "scripts/stage_shared_physx_contact_model.py",
        "scripts/generate_push_anything_isaaclab_manifest.py",
        "scripts/run_push_anything_isaaclab_online.sh",
        "scripts/run_c3_online_isaaclab_task.py",
        "source/IsaacLab_nonPrehensile/dapl/contact_planner/isaac_trajectory_action.py",
        "source/IsaacLab_nonPrehensile/dapl/contact_planner/isaac_bridge.py",
        "source/IsaacLab_nonPrehensile/dapl/contact_planner/c3_online_protocol.py",
        "third_party/push_anything/online_bridge_relay.py"]]
    source_paths += sorted((repo_root / "third_party/push_anything/patches").glob("*.patch"))
    source_paths = sorted(set(source_paths) | set((repo_root / "source/IsaacLab_nonPrehensile").rglob("*.py")))
    config = {
        "schema": "nonprehensile.push_anything_isaaclab_c1_config.v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "native_binary_sha256": hashlib.sha256(native_binary.read_bytes()).hexdigest() if native_binary.is_file() else None,
        "execution_source_sha256": {str(p.relative_to(repo_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "upstream_root": str(args.upstream_root.resolve()),
        "binary_root": str(binary_root),
        "objective": args.objective_mode,
        "planner": "Sampling+C3+ receding-horizon",
        "native_goal_mode": 2,
        "diagnostic_reference_replay_start_s": args.diagnostic_reference_replay_start_s,
        "diagnostic_reference_replay_duration_s": args.diagnostic_reference_replay_duration_s,
        "diagnostic_compare_pd_references": args.diagnostic_compare_pd_references,
        "diagnostic_pd_contact_relinearization": args.diagnostic_pd_contact_relinearization,
        "diagnostic_pd_inertia_calibration": args.diagnostic_pd_inertia_calibration,

        "physical_end_effector": args.physical_end_effector,
        "video": args.video,
        "planner_clock_mode": args.planner_clock_mode,
        "enforce_actor_workspace_bounds": args.enforce_actor_workspace,
        "c3_force_action_sign": args.c3_force_action_sign,
        "c3_admm_iterations": args.c3_admm_iterations,
        "c3_qp_max_iterations": args.c3_qp_max_iterations,
        "c3_nonnegative_contact_forces": args.c3_nonnegative_contact_forces,
        "c3_quaternion_cost_weight": args.c3_quaternion_cost_weight,
        "planning_horizon": args.planning_horizon,
        "c3_osc_matched_coarse_model": args.c3_osc_matched_coarse_model,
        "enforce_planner_finger_floor": args.enforce_planner_finger_floor,
        "c3_relinearized_pd_cost": args.c3_relinearized_pd_cost,
        "c3_pd_rollout_interpolation": args.c3_pd_rollout_interpolation,
        "c3_pd_rollout_kp": args.c3_pd_rollout_kp,
        "c3_pd_rollout_kd": args.c3_pd_rollout_kd,
        "c3_progress_window_loops": args.c3_progress_window_loops,
        "c3_progress_cost_drop": args.c3_progress_cost_drop,
        "c3_final_contact_scaling_mode": args.c3_final_contact_scaling_mode,
        "task_height_floor_mode": args.task_height_floor_mode,
        "task_height_clearance_m": args.task_height_clearance_m,
        "semantic_c1_guard_mode": args.semantic_c1_guard_mode,
        "spatial_safe_sampling": args.spatial_safe_sampling,
        "c3_end_on_qp_step": args.c3_end_on_qp_step,
        "execution_height_mode": args.execution_height_mode,
        "shared_physx_contact_export": str(args.shared_physx_contact_export.resolve()) if args.shared_physx_contact_export else None,
        "shared_physx_contact_export_sha256": hashlib.sha256(args.shared_physx_contact_export.read_bytes()).hexdigest() if args.shared_physx_contact_export else None,
        "local_controller": args.local_controller,
        "osc_translation_stiffness": args.osc_translation_stiffness,
        "osc_rotation_stiffness": args.osc_rotation_stiffness,
        "osc_damping_ratio": args.osc_damping_ratio,
        "osc_control_point": args.osc_control_point,
        "osc_track_trajectory_velocity": args.osc_track_trajectory_velocity,
        "osc_reference_velocity_mode": args.osc_reference_velocity_mode,
        "osc_dynamics_audit": args.osc_dynamics_audit,
        "task_lookahead_ms": args.task_lookahead_ms,
        "force_c3_on_contact": args.force_c3_on_contact,
        "reposition_speed_m_s": args.reposition_speed_m_s,
        "contact_acquisition_speed_m_s": contact_acquisition_speed,
        "disabled_experiments": [
            *([] if args.force_c3_on_contact else ["force_c3_on_contact"]),
            "position_force_feedforward", "semantic_push",
            "measured_yaw_brake", "pose_effect_filter", "signed_yaw_filter",
        ],
        "max_sim_time_s": args.max_sim_time_s,
        "gpu": args.gpu,
    }
    previous_config_path = output_root / "evaluation_config.json"
    if previous_config_path.is_file() and not args.rerun:
        previous = json.loads(previous_config_path.read_text())
        comparable_previous = {k: v for k, v in previous.items() if k != "gpu"}
        comparable_current = {k: v for k, v in config.items() if k != "gpu"}
        if comparable_previous != comparable_current:
            changed = sorted(k for k in comparable_previous.keys() | comparable_current.keys()
                             if comparable_previous.get(k) != comparable_current.get(k))
            raise ValueError(f"Cannot resume changed or unversioned configuration ({changed}); use a new output root")
    snapshot_sources(output_root, config, repo_root, native_binary)
    previous_config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        run_dir = output_root / scene["scene_id"]
        result_path = run_dir / "result.json"
        if result_path.is_file() and not args.rerun:
            item = read_result(scene, run_dir, 0)
            results.append(item)
            write_summary(output_root, results, len(scenes))
            print(f"RESUME {scene['scene_id']} success={int(item['success'])}", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"START {index + 1}/{len(scenes)} {scene['scene_id']} "
            f"relative={scene['goal_direction_relative_deg']:+.1f}deg "
            f"yaw={scene['goal_yaw_deg']:+.1f}deg",
            flush=True,
        )
        # Stage mutates the shared native checkout before copying its runtime.
        # Serialize the whole snapshot so concurrent objectives cannot mix.
        lock_path = args.upstream_root.resolve() / '.isaac_staging.lock'
        with lock_path.open('a') as staging_lock:
            fcntl.flock(staging_lock.fileno(), fcntl.LOCK_EX)
            stage = run_dir / "stage_manifest.json"
            stage_status = run_logged([
                str(args.stage_python), "scripts/stage_domino_hammer_push_anything.py",
                "--upstream-root", str(args.upstream_root.resolve()),
                "--generator-python", str(args.stage_python),
                "--initial-x", str(scene["initial_xy_m"][0]),
                "--initial-yaw-deg", str(scene.get("initial_yaw_deg", 0.0)),
                "--initial-y", str(scene["initial_xy_m"][1]),
                "--goal-distance", str(scene["goal_distance_m"]),
                "--goal-direction-deg", str(scene["goal_direction_deg"]),
                "--goal-yaw-deg", str(scene["goal_yaw_deg"]),
                "--objective-mode", args.objective_mode,
                "--end-effector-mode", (
                    "closed-gripper-proxy"
                    if args.physical_end_effector == "closed_gripper"
                    else "push-anything-sphere"
                ),
                "--reposition-speed", str(args.reposition_speed_m_s),
                "--num-additional-samples-repos", "5",
                "--num-additional-samples-c3", "5",
                "--planning-horizon", str(args.planning_horizon),
                "--sampling-seed", str(scene["sampling_seed"]),
                "--output-manifest", str(stage),
            ], run_dir / "stage.log", cwd=repo_root)

            isaac_manifest = run_dir / "isaac_manifest.jsonl"
            bridge_env = os.environ.copy()
            bridge_env["PYTHONPATH"] = str(repo_root / "source/IsaacLab_nonPrehensile")
            bridge_status = 1
            if stage_status == 0:
                bridge_status = run_logged([
                    str(args.isaac_python), "scripts/generate_push_anything_isaaclab_manifest.py",
                    "--stage-manifest", str(stage),
                    "--template-manifest", str(args.template_manifest.resolve()),
                    "--output", str(isaac_manifest),
                    "--scene-id", scene["scene_id"],
                ], run_dir / "bridge.log", cwd=repo_root, env=bridge_env)

            execution_runtime = args.upstream_root.resolve()
            geometry_status = None
            if bridge_status == 0 and args.shared_physx_contact_export:
                execution_runtime = run_dir / f"shared_runtime_{time.time_ns()}"
                geometry_status = run_logged([
                    str(args.stage_python), "scripts/stage_shared_physx_contact_model.py",
                    "--export", str(args.shared_physx_contact_export.resolve()),
                    "--stage-manifest", str(stage),
                    "--upstream-root", str(args.upstream_root.resolve()),
                    "--runtime", str(execution_runtime),
                    "--planner-clock-mode", args.planner_clock_mode,
                    "--execution-height-mode", args.execution_height_mode,
                    "--c3-admm-iterations", str(args.c3_admm_iterations),
                    "--c3-qp-max-iterations", str(args.c3_qp_max_iterations),
                    *(["--c3-nonnegative-contact-forces"] if args.c3_nonnegative_contact_forces else []),
                    "--c3-quaternion-cost-weight", str(args.c3_quaternion_cost_weight),
                    "--c3-pd-rollout-interpolation", args.c3_pd_rollout_interpolation,
                    *(["--c3-relinearized-pd-cost"] if args.c3_relinearized_pd_cost else []),
                    *(["--c3-osc-matched-coarse-model"] if args.c3_osc_matched_coarse_model else []),
                    *(["--planner-finger-table-clearance", str(args.task_height_clearance_m)] if args.enforce_planner_finger_floor else []),
                    *(["--c3-pd-rollout-kp", *map(str, args.c3_pd_rollout_kp)]
                      if args.c3_pd_rollout_kp is not None else []),
                    *(["--c3-pd-rollout-kd", *map(str, args.c3_pd_rollout_kd)]
                      if args.c3_pd_rollout_kd is not None else []),
                    *(["--c3-progress-window-loops", str(args.c3_progress_window_loops)]
                      if args.c3_progress_window_loops is not None else []),
                    *(["--c3-progress-cost-drop", str(args.c3_progress_cost_drop)]
                      if args.c3_progress_cost_drop is not None else []),
                    "--c3-final-contact-scaling-mode", args.c3_final_contact_scaling_mode,
                    *(["--c3-end-on-qp-step"] if args.c3_end_on_qp_step else []),
                    *(["--spatial-safe-sampling"] if args.spatial_safe_sampling else []),
                    *(["--enforce-actor-workspace"] if args.enforce_actor_workspace else []),
                ], run_dir / "shared_geometry_stage.log", cwd=repo_root)
        status = 125
        start = time.monotonic()
        if bridge_status == 0 and geometry_status in (None, 0):
            controller_params = yaml.safe_load((execution_runtime /
                "examples/sampling_c3/anything/parameters/sampling_c3_controller_params.yaml").read_text())
            if bool(controller_params.get("use_foh_pd_rollout", False)) != (args.c3_pd_rollout_interpolation == "foh"):
                raise ValueError("Runtime PD interpolation does not match evaluation configuration")
            expected_planner_clearance = args.task_height_clearance_m if args.enforce_planner_finger_floor else None
            if controller_params.get("planner_finger_table_clearance") != expected_planner_clearance:
                raise ValueError("Runtime planner finger clearance does not match execution configuration")
            if bool(controller_params.get("use_osc_matched_coarse_model", False)) != args.c3_osc_matched_coarse_model:
                raise ValueError("Runtime coarse model does not match evaluation configuration")
            if bool(controller_params.get("use_relinearized_pd_cost", False)) != args.c3_relinearized_pd_cost:
                raise ValueError("Runtime relinearized PD cost does not match evaluation configuration")
            if args.c3_relinearized_pd_cost:
                progress = yaml.safe_load((execution_runtime / controller_params["progress_params_file"]).read_text())
                if progress.get('cost_type') != 5 or progress.get('cost_type_position') != 5:
                    raise ValueError('Relinearized model supports existing object-only PD cost (type 5)')
            sampling_options = yaml.safe_load((execution_runtime / controller_params["sampling_c3_options_file"]).read_text())
            if sampling_options.get("N") != args.planning_horizon:
                raise ValueError("Runtime planning horizon does not match evaluation configuration")
            goal_params = yaml.safe_load((execution_runtime / controller_params["goal_params_file"]).read_text())
            if goal_params.get("goal_mode") != 2:
                raise ValueError("Fixed-scene evaluation requires native goal_mode=2")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": args.gpu,
                "PUSH_ANYTHING_DIAGNOSTIC_PD_RELINEARIZE": "1" if args.diagnostic_pd_contact_relinearization else "0",
                "PUSH_ANYTHING_DIAGNOSTIC_PD_CALIBRATION": "1" if args.diagnostic_pd_inertia_calibration else "0",
                "PUSH_ANYTHING_DIAGNOSTIC_PD_COMPARE": "1" if args.diagnostic_compare_pd_references else "0",
                "PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_START_S": (
                    "" if args.diagnostic_reference_replay_start_s is None else str(args.diagnostic_reference_replay_start_s)),
                "PUSH_ANYTHING_DIAGNOSTIC_REFERENCE_REPLAY_DURATION_S": str(args.diagnostic_reference_replay_duration_s),
                "PUSH_ANYTHING_ROOT": str(execution_runtime),
                "PUSH_ANYTHING_BINARY_ROOT": str(binary_root),
                "PUSH_ANYTHING_DEMO_NAME": "anything",
                "PUSH_ANYTHING_TCPQ_PORT": str(args.base_port + 2 * index),
                "PUSH_ANYTHING_RELAY_PORT": str(args.base_port + 2 * index + 1),
                "PUSH_ANYTHING_TASK_LOOKAHEAD_MS": str(args.task_lookahead_ms),
            })
            command = [
                "bash", "scripts/run_push_anything_isaaclab_online.sh",
                str(isaac_manifest), str(result_path),
                "--planner-frequency-hz", "20",
                "--max-sim-time-s", str(args.max_sim_time_s),
                "--local-controller", args.local_controller,
                "--osc-translation-stiffness",
                str(args.osc_translation_stiffness),
                "--osc-rotation-stiffness",
                str(args.osc_rotation_stiffness),
                "--osc-damping-ratio", str(args.osc_damping_ratio),
                "--osc-control-point", args.osc_control_point,
                "--task-height-floor-mode", args.task_height_floor_mode,
                "--task-height-clearance-m", str(args.task_height_clearance_m),
                "--semantic-c1-guard-mode", args.semantic_c1_guard_mode,
                "--c3-force-action-sign", str(args.c3_force_action_sign),
                "--osc-reference-velocity-mode", args.osc_reference_velocity_mode,
                *(["--osc-dynamics-audit"] if args.osc_dynamics_audit else []),
                (
                    "--osc-track-trajectory-velocity"
                    if args.osc_track_trajectory_velocity
                    else "--no-osc-track-trajectory-velocity"
                ),
                (
                    "--force-c3-on-legal-safe-contact"
                    if args.force_c3_on_contact
                    else "--no-force-c3-on-legal-safe-contact"
                ),
                "--contact-acquisition-speed-m-s", str(contact_acquisition_speed),
                "--position-feedforward-compliance-m-per-n", "0",
                "--semantic-contact-push-speed-m-s", "0",
                "--no-measured-c3-yaw-brake",
                "--execution-rotation-guard-limit-rad", "3",
                "--headless", "--device", "cuda:0",
            ]
            if args.physical_end_effector == "closed_gripper":
                command.extend([
                    "--no-use-push-anything-end-effector",
                    "--drake-tip-from-hand-m",
                    str(CLOSED_GRIPPER_PROXY_FROM_HAND_M),
                ])
            else:
                # The online runner defaults to the deployable closed-gripper
                # setup.  An upstream spherical-pusher baseline must opt in
                # explicitly or it silently combines the sphere C3 model with
                # the shorter closed-gripper control frame.
                command.extend([
                    "--use-push-anything-end-effector",
                    "--drake-tip-from-hand-m", "0.1265",
                ])
            if args.local_controller == "cartesian_impedance":
                command.extend([
                    "--osc-nullspace-target", "push_anything_joint2",
                    "--osc-inertial-dynamics-decoupling",
                ])
                if args.physical_end_effector == "closed_gripper":
                    # C3 optimizes translation and holds measured finger
                    # orientation over each prediction horizon. Keep wrist
                    # orientation tracking enabled for that approximation.
                    command.append("--osc-track-orientation")
            if args.video:
                command.extend([
                    "--video", "--video-folder", str(run_dir / "videos"),
                    "--video-name-prefix", scene["scene_id"],
                ])
            status = run_logged(command, run_dir / "runner.log", cwd=repo_root, env=env)
        item = read_result(scene, run_dir, status)
        item.update({
            "stage_returncode": stage_status,
            "bridge_returncode": bridge_status,
            "shared_geometry_returncode": geometry_status,
            "execution_runtime": str(execution_runtime),
            "wall_time_s": time.monotonic() - start,
        })
        results.append(item)
        write_summary(output_root, results, len(scenes))
        print(
            f"DONE {scene['scene_id']} success={int(item['success'])} "
            f"c1={int(item['c1_pass'])} reason={item['stopped_reason']}",
            flush=True,
        )
    return 0 if all(item["success"] for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
