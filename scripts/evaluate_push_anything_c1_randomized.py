#!/usr/bin/env python3
"""Run and incrementally summarize a resumable Push Anything C1 scene set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            repo_root
            / "data/manifests/contact_planner_m3"
            / "hammer_c1_front180_eval50_seed20260901.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repo_root
            / "outputs/contact_planner_m3/hammer_c1_front180_eval50_seed20260901"
        ),
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/data1/linsixu/dairlib-push-anything"),
    )
    parser.add_argument(
        "--stage-python",
        type=Path,
        default=Path("/data1/linsixu/miniconda3/envs/domino/bin/python"),
    )
    parser.add_argument(
        "--audit-python",
        type=Path,
        default=Path("/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python"),
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--tcpq-port", type=int, default=7730)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    scenes = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            scene = json.loads(line)
            if scene.get("schema") != "nonprehensile.push_anything_c1_scene.v1":
                raise ValueError(f"invalid schema at {path}:{line_number}")
            if scene.get("asset_id") != "020_hammer:0" or scene.get("clutter_count") != 0:
                raise ValueError(f"scene outside the C1 single-hammer gate: {scene}")
            scenes.append(scene)
    if not scenes:
        raise ValueError(f"empty manifest: {path}")
    scene_ids = [scene["scene_id"] for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("manifest contains duplicate scene IDs")
    return scenes


def run_logged(command: list[str], log_path: Path, **kwargs: Any) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            **kwargs,
        )
    return result.returncode


def result_from_artifacts(scene: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    geometry_path = run_dir / "acceptance.json"
    c1_path = run_dir / "c1_semantic_audit.json"
    joint_path = run_dir / "joint_acceptance.json"
    geometry = load_json(geometry_path) if geometry_path.is_file() else {}
    c1 = load_json(c1_path) if c1_path.is_file() else {}
    joint = load_json(joint_path) if joint_path.is_file() else {}
    return {
        "scene_id": scene["scene_id"],
        "initial_xy_m": scene["initial_xy_m"],
        "goal_xy_m": scene["goal_xy_m"],
        "goal_distance_m": scene["goal_distance_m"],
        "goal_direction_deg": scene["goal_direction_deg"],
        "goal_yaw_deg": scene["goal_yaw_deg"],
        "sampling_seed": scene["sampling_seed"],
        "geometry_pass": bool(geometry.get("accepted", False)),
        "c1_pass": bool(c1.get("c1_pass", False)),
        "accepted": bool(joint.get("accepted", False)),
        "final_position_error_m": geometry.get("final_position_error_m"),
        "final_rotation_error_rad": geometry.get("final_rotation_error_rad"),
        "accepted_at_s": geometry.get("accepted_at_s"),
        "legal_safe_contact_rows": c1.get("legal_safe_contact_rows"),
        "c1_violation_rows": c1.get("c1_violation_rows"),
        "run_dir": run_dir.name,
    }


def direction_bin(angle: float) -> str:
    if angle < -45.0:
        return "[-90,-45)"
    if angle < 0.0:
        return "[-45,0)"
    if angle < 45.0:
        return "[0,45)"
    return "[45,90]"


def summarize(results: list[dict[str, Any]], total_scenes: int) -> dict[str, Any]:
    bins: dict[str, dict[str, int]] = {}
    for result in results:
        name = direction_bin(float(result["goal_direction_deg"]))
        bucket = bins.setdefault(name, {"attempted": 0, "accepted": 0})
        bucket["attempted"] += 1
        bucket["accepted"] += int(result["accepted"])
    attempted = len(results)
    geometry_successes = sum(int(item["geometry_pass"]) for item in results)
    c1_successes = sum(int(item["c1_pass"]) for item in results)
    joint_successes = sum(int(item["accepted"]) for item in results)
    return {
        "schema": "nonprehensile.push_anything_c1_eval_summary.v1",
        "total_scenes": total_scenes,
        "attempted": attempted,
        "remaining": total_scenes - attempted,
        "geometry_successes": geometry_successes,
        "c1_successes": c1_successes,
        "joint_successes": joint_successes,
        "geometry_success_rate": geometry_successes / attempted if attempted else None,
        "c1_success_rate": c1_successes / attempted if attempted else None,
        "joint_success_rate": joint_successes / attempted if attempted else None,
        "by_direction_deg": bins,
        "failed_scene_ids": [
            item["scene_id"] for item in results if not item["accepted"]
        ],
    }


def write_progress(
    output_root: Path, results: list[dict[str, Any]], total_scenes: int
) -> dict[str, Any]:
    results_path = output_root / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    summary = summarize(results, total_scenes)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.timeout_s <= 0.0 or not 1 <= args.tcpq_port <= 65535:
        raise ValueError("timeout/port is invalid")
    repo_root = Path(__file__).resolve().parents[1]
    scenes = load_manifest(args.manifest.resolve())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        scenes = scenes[: args.limit]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation_config.json").write_text(
        json.dumps(
            {
                "schema": "nonprehensile.push_anything_c1_eval_config.v1",
                "manifest": str(args.manifest.resolve()),
                "scene_count": len(scenes),
                "timeout_s": args.timeout_s,
                "tcpq_port": args.tcpq_port,
                "upstream_root": str(args.upstream_root.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.build:
        first = scenes[0]
        subprocess.run(
            [
                str(args.stage_python),
                "scripts/stage_domino_hammer_push_anything.py",
                "--upstream-root", str(args.upstream_root),
                "--generator-python", str(args.stage_python),
                "--initial-x", str(first["initial_xy_m"][0]),
                "--initial-y", str(first["initial_xy_m"][1]),
                "--goal-distance", str(first["goal_distance_m"]),
                "--goal-direction-deg", str(first["goal_direction_deg"]),
                "--goal-yaw-deg", str(first["goal_yaw_deg"]),
                "--quaternion-weight", "5",
                "--sampling-seed", str(first["sampling_seed"]),
                "--output-manifest", str(output_root / "initial_stage_manifest.json"),
            ],
            cwd=repo_root,
            check=True,
        )
        build_environment = os.environ.copy()
        build_environment["PUSH_ANYTHING_ROOT"] = str(args.upstream_root)
        subprocess.run(
            ["bash", "scripts/build_push_anything_native.sh", "--build"],
            cwd=repo_root,
            env=build_environment,
            check=True,
        )

    results = []
    for scene_index, scene in enumerate(scenes, start=1):
        run_dir = output_root / scene["scene_id"]
        joint_path = run_dir / "joint_acceptance.json"
        if joint_path.is_file() and not args.rerun:
            result = result_from_artifacts(scene, run_dir)
            results.append(result)
            summary = write_progress(output_root, results, len(scenes))
            print(
                f"RESUME {scene['scene_id']} "
                f"accepted={result['accepted']} "
                f"joint={summary['joint_successes']}/{summary['attempted']}",
                flush=True,
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"START {scene_index}/{len(scenes)} {scene['scene_id']} "
            f"dir={scene['goal_direction_deg']}deg "
            f"yaw={scene['goal_yaw_deg']}deg",
            flush=True,
        )
        start = time.monotonic()
        stage_command = [
            str(args.stage_python),
            "scripts/stage_domino_hammer_push_anything.py",
            "--upstream-root", str(args.upstream_root),
            "--generator-python", str(args.stage_python),
            "--initial-x", str(scene["initial_xy_m"][0]),
            "--initial-y", str(scene["initial_xy_m"][1]),
            "--goal-distance", str(scene["goal_distance_m"]),
            "--goal-direction-deg", str(scene["goal_direction_deg"]),
            "--goal-yaw-deg", str(scene["goal_yaw_deg"]),
            "--quaternion-weight", "5",
            "--sampling-seed", str(scene["sampling_seed"]),
            "--semantic-guard-clearance", "0.025",
            "--semantic-guard-stop-distance", "0.055",
            "--output-manifest", str(run_dir / "stage_manifest.json"),
        ]
        stage_status = run_logged(
            stage_command, run_dir / "stage.log", cwd=repo_root
        )
        geometry_status = 1
        semantic_status = 1
        joint_status = 1
        if stage_status == 0:
            environment = os.environ.copy()
            environment.update(
                {
                    "PUSH_ANYTHING_ROOT": str(args.upstream_root),
                    "PUSH_ANYTHING_RUN_NAME": scene["scene_id"],
                    "PUSH_ANYTHING_OUTPUT_DIR": str(run_dir),
                    "PUSH_ANYTHING_TIMEOUT_S": str(args.timeout_s),
                    "PUSH_ANYTHING_TCPQ_PORT": str(args.tcpq_port),
                }
            )
            geometry_status = run_logged(
                ["bash", "scripts/run_push_anything_native_baseline.sh"],
                run_dir / "runner.log",
                cwd=repo_root,
                env=environment,
            )
        trajectory_path = run_dir / "sampling_c3_debug.csv"
        if trajectory_path.is_file():
            semantic_status = run_logged(
                [
                    str(args.audit_python),
                    "scripts/audit_push_anything_c1.py",
                    "--trajectory-csv", str(trajectory_path),
                    "--semantic-dir", "data/push_anything_semantics/020_hammer_0",
                    "--output-json", str(run_dir / "c1_semantic_audit.json"),
                    "--output-csv", str(run_dir / "c1_semantic_audit.csv"),
                ],
                run_dir / "audit.log",
                cwd=repo_root,
            )
        if (run_dir / "acceptance.json").is_file() and (
            run_dir / "c1_semantic_audit.json"
        ).is_file():
            joint_status = run_logged(
                [
                    str(args.audit_python),
                    "scripts/verify_push_anything_c1_acceptance.py",
                    "--run-dir", str(run_dir),
                ],
                run_dir / "verify.log",
                cwd=repo_root,
            )

        result = result_from_artifacts(scene, run_dir)
        result.update(
            {
                "stage_status": stage_status,
                "geometry_status": geometry_status,
                "semantic_status": semantic_status,
                "joint_status": joint_status,
                "wall_time_s": time.monotonic() - start,
            }
        )
        (run_dir / "scene_result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        summary = write_progress(output_root, results, len(scenes))
        print(
            f"DONE {scene['scene_id']} accepted={result['accepted']} "
            f"geometry={result['geometry_pass']} c1={result['c1_pass']} "
            f"joint={summary['joint_successes']}/{summary['attempted']}",
            flush=True,
        )
        time.sleep(0.5)

    summary = write_progress(output_root, results, len(scenes))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["joint_successes"] == len(scenes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
