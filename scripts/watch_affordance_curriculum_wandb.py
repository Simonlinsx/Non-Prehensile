#!/usr/bin/env python3
"""Start one resumable TensorBoard-to-W&B sidecar per curriculum run."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


RUN_PATTERN = re.compile(r".*_seed(?P<seed>\d+)_stage(?P<stage>[0-3])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--entity", default="simonlsx")
    parser.add_argument("--project", default="isaaclab")
    parser.add_argument("--date-tag", default="20260821")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--sidecar-idle-seconds", type=float, default=1800.0)
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def discover_runs(log_root: Path) -> list[tuple[int, int, Path]]:
    runs = []
    for seed_root in sorted(log_root.glob("franka_affordance_curriculum_seed*")):
        if not seed_root.is_dir():
            continue
        for run_dir in sorted(seed_root.iterdir()):
            match = RUN_PATTERN.fullmatch(run_dir.name)
            if match is None or not list(run_dir.glob("events.out.tfevents.*")):
                continue
            runs.append((int(match["seed"]), int(match["stage"]), run_dir))
    return runs


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sync_script = repo_root / "scripts" / "sync_tensorboard_to_wandb.py"
    log_root = args.log_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started: set[Path] = set()
    children: dict[Path, tuple[subprocess.Popen, object]] = {}
    while True:
        for seed, stage, run_dir in discover_runs(log_root):
            if run_dir in started:
                continue
            run_id = f"parl-hammer-s{seed}-stage{stage}-{args.date_tag}"
            run_name = f"PARL hammer seed {seed} stage {stage}"
            sidecar_log = output_dir / f"wandb_seed{seed}_stage{stage}.log"
            stream = sidecar_log.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(sync_script),
                "--log-dir",
                str(run_dir),
                "--entity",
                args.entity,
                "--project",
                args.project,
                "--run-id",
                run_id,
                "--run-name",
                run_name,
                "--task",
                f"Isaac-AffordanceHammer-Stage{stage}-Franka-v0",
                "--num-envs",
                "1024",
                "--follow",
                "--poll-seconds",
                str(args.poll_seconds),
                "--idle-timeout-seconds",
                str(args.sidecar_idle_seconds),
            ]
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            started.add(run_dir)
            children[run_dir] = (process, stream)
            print(
                f"[watcher] seed={seed} stage={stage} pid={process.pid} "
                f"log={sidecar_log}",
                flush=True,
            )

        for run_dir, (process, stream) in list(children.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del children[run_dir]
            print(
                f"[watcher] sidecar exited code={return_code} run={run_dir.name}",
                flush=True,
            )

        if len(started) >= args.expected_runs and not children:
            print("[watcher] all expected curriculum runs synchronized", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
