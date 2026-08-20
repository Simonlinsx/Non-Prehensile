#!/usr/bin/env python3
"""Upload an existing TensorBoard scalar stream to a resumable W&B run.

This utility is intended to run as a sidecar for training jobs that were
started with the TensorBoard logger.  It never touches the training process or
its event file; it only tails scalar events and records the last uploaded
iteration in a small JSON state file.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import wandb
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# RSL-RL writes these duplicate charts against elapsed seconds instead of
# policy iterations.  Mixing them into an explicitly stepped W&B run would
# advance the global W&B step past the actual training iteration.
ELAPSED_TIME_TAG_SUFFIX = "/time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help="RSL-RL run directory containing tfevents.")
    parser.add_argument("--entity", required=True, help="W&B entity/account name.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--run-id", required=True, help="Stable W&B run ID used for resume.")
    parser.add_argument("--run-name", required=True, help="Human-readable W&B run name.")
    parser.add_argument("--task", default=None, help="Optional Isaac Lab task name added to the W&B config.")
    parser.add_argument("--num-envs", type=int, default=None, help="Optional environment count added to the config.")
    parser.add_argument("--follow", action="store_true", help="Continue following the event stream after backfill.")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Seconds between event-file reloads.")
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=600.0,
        help="In follow mode, exit after this long without a new scalar step; zero disables the timeout.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Resume-state JSON path (default: LOG_DIR/.wandb_tensorboard_sync.json).",
    )
    return parser.parse_args()


def load_agent_config(log_dir: Path) -> dict:
    config_path = log_dir / "params" / "agent.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config if isinstance(config, dict) else {}


def load_state(state_file: Path, run_id: str) -> int:
    if not state_file.is_file():
        return -1
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state_run_id = state.get("run_id")
    if state_run_id != run_id:
        raise ValueError(f"State file belongs to W&B run {state_run_id!r}, not {run_id!r}.")
    return int(state.get("last_step", -1))


def save_state(state_file: Path, run_id: str, last_step: int) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(f".{state_file.name}.tmp")
    temporary.write_text(
        json.dumps({"run_id": run_id, "last_step": last_step}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_file)


def collect_new_scalars(accumulator: EventAccumulator, last_step: int) -> list[tuple[int, dict[str, float]]]:
    accumulator.Reload()
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    for tag in accumulator.Tags().get("scalars", []):
        if tag.endswith(ELAPSED_TIME_TAG_SUFFIX):
            continue
        for event in accumulator.Scalars(tag):
            if event.step > last_step:
                by_step[event.step][tag] = event.value
    return sorted(by_step.items())


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory does not exist: {log_dir}")
    if not list(log_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"No TensorBoard event file found in: {log_dir}")

    state_file = args.state_file or log_dir / ".wandb_tensorboard_sync.json"
    state_file = state_file.expanduser().resolve()
    last_step = load_state(state_file, args.run_id)

    config = load_agent_config(log_dir)
    config.update(
        {
            "source_logger": "tensorboard-sidecar",
            "tensorboard_log_dir": str(log_dir),
        }
    )
    if args.task is not None:
        config["task"] = args.task
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs

    wandb_dir = log_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        job_type="policy-training",
        config=config,
        tags=["DAPL", "IsaacLab", "PPO", "TensorBoard-backfill"],
        dir=str(wandb_dir),
    )
    if run is None:
        raise RuntimeError("wandb.init() did not create a run.")

    agent_config_path = log_dir / "params" / "agent.yaml"
    if agent_config_path.is_file():
        run.save(str(agent_config_path), base_path=str(log_dir), policy="now")

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    idle_since = time.monotonic()
    print(f"[sync] W&B URL: {run.url}", flush=True)
    print(f"[sync] Resuming after TensorBoard step {last_step}", flush=True)

    try:
        while True:
            records = collect_new_scalars(accumulator, last_step)
            if records:
                for step, scalars in records:
                    run.log(scalars, step=step)
                    last_step = step
                save_state(state_file, args.run_id, last_step)
                run.summary["last_tensorboard_step"] = last_step
                idle_since = time.monotonic()
                print(f"[sync] Uploaded through TensorBoard step {last_step}", flush=True)

            if not args.follow:
                break
            if args.idle_timeout_seconds > 0 and time.monotonic() - idle_since >= args.idle_timeout_seconds:
                print(f"[sync] No new scalar step for {args.idle_timeout_seconds:g}s; exiting.", flush=True)
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
