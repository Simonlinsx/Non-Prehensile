"""Run a fixed shard once, from verified FR3 controller snapshots; never select reruns."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess

try:
    from .fr3_frozen_batch import sha256, verify_batch, verify_scene
except ImportError:
    from fr3_frozen_batch import sha256, verify_batch, verify_scene


def command_for(root, scene, port):
    return ["bash", str(root / "execution_source/scripts/run_push_anything_isaaclab_online.sh"),
            str(scene / "isaac_manifest.jsonl"), str(scene / "result.json"),
            "--stock-closed-gripper", "--headless", "--device", "cuda:0",
            "--max-sim-time-s", "180", "--physics-dt-s", "0.001", "--control-decimation", "1",
            "--trace-stride", "10", "--socket-timeout-s", "10"]


def execution_env(root, scene, freeze, gpu, port):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PUSH_ANYTHING_", "DAPL_"))}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), ISAACLAB_PYTHON=freeze["python_executable"],
        PUSH_ANYTHING_ROOT=str(scene / "runtime"), PUSH_ANYTHING_BINARY_ROOT=str(root / "binary"),
        PUSH_ANYTHING_EXECUTOR_MODE="effort", PUSH_ANYTHING_TCPQ_QUICK_ACK="1",
        PUSH_ANYTHING_PLANNER_PERIOD_MS="50", PUSH_ANYTHING_FRESH_TASK_TIMEOUT_MS="10000",
        PUSH_ANYTHING_ROBOT_MODEL_MANIFEST=freeze["robot_model_manifest"],
        PUSH_ANYTHING_DIAGNOSTIC_OSC_HOLD="0", PUSH_ANYTHING_FRESH_EFFORT_TIMEOUT_MS="1000",
        PUSH_ANYTHING_TCPQ_PORT=str(port), PUSH_ANYTHING_RELAY_PORT=str(port + 1),
        PUSH_ANYTHING_RELAY_SOCKET_TIMEOUT_S="60")
    return env


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args):
    root = args.batch_root.resolve()
    if not 0 <= args.shard < args.shards or args.shards < 1:
        raise ValueError("Invalid fixed shard")
    freeze, protocol, scenes, semantic = verify_batch(root, args.freeze_sha256)
    with (root / f"worker{args.shard}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, specification in enumerate(scenes):
            if index % args.shards != args.shard:
                continue
            scene = verify_scene(root, freeze, specification, semantic)
            status_path = scene / "execution.json"
            if status_path.exists():
                previous = json.loads(status_path.read_text())
                if previous.get("status") == "finished":
                    print(f"SKIP completed {specification['scene_id']}; no rerun", flush=True)
                    continue
                raise RuntimeError(f"Existing unfinished attempt must be inspected, never rerun: {scene}")
            if (scene / "result.json").exists():
                raise RuntimeError("Unattributed result already exists")
            # Revalidate common source/binaries/external inputs before every launch.
            verify_batch(root, args.freeze_sha256)
            port = args.base_port + 2 * index
            for value in (port, port + 1):
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", value))
            command = command_for(root, scene, port)
            env = execution_env(root, scene, freeze, args.gpu, port)
            state = dict(schema="nonprehensile.fr3_formal_execution.v1", status="starting",
                         scene_id=specification["scene_id"], started_utc=now(),
                         freeze_sha256=args.freeze_sha256, worker_pid=os.getpid(),
                         shard=args.shard, shards=args.shards, gpu=str(args.gpu), command=command,
                         environment={k: v for k, v in env.items() if k.startswith("PUSH_ANYTHING_") or k in ("CUDA_VISIBLE_DEVICES", "ISAACLAB_PYTHON")},
                         worker_source_sha256=sha256(Path(__file__)),
                         verification_source_sha256=sha256(Path(__file__).with_name("fr3_frozen_batch.py")))
            with status_path.open("x") as stream:
                json.dump(state, stream, indent=2)
            print(f"START {specification['scene_id']} gpu={args.gpu}", flush=True)
            with (scene / "runner.log").open("x") as log:
                process = subprocess.Popen(command, cwd=root / "execution_source", env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status="running", runner_pid=process.pid)
                status_path.write_text(json.dumps(state, indent=2) + "\n")
                try:
                    code = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    state.update(status="interrupted", ended_utc=now())
                    status_path.write_text(json.dumps(state, indent=2) + "\n")
                    raise
            state.update(status="finished", exit_code=code, ended_utc=now())
            if (scene / "result.json").exists():
                state["result_sha256"] = sha256(scene / "result.json")
            status_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"DONE {specification['scene_id']} exit={code}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=21000)
    run(parser.parse_args())
