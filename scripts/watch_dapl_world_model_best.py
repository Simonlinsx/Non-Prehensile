"""Preserve each atomically published best checkpoint from a running DAPL job."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import time

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--poll-seconds", type=float, default=5.0)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    source = run_dir / "world_model_best.pt"
    candidates = run_dir / "best_candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    previous_identity: tuple[int, int, int] | None = None

    while True:
        if source.is_file():
            stat = source.stat()
            identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if identity != previous_identity:
                payload = torch.load(source, map_location="cpu", weights_only=False)
                if payload.get("schema_version") != 1:
                    raise ValueError(f"unsupported checkpoint schema in {source}")
                step = int(payload["step"])
                destination = candidates / f"world_model_best_step_{step:07d}.pt"
                if not destination.exists():
                    temporary = destination.with_suffix(".pt.tmp")
                    shutil.copy2(source, temporary)
                    copied = torch.load(temporary, map_location="cpu", weights_only=False)
                    if int(copied["step"]) != step:
                        temporary.unlink(missing_ok=True)
                        # The source changed between load and copy. Retry on the
                        # next poll rather than publishing a mislabeled file.
                        previous_identity = None
                        time.sleep(args.poll_seconds)
                        continue
                    temporary.replace(destination)
                    print(
                        "DAPL_WORLD_MODEL_BEST_CANDIDATE",
                        f"step={step}",
                        f"sha256={_sha256(destination)}",
                        f"path={destination}",
                        flush=True,
                    )
                previous_identity = identity

        if (run_dir / "summary.json").is_file():
            break
        time.sleep(args.poll_seconds)

    print("DAPL_WORLD_MODEL_BEST_WATCH_OK", f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
