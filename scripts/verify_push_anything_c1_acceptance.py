#!/usr/bin/env python3
"""Combine geometric and semantic Push Anything C1 acceptance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required acceptance artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def combine_acceptance(run_dir: Path) -> dict[str, object]:
    geometry_path = run_dir / "acceptance.json"
    semantic_path = run_dir / "c1_semantic_audit.json"
    geometry = load_json(geometry_path)
    semantic = load_json(semantic_path)

    geometry_pass = bool(geometry.get("accepted", False))
    legal_contacts = int(semantic.get("legal_safe_contact_rows", 0))
    protected_contacts = int(semantic.get("protected_contact_rows", -1))
    neutral_contacts = int(semantic.get("neutral_contact_rows", -1))
    violations = int(semantic.get("c1_violation_rows", -1))
    c1_pass = (
        bool(semantic.get("c1_pass", False))
        and legal_contacts > 0
        and protected_contacts == 0
        and neutral_contacts == 0
        and violations == 0
    )
    return {
        "schema": "nonprehensile.push_anything_joint_acceptance.v1",
        "run_name": run_dir.name,
        "geometry_pass": geometry_pass,
        "c1_pass": c1_pass,
        "accepted": geometry_pass and c1_pass,
        "pose_gate": {
            "position_threshold_m": geometry.get("position_threshold_m"),
            "rotation_threshold_rad": geometry.get("rotation_threshold_rad"),
            "dwell_messages_required": geometry.get("dwell_messages_required"),
            "final_position_error_m": geometry.get("final_position_error_m"),
            "final_rotation_error_rad": geometry.get("final_rotation_error_rad"),
            "accepted_at_s": geometry.get("accepted_at_s"),
        },
        "c1_gate": {
            "contact_threshold_m": semantic.get("contact_threshold_m"),
            "legal_safe_contact_rows": semantic.get("legal_safe_contact_rows"),
            "protected_contact_rows": semantic.get("protected_contact_rows"),
            "neutral_contact_rows": semantic.get("neutral_contact_rows"),
            "c1_violation_rows": semantic.get("c1_violation_rows"),
            "minimum_surface_distance_m": semantic.get(
                "minimum_surface_distance_m"
            ),
            "trajectory_sha256": semantic.get("trajectory_sha256"),
        },
        "artifacts": {
            "geometry": geometry_path.name,
            "semantic": semantic_path.name,
            "trajectory": "sampling_c3_debug.csv",
            "semantic_frames": "c1_semantic_audit.csv",
        },
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = combine_acceptance(run_dir)
    output = args.output_json or run_dir / "joint_acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
