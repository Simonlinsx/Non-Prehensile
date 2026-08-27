#!/usr/bin/env python3
"""Summarize balanced teacher evaluations by goal-direction bin."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--eval-root", type=Path, default=Path("outputs/teacher_eval")
)
parser.add_argument(
    "--run-prefix",
    action="append",
    required=True,
    help="Evaluation directory prefix; repeat to compare multiple runs.",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--endpoint-min-abs-deg",
    type=float,
    default=70.0,
    help="Treat directions with |angle| at least this value as audited endpoints.",
)
parser.add_argument(
    "--require-negative-endpoint",
    action="store_true",
    help="Require both signed endpoint bins instead of only the positive endpoint.",
)
parser.add_argument(
    "--skip-bidirectional-yaw-gate",
    action="store_true",
    help="Report yaw-sign bins but do not require both signs for checkpoint eligibility.",
)
args = parser.parse_args()

if not 0.0 < args.endpoint_min_abs_deg <= 90.0:
    parser.error("--endpoint-min-abs-deg must be in (0, 90]")


DIRECTION_BINS = (
    ("negative_endpoint", -90.0001, -70.0),
    ("negative_mid", -70.0, -35.0),
    ("near_forward_negative", -35.0, 0.0),
    ("near_forward_positive", 0.0, 35.0),
    ("positive_mid", 35.0, 70.0),
    ("positive_endpoint", 70.0, 90.0001),
)
MIN_OVERALL_CONSTRAINED_SUCCESS_RATE = 0.85
MAX_OVERALL_C1_VIOLATION_RATE = 0.01
MIN_POSITIVE_ENDPOINT_CONSTRAINED_SUCCESS_RATE = 0.75
MIN_EACH_YAW_SIGN_CONSTRAINED_SUCCESS_RATE = 0.75


def _checkpoint_index(path: Path, summary: dict) -> int:
    checkpoint = Path(str(summary["checkpoint"])).stem
    match = re.fullmatch(r"model_(\d+)", checkpoint)
    if match is None:
        match = re.search(r"model[_-]?(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot resolve checkpoint index from {path}")
    return int(match.group(1))


def _empty_bin() -> dict[str, float | int]:
    return {
        "episodes": 0,
        "successes": 0,
        "constrained_successes": 0,
        "legal_safe_contact_episodes": 0.0,
        "legal_safe_contact_metric_episodes": 0,
        "c1_violation_episodes": 0.0,
        "terminal_planar_error_sum_m": 0.0,
        "terminal_rotation_error_sum_rad": 0.0,
        "terminal_signed_yaw_error_sum_rad": 0.0,
        "terminal_signed_yaw_metric_episodes": 0,
        "terminal_yaw_progress_ratio_sum": 0.0,
        "terminal_yaw_progress_metric_episodes": 0,
    }


def _finalize_bin(values: dict[str, float | int]) -> dict[str, float | int]:
    episodes = int(values["episodes"])
    if episodes == 0:
        return {"episodes": 0}
    return {
        "episodes": episodes,
        "successes": int(values["successes"]),
        "success_rate": float(values["successes"]) / episodes,
        "constrained_successes": int(values["constrained_successes"]),
        "constrained_success_rate": (
            float(values["constrained_successes"]) / episodes
        ),
        "legal_safe_contact_episode_rate": (
            float(values["legal_safe_contact_episodes"])
            / int(values["legal_safe_contact_metric_episodes"])
            if int(values["legal_safe_contact_metric_episodes"]) > 0
            else None
        ),
        "c1_violation_episodes": float(values["c1_violation_episodes"]),
        "c1_violation_rate": float(values["c1_violation_episodes"]) / episodes,
        "terminal_planar_error_mean_m": (
            float(values["terminal_planar_error_sum_m"]) / episodes
        ),
        "terminal_rotation_error_mean_rad": (
            float(values["terminal_rotation_error_sum_rad"]) / episodes
        ),
        "terminal_signed_yaw_error_mean_rad": (
            float(values["terminal_signed_yaw_error_sum_rad"])
            / int(values["terminal_signed_yaw_metric_episodes"])
            if int(values["terminal_signed_yaw_metric_episodes"]) > 0
            else None
        ),
        "terminal_yaw_progress_ratio_mean": (
            float(values["terminal_yaw_progress_ratio_sum"])
            / int(values["terminal_yaw_progress_metric_episodes"])
            if int(values["terminal_yaw_progress_metric_episodes"]) > 0
            else None
        ),
    }


def _direction_bin(angle_deg: float) -> str:
    for name, lower, upper in DIRECTION_BINS:
        if lower <= angle_deg < upper:
            return name
    raise ValueError(f"goal direction {angle_deg} is outside [-90, 90]")


def _accumulate(values: dict, row: dict[str, str]) -> None:
    episodes = int(row["episodes"])
    values["episodes"] += episodes
    if episodes == 0:
        return
    values["successes"] += int(row["successes"])
    values["constrained_successes"] += int(row["constrained_successes"])
    legal_contact_rate = row.get("legal_safe_contact_episode_rate", "")
    if legal_contact_rate not in (None, ""):
        values["legal_safe_contact_episodes"] += (
            float(legal_contact_rate) * episodes
        )
        values["legal_safe_contact_metric_episodes"] += episodes
    values["c1_violation_episodes"] += float(row["c1_violation_rate"]) * episodes
    values["terminal_planar_error_sum_m"] += (
        float(row["terminal_planar_error_mean_m"]) * episodes
    )
    values["terminal_rotation_error_sum_rad"] += (
        float(row["terminal_rotation_error_mean_rad"]) * episodes
    )
    signed_yaw = row.get("terminal_signed_yaw_error_mean_rad", "")
    yaw_progress = row.get("terminal_yaw_progress_ratio_mean", "")
    if signed_yaw not in (None, ""):
        values["terminal_signed_yaw_error_sum_rad"] += (
            float(signed_yaw) * episodes
        )
        values["terminal_signed_yaw_metric_episodes"] += episodes
    if yaw_progress not in (None, ""):
        values["terminal_yaw_progress_ratio_sum"] += (
            float(yaw_progress) * episodes
        )
        values["terminal_yaw_progress_metric_episodes"] += episodes


def _summarize_evaluation(directory: Path, run_prefix: str) -> dict:
    summary_path = directory / "eval_summary.json"
    per_scene_path = directory / "eval_per_scene.csv"
    summary = json.loads(summary_path.read_text())
    if summary.get("episode_allocation") != "balanced_per_environment":
        raise ValueError(f"{directory} is not a balanced-per-environment evaluation")

    bins = {name: _empty_bin() for name, *_ in DIRECTION_BINS}
    audited_endpoint_bins = {
        "negative": _empty_bin(),
        "positive": _empty_bin(),
    }
    yaw_sign_bins = {"negative": _empty_bin(), "positive": _empty_bin()}
    has_yaw_sign_column = False
    with per_scene_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            direction_deg = float(row["goal_direction_deg"])
            name = _direction_bin(direction_deg)
            _accumulate(bins[name], row)
            if direction_deg <= -args.endpoint_min_abs_deg:
                _accumulate(audited_endpoint_bins["negative"], row)
            elif direction_deg >= args.endpoint_min_abs_deg:
                _accumulate(audited_endpoint_bins["positive"], row)
            yaw_value = row.get("goal_yaw_delta_rad", "")
            if yaw_value not in (None, ""):
                has_yaw_sign_column = True
                yaw_sign = "positive" if float(yaw_value) >= 0.0 else "negative"
                _accumulate(yaw_sign_bins[yaw_sign], row)

    finalized_bins = {
        name: _finalize_bin(values) for name, values in bins.items()
    }
    finalized_yaw_bins = (
        {
            name: _finalize_bin(values)
            for name, values in yaw_sign_bins.items()
        }
        if has_yaw_sign_column
        else {}
    )
    finalized_endpoint_bins = {
        name: _finalize_bin(values)
        for name, values in audited_endpoint_bins.items()
    }
    constrained_success_rate = float(summary["constrained_success_rate"])
    c1_violation_rate = float(summary["typed_violation_rates"]["c1"])
    positive_endpoint = finalized_endpoint_bins["positive"]
    negative_endpoint = finalized_endpoint_bins["negative"]
    passes_positive_endpoint_gate = (
        positive_endpoint.get("constrained_success_rate", 0.0)
        >= MIN_POSITIVE_ENDPOINT_CONSTRAINED_SUCCESS_RATE
        and positive_endpoint.get("c1_violation_rate", 1.0) == 0.0
    )
    passes_negative_endpoint_gate = (
        negative_endpoint.get("constrained_success_rate", 0.0)
        >= MIN_POSITIVE_ENDPOINT_CONSTRAINED_SUCCESS_RATE
        and negative_endpoint.get("c1_violation_rate", 1.0) == 0.0
    )
    passes_required_endpoint_gate = (
        passes_positive_endpoint_gate
        and (
            passes_negative_endpoint_gate
            if args.require_negative_endpoint
            else True
        )
    )
    passes_overall_gate = (
        constrained_success_rate >= MIN_OVERALL_CONSTRAINED_SUCCESS_RATE
        and c1_violation_rate <= MAX_OVERALL_C1_VIOLATION_RATE
    )
    passes_bidirectional_yaw_gate = (
        all(
            values.get("constrained_success_rate", 0.0)
            >= MIN_EACH_YAW_SIGN_CONSTRAINED_SUCCESS_RATE
            and values.get("c1_violation_rate", 1.0) == 0.0
            for values in finalized_yaw_bins.values()
        )
        if finalized_yaw_bins
        else None
    )
    return {
        "run_prefix": run_prefix,
        "checkpoint": _checkpoint_index(directory, summary),
        "evaluation_directory": str(directory.resolve()),
        "episodes": int(summary["episodes"]),
        "success_rate": float(summary["success_rate"]),
        "constrained_success_rate": constrained_success_rate,
        "c1_violation_rate": c1_violation_rate,
        "passes_positive_endpoint_gate": passes_positive_endpoint_gate,
        "passes_negative_endpoint_gate": passes_negative_endpoint_gate,
        "passes_required_endpoint_gate": passes_required_endpoint_gate,
        "passes_overall_c1_gate": passes_overall_gate,
        "passes_bidirectional_yaw_gate": passes_bidirectional_yaw_gate,
        "eligible_for_c1_selection": (
            passes_required_endpoint_gate
            and passes_overall_gate
            and (
                args.skip_bidirectional_yaw_gate
                or passes_bidirectional_yaw_gate is not False
            )
        ),
        "diagnostic_summary": summary.get("diagnostic_summary", {}),
        "direction_bins": finalized_bins,
        "audited_endpoint_bins": finalized_endpoint_bins,
        "yaw_sign_bins": finalized_yaw_bins,
    }


def main() -> None:
    records = []
    for run_prefix in args.run_prefix:
        for directory in sorted(args.eval_root.glob(f"{run_prefix}_model*")):
            if not directory.is_dir():
                continue
            if not (directory / "eval_summary.json").is_file():
                continue
            if not (directory / "eval_per_scene.csv").is_file():
                continue
            records.append(_summarize_evaluation(directory, run_prefix))
    records.sort(key=lambda item: (item["run_prefix"], item["checkpoint"]))
    result = {
        "selection_gates": {
            "endpoint_min_abs_deg": args.endpoint_min_abs_deg,
            "negative_endpoint_required": args.require_negative_endpoint,
            "positive_endpoint_min_constrained_success_rate": (
                MIN_POSITIVE_ENDPOINT_CONSTRAINED_SUCCESS_RATE
            ),
            "positive_endpoint_max_c1_violation_rate": 0.0,
            "overall_min_constrained_success_rate": (
                MIN_OVERALL_CONSTRAINED_SUCCESS_RATE
            ),
            "overall_max_c1_violation_rate": MAX_OVERALL_C1_VIOLATION_RATE,
            "each_yaw_sign_min_constrained_success_rate": (
                MIN_EACH_YAW_SIGN_CONSTRAINED_SUCCESS_RATE
            ),
            "each_yaw_sign_max_c1_violation_rate": 0.0,
            "bidirectional_yaw_gate_required": (
                not args.skip_bidirectional_yaw_gate
            ),
        },
        "direction_bins_deg": {
            name: [lower, upper] for name, lower, upper in DIRECTION_BINS
        },
        "evaluations": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output} with {len(records)} evaluations")
    for item in records:
        negative_endpoint = item["audited_endpoint_bins"]["negative"]
        positive_endpoint = item["audited_endpoint_bins"]["positive"]
        print(
            f"{item['run_prefix']} model_{item['checkpoint']}: "
            f"overall={item['constrained_success_rate']:.2%}, "
            f"C1={item['c1_violation_rate']:.2%}, "
            f"-endpoint={negative_endpoint.get('constrained_success_rate', 0.0):.2%} "
            f"({negative_endpoint.get('constrained_successes', 0)}/"
            f"{negative_endpoint.get('episodes', 0)}), "
            f"+endpoint={positive_endpoint.get('constrained_success_rate', 0.0):.2%} "
            f"({positive_endpoint.get('constrained_successes', 0)}/"
            f"{positive_endpoint.get('episodes', 0)})"
        )


if __name__ == "__main__":
    main()
