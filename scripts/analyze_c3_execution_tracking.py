#!/usr/bin/env python3
"""Separate reference limiting, servo tracking, and measured C3 contact.

Contact fractions describe sampled trace rows, not force duration or episode
success. Sparse historical traces can miss short contacts; result-level
success and safety flags remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def stats(values):
    values = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None,
        "max": max(values) if values else None,
    }


def summarize(rows):
    distances = {}
    for name, left, right in [
        ("raw_to_measured_m", "raw_task_target_c3_m", "planner_tip_position_m"),
        ("governed_to_measured_m", "governed_task_target_c3_m", "planner_tip_position_m"),
        ("raw_to_governed_m", "raw_task_target_c3_m", "governed_task_target_c3_m"),
    ]:
        distances[name] = stats([
            math.dist(row[left], row[right]) for row in rows
            if row.get(left) is not None and row.get(right) is not None
        ])
    contacts = sum(bool(row.get("legal_safe_robot_contact")) for row in rows)
    return {
        "sample_count": len(rows),
        "physical_safe_contact_samples": contacts,
        "physical_safe_contact_sample_fraction": contacts / len(rows) if rows else None,
        **distances,
    }


def analyze(result):
    trace = result.get("trace", [])
    fresh = [row for row in trace if int(row.get("flags", 0)) & 1
             and not int(row.get("flags", 0)) & (2 | 4 | 8)]
    c3 = [row for row in fresh if row.get("c3_mode")]
    contact = [row for row in trace if row.get("legal_safe_robot_contact")]
    return {
        "schema": "nonprehensile.c3_execution_tracking.v1",
        "sample_period_s": (None if result.get("trace_sampling_policy") == "stride_plus_all_physical_contact_steps"
                            else result["control_period_s"] * result["trace_stride"]),
        "nominal_stride_period_s": result["control_period_s"] * result["trace_stride"],
        "trace_sampling_policy": result.get("trace_sampling_policy", "uniform_stride"),
        "physical_end_effector": result.get("physical_end_effector"),
        "controller_parameters": result.get("controller_parameters"),
        "executed_sim_time_s": result.get("executed_sim_time_s"),
        "success": result.get("online_closed_loop_success"),
        "forbidden_contact_ever": result.get("forbidden_robot_contact_ever"),
        "final_planar_error_m": result.get("final_planar_error_m"),
        "final_rotation_error_rad": result.get("final_rotation_error_rad"),
        "first_physical_contact_time_s": (
            result["first_safe_contact_step"] * result["control_period_s"]
            if result.get("first_safe_contact_step") is not None else None
        ),
        "all": summarize(trace),
        "fresh_reposition": summarize([row for row in fresh if not row.get("c3_mode")]),
        "fresh_c3": summarize(c3),
        "physical_contact": summarize(contact),
        "sampled_contact_outside_fresh_c3": sum(row not in c3 for row in contact),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(json.loads(args.result.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "success", "first_physical_contact_time_s", "fresh_c3", "physical_contact"
    )}, indent=2))


if __name__ == "__main__":
    main()
