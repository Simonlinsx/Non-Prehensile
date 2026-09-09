#!/usr/bin/env python3
"""Check task trajectory timing against the synchronous simulator clock."""
import argparse
import json
import math
from pathlib import Path
import statistics


def analyze(records, tolerance_s=1e-5):
    rows = [r for r in records if r.get("event") == "task_reference"]
    if not rows:
        raise ValueError("No task-reference clock observations")
    leads = []
    query_offsets = []
    for row in rows:
        state_s = row["relay_state_utime_us"] * 1e-6
        lead = row["plan_start_time_s"] - state_s
        query_offset = row["query_time_s"] - state_s
        if not all(math.isfinite(v) for v in (state_s, lead, query_offset)):
            raise ValueError("Non-finite task-reference clock observation")
        leads.append(lead)
        query_offsets.append(query_offset)
    return {
        "schema": "nonprehensile.synchronous_plan_clock_audit.v1",
        "sample_count": len(rows),
        "c3_mode_samples": sum(bool(r["c3_mode"]) for r in rows),
        "median_plan_start_lead_s": statistics.median(leads),
        "maximum_absolute_plan_start_error_s": max(map(abs, leads)),
        "maximum_absolute_query_offset_s": max(map(abs, query_offsets)),
        "tolerance_s": tolerance_s,
        "passed": max(map(abs, leads)) <= tolerance_s and max(map(abs, query_offsets)) <= tolerance_s,
        "scope": "Fresh task references with zero lookahead on the synchronous Isaac bridge; not task acceptance.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = analyze([json.loads(line) for line in args.input.read_text().splitlines() if line.strip()])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
