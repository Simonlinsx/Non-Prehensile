"""Audit measured closed-gripper motion on every physics step, without interpolation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit_rows(result, rows, tolerance_m):
    if type(tolerance_m) not in (int, float) or not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Closure tolerance must be finite and positive")
    steps, period = result["executed_steps"], result["control_period_s"]
    if type(steps) is not int or steps <= 0 or type(period) not in (int, float) or not math.isfinite(period) or period <= 0:
        raise ValueError("Invalid execution timing")
    if result.get("control_decimation") != 1 or result.get("physics_dt_s") != period:
        raise ValueError("Closure audit requires one measured sample per physics step")
    maximum = [0.0, 0.0]
    mismatch = 0.0
    violations = []
    maximum_target = 0.0
    count = 0
    for row in rows:
        if type(row["step"]) is not int or row["step"] != count or count >= steps:
            raise ValueError("Missing, duplicate, unordered, or excess finger sample")
        stamp = 100000 + round((count + 1) * period * 1e6)
        if type(row["measurement_utime_us"]) is not int or row["measurement_utime_us"] != stamp:
            raise ValueError("Finger measurement timestamp differs from physics clock")
        q, target = row["position_m"], row["target_m"]
        if (len(q) != 2 or len(target) != 2 or any(
                type(x) not in (float, int) or not math.isfinite(x) for x in q + target)):
            raise ValueError("Invalid measured finger state or commanded target")
        maximum = [max(previous, abs(value)) for previous, value in zip(maximum, q)]
        mismatch = max(mismatch, abs(q[0] - q[1]))
        maximum_target = max(maximum_target, *map(abs, target))
        if max(map(abs, q)) > tolerance_m or abs(q[0] - q[1]) > tolerance_m or any(x != 0 for x in target):
            violations.append(count)
        count += 1
    if count != steps:
        raise ValueError("Incomplete finger measurements")
    return dict(schema="nonprehensile.fr3_finger_closure_audit.v1",
                scope="Dynamic finger closure only; not model, contact, C1, or task acceptance",
                closure_pass=not violations, tolerance_m=tolerance_m,
                observed_physics_steps=count, maximum_absolute_position_m=maximum,
                maximum_finger_mismatch_m=mismatch, maximum_absolute_command_target_m=maximum_target,
                violation_steps=len(violations), first_violation_step=violations[0] if violations else None)


def audit_file(result_path, tolerance_m):
    result = json.loads(result_path.read_text())
    artifact = result["finger_state_artifact"]
    path = Path(artifact["path"])
    if (not path.resolve().is_relative_to(result_path.parent.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            or artifact["recorded_steps"] != result["executed_steps"]):
        raise ValueError("Missing or changed finger state artifact")
    with path.open() as stream:
        report = audit_rows(result, (json.loads(line) for line in stream), tolerance_m)
    report.update(result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                  finger_state_sha256=artifact["sha256"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tolerance-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_file(args.result, args.tolerance_m)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["closure_pass"] else 2)
