"""Retiming for the frozen FR3 M1 executor; legacy step arguments denote 10 Hz ticks.

This changes action cadence, not the planner algorithm or physics timestep.
"""
import time

REFERENCE_HZ = 10
PHYSICS_DT_S = 0.001
DURATION_FIELDS = (
    "settle_steps", "approach_steps", "contact_steps", "endpoint_hold_steps",
    "push_steps", "retreat_steps", "inter_push_settle_steps", "final_hold_steps",
    "dwell_steps", "persistent_micro_steps",
)


def decimation(rate_hz):
    if rate_hz not in (10, 20):
        raise ValueError("Only the explicit 10 Hz reference and 20 Hz alignment are supported")
    return round(1 / (rate_hz * PHYSICS_DT_S))


def retime(args, rate_hz):
    """Convert duration counts exactly once, leaving distances and budgets alone."""
    substeps = decimation(rate_hz)
    if getattr(args, "_m1_retimed", False):
        raise ValueError("Arguments have already been retimed")
    converted = {}
    for name in DURATION_FIELDS:
        before = getattr(args, name)
        if not isinstance(before, int) or before < 0:
            raise ValueError(f"Invalid duration count {name}: {before}")
        after = before * rate_hz // REFERENCE_HZ
        converted[name] = {"reference_steps": before, "execution_steps": after,
                           "duration_s": before / REFERENCE_HZ}
    for name, values in converted.items():
        setattr(args, name, values["execution_steps"])
    args._m1_retimed = True
    return {"action_rate_hz": rate_hz, "physics_dt_s": PHYSICS_DT_S,
            "decimation": substeps, "action_dt_s": 1 / rate_hz,
            "duration_arguments_reference_hz": REFERENCE_HZ, "durations": converted,
            "planner_schedule": "event-driven; persistent micro horizon remains 0.3 s at defaults",
            "controller": "M1 relative joint-position feedback, original PD; no hardware interpolator"}


def install_action_clock(base, state, out):
    """Observe actual process_action calls; reject an unexpected physics cadence."""
    import json
    original = base.action_manager.process_action
    rows = (out / "action_clock.jsonl").open("x")
    previous_step = None
    expected = round(base.step_dt / base.physics_dt)
    state["action_count"] = 0
    def process(action):
        nonlocal previous_step
        step = state["steps"]
        if previous_step is not None and step - previous_step != expected:
            raise RuntimeError(f"Action cadence changed: {step - previous_step} != {expected}")
        result = original(action)
        rows.write(json.dumps({"physics_step": step, "sim_s": step * base.physics_dt,
                               "wall_monotonic_s": time.monotonic()}) + "\n")
        previous_step = step
        state["action_count"] += 1
        return result
    base.action_manager.process_action = process
    return rows
