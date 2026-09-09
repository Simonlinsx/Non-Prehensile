"""Pure-data bridge from Push Anything rollouts to Isaac Lab scenes.

This module deliberately has no Drake, Isaac Sim, or Pinocchio dependency.  It
owns the coordinate and time-series conventions shared by the staging CLI,
unit tests, and the Isaac Lab trajectory executor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import math
from pathlib import Path
from typing import Sequence

from ..scene import ClutterScene, ManipulationTask, SceneObject


PUSH_ANYTHING_TABLE_TO_ISAAC_Z_M = 0.029


def quaternion_multiply_wxyz(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return the normalized Hamilton product ``left * right`` in wxyz order."""

    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    value = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion product is not normalizable")
    return tuple(component / norm for component in value)


def minimum_reference_height(vertices, quaternion_wxyz, table_z_m, clearance_m=.002):
    """Reference height that keeps all hand-frame vertices above the table."""
    if not vertices or not math.isfinite(table_z_m) or not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("Invalid finger/table geometry")
    q = tuple(float(v) for v in quaternion_wxyz)
    norm = math.sqrt(sum(v*v for v in q))
    if len(q) != 4 or not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid hand orientation")
    w, x, y, z = (v/norm for v in q)
    row = (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))
    offsets = []
    for vertex in vertices:
        if len(vertex) != 3 or not all(math.isfinite(v) for v in vertex):
            raise ValueError("Invalid finger vertex")
        offsets.append(sum(a*b for a, b in zip(row, vertex)))
    return table_z_m + clearance_m - min(offsets)


def measured_contact_is_yaw_recovery(
    *,
    start_signed_error_rad: float,
    current_signed_error_rad: float,
    current_yaw_rate_rad_s: float,
    activation_threshold_rad: float,
    minimum_progress_rad: float,
    minimum_rate_rad_s: float,
) -> bool:
    """Return whether a measured contact is correcting a material yaw error.

    This is an execution-time effect test, not a simulator contact label.  It
    requires the signed error to retain its initial sign, decrease by a
    measurable amount, and keep moving toward zero.  The inputs are available
    from object pose tracking on a real robot, so the same predictive braking
    decision can be used in Isaac and on hardware.
    """

    values = (
        start_signed_error_rad,
        current_signed_error_rad,
        current_yaw_rate_rad_s,
        activation_threshold_rad,
        minimum_progress_rad,
        minimum_rate_rad_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if min(activation_threshold_rad, minimum_progress_rad, minimum_rate_rad_s) < 0:
        raise ValueError("yaw-recovery thresholds must be non-negative")
    if abs(start_signed_error_rad) < activation_threshold_rad:
        return False
    if start_signed_error_rad * current_signed_error_rad <= 0.0:
        return False
    if (
        abs(start_signed_error_rad) - abs(current_signed_error_rad)
        < minimum_progress_rad
    ):
        return False
    return (
        math.copysign(1.0, start_signed_error_rad) * current_yaw_rate_rad_s
        <= -minimum_rate_rad_s
    )


def measured_yaw_brake_requested(
    *,
    start_signed_error_rad: float,
    current_signed_error_rad: float,
    current_yaw_rate_rad_s: float,
    recovery_latched: bool,
    exit_threshold_rad: float,
    lookahead_s: float,
) -> bool:
    """Request contact release before a corrective yaw pulse overshoots.

    The caller first latches ``recovery_latched`` from measured progress using
    :func:`measured_contact_is_yaw_recovery`.  Once latched, release is requested
    when the measured error enters the inner terminal band, crosses the target,
    or is predicted to do either under the measured angular velocity.  This is
    deliberately an execution guard over observable object pose and velocity;
    it does not depend on a simulator contact impulse or privileged rollout.
    """

    values = (
        start_signed_error_rad,
        current_signed_error_rad,
        current_yaw_rate_rad_s,
        exit_threshold_rad,
        lookahead_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if exit_threshold_rad < 0.0 or lookahead_s < 0.0:
        raise ValueError("yaw-brake thresholds must be non-negative")
    if not recovery_latched:
        return False
    if (
        abs(current_signed_error_rad) <= exit_threshold_rad
        or start_signed_error_rad * current_signed_error_rad <= 0.0
    ):
        return True
    if current_signed_error_rad * current_yaw_rate_rad_s >= 0.0:
        return False
    projected_error = (
        current_signed_error_rad + lookahead_s * current_yaw_rate_rad_s
    )
    return (
        abs(projected_error) <= exit_threshold_rad
        or current_signed_error_rad * projected_error <= 0.0
    )


def measured_yaw_regression_brake_requested(
    *,
    current_signed_error_rad: float,
    best_abs_error_rad: float,
    regression_tolerance_rad: float,
) -> bool:
    """Return whether a contact has measurably worsened planar yaw error.

    Unlike the predictive contact model, this guard uses only tracked object
    pose.  It is therefore identical in Isaac and on hardware, and catches a
    model-sign or contact-location mismatch before the wider rotation envelope
    is reached.
    """

    values = (
        current_signed_error_rad,
        best_abs_error_rad,
        regression_tolerance_rad,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if best_abs_error_rad < 0.0 or regression_tolerance_rad < 0.0:
        raise ValueError("yaw regression values must be non-negative")
    return (
        abs(current_signed_error_rad)
        > best_abs_error_rad + regression_tolerance_rad
    )


def execution_rotation_guard_requires_retreat(
    *,
    signed_error_rad: float,
    yaw_rate_rad_s: float,
    limit_rad: float,
    lookahead_s: float,
    recovery_outward_rate_tolerance_rad_s: float,
    planner_marks_yaw_recovery: bool,
) -> bool:
    """Return whether contact execution must retreat from the target.

    Within the yaw envelope this applies the usual velocity look-ahead.  Once
    measured yaw is outside the envelope, approaching contact is authorized
    only for a C3 trajectory that explicitly predicts yaw recovery.  This
    closes a fail-open interface bug where a stationary ordinary push was
    repeatedly allowed after every guard retreat.
    """

    values = (
        signed_error_rad,
        yaw_rate_rad_s,
        limit_rad,
        lookahead_s,
        recovery_outward_rate_tolerance_rad_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return True
    if min(limit_rad, lookahead_s, recovery_outward_rate_tolerance_rad_s) < 0.0:
        raise ValueError("rotation-guard thresholds must be non-negative")
    if abs(signed_error_rad) > 1.0e-6:
        outward_yaw_rate = max(
            0.0,
            math.copysign(1.0, signed_error_rad) * yaw_rate_rad_s,
        )
    else:
        outward_yaw_rate = abs(yaw_rate_rad_s)
    if abs(signed_error_rad) < limit_rad:
        predicted_risk = (
            abs(signed_error_rad) + lookahead_s * outward_yaw_rate
        )
        return predicted_risk >= limit_rad
    return (
        not planner_marks_yaw_recovery
        or outward_yaw_rate > recovery_outward_rate_tolerance_rad_s
    )


def _require_pair(stage: dict, key: str) -> tuple[float, float]:
    value = stage.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"stage field {key!r} must contain two values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"stage field {key!r} must be finite")
    return result


def stage_to_isaac_scene(
    stage: dict,
    template_scene: ClutterScene,
    *,
    scene_id: str,
    table_height_offset_m: float = PUSH_ANYTHING_TABLE_TO_ISAAC_Z_M,
    target_mass_kg: float | None = 0.05,
    target_static_friction: float | None = 0.3,
    target_dynamic_friction: float | None = 0.3,
) -> ClutterScene:
    """Map one staged C3+ hammer task into the Isaac Lab manifest schema.

    Push Anything bakes the hammer's stable support transform into its staged
    mesh.  Isaac Lab loads the original DOMINO asset, so the exact support
    quaternion must travel with the stage manifest.  It cannot be copied from
    ``template_scene`` because that template may contain an additional planar
    yaw.  A requested world-Z goal yaw is left-multiplied onto the staged
    support orientation.
    """

    if stage.get("schema") != "nonprehensile.push_anything_stage.v1":
        raise ValueError("unsupported Push Anything stage schema")
    if stage.get("asset_name") != "DOMINO_020_hammer_safe":
        raise ValueError("the current Isaac bridge only supports DOMINO hammer 020")
    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    offset = float(table_height_offset_m)
    if not math.isfinite(offset):
        raise ValueError("table_height_offset_m must be finite")

    initial_xy = _require_pair(stage, "initial_xy_m")
    goal_xy = _require_pair(stage, "goal_xy_m")
    root_height = float(stage["root_height_m"])
    initial_yaw = math.radians(float(stage.get("initial_yaw_deg", 0.0)))
    if not math.isfinite(initial_yaw):
        raise ValueError("stage initial yaw must be finite")
    goal_yaw = math.radians(float(stage["goal_yaw_deg"]))
    if not math.isfinite(root_height) or not math.isfinite(goal_yaw):
        raise ValueError("stage root height and goal yaw must be finite")

    template_target = template_scene.target_object
    support_value = stage.get("support_quaternion_wxyz")
    if not isinstance(support_value, (list, tuple)) or len(support_value) != 4:
        raise ValueError(
            "stage field 'support_quaternion_wxyz' must contain four values"
        )
    support_quaternion = tuple(float(value) for value in support_value)
    support_norm = math.sqrt(sum(value * value for value in support_quaternion))
    if not math.isfinite(support_norm) or support_norm <= 0.0:
        raise ValueError("stage support quaternion is not normalizable")
    support_quaternion = tuple(value / support_norm for value in support_quaternion)
    yaw_quaternion = (
        math.cos(0.5 * goal_yaw),
        0.0,
        0.0,
        math.sin(0.5 * goal_yaw),
    )
    goal_quaternion = quaternion_multiply_wxyz(yaw_quaternion, support_quaternion)
    isaac_z = root_height + offset
    initial_quaternion = quaternion_multiply_wxyz(
        (math.cos(initial_yaw / 2), 0.0, 0.0, math.sin(initial_yaw / 2)), support_quaternion)
    initial_pose = (*initial_xy, isaac_z, *initial_quaternion)
    goal_pose = (*goal_xy, isaac_z, *goal_quaternion)

    physics_overrides = {
        "mass_kg": target_mass_kg,
        "static_friction": target_static_friction,
        "dynamic_friction": target_dynamic_friction,
    }
    replacement_kwargs = {
        name: float(value)
        for name, value in physics_overrides.items()
        if value is not None
    }
    target = replace(template_target, pose=initial_pose, **replacement_kwargs)
    objects: list[SceneObject] = []
    for item in template_scene.objects:
        objects.append(target if item.instance_id == template_target.instance_id else item)

    task = ManipulationTask(
        task_id=f"{scene_id}-joint-pose",
        target_instance_id=template_target.instance_id,
        initial_pose=initial_pose,
        goal_pose=goal_pose,
    )
    return ClutterScene(
        scene_id=scene_id,
        split="eval",
        track=template_scene.track,
        objects=tuple(objects),
        tasks=(task,),
    )


@dataclass(frozen=True)
class C3JointSample:
    """One time-aligned Franka joint sample from the C3+ monitor."""

    time_s: float
    q: tuple[float, ...]
    ee_xyz_m: tuple[float, float, float] | None = None
    object_pose_wxyz_xyz: tuple[float, ...] | None = None


def _optional_floats(
    row: dict[str, str], names: Sequence[str]
) -> tuple[float, ...] | None:
    raw = [row.get(name, "").strip() for name in names]
    if any(not value for value in raw):
        return None
    values = tuple(float(value) for value in raw)
    return values if all(math.isfinite(value) for value in values) else None


def load_c3_joint_trajectory(path: str | Path) -> tuple[C3JointSample, ...]:
    """Load complete, unique joint-state samples from a C3+ debug CSV."""

    source = Path(path)
    samples: list[C3JointSample] = []
    last_utime: int | None = None
    first_utime: int | None = None
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"franka_state_utime", *(f"franka_q_{index}" for index in range(7))}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"C3+ CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            timestamp_text = row["franka_state_utime"].strip()
            q_text = [row[f"franka_q_{index}"].strip() for index in range(7)]
            if not timestamp_text or any(not value for value in q_text):
                continue
            try:
                timestamp = int(timestamp_text)
                q = tuple(float(value) for value in q_text)
            except ValueError as exc:
                raise ValueError(f"invalid C3+ joint row at {source}:{line_number}") from exc
            if timestamp < 0 or not all(math.isfinite(value) for value in q):
                raise ValueError(f"invalid C3+ joint row at {source}:{line_number}")
            if last_utime is not None and timestamp < last_utime:
                raise ValueError("C3+ joint timestamps must be monotonic")
            if timestamp == last_utime:
                continue
            if first_utime is None:
                first_utime = timestamp
            ee_xyz = _optional_floats(row, ("ee_x_m", "ee_y_m", "ee_z_m"))
            object_pose = _optional_floats(
                row,
                (
                    "object_qw",
                    "object_qx",
                    "object_qy",
                    "object_qz",
                    "object_x_m",
                    "object_y_m",
                    "object_z_m",
                ),
            )
            samples.append(
                C3JointSample(
                    time_s=(timestamp - first_utime) * 1.0e-6,
                    q=q,
                    ee_xyz_m=ee_xyz,
                    object_pose_wxyz_xyz=object_pose,
                )
            )
            last_utime = timestamp
    if len(samples) < 2:
        raise ValueError(f"C3+ CSV contains fewer than two complete joint samples: {source}")
    return tuple(samples)


def resample_c3_joint_trajectory(
    samples: Sequence[C3JointSample], control_period_s: float
) -> tuple[C3JointSample, ...]:
    """Linearly resample joint states at an Isaac Lab control period."""

    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    period = float(control_period_s)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("control_period_s must be positive and finite")
    if any(right.time_s <= left.time_s for left, right in zip(samples, samples[1:])):
        raise ValueError("sample times must be strictly increasing")

    end_time = samples[-1].time_s
    count = int(math.floor(end_time / period))
    query_times = [index * period for index in range(count + 1)]
    if not math.isclose(query_times[-1], end_time, abs_tol=1.0e-9):
        query_times.append(end_time)

    result: list[C3JointSample] = []
    right_index = 1
    for query_time in query_times:
        while right_index < len(samples) - 1 and samples[right_index].time_s < query_time:
            right_index += 1
        left = samples[right_index - 1]
        right = samples[right_index]
        span = right.time_s - left.time_s
        alpha = min(1.0, max(0.0, (query_time - left.time_s) / span))
        q = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(left.q, right.q))
        # Auxiliary poses are diagnostic only.  Keep the nearest observed
        # sample rather than interpolating quaternions without SLERP.
        nearest = left if alpha < 0.5 else right
        result.append(
            C3JointSample(
                time_s=query_time,
                q=q,
                ee_xyz_m=nearest.ee_xyz_m,
                object_pose_wxyz_xyz=nearest.object_pose_wxyz_xyz,
            )
        )
    return tuple(result)
