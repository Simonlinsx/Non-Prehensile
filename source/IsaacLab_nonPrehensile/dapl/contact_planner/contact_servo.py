"""Small planar contact-control helpers shared by simulation and hardware.

The functions in this module deliberately depend only on measured SE(2) state
and semantic geometry.  They do not use simulator contact impulses or future
rollouts, so the same calculation can be driven by an RGB-D object tracker on
the real robot.
"""

from __future__ import annotations

import math

import numpy as np


def _wrap_to_pi(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def goal_wrench_contact_axis(
    *,
    goal_delta_xy_m: np.ndarray,
    contact_point_xy_m: np.ndarray,
    object_com_xy_m: np.ndarray,
    signed_yaw_error_rad: float,
    object_yaw_rate_rad_s: float,
    yaw_moment_gain_m_per_rad: float,
    yaw_rate_moment_gain_m_s_per_rad: float,
    max_axis_deviation_rad: float,
) -> tuple[np.ndarray, float, float]:
    """Choose a push direction with the requested planar force moment.

    ``signed_yaw_error_rad`` is current-minus-goal yaw.  The desired moment arm
    per unit push force is a PD command, ``-kp * error - kd * yaw_rate``.  For
    the measured safe contact radius ``r``, the returned unit axis is the
    direction nearest to the translational goal that satisfies
    ``cross(r, axis) == desired_moment_arm`` whenever the deviation limit allows
    it.  The two scalar returns are the requested and achieved moment arms.

    This is a direction calculation, not a dynamics model: the high-rate C1
    guard still owns contact admission and the outer planner still owns contact
    selection and obstacle avoidance.
    """

    goal_delta = np.asarray(goal_delta_xy_m, dtype=float).reshape(2)
    contact = np.asarray(contact_point_xy_m, dtype=float).reshape(2)
    com = np.asarray(object_com_xy_m, dtype=float).reshape(2)
    if not (
        np.isfinite(goal_delta).all()
        and np.isfinite(contact).all()
        and np.isfinite(com).all()
        and math.isfinite(signed_yaw_error_rad)
        and math.isfinite(object_yaw_rate_rad_s)
    ):
        raise ValueError("contact-axis inputs must be finite")
    if yaw_moment_gain_m_per_rad < 0.0:
        raise ValueError("yaw moment gain must be non-negative")
    if yaw_rate_moment_gain_m_s_per_rad < 0.0:
        raise ValueError("yaw-rate moment gain must be non-negative")
    if not 0.0 <= max_axis_deviation_rad <= math.pi:
        raise ValueError("maximum axis deviation must lie in [0, pi]")

    goal_norm = float(np.linalg.norm(goal_delta))
    if goal_norm <= 1.0e-9:
        raise ValueError("goal delta is too small to define a push direction")
    goal_axis = goal_delta / goal_norm

    radius = contact - com
    radius_norm = float(np.linalg.norm(radius))
    if radius_norm <= 1.0e-9:
        return goal_axis, 0.0, 0.0

    requested_moment_arm = (
        -yaw_moment_gain_m_per_rad * float(signed_yaw_error_rad)
        - yaw_rate_moment_gain_m_s_per_rad * float(object_yaw_rate_rad_s)
    )
    # No unit force can create a moment arm larger than |r|.  Leave a small
    # numerical margin so asin remains well conditioned.
    requested_moment_arm = float(np.clip(
        requested_moment_arm,
        -0.999 * radius_norm,
        0.999 * radius_norm,
    ))

    radius_angle = math.atan2(radius[1], radius[0])
    sine = requested_moment_arm / radius_norm
    contact_angle = math.asin(sine)
    candidate_angles = (
        radius_angle + contact_angle,
        radius_angle + math.pi - contact_angle,
    )
    goal_angle = math.atan2(goal_axis[1], goal_axis[0])
    # The trigonometric equation has two solutions with opposite radial force
    # components.  A unilateral point contact can push toward the object but
    # cannot pull it, so discard the outward solution before comparing it with
    # the translational goal direction.
    inward_axis = -radius / radius_norm
    inward_candidates = tuple(
        angle
        for angle in candidate_angles
        if float(np.dot(
            np.asarray((math.cos(angle), math.sin(angle))), inward_axis
        )) >= -1.0e-9
    )
    if not inward_candidates:
        inward_candidates = candidate_angles
    deviations = tuple(
        _wrap_to_pi(angle - goal_angle) for angle in inward_candidates
    )
    desired_deviation = min(deviations, key=abs)
    applied_deviation = float(np.clip(
        desired_deviation,
        -max_axis_deviation_rad,
        max_axis_deviation_rad,
    ))
    applied_angle = goal_angle + applied_deviation
    axis = np.asarray((math.cos(applied_angle), math.sin(applied_angle)))
    if float(np.dot(axis, inward_axis)) < 0.0:
        # If the translational goal itself lies outside the unilateral contact
        # half-plane and the deviation cap cannot reach the ideal solution,
        # fail bounded at the nearest slightly-inward direction.  Violating a
        # heuristic angular preference is preferable to commanding an
        # impossible pulling contact.
        inward_angle = math.atan2(inward_axis[1], inward_axis[0])
        relative_goal_angle = _wrap_to_pi(goal_angle - inward_angle)
        applied_angle = inward_angle + float(np.clip(
            relative_goal_angle,
            -0.5 * math.pi + 1.0e-3,
            0.5 * math.pi - 1.0e-3,
        ))
        axis = np.asarray((math.cos(applied_angle), math.sin(applied_angle)))
    achieved_moment_arm = float(
        radius[0] * axis[1] - radius[1] * axis[0]
    )
    return axis, requested_moment_arm, achieved_moment_arm
