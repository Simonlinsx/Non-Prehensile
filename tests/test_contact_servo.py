from __future__ import annotations

import math

import numpy as np
import pytest

from dapl.contact_planner.contact_servo import goal_wrench_contact_axis


def test_zero_yaw_command_cancels_goal_axis_moment() -> None:
    axis, requested, achieved = goal_wrench_contact_axis(
        goal_delta_xy_m=np.asarray((0.08, 0.025)),
        contact_point_xy_m=np.asarray((0.35, 0.208)),
        object_com_xy_m=np.asarray((0.427, 0.218)),
        signed_yaw_error_rad=0.0,
        object_yaw_rate_rad_s=0.0,
        yaw_moment_gain_m_per_rad=0.04,
        yaw_rate_moment_gain_m_s_per_rad=0.02,
        max_axis_deviation_rad=math.radians(70.0),
    )
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert requested == pytest.approx(0.0)
    assert achieved == pytest.approx(0.0, abs=1.0e-9)


def test_positive_current_minus_goal_yaw_requests_negative_moment() -> None:
    axis, requested, achieved = goal_wrench_contact_axis(
        goal_delta_xy_m=np.asarray((0.08, 0.025)),
        contact_point_xy_m=np.asarray((0.35, 0.208)),
        object_com_xy_m=np.asarray((0.427, 0.218)),
        signed_yaw_error_rad=0.1,
        object_yaw_rate_rad_s=0.0,
        yaw_moment_gain_m_per_rad=0.04,
        yaw_rate_moment_gain_m_s_per_rad=0.02,
        max_axis_deviation_rad=math.radians(70.0),
    )
    assert requested == pytest.approx(-0.004)
    assert achieved == pytest.approx(requested, abs=1.0e-9)
    assert axis[0] > 0.0


def test_yaw_rate_term_brakes_corrective_overshoot() -> None:
    _, requested, achieved = goal_wrench_contact_axis(
        goal_delta_xy_m=np.asarray((0.08, 0.025)),
        contact_point_xy_m=np.asarray((0.35, 0.208)),
        object_com_xy_m=np.asarray((0.427, 0.218)),
        signed_yaw_error_rad=0.01,
        object_yaw_rate_rad_s=-0.1,
        yaw_moment_gain_m_per_rad=0.04,
        yaw_rate_moment_gain_m_s_per_rad=0.02,
        max_axis_deviation_rad=math.radians(70.0),
    )
    assert requested > 0.0
    assert achieved == pytest.approx(requested, abs=1.0e-9)


def test_axis_deviation_limit_is_fail_bounded() -> None:
    goal = np.asarray((1.0, 0.0))
    axis, requested, achieved = goal_wrench_contact_axis(
        goal_delta_xy_m=goal,
        contact_point_xy_m=np.asarray((-0.01, 0.08)),
        object_com_xy_m=np.zeros(2),
        signed_yaw_error_rad=-1.0,
        object_yaw_rate_rad_s=0.0,
        yaw_moment_gain_m_per_rad=0.08,
        yaw_rate_moment_gain_m_s_per_rad=0.0,
        max_axis_deviation_rad=math.radians(20.0),
    )
    deviation = math.atan2(axis[1], axis[0])
    assert abs(deviation) == pytest.approx(math.radians(20.0))
    assert achieved != pytest.approx(requested)


def test_contact_axis_never_commands_an_outward_pull() -> None:
    contact = np.asarray((0.405, 0.208))
    com = np.asarray((0.492, 0.193))
    axis, requested, achieved = goal_wrench_contact_axis(
        goal_delta_xy_m=np.asarray((0.008, 0.044)),
        contact_point_xy_m=contact,
        object_com_xy_m=com,
        signed_yaw_error_rad=-0.17,
        object_yaw_rate_rad_s=0.0,
        yaw_moment_gain_m_per_rad=0.04,
        yaw_rate_moment_gain_m_s_per_rad=0.02,
        max_axis_deviation_rad=math.radians(70.0),
    )
    assert float(np.dot(axis, com - contact)) > 0.0
    assert requested > 0.0
    goal_axis = np.asarray((0.008, 0.044), dtype=float)
    goal_axis /= np.linalg.norm(goal_axis)
    radius = contact - com
    goal_moment = radius[0] * goal_axis[1] - radius[1] * goal_axis[0]
    assert achieved > goal_moment
