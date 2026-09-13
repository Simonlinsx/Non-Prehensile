"""Cheap, shared directional check; does not replace IK or semantic validation."""
import math
import numpy as np

from dapl.contact_planner.contact_servo import goal_wrench_contact_axis


def check_candidate(*, goal_delta, contact, com, yaw_error, yaw_rate, fallback_axis,
                    yaw_gain=.04, rate_gain=.02, max_deviation=1.56,
                    minimum_cosine=.2, maximum_moment_error=.01):
    goal=np.asarray(goal_delta,dtype=float)
    if np.linalg.norm(goal)<=1e-6:
        goal=np.asarray(fallback_axis,dtype=float)
    axis,requested,achieved=goal_wrench_contact_axis(
        goal_delta_xy_m=goal,contact_point_xy_m=np.asarray(contact,dtype=float),
        object_com_xy_m=np.asarray(com,dtype=float),signed_yaw_error_rad=float(yaw_error),
        object_yaw_rate_rad_s=float(yaw_rate),yaw_moment_gain_m_per_rad=yaw_gain,
        yaw_rate_moment_gain_m_s_per_rad=rate_gain,max_axis_deviation_rad=max_deviation)
    cosine=float(np.dot(axis,goal/np.linalg.norm(goal)))
    reason=('contact_cannot_advance_goal' if cosine<minimum_cosine else
            'requested_moment_unreachable' if abs(achieved-requested)>maximum_moment_error else 'compatible')
    return dict(compatible=reason=='compatible',reason=reason,axis=axis.tolist(),
        goal_axis_cosine=cosine,requested_moment=requested,achieved_moment=achieved,
        goal_delta=goal.tolist(),contact=list(map(float,contact)),com=list(map(float,com)),
        yaw_error=float(yaw_error),yaw_rate=float(yaw_rate))


def rank_candidates(scores,checks):
    # Keep every candidate. If the preferred contacts fail existing IK/C1
    # checks, the existing candidate loop can still reach the other options.
    return sorted(scores,key=lambda rank:(not checks[rank]['compatible'],scores[rank]))
