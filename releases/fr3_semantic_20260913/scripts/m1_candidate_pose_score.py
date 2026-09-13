"""Candidate pose cost expressed in metres with acceptance-scaled yaw cost."""


def acceptance_scaled_pose_score(
    planar_error_m, yaw_error_rad, candidate_cost, *, yaw_is_inside,
    planar_threshold_m=.02, yaw_threshold_rad=.10, yaw_guard_rad=.075,
):
    """Change pose weights only; preserve the legacy secondary candidate cost.

    Equivalent to multiplying a dimensionless pose cost by the XY threshold.
    The existing inner yaw guard remains a soft penalty, not an admission gate.
    This function neither changes candidate generation nor certifies safety.
    """
    yaw_weight = planar_threshold_m / yaw_threshold_rad
    guard = yaw_weight * max(0., yaw_error_rad - yaw_guard_rad) if yaw_is_inside else 0.
    return planar_error_m + yaw_weight * yaw_error_rad + guard + .05 * candidate_cost
