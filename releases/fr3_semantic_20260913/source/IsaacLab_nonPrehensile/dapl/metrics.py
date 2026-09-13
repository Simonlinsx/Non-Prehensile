"""Simulator-independent DAPL task metrics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def dapl_tanh_proximity_reward(
    distance: torch.Tensor, *, standard_deviation: float
) -> torch.Tensor:
    """DAPL's reported ``1 - tanh(distance / sigma)`` shaping."""

    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive")
    if torch.any(distance < 0.0):
        raise ValueError("distance must be non-negative")
    return 1.0 - torch.tanh(distance / standard_deviation)


def dapl_combined_pose_error(
    position_distance: torch.Tensor, rotation_distance: torch.Tensor
) -> torch.Tensor:
    """DAPL's full-3D pose error ``d_p + d_r / 5``."""

    if position_distance.shape != rotation_distance.shape:
        raise ValueError("position and rotation distances must have matching shapes")
    if torch.any(position_distance < 0.0) or torch.any(rotation_distance < 0.0):
        raise ValueError("pose distances must be non-negative")
    return position_distance + rotation_distance / 5.0


def dapl_multiscale_pose_score(
    position_distance: torch.Tensor,
    rotation_distance: torch.Tensor,
    *,
    coarse_standard_deviation: float = 0.6,
    fine_standard_deviation: float = 0.3,
    coarse_weight: float = 5.0,
    fine_weight: float = 16.0,
) -> torch.Tensor:
    """Combine DAPL's reported coarse and fine full-pose rewards.

    Keeping the published kernels in one pure primitive lets reward terms
    zero-center the *same* score without silently changing its geometry or
    relative coarse/fine weighting.
    """

    if coarse_standard_deviation <= 0.0 or fine_standard_deviation <= 0.0:
        raise ValueError("pose-score standard deviations must be positive")
    if coarse_weight < 0.0 or fine_weight < 0.0:
        raise ValueError("pose-score weights must be non-negative")
    pose_error = dapl_combined_pose_error(
        position_distance, rotation_distance
    )
    return (
        coarse_weight
        * dapl_tanh_proximity_reward(
            pose_error, standard_deviation=coarse_standard_deviation
        )
        + fine_weight
        * dapl_tanh_proximity_reward(
            pose_error, standard_deviation=fine_standard_deviation
        )
    )


def positive_reference_relative_score(
    reference_score: torch.Tensor, current_score: torch.Tensor
) -> torch.Tensor:
    """Return only improvement above an episode-constant score reference.

    This preserves zero payoff for a stationary initial state while removing
    the early-training incentive to avoid all object contact after exploratory
    pushes happen to worsen the pose.  It is a rectification of one score, not
    a new objective, phase variable, or waypoint.
    """

    if reference_score.shape != current_score.shape:
        raise ValueError("reference and current scores must have matching shapes")
    return torch.clamp(current_score - reference_score, min=0.0)


def positive_reference_relative_error_improvement(
    reference_error: torch.Tensor,
    current_error: torch.Tensor,
    *,
    reference_error_floor: float = 1.0,
) -> torch.Tensor:
    """Return a bounded fraction of improvement in one joint error.

    The reference and current values are expected to be the same monotone
    joint error, such as the worst normalized XY/Z/SO(3) component. Dividing
    by the reset error makes the result dimensionless without introducing a
    task-distance tuning constant. Stationary and regressed states receive
    zero, while reaching zero error receives at most one.
    """

    if reference_error.shape != current_error.shape:
        raise ValueError("reference and current errors must have matching shapes")
    if reference_error_floor <= 0.0:
        raise ValueError("reference_error_floor must be positive")
    if torch.any(reference_error < 0.0) or torch.any(current_error < 0.0):
        raise ValueError("reference and current errors must be non-negative")
    denominator = torch.clamp(reference_error, min=reference_error_floor)
    return torch.clamp(
        (reference_error - current_error) / denominator,
        min=0.0,
        max=1.0,
    )


def signed_reference_relative_error_improvement(
    reference_error: torch.Tensor,
    current_error: torch.Tensor,
    *,
    reference_error_floor: float = 1.0,
    regression_scale: float = 1.0,
) -> torch.Tensor:
    """Return bounded signed improvement in one reset-relative error.

    This is the symmetric counterpart of
    :func:`positive_reference_relative_error_improvement`.  It preserves the
    same reset normalization and zero stationary payoff, while assigning
    negative credit to regressions instead of treating a wrong-direction push
    like an unchanged object.  ``regression_scale`` permits a pre-registered
    leaky-signed ablation while preserving the positive-improvement branch.
    """

    if reference_error.shape != current_error.shape:
        raise ValueError("reference and current errors must have matching shapes")
    if reference_error_floor <= 0.0:
        raise ValueError("reference_error_floor must be positive")
    if not 0.0 <= regression_scale <= 1.0:
        raise ValueError("regression_scale must be in [0, 1]")
    if torch.any(reference_error < 0.0) or torch.any(current_error < 0.0):
        raise ValueError("reference and current errors must be non-negative")
    denominator = torch.clamp(reference_error, min=reference_error_floor)
    improvement = torch.clamp(
        (reference_error - current_error) / denominator,
        min=-1.0,
        max=1.0,
    )
    return torch.where(
        improvement >= 0.0,
        improvement,
        improvement * regression_scale,
    )


def bounded_linear_distance_score(
    distance: torch.Tensor, *, maximum_distance: float
) -> torch.Tensor:
    """Map distance to ``[0, 1]`` without the far-field tanh saturation."""

    if maximum_distance <= 0.0:
        raise ValueError("maximum_distance must be positive")
    if torch.any(distance < 0.0):
        raise ValueError("distance must be non-negative")
    return 1.0 - torch.clamp(distance / maximum_distance, min=0.0, max=1.0)


def normalized_contact_distance_excess(
    distance: torch.Tensor,
    *,
    contact_distance: float,
    normalization_distance: float,
) -> torch.Tensor:
    """Return a bounded distance cost that is exactly zero at contact.

    Unlike a positive proximity score, this cost has no positive stationary
    payoff just outside the contact boundary and no reward discontinuity when
    contact starts.
    """

    if contact_distance < 0.0:
        raise ValueError("contact_distance must be non-negative")
    if normalization_distance <= 0.0:
        raise ValueError("normalization_distance must be positive")
    return torch.clamp(
        (distance - contact_distance) / normalization_distance,
        min=0.0,
        max=1.0,
    )


def normalized_clearance_violation(
    distance: torch.Tensor,
    *,
    contact_distance: float,
    activation_distance: float,
) -> torch.Tensor:
    """Ramp from zero clearance cost to one at the contact boundary."""

    if contact_distance < 0.0:
        raise ValueError("contact_distance must be non-negative")
    if activation_distance <= contact_distance:
        raise ValueError("activation_distance must exceed contact_distance")
    if torch.any(distance < 0.0):
        raise ValueError("distance must be non-negative")
    return torch.clamp(
        (activation_distance - distance)
        / (activation_distance - contact_distance),
        min=0.0,
        max=1.0,
    )


def clearance_log_barrier(
    distance: torch.Tensor,
    *,
    contact_distance: float,
    activation_distance: float,
    minimum_free_fraction: float = 0.05,
) -> torch.Tensor:
    """Return a finite log barrier near a clearance constraint.

    The value is zero at and beyond ``activation_distance`` and increases
    smoothly as free margin vanishes.  Clamping the normalized free fraction
    keeps simulator contacts finite while retaining a substantially stronger
    near-boundary signal than a saturated linear hinge.
    """

    if contact_distance < 0.0:
        raise ValueError("contact_distance must be non-negative")
    if activation_distance <= contact_distance:
        raise ValueError("activation_distance must exceed contact_distance")
    if not 0.0 < minimum_free_fraction < 1.0:
        raise ValueError("minimum_free_fraction must be in (0, 1)")
    if torch.any(distance < 0.0):
        raise ValueError("distance must be non-negative")
    free_fraction = (distance - contact_distance) / (
        activation_distance - contact_distance
    )
    free_fraction = torch.clamp(
        free_fraction, min=float(minimum_free_fraction), max=1.0
    )
    return -torch.log(free_fraction)


def normalized_distance_progress(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    *,
    normalization_distance: float,
) -> torch.Tensor:
    """Return clipped signed progress, positive only when distance decreases."""

    if previous_distance.shape != current_distance.shape:
        raise ValueError("previous and current distance shapes must match")
    if normalization_distance <= 0.0:
        raise ValueError("normalization_distance must be positive")
    return torch.clamp(
        (previous_distance - current_distance) / normalization_distance,
        min=-1.0,
        max=1.0,
    )


def planar_lateral_escape_axis(
    start: torch.Tensor,
    end: torch.Tensor,
    detour_side: torch.Tensor,
) -> torch.Tensor:
    """Return the signed planar unit normal to a blocked direct route.

    ``detour_side`` follows the usual 2-D cross-product convention: ``+1``
    selects the left side of ``start -> end``, ``-1`` the right side, and zero
    disables lateral shaping for an unobstructed route.  The construction is
    translation/rotation equivariant in the support plane and contains no
    scene-specific waypoint.
    """

    if start.ndim != 2 or start.shape[-1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have matching [batch, 3] shapes")
    if detour_side.shape != start.shape[:1]:
        raise ValueError("detour_side must match the batch dimension")
    if torch.any(torch.abs(detour_side) > 1):
        raise ValueError("detour_side values must be in {-1, 0, 1}")

    direct_xy = end[:, :2] - start[:, :2]
    direct_norm = torch.linalg.vector_norm(direct_xy, dim=1, keepdim=True)
    if torch.any((direct_norm.squeeze(1) <= 1.0e-6) & (detour_side != 0)):
        raise ValueError("a non-zero detour side requires a planar route")
    left_xy = torch.stack((-direct_xy[:, 1], direct_xy[:, 0]), dim=1)
    left_xy = left_xy / torch.clamp(direct_norm, min=1.0e-6)
    signed_xy = left_xy * detour_side.to(start.dtype)[:, None]
    axis = torch.zeros_like(start)
    axis[:, :2] = signed_xy
    return axis


def normalized_directional_displacement(
    previous_point: torch.Tensor,
    current_point: torch.Tensor,
    direction: torch.Tensor,
    *,
    normalization_distance: float,
) -> torch.Tensor:
    """Return bounded signed motion along a supplied unit-vector field."""

    if (
        previous_point.ndim != 2
        or previous_point.shape[-1] != 3
        or current_point.shape != previous_point.shape
        or direction.shape != previous_point.shape
    ):
        raise ValueError(
            "previous_point, current_point, and direction must match [batch, 3]"
        )
    if normalization_distance <= 0.0:
        raise ValueError("normalization_distance must be positive")
    direction_norm = torch.linalg.vector_norm(direction, dim=1)
    if torch.any(direction_norm > 1.0 + 1.0e-5):
        raise ValueError("direction magnitude must not exceed one")
    displacement = current_point - previous_point
    projected = torch.sum(displacement * direction, dim=1)
    return torch.clamp(
        projected / float(normalization_distance), min=-1.0, max=1.0
    )


def potential_consistent_progress(
    previous_potential: torch.Tensor,
    current_potential: torch.Tensor,
    field_alignment: torch.Tensor,
    *,
    potential_scale: float = 1.0,
    descent_gate_floor: float = 0.25,
) -> torch.Tensor:
    """Return bounded conservative progress filtered by local-field alignment.

    ``potential / (potential + potential_scale)`` maps a non-negative
    navigation potential into ``[0, 1)`` before taking the exact transition
    difference.  Positive descent is attenuated when motion disagrees with the
    local semantic field, while potential ascent is never attenuated.  Thus
    every transition reward is no greater than the conservative potential
    difference and a closed state-space loop has non-positive total reward.

    This makes a local vector field useful as a direction discriminator
    without allowing a non-integrable tangential field to pay indefinitely for
    circulating around protected geometry.
    """

    if previous_potential.shape != current_potential.shape:
        raise ValueError("previous and current potential shapes must match")
    if field_alignment.shape != previous_potential.shape:
        raise ValueError("field alignment must match potential shape")
    if potential_scale <= 0.0:
        raise ValueError("potential_scale must be positive")
    if not 0.0 <= descent_gate_floor <= 1.0:
        raise ValueError("descent_gate_floor must be in [0, 1]")
    if torch.any(previous_potential < 0.0) or torch.any(current_potential < 0.0):
        raise ValueError("navigation potentials must be non-negative")

    previous_bounded = previous_potential / (
        previous_potential + float(potential_scale)
    )
    current_bounded = current_potential / (
        current_potential + float(potential_scale)
    )
    conservative_progress = previous_bounded - current_bounded
    alignment_gate = float(descent_gate_floor) + (
        1.0 - float(descent_gate_floor)
    ) * torch.clamp(field_alignment, min=0.0, max=1.0)
    return torch.where(
        conservative_progress > 0.0,
        conservative_progress * alignment_gate,
        conservative_progress,
    )


def lexicographic_route_potential(
    route_length: torch.Tensor,
    route_clearance: torch.Tensor,
    *,
    required_clearance: float,
    length_scale: float,
    violation_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a route without trading task distance against safety margin.

    Feasible routes are ordered *only* by length.  If no route satisfies the
    clearance constraint, routes are ordered first by maximum clearance and
    then by length for exact ties.  The returned scalar preserves that
    lexicographic ordering: every feasible cost lies in ``[0, 1)`` and every
    infeasible cost lies in ``[1, 2)``.

    This differs deliberately from a weighted distance-plus-barrier score.
    Once the required clearance is met, moving farther away from the protected
    geometry can never compensate for moving away from the safe contact set.
    """

    if route_length.ndim != 2 or route_clearance.shape != route_length.shape:
        raise ValueError(
            "route length and clearance must have matching [batch, routes] shapes"
        )
    if route_length.shape[1] == 0:
        raise ValueError("at least one route is required")
    if required_clearance < 0.0:
        raise ValueError("required_clearance must be non-negative")
    if length_scale <= 0.0 or violation_scale <= 0.0:
        raise ValueError("route potential scales must be positive")
    if torch.any(route_length < 0.0):
        raise ValueError("route lengths must be non-negative")
    if torch.any(route_clearance < 0.0):
        raise ValueError("route clearances must be non-negative")

    legal = route_clearance >= float(required_clearance)
    has_legal = legal.any(dim=1)

    legal_length = route_length.masked_fill(~legal, torch.inf)
    selected_legal = legal_length.argmin(dim=1)

    best_clearance = route_clearance.amax(dim=1, keepdim=True)
    maximum_clearance = route_clearance == best_clearance
    fallback_length = route_length.masked_fill(~maximum_clearance, torch.inf)
    selected_fallback = fallback_length.argmin(dim=1)
    selected_index = torch.where(has_legal, selected_legal, selected_fallback)

    batch_index = torch.arange(route_length.shape[0], device=route_length.device)
    selected_length = route_length[batch_index, selected_index]
    selected_clearance = route_clearance[batch_index, selected_index]

    feasible_cost = selected_length / (
        selected_length + float(length_scale)
    )
    violation = torch.clamp(
        float(required_clearance) - selected_clearance, min=0.0
    )
    normalized_violation = violation / float(violation_scale)
    infeasible_cost = 1.0 + normalized_violation / (
        normalized_violation + 1.0
    )
    potential = torch.where(has_legal, feasible_cost, infeasible_cost)
    return potential, selected_index, has_legal


def wrench_aware_contact_support_score(
    point_offset_xy: torch.Tensor,
    desired_translation_xy: torch.Tensor,
    desired_yaw: torch.Tensor,
    *,
    yaw_moment_weight: float,
    yaw_activation_rad: float,
) -> torch.Tensor:
    """Score object-centric contact points by useful planar push wrench.

    The translational term recovers the usual trailing-side contact rule.  The
    moment term is the signed planar moment arm ``(r x f)_z`` for a unit push
    in the desired translation direction.  Its sign is gated by the desired
    yaw, so opposite yaw goals select opposite sides of the same safe region.

    This is a contact-manifold score, not a waypoint or an action label.  It is
    computed entirely from point geometry and the recoverable relative goal.
    Setting ``yaw_moment_weight`` to zero exactly recovers the old rule.
    """

    if point_offset_xy.ndim != 3 or point_offset_xy.shape[-1] != 2:
        raise ValueError("point offsets must have shape [batch, points, 2]")
    if (
        desired_translation_xy.ndim != 2
        or desired_translation_xy.shape != (point_offset_xy.shape[0], 2)
    ):
        raise ValueError("desired translation must have shape [batch, 2]")
    if desired_yaw.shape != (point_offset_xy.shape[0],):
        raise ValueError("desired yaw must have shape [batch]")
    if yaw_moment_weight < 0.0:
        raise ValueError("yaw moment weight must be non-negative")
    if yaw_activation_rad <= 0.0:
        raise ValueError("yaw activation must be positive")

    translation_direction = desired_translation_xy / torch.clamp(
        torch.linalg.vector_norm(
            desired_translation_xy, dim=1, keepdim=True
        ),
        min=1.0e-6,
    )
    trailing_projection = torch.sum(
        point_offset_xy * -translation_direction[:, None, :], dim=-1
    )
    return trailing_projection + (
        float(yaw_moment_weight)
        * signed_yaw_contact_moment_score(
            point_offset_xy,
            desired_translation_xy,
            desired_yaw,
            yaw_activation_rad=yaw_activation_rad,
        )
    )


def signed_yaw_contact_moment_score(
    point_offset_xy: torch.Tensor,
    desired_translation_xy: torch.Tensor,
    desired_yaw: torch.Tensor,
    *,
    yaw_activation_rad: float,
) -> torch.Tensor:
    """Return the signed yaw-compatible moment arm of each contact point.

    A positive value means that pushing at the point in the desired
    translation direction also produces torque with the sign of the remaining
    yaw error.  Keeping this score separate from trailing-side translation
    support lets a state-conditioned contact *set* prioritize rotation without
    allowing the two objectives to cancel in a weighted sum.
    """

    if point_offset_xy.ndim != 3 or point_offset_xy.shape[-1] != 2:
        raise ValueError("point offsets must have shape [batch, points, 2]")
    if (
        desired_translation_xy.ndim != 2
        or desired_translation_xy.shape != (point_offset_xy.shape[0], 2)
    ):
        raise ValueError("desired translation must have shape [batch, 2]")
    if desired_yaw.shape != (point_offset_xy.shape[0],):
        raise ValueError("desired yaw must have shape [batch]")
    if yaw_activation_rad <= 0.0:
        raise ValueError("yaw activation must be positive")

    translation_direction = desired_translation_xy / torch.clamp(
        torch.linalg.vector_norm(
            desired_translation_xy, dim=1, keepdim=True
        ),
        min=1.0e-6,
    )
    signed_moment_arm = (
        point_offset_xy[..., 0] * translation_direction[:, None, 1]
        - point_offset_xy[..., 1] * translation_direction[:, None, 0]
    )
    yaw_gate = torch.tanh(desired_yaw / float(yaw_activation_rad))
    return yaw_gate[:, None] * signed_moment_arm


def yaw_compatible_safe_point_mask(
    compatibility_score: torch.Tensor,
    safe_mask: torch.Tensor,
    *,
    selection_mode: str,
    near_best_band_m: float,
    minimum_compatibility_m: float = 0.002,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a yaw-compatible safe subset without mixing task objectives.

    ``near_best`` retains points close to the maximum signed moment.  The
    broader ``positive_halfspace`` retains every safe point with a guaranteed
    positive signed moment, preserving reachability diversity.  If an
    unaudited asset has no point above the requested floor, the latter falls
    back to the finite near-best set and reports that fallback per batch item.
    """

    if compatibility_score.ndim != 2:
        raise ValueError("compatibility scores must have shape [batch, points]")
    if safe_mask.shape != compatibility_score.shape:
        raise ValueError("safe mask must match compatibility score shape")
    if safe_mask.dtype != torch.bool:
        raise ValueError("safe mask must be boolean")
    if not bool(safe_mask.any(dim=1).all()):
        raise ValueError("every batch item must contain at least one safe point")
    if near_best_band_m < 0.0 or minimum_compatibility_m < 0.0:
        raise ValueError("selection margins must be non-negative")
    if selection_mode not in ("near_best", "positive_halfspace"):
        raise ValueError(
            "selection mode must be 'near_best' or 'positive_halfspace'"
        )

    best_score = compatibility_score.masked_fill(~safe_mask, -torch.inf).max(
        dim=1, keepdim=True
    ).values
    near_best = safe_mask & (
        compatibility_score >= best_score - float(near_best_band_m)
    )
    if selection_mode == "near_best":
        return near_best, torch.zeros(
            compatibility_score.shape[0],
            dtype=torch.bool,
            device=compatibility_score.device,
        )

    positive = safe_mask & (
        compatibility_score >= float(minimum_compatibility_m)
    )
    has_positive = positive.any(dim=1)
    selected = torch.where(has_positive[:, None], positive, near_best)
    return selected, ~has_positive


def flip_relative_goal_yaw_in_actor_observation(
    observation: torch.Tensor,
    *,
    rel_goal_start: int = 4096 + 9 + 14 + 7,
) -> torch.Tensor:
    """Return an actor observation with only planar goal yaw reflected.

    The oracle teacher's frozen external actor contract concatenates the
    4,096-D semantic scene, hand(9), robot(14), previous action(7), and then a
    9-D relative goal ``[xyz, R00,R01,R02,R10,R11,R12]``.  Reflecting ``R01``
    and ``R10`` changes ``yaw`` to ``-yaw`` for the support-preserving planar
    task while leaving geometry, robot state, translation, and dynamics
    identical.  This provides a strict counterfactual audit of whether a
    learned policy actually uses the recoverable yaw input.
    """

    if observation.ndim != 2:
        raise ValueError("actor observation must have shape [batch, features]")
    if rel_goal_start < 0 or observation.shape[1] < rel_goal_start + 9:
        raise ValueError("actor observation does not contain the 9-D relative goal")
    counterfactual = observation.clone()
    counterfactual[:, rel_goal_start + 4] *= -1.0  # R01
    counterfactual[:, rel_goal_start + 6] *= -1.0  # R10
    return counterfactual


def discounted_potential_shaping(
    previous_cost: torch.Tensor,
    current_cost: torch.Tensor,
    *,
    discount_factor: float,
) -> torch.Tensor:
    """Return policy-invariant shaping for a non-negative state cost.

    With potential ``phi(s) = -cost(s)``, the standard shaping transition is
    ``gamma * phi(s') - phi(s) = cost(s) - gamma * cost(s')``.  Discounted
    sums therefore telescope, so reaching a lower cost and later retreating
    cannot earn more shaping return than ending at the same state directly.
    """

    if previous_cost.shape != current_cost.shape:
        raise ValueError("previous and current costs must have matching shapes")
    if not 0.0 < discount_factor <= 1.0:
        raise ValueError("discount_factor must be in (0, 1]")
    if torch.any(previous_cost < 0.0) or torch.any(current_cost < 0.0):
        raise ValueError("potential costs must be non-negative")
    return previous_cost - float(discount_factor) * current_cost


def axis_aligned_bounding_box_keypoints(points: torch.Tensor) -> torch.Tensor:
    """Return the eight canonical AABB corners for a point set.

    DyWA's released pose reward uses the object's canonical bounding box as
    keypoints.  Building the corners from the already-loaded canonical surface
    samples keeps the reward asset-agnostic and does not add privileged actor
    observations.
    """

    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [..., points, 3]")
    if points.shape[-2] < 1:
        raise ValueError("points must contain at least one point")
    lower = points.amin(dim=-2)
    upper = points.amax(dim=-2)
    x0, y0, z0 = lower.unbind(dim=-1)
    x1, y1, z1 = upper.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((x0, y0, z0), dim=-1),
            torch.stack((x0, y0, z1), dim=-1),
            torch.stack((x0, y1, z0), dim=-1),
            torch.stack((x0, y1, z1), dim=-1),
            torch.stack((x1, y0, z0), dim=-1),
            torch.stack((x1, y0, z1), dim=-1),
            torch.stack((x1, y1, z0), dim=-1),
            torch.stack((x1, y1, z1), dim=-1),
        ),
        dim=-2,
    )


def dywa_exponential_keypoint_potential(
    point_distances: torch.Tensor,
    *,
    amplitude: float = 0.302,
    distance_rate: float = 243.12,
    exponential_base: float = 0.995,
) -> torch.Tensor:
    """Return DyWA's mean exponential potential over object keypoints.

    ``point_distances`` contains corresponding current-to-goal keypoint
    distances in metres.  DyWA applies the exponential to every keypoint
    before averaging, rather than collapsing translation and rotation into
    independently weighted reward terms.
    """

    if point_distances.ndim < 1:
        raise ValueError("point_distances must have at least one dimension")
    if amplitude <= 0.0 or distance_rate <= 0.0:
        raise ValueError("potential amplitude and distance rate must be positive")
    if not 0.0 < exponential_base < 1.0:
        raise ValueError("exponential_base must be in (0, 1)")
    if torch.any(point_distances < 0.0):
        raise ValueError("point distances must be non-negative")
    per_point = float(amplitude) * torch.pow(
        float(exponential_base), float(distance_rate) * point_distances
    )
    return per_point.mean(dim=-1)


def discounted_score_potential_shaping(
    previous_score: torch.Tensor,
    current_score: torch.Tensor,
    *,
    discount_factor: float,
) -> torch.Tensor:
    """Return ``gamma * phi(s') - phi(s)`` for a non-negative score."""

    if previous_score.shape != current_score.shape:
        raise ValueError("previous and current scores must have matching shapes")
    if not 0.0 < discount_factor <= 1.0:
        raise ValueError("discount_factor must be in (0, 1]")
    if torch.any(previous_score < 0.0) or torch.any(current_score < 0.0):
        raise ValueError("potential scores must be non-negative")
    return float(discount_factor) * current_score - previous_score


def gate_navigation_at_legal_contact(
    navigation_cost: torch.Tensor,
    target_distance: torch.Tensor,
    legal_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Disable pre-contact navigation once a legal contact is observable.

    The gate is a deterministic function of the same semantic cloud and hand
    geometry available to the actor.  It prevents a moving, goal-conditioned
    contact anchor from continuing to pull the hand around the handle while
    the object-pose progress reward is already asking it to push.
    """

    if navigation_cost.shape != target_distance.shape:
        raise ValueError("navigation cost and target distance must match")
    if legal_contact.shape != navigation_cost.shape:
        raise ValueError("legal contact must match navigation cost")
    if legal_contact.dtype != torch.bool:
        raise ValueError("legal contact gate must be boolean")
    zero_cost = torch.zeros_like(navigation_cost)
    zero_distance = torch.zeros_like(target_distance)
    return (
        torch.where(legal_contact, zero_cost, navigation_cost),
        torch.where(legal_contact, zero_distance, target_distance),
    )


def sampled_segment_minimum_clearance(
    start: torch.Tensor,
    end: torch.Tensor,
    obstacle_points: torch.Tensor,
    *,
    obstacle_mask: torch.Tensor | None = None,
    num_samples: int = 9,
    start_fraction: float = 0.10,
    end_fraction: float = 0.85,
) -> torch.Tensor:
    """Return minimum point clearance along a batched straight corridor.

    This simulator-independent primitive deliberately samples only the open
    segment.  A caller can therefore test whether the route *toward* a legal
    contact surface is occluded without treating the adjacent object surface
    at the contact endpoint as an obstacle.  It is useful for semantic
    navigation potentials, but does not expose or prescribe a waypoint.
    """

    if start.ndim != 2 or start.shape[-1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have matching [batch, 3] shapes")
    if (
        obstacle_points.ndim != 3
        or obstacle_points.shape[0] != start.shape[0]
        or obstacle_points.shape[-1] != 3
    ):
        raise ValueError("obstacle points must have shape [batch, points, 3]")
    if obstacle_mask is not None and obstacle_mask.shape != obstacle_points.shape[:2]:
        raise ValueError("obstacle mask must match [batch, points]")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if not 0.0 <= start_fraction < end_fraction < 1.0:
        raise ValueError("segment fractions must satisfy 0 <= start < end < 1")

    fractions = torch.linspace(
        float(start_fraction),
        float(end_fraction),
        int(num_samples),
        device=start.device,
        dtype=start.dtype,
    )
    samples = start[:, None, :] + fractions[None, :, None] * (
        end - start
    )[:, None, :]
    distance = torch.cdist(samples, obstacle_points)
    if obstacle_mask is not None:
        distance = distance.masked_fill(~obstacle_mask[:, None, :], torch.inf)
    return distance.amin(dim=(1, 2))


def semantic_ring_route_geometry(
    start: torch.Tensor,
    end: torch.Tensor,
    obstacle_points: torch.Tensor,
    *,
    obstacle_mask: torch.Tensor | None = None,
    body_radius: float = 0.03,
    contact_clearance: float = 0.01,
    detour_margin: float = 0.02,
    num_candidates: int = 12,
    num_segment_samples: int = 7,
    obstacle_sample_count: int = 96,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build direct and point-cloud-derived ring routes around an obstacle.

    The returned tensors have shape ``[batch, 1 + num_candidates]`` and hold
    route length and body-inflated clearance respectively.  Candidate zero is
    always the direct route.  The remaining candidates pass through support
    points on a conservative ring around the masked obstacle cloud.  This is
    a geometry primitive: it does not choose a route or expose any waypoint to
    a policy.

    A deterministic masked subsample bounds the segment-clearance cost.  The
    full cloud is still used to construct every support point, so the ring
    encloses the complete annotated non-safe geometry rather than the sample.
    """

    if start.ndim != 2 or start.shape[-1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have matching [batch, 3] shapes")
    if (
        obstacle_points.ndim != 3
        or obstacle_points.shape[0] != start.shape[0]
        or obstacle_points.shape[-1] != 3
    ):
        raise ValueError("obstacle points must have shape [batch, points, 3]")
    if obstacle_mask is not None and obstacle_mask.shape != obstacle_points.shape[:2]:
        raise ValueError("obstacle mask must match [batch, points]")
    if body_radius < 0.0 or contact_clearance < 0.0 or detour_margin < 0.0:
        raise ValueError("route radii and clearances must be non-negative")
    if num_candidates < 4:
        raise ValueError("num_candidates must be at least four")
    if num_segment_samples < 2:
        raise ValueError("num_segment_samples must be at least two")
    if obstacle_sample_count <= 0:
        raise ValueError("obstacle_sample_count must be positive")

    batch_size, point_count, _ = obstacle_points.shape
    if point_count == 0:
        raise ValueError("obstacle_points must contain at least one point")
    if obstacle_mask is None:
        obstacle_mask = torch.ones(
            (batch_size, point_count), device=start.device, dtype=torch.bool
        )
    else:
        obstacle_mask = obstacle_mask.bool()
    has_obstacle = obstacle_mask.any(dim=1)
    weights = obstacle_mask.to(obstacle_points.dtype)
    center = torch.sum(obstacle_points * weights[..., None], dim=1) / torch.clamp(
        weights.sum(dim=1, keepdim=True), min=1.0
    )

    angles = torch.arange(
        int(num_candidates), device=start.device, dtype=start.dtype
    ) * (2.0 * math.pi / float(num_candidates))
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    relative_xy = obstacle_points[..., :2] - center[:, None, :2]
    support = torch.einsum("bpd,kd->bpk", relative_xy, directions)
    support = support.masked_fill(~obstacle_mask[..., None], -torch.inf).amax(dim=1)
    support = torch.where(has_obstacle[:, None], support, torch.zeros_like(support))
    ring_radius = support + float(body_radius + contact_clearance + detour_margin)
    candidate_xy = center[:, None, :2] + ring_radius[..., None] * directions[None]
    # Keep a legal planar approach while allowing a high reset pose to remain
    # high until it has gone around the semantic obstacle.
    candidate_z = torch.maximum(start[:, 2], end[:, 2])[:, None, None].expand(
        -1, int(num_candidates), 1
    )
    candidates = torch.cat((candidate_xy, candidate_z), dim=-1)

    # Select evenly spaced valid canonical indices without a [B,S,P] ranking
    # tensor.  Repetition when an asset has fewer than S non-safe points is
    # harmless and keeps the vectorized shape fixed.
    sample_count = min(int(obstacle_sample_count), point_count)
    canonical_indices = torch.arange(point_count, device=start.device)[None].expand(
        batch_size, -1
    )
    sorted_valid_indices = torch.where(
        obstacle_mask, canonical_indices, point_count + canonical_indices
    ).sort(dim=1).values
    valid_count = obstacle_mask.sum(dim=1)
    rank_fraction = torch.linspace(
        0.0, 1.0, sample_count, device=start.device, dtype=start.dtype
    )
    sample_ranks = torch.round(
        rank_fraction[None] * torch.clamp(valid_count - 1, min=0)[:, None]
    ).long()
    sampled_indices = torch.gather(sorted_valid_indices, 1, sample_ranks).clamp(
        max=point_count - 1
    )
    sampled_obstacles = torch.gather(
        obstacle_points, 1, sampled_indices[..., None].expand(-1, -1, 3)
    )
    sampled_mask = has_obstacle[:, None].expand(-1, sample_count)

    direct_raw_clearance = sampled_segment_minimum_clearance(
        start,
        end,
        sampled_obstacles,
        obstacle_mask=sampled_mask,
        num_samples=num_segment_samples,
        start_fraction=0.0,
        end_fraction=0.85,
    )

    expanded_obstacles = sampled_obstacles[:, None].expand(
        -1, int(num_candidates), -1, -1
    ).reshape(batch_size * int(num_candidates), sample_count, 3)
    expanded_mask = sampled_mask[:, None].expand(
        -1, int(num_candidates), -1
    ).reshape(batch_size * int(num_candidates), sample_count)
    start_to_ring = sampled_segment_minimum_clearance(
        start[:, None].expand(-1, int(num_candidates), -1).reshape(-1, 3),
        candidates.reshape(-1, 3),
        expanded_obstacles,
        obstacle_mask=expanded_mask,
        num_samples=num_segment_samples,
        start_fraction=0.0,
        end_fraction=0.95,
    ).reshape(batch_size, int(num_candidates))
    ring_to_end = sampled_segment_minimum_clearance(
        candidates.reshape(-1, 3),
        end[:, None].expand(-1, int(num_candidates), -1).reshape(-1, 3),
        expanded_obstacles,
        obstacle_mask=expanded_mask,
        num_samples=num_segment_samples,
        start_fraction=0.05,
        end_fraction=0.85,
    ).reshape(batch_size, int(num_candidates))

    direct_length = torch.linalg.vector_norm(end - start, dim=1)
    detour_length = torch.linalg.vector_norm(candidates - start[:, None], dim=2)
    detour_length += torch.linalg.vector_norm(end[:, None] - candidates, dim=2)
    route_length = torch.cat((direct_length[:, None], detour_length), dim=1)
    raw_clearance = torch.cat(
        (direct_raw_clearance[:, None], torch.minimum(start_to_ring, ring_to_end)),
        dim=1,
    )
    clearance = torch.clamp(raw_clearance - float(body_radius), min=0.0)
    clearance = torch.where(
        has_obstacle[:, None], clearance, torch.full_like(clearance, torch.inf)
    )
    first_edge_target = torch.cat((end[:, None, :], candidates), dim=1)
    return route_length, clearance, first_edge_target


def goal_swept_semantic_point_index(
    start_points: torch.Tensor,
    end_points: torch.Tensor,
    obstacle_points: torch.Tensor,
    *,
    point_mask: torch.Tensor | None = None,
    num_samples: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the semantic point whose goal sweep passes closest to a blocker.

    The selection is computed from a pointwise interpolation between the
    current and goal clouds.  It therefore includes the goal-conditioned
    rigid-body motion while avoiding a quadratic target--obstacle distance
    tensor.  The obstacle centroid is used only to identify the most relevant
    semantic point; route legality is still evaluated against the full
    obstacle cloud by :func:`semantic_ring_route_geometry`.
    """

    if (
        start_points.ndim != 3
        or start_points.shape[-1] != 3
        or end_points.shape != start_points.shape
    ):
        raise ValueError(
            "start_points and end_points must have matching [batch, points, 3] shapes"
        )
    if (
        obstacle_points.ndim != 3
        or obstacle_points.shape[0] != start_points.shape[0]
        or obstacle_points.shape[-1] != 3
        or obstacle_points.shape[1] == 0
    ):
        raise ValueError("obstacle_points must have shape [batch, obstacles, 3]")
    if point_mask is None:
        point_mask = torch.ones(
            start_points.shape[:2], device=start_points.device, dtype=torch.bool
        )
    elif point_mask.shape != start_points.shape[:2]:
        raise ValueError("point_mask must match [batch, points]")
    else:
        point_mask = point_mask.bool()
    if not torch.all(point_mask.any(dim=1)):
        raise ValueError("every batch item must contain at least one semantic point")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")

    fractions = torch.linspace(
        0.0,
        1.0,
        int(num_samples),
        device=start_points.device,
        dtype=start_points.dtype,
    )
    swept_points = start_points[:, :, None, :] + fractions[None, None, :, None] * (
        end_points - start_points
    )[:, :, None, :]
    obstacle_center = obstacle_points.mean(dim=1)
    center_distance = torch.linalg.vector_norm(
        swept_points - obstacle_center[:, None, None, :], dim=-1
    ).amin(dim=2)
    center_distance = center_distance.masked_fill(~point_mask, torch.inf)
    selected_distance, selected_index = center_distance.min(dim=1)
    return selected_index, selected_distance


def rigid_body_ring_route_aabb_clearance(
    start_points: torch.Tensor,
    end_points: torch.Tensor,
    obstacle_points: torch.Tensor,
    first_edge_target: torch.Tensor,
    critical_start: torch.Tensor,
    critical_end: torch.Tensor,
    *,
    point_mask: torch.Tensor | None = None,
    num_segment_samples: int = 5,
) -> torch.Tensor:
    """Conservatively screen ring-route homotopies for a protected rigid body.

    A point route can pass on either side of a blocker even when only one side
    is feasible for the complete protected part.  This helper translates the
    complete start/goal semantic clouds along every two-edge ring route and
    returns their minimum clearance from the obstacle AABB.  It is intended as
    a cheap, reset-time homotopy selector for an isolated blocker; exact C2
    accounting continues to use the full point-cloud/PhysX predicates.

    Candidate zero is the direct start-to-goal sweep.  Remaining candidates
    pass through the corresponding ``first_edge_target`` while the protected
    cloud interpolates between its current and goal configurations.
    """

    if (
        start_points.ndim != 3
        or start_points.shape[-1] != 3
        or end_points.shape != start_points.shape
    ):
        raise ValueError(
            "start_points and end_points must have matching [batch, points, 3] shapes"
        )
    batch_size, _, _ = start_points.shape
    if (
        obstacle_points.ndim != 3
        or obstacle_points.shape[0] != batch_size
        or obstacle_points.shape[-1] != 3
        or obstacle_points.shape[1] == 0
    ):
        raise ValueError("obstacle_points must have shape [batch, obstacles, 3]")
    if (
        first_edge_target.ndim != 3
        or first_edge_target.shape[0] != batch_size
        or first_edge_target.shape[-1] != 3
        or first_edge_target.shape[1] == 0
    ):
        raise ValueError("first_edge_target must have shape [batch, routes, 3]")
    if critical_start.shape != (batch_size, 3) or critical_end.shape != (
        batch_size,
        3,
    ):
        raise ValueError("critical_start and critical_end must have shape [batch, 3]")
    if point_mask is None:
        point_mask = torch.ones(
            start_points.shape[:2], device=start_points.device, dtype=torch.bool
        )
    elif point_mask.shape != start_points.shape[:2]:
        raise ValueError("point_mask must match [batch, points]")
    else:
        point_mask = point_mask.bool()
    if not torch.all(point_mask.any(dim=1)):
        raise ValueError("every batch item must contain at least one semantic point")
    if num_segment_samples < 2:
        raise ValueError("num_segment_samples must be at least two")

    lower = obstacle_points.amin(dim=1)
    upper = obstacle_points.amax(dim=1)
    alpha = torch.linspace(
        0.0,
        1.0,
        int(num_segment_samples),
        device=start_points.device,
        dtype=start_points.dtype,
    )

    def aabb_clearance(path_points: torch.Tensor) -> torch.Tensor:
        # path_points: [B, samples, points, 3]
        below = lower[:, None, None, :] - path_points
        above = path_points - upper[:, None, None, :]
        outside = torch.clamp(torch.maximum(below, above), min=0.0)
        distance = torch.linalg.vector_norm(outside, dim=-1)
        distance = distance.masked_fill(~point_mask[:, None, :], torch.inf)
        return distance.amin(dim=(1, 2))

    direct_alpha = torch.linspace(
        0.0,
        1.0,
        2 * int(num_segment_samples) - 1,
        device=start_points.device,
        dtype=start_points.dtype,
    )
    direct = start_points[:, None, :, :] + direct_alpha[
        None, :, None, None
    ] * (end_points - start_points)[:, None, :, :]
    clearances = [aabb_clearance(direct)]

    midpoint_points = 0.5 * (start_points + end_points)
    critical_midpoint = 0.5 * (critical_start + critical_end)
    for route_index in range(1, first_edge_target.shape[1]):
        shift = first_edge_target[:, route_index] - critical_midpoint
        route_midpoint = midpoint_points + shift[:, None, :]
        first_segment = start_points[:, None, :, :] + alpha[
            None, :, None, None
        ] * (route_midpoint - start_points)[:, None, :, :]
        second_segment = route_midpoint[:, None, :, :] + alpha[
            None, 1:, None, None
        ] * (end_points - route_midpoint)[:, None, :, :]
        path = torch.cat((first_segment, second_segment), dim=1)
        clearances.append(aabb_clearance(path))
    return torch.stack(clearances, dim=1)


def semantic_ring_route_candidates(
    start: torch.Tensor,
    end: torch.Tensor,
    obstacle_points: torch.Tensor,
    *,
    obstacle_mask: torch.Tensor | None = None,
    body_radius: float = 0.03,
    contact_clearance: float = 0.01,
    detour_margin: float = 0.02,
    num_candidates: int = 12,
    num_segment_samples: int = 7,
    obstacle_sample_count: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lengths and clearances for direct and semantic-ring routes."""

    route_length, clearance, _ = semantic_ring_route_geometry(
        start,
        end,
        obstacle_points,
        obstacle_mask=obstacle_mask,
        body_radius=body_radius,
        contact_clearance=contact_clearance,
        detour_margin=detour_margin,
        num_candidates=num_candidates,
        num_segment_samples=num_segment_samples,
        obstacle_sample_count=obstacle_sample_count,
    )
    return route_length, clearance


def semantic_clearance_recovery_direction(
    start: torch.Tensor,
    obstacle_points: torch.Tensor,
    *,
    obstacle_mask: torch.Tensor | None = None,
    safety_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an outward field and raw clearance for an inflated obstacle.

    This is the recovery branch for a state in which every sampled route is
    illegal, including the case where a coarse route graph loses feasibility
    before the controlled point itself enters the inflated semantic obstacle.
    It is a lexicographic feasibility direction, not a repulsive term added to
    goal attraction: while recovery is required its outward component cannot
    be cancelled by the goal direction.
    """

    if start.ndim != 2 or start.shape[-1] != 3:
        raise ValueError("start must have shape [batch, 3]")
    if (
        obstacle_points.ndim != 3
        or obstacle_points.shape[0] != start.shape[0]
        or obstacle_points.shape[-1] != 3
    ):
        raise ValueError("obstacle points must have shape [batch, points, 3]")
    if obstacle_mask is not None and obstacle_mask.shape != obstacle_points.shape[:2]:
        raise ValueError("obstacle mask must match [batch, points]")
    if safety_radius <= 0.0:
        raise ValueError("safety_radius must be positive")

    if obstacle_mask is None:
        obstacle_mask = torch.ones(
            obstacle_points.shape[:2], device=start.device, dtype=torch.bool
        )
    else:
        obstacle_mask = obstacle_mask.bool()
    offset = start[:, None, :] - obstacle_points
    distance = torch.linalg.vector_norm(offset, dim=2)
    masked_distance = distance.masked_fill(~obstacle_mask, torch.inf)
    minimum_raw_clearance = masked_distance.amin(dim=1)

    penetration = torch.clamp(float(safety_radius) - distance, min=0.0)
    penetration = penetration * obstacle_mask.to(distance.dtype)
    outward = torch.sum(
        penetration[..., None]
        * offset
        / torch.clamp(distance[..., None], min=1.0e-6),
        dim=1,
    )

    # Degenerate symmetric samples can cancel exactly.  Use the vector away
    # from the masked cloud centroid, then a deterministic vertical direction
    # only if the start is also exactly at that centroid.
    weights = obstacle_mask.to(obstacle_points.dtype)
    center = torch.sum(obstacle_points * weights[..., None], dim=1) / torch.clamp(
        weights.sum(dim=1, keepdim=True), min=1.0
    )
    center_outward = start - center
    outward_norm = torch.linalg.vector_norm(outward, dim=1, keepdim=True)
    outward = torch.where(outward_norm > 1.0e-6, outward, center_outward)
    outward_norm = torch.linalg.vector_norm(outward, dim=1, keepdim=True)
    fixed_outward = torch.zeros_like(outward)
    fixed_outward[:, 2] = 1.0
    outward = torch.where(outward_norm > 1.0e-6, outward, fixed_outward)
    direction = outward / torch.clamp(
        torch.linalg.vector_norm(outward, dim=1, keepdim=True), min=1.0e-6
    )
    return direction, minimum_raw_clearance


def semantic_tangential_recovery_direction(
    start: torch.Tensor,
    end: torch.Tensor,
    outward_direction: torch.Tensor,
    route_clearance: torch.Tensor,
    first_edge_target: torch.Tensor,
    *,
    contact_clearance: float,
    outward_weight: float = 0.5,
    tangent_weight: float = 1.0,
    side_preference_weight: float = 1.0,
) -> torch.Tensor:
    """Add a boundary-following component to an outward recovery field.

    Pure repulsion can create an escape attractor: the controlled point moves
    farther from both the obstacle and its legal contact target without ever
    making a sampled route feasible.  This primitive keeps a strictly positive
    outward component while choosing a detour-edge tangent that both has high
    current clearance and is lateral to the illegal direct route.  The tangent
    is projected onto the outward field's null space, so it cannot cancel the
    safety component.

    Candidate zero is the direct edge and is deliberately excluded.  No route
    node or selected index is exposed to the policy; the returned vector is a
    deterministic function of the current geometry.
    """

    if start.ndim != 2 or start.shape[-1] != 3 or end.shape != start.shape:
        raise ValueError("start and end must have matching [batch, 3] shapes")
    if outward_direction.shape != start.shape:
        raise ValueError("outward direction must match start")
    if (
        first_edge_target.ndim != 3
        or first_edge_target.shape[0] != start.shape[0]
        or first_edge_target.shape[-1] != 3
    ):
        raise ValueError("first edge targets must have shape [batch, routes, 3]")
    if route_clearance.shape != first_edge_target.shape[:2]:
        raise ValueError("route clearance must match [batch, routes]")
    if route_clearance.shape[1] < 2:
        raise ValueError("at least one non-direct route is required")
    if contact_clearance <= 0.0:
        raise ValueError("contact clearance must be positive")
    if (
        outward_weight <= 0.0
        or tangent_weight < 0.0
        or side_preference_weight < 0.0
    ):
        raise ValueError(
            "outward weight must be positive; tangent and side weights must be non-negative"
        )

    outward_direction = outward_direction / torch.clamp(
        torch.linalg.vector_norm(outward_direction, dim=1, keepdim=True),
        min=1.0e-6,
    )
    direct_direction = end - start
    direct_direction = direct_direction / torch.clamp(
        torch.linalg.vector_norm(direct_direction, dim=1, keepdim=True),
        min=1.0e-6,
    )
    candidate_direction = first_edge_target[:, 1:] - start[:, None, :]
    candidate_direction = candidate_direction / torch.clamp(
        torch.linalg.vector_norm(candidate_direction, dim=2, keepdim=True),
        min=1.0e-6,
    )

    # Boundary-following motion lies in the null space of the outward normal,
    # hence cannot reduce clearance to first order.
    tangent = candidate_direction - torch.sum(
        candidate_direction * outward_direction[:, None, :], dim=2, keepdim=True
    ) * outward_direction[:, None, :]
    tangent_norm = torch.linalg.vector_norm(tangent, dim=2)
    tangent_direction = tangent / torch.clamp(tangent_norm[..., None], min=1.0e-6)

    direct_lateral = candidate_direction - torch.sum(
        candidate_direction * direct_direction[:, None, :], dim=2, keepdim=True
    ) * direct_direction[:, None, :]
    direct_lateral_norm = torch.linalg.vector_norm(direct_lateral, dim=2)
    world_up = torch.zeros_like(direct_direction)
    world_up[:, 2] = 1.0
    preferred_side = torch.linalg.cross(world_up, direct_direction, dim=1)
    # Keep the side convention inside the outward null space as well.  This
    # gives symmetric candidate sets a stable, observation-derived tie-breaker
    # instead of alternating left/right every step.
    preferred_side -= torch.sum(
        preferred_side * outward_direction, dim=1, keepdim=True
    ) * outward_direction
    preferred_norm = torch.linalg.vector_norm(preferred_side, dim=1, keepdim=True)
    fixed_side = torch.zeros_like(preferred_side)
    fixed_side[:, 1] = 1.0
    preferred_side = torch.where(
        preferred_norm > 1.0e-6, preferred_side, fixed_side
    )
    preferred_side /= torch.clamp(
        torch.linalg.vector_norm(preferred_side, dim=1, keepdim=True), min=1.0e-6
    )
    side_alignment = torch.sum(
        tangent_direction * preferred_side[:, None, :], dim=2
    )
    clearance_score = torch.clamp(
        route_clearance[:, 1:] / float(contact_clearance), min=0.0, max=1.0
    )
    # Among still-illegal detours, prefer clearance, a usable boundary tangent,
    # and motion lateral to the direct shortcut.  All terms are bounded, so no
    # single geometric tie-breaker can erase the other two.
    score = (
        clearance_score
        + tangent_norm
        + direct_lateral_norm
        + float(side_preference_weight) * side_alignment
    )
    selected_index = score.argmax(dim=1)
    batch_index = torch.arange(start.shape[0], device=start.device)
    selected_tangent = tangent_direction[batch_index, selected_index]

    recovery = (
        float(outward_weight) * outward_direction
        + float(tangent_weight) * selected_tangent
    )
    return recovery / torch.clamp(
        torch.linalg.vector_norm(recovery, dim=1, keepdim=True), min=1.0e-6
    )


def semantic_route_vector_field(
    start: torch.Tensor,
    end: torch.Tensor,
    obstacle_points: torch.Tensor,
    *,
    obstacle_mask: torch.Tensor | None = None,
    body_radius: float = 0.03,
    contact_clearance: float = 0.01,
    detour_margin: float = 0.02,
    num_candidates: int = 12,
    num_segment_samples: int = 7,
    obstacle_sample_count: int = 96,
    recover_illegal_route: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a local free-space direction derived from semantic geometry.

    The vector is the normalized first edge of the shortest currently legal
    route.  It is recomputed from the current point cloud at every state, so it
    is a local navigation field rather than a stored task waypoint.  The other
    outputs are the detour mask, direct-route clearance, and selected-route
    clearance and are intended for reward gating and diagnostics.
    """

    route_length, route_clearance, first_edge_target = (
        semantic_ring_route_geometry(
            start,
            end,
            obstacle_points,
            obstacle_mask=obstacle_mask,
            body_radius=body_radius,
            contact_clearance=contact_clearance,
            detour_margin=detour_margin,
            num_candidates=num_candidates,
            num_segment_samples=num_segment_samples,
            obstacle_sample_count=obstacle_sample_count,
        )
    )
    legal_route = route_clearance >= float(contact_clearance)
    has_legal_route = legal_route.any(dim=1)
    selected_legal = route_length.masked_fill(~legal_route, torch.inf).argmin(dim=1)
    selected_fallback = route_length.argmin(dim=1)
    selected_index = torch.where(has_legal_route, selected_legal, selected_fallback)
    batch_index = torch.arange(start.shape[0], device=start.device)
    first_edge = first_edge_target[batch_index, selected_index] - start
    direction = first_edge / torch.clamp(
        torch.linalg.vector_norm(first_edge, dim=1, keepdim=True), min=1.0e-6
    )
    recovery_required = torch.zeros_like(has_legal_route)
    if recover_illegal_route:
        recovery_direction, _ = (
            semantic_clearance_recovery_direction(
                start,
                obstacle_points,
                obstacle_mask=obstacle_mask,
                safety_radius=float(body_radius + contact_clearance),
            )
        )
        recovery_direction = semantic_tangential_recovery_direction(
            start,
            end,
            recovery_direction,
            route_clearance,
            first_edge_target,
            contact_clearance=contact_clearance,
        )
        # Never reward an illegal straight shortcut.  Even before the start
        # point itself enters the inflation, a coarse ring graph can have an
        # empty legal set; move outward until at least one legal route exists.
        recovery_required = ~has_legal_route
        direction = torch.where(
            recovery_required[:, None], recovery_direction, direction
        )
    selected_clearance = route_clearance[batch_index, selected_index]
    return (
        direction,
        (selected_index > 0) | recovery_required,
        route_clearance[:, 0],
        selected_clearance,
    )


def route_conditioned_alignment(
    alignment: torch.Tensor,
    used_detour: torch.Tensor,
    *,
    direct_route_scale: float = 0.0,
) -> torch.Tensor:
    """Scale navigation-field progress without dropping direct-route learning.

    A value of zero preserves the original detour-only shaping.  A small
    positive direct-route scale lets one field replace Euclidean progress on
    unobstructed scenes while retaining a stronger lateral signal wherever a
    semantic detour is required.
    """

    if alignment.shape != used_detour.shape:
        raise ValueError("alignment and detour mask shapes must match")
    if not 0.0 <= direct_route_scale <= 1.0:
        raise ValueError("direct route scale must be in [0, 1]")
    scale = torch.where(
        used_detour.bool(),
        torch.ones_like(alignment),
        torch.full_like(alignment, float(direct_route_scale)),
    )
    return alignment * scale


def update_route_detour_commitment(
    previous_commitment: torch.Tensor, used_detour: torch.Tensor
) -> torch.Tensor:
    """Latch per-episode detour demand until the reward term is reset."""

    if previous_commitment.shape != used_detour.shape:
        raise ValueError("commitment and detour mask shapes must match")
    return previous_commitment.bool() | used_detour.bool()


def clearance_conditioned_route_scale(
    direct_clearance: torch.Tensor,
    *,
    contact_clearance: float,
    activation_clearance: float,
    direct_route_scale: float,
) -> torch.Tensor:
    """Blend direct-route shaping strength from the current route clearance.

    A direct segment that still grazes the inflated protected region retains
    full navigation strength.  The strength falls continuously to the
    unobstructed baseline only after the segment has acquired free-space
    margin.  Unlike a per-episode latch, this scale is fully determined by the
    current geometry and is therefore recoverable from the actor observation.
    """

    if contact_clearance < 0.0:
        raise ValueError("contact_clearance must be non-negative")
    if activation_clearance <= contact_clearance:
        raise ValueError("activation_clearance must exceed contact_clearance")
    if not 0.0 <= direct_route_scale <= 1.0:
        raise ValueError("direct_route_scale must be in [0, 1]")
    if torch.any(direct_clearance < 0.0):
        raise ValueError("direct_clearance must be non-negative")
    obstruction = torch.clamp(
        (float(activation_clearance) - direct_clearance)
        / float(activation_clearance - contact_clearance),
        min=0.0,
        max=1.0,
    )
    return float(direct_route_scale) + (
        1.0 - float(direct_route_scale)
    ) * obstruction


def distance_progress_during_contact(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    normalization_distance: float,
) -> torch.Tensor:
    """Return signed distance progress only while valid contact is active."""

    if contact_mask.shape != current_distance.shape:
        raise ValueError("contact mask and distance shapes must match")
    progress = normalized_distance_progress(
        previous_distance,
        current_distance,
        normalization_distance=normalization_distance,
    )
    return torch.where(contact_mask.bool(), progress, torch.zeros_like(progress))


def componentwise_progress_during_contact(
    previous_errors: torch.Tensor,
    current_errors: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    normalization_scales: tuple[float, ...],
) -> torch.Tensor:
    """Return independent signed progress for each pose-error component.

    Keeping the components separate prevents a large translation error from
    hiding useful orientation or support-height progress, as happens with a
    max-combined pose potential.
    """

    if previous_errors.shape != current_errors.shape:
        raise ValueError("previous and current error shapes must match")
    if current_errors.ndim < 2:
        raise ValueError("component errors must have a final component dimension")
    if contact_mask.shape != current_errors.shape[:-1]:
        raise ValueError("contact mask must match the error batch shape")
    if len(normalization_scales) != current_errors.shape[-1]:
        raise ValueError("one normalization scale is required per error component")
    scales = torch.as_tensor(
        normalization_scales,
        device=current_errors.device,
        dtype=current_errors.dtype,
    )
    if torch.any(scales <= 0.0):
        raise ValueError("normalization scales must be positive")
    progress = torch.clamp(
        (previous_errors - current_errors) / scales,
        min=-1.0,
        max=1.0,
    )
    return torch.where(
        contact_mask.bool().unsqueeze(-1),
        progress,
        torch.zeros_like(progress),
    )


def weighted_componentwise_pose_progress(
    previous_errors: torch.Tensor,
    current_errors: torch.Tensor,
    *,
    normalization_scales: tuple[float, ...],
    component_weights: tuple[float, ...],
) -> torch.Tensor:
    """Combine signed pose-component progress without a max bottleneck.

    Every component is normalized and clipped independently before the fixed
    weighted sum.  Consequently, improving XY still produces a signal while
    SO(3) is the largest normalized error (and vice versa).  This is a single
    current-transition potential: it contains no waypoint, contact latch, or
    episode phase.
    """

    if previous_errors.shape != current_errors.shape:
        raise ValueError("previous and current error shapes must match")
    if current_errors.ndim < 2:
        raise ValueError("component errors must have a final component dimension")
    if len(normalization_scales) != current_errors.shape[-1]:
        raise ValueError("one normalization scale is required per error component")
    if len(component_weights) != current_errors.shape[-1]:
        raise ValueError("one weight is required per error component")
    scales = torch.as_tensor(
        normalization_scales,
        device=current_errors.device,
        dtype=current_errors.dtype,
    )
    weights = torch.as_tensor(
        component_weights,
        device=current_errors.device,
        dtype=current_errors.dtype,
    )
    if torch.any(scales <= 0.0):
        raise ValueError("normalization scales must be positive")
    if torch.any(weights < 0.0) or not torch.any(weights > 0.0):
        raise ValueError("component weights must be non-negative with at least one positive")
    component_progress = torch.clamp(
        (previous_errors - current_errors) / scales,
        min=-1.0,
        max=1.0,
    )
    return torch.sum(component_progress * weights, dim=-1)


def positive_reference_relative_component_improvement(
    reference_errors: torch.Tensor,
    current_errors: torch.Tensor,
    *,
    reference_error_floors: tuple[float, ...],
    component_weights: tuple[float, ...],
) -> torch.Tensor:
    """Return one bounded score from independently visible pose improvements.

    Each XY/Z/SO(3) component is compared with its episode-constant reset
    reference before rectification.  Therefore, improvement in one component
    remains visible while another component is larger, but regressions do not
    create the contact-entry cost observed with signed transition progress.
    The non-negative weights are normalized internally, so the combined score
    remains in ``[0, 1]`` and has the same return scale as the preceding joint
    reset-relative diagnostic.
    """

    if reference_errors.shape != current_errors.shape:
        raise ValueError("reference and current error shapes must match")
    if current_errors.ndim < 2:
        raise ValueError("component errors must have a final component dimension")
    if len(reference_error_floors) != current_errors.shape[-1]:
        raise ValueError("one reference-error floor is required per component")
    if len(component_weights) != current_errors.shape[-1]:
        raise ValueError("one weight is required per error component")
    floors = torch.as_tensor(
        reference_error_floors,
        device=current_errors.device,
        dtype=current_errors.dtype,
    )
    weights = torch.as_tensor(
        component_weights,
        device=current_errors.device,
        dtype=current_errors.dtype,
    )
    if torch.any(floors <= 0.0):
        raise ValueError("reference-error floors must be positive")
    if torch.any(weights < 0.0) or not torch.any(weights > 0.0):
        raise ValueError("component weights must be non-negative with at least one positive")
    if torch.any(reference_errors < 0.0) or torch.any(current_errors < 0.0):
        raise ValueError("reference and current errors must be non-negative")
    denominator = torch.maximum(reference_errors, floors)
    component_improvement = torch.clamp(
        (reference_errors - current_errors) / denominator,
        min=0.0,
        max=1.0,
    )
    normalized_weights = weights / torch.sum(weights)
    return torch.sum(component_improvement * normalized_weights, dim=-1)


def positive_reference_relative_pareto_pose_improvement(
    reference_errors: torch.Tensor,
    current_errors: torch.Tensor,
    *,
    reference_planar_error_floor_m: float = 0.02,
    reference_rotation_error_floor_rad: float = 0.10,
    support_height_tolerance_m: float = 0.01,
) -> torch.Tensor:
    """Reward only joint XY/SO(3) improvement on the support manifold.

    The minimum acts as a Pareto conjunction: neither planar nor full-rotation
    progress can compensate for regression in the other.  The third argument
    is the remaining margin to the strict support-height tolerance, so tipping
    or lifting beyond that tolerance also zeroes the score.  A stationary
    reset pose and every non-Pareto-improving pose receive exactly zero.  The
    result is one bounded current-state scalar with no waypoint or phase.
    """

    if reference_errors.shape != current_errors.shape:
        raise ValueError("reference and current error shapes must match")
    if current_errors.ndim < 2 or current_errors.shape[-1] != 3:
        raise ValueError("pose errors must have final components [XY, Z, SO(3)]")
    if reference_planar_error_floor_m <= 0.0:
        raise ValueError("reference planar-error floor must be positive")
    if reference_rotation_error_floor_rad <= 0.0:
        raise ValueError("reference rotation-error floor must be positive")
    if support_height_tolerance_m <= 0.0:
        raise ValueError("support height tolerance must be positive")
    if torch.any(reference_errors < 0.0) or torch.any(current_errors < 0.0):
        raise ValueError("reference and current errors must be non-negative")

    planar_denominator = torch.clamp(
        reference_errors[..., 0], min=reference_planar_error_floor_m
    )
    rotation_denominator = torch.clamp(
        reference_errors[..., 2], min=reference_rotation_error_floor_rad
    )
    planar_improvement = (
        reference_errors[..., 0] - current_errors[..., 0]
    ) / planar_denominator
    rotation_improvement = (
        reference_errors[..., 2] - current_errors[..., 2]
    ) / rotation_denominator
    support_height_margin = 1.0 - (
        current_errors[..., 1] / float(support_height_tolerance_m)
    )
    joint_improvement = torch.minimum(
        torch.minimum(planar_improvement, rotation_improvement),
        support_height_margin,
    )
    return torch.clamp(joint_improvement, min=0.0, max=1.0)


def positive_distance_progress_during_contact(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    normalization_distance: float,
) -> torch.Tensor:
    """Return only goal-directed progress made during valid contact."""

    return torch.clamp(
        distance_progress_during_contact(
            previous_distance,
            current_distance,
            contact_mask,
            normalization_distance=normalization_distance,
        ),
        min=0.0,
    )


def near_goal_motion_cost(
    pose_error: torch.Tensor,
    linear_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    *,
    activation_pose_error: float,
    linear_speed_scale: float,
    angular_speed_scale: float,
) -> torch.Tensor:
    """Penalize normalized target motion increasingly close to the pose goal."""

    if pose_error.shape != linear_speed.shape or pose_error.shape != angular_speed.shape:
        raise ValueError("pose error and speed shapes must match")
    if min(activation_pose_error, linear_speed_scale, angular_speed_scale) <= 0.0:
        raise ValueError("motion-cost scales must be positive")
    gate = torch.clamp(
        (activation_pose_error - pose_error) / activation_pose_error,
        min=0.0,
        max=1.0,
    )
    normalized_motion = 0.5 * (
        torch.clamp(linear_speed / linear_speed_scale, min=0.0, max=1.0)
        + torch.clamp(angular_speed / angular_speed_scale, min=0.0, max=1.0)
    )
    return gate * normalized_motion


def smooth_max_normalized_pose_error(
    planar_error: torch.Tensor,
    height_error: torch.Tensor,
    rotation_error: torch.Tensor,
    *,
    planar_threshold: float,
    height_threshold: float,
    rotation_threshold: float,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Combine support-aware pose errors while emphasizing the worst one.

    Each component is expressed in units of its success threshold.  The
    zero-adjusted log-sum-exp is a smooth maximum: it is zero at the exact
    goal and equals one when all three errors are exactly at threshold.
    """

    if planar_error.shape != height_error.shape or planar_error.shape != rotation_error.shape:
        raise ValueError("pose error component shapes must match")
    if min(planar_threshold, height_threshold, rotation_threshold, temperature) <= 0.0:
        raise ValueError("pose thresholds and temperature must be positive")
    normalized = torch.stack(
        (
            planar_error / planar_threshold,
            height_error / height_threshold,
            rotation_error / rotation_threshold,
        ),
        dim=-1,
    )
    return temperature * (
        torch.logsumexp(normalized / temperature, dim=-1) - math.log(3.0)
    )


def bounded_joint_pose_tracking_cost(
    planar_error: torch.Tensor,
    height_error: torch.Tensor,
    rotation_error: torch.Tensor,
    *,
    planar_scale: float,
    height_scale: float,
    rotation_scale: float,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Bounded state cost that remains sensitive to every pose component.

    The smooth maximum prevents a policy from collecting a high score by
    fixing yaw while ignoring XY (or vice versa).  Converting its Gaussian
    tracking score to a cost yields zero only at the joint pose goal and
    approaches one for a large error, avoiding a positive living reward for
    merely holding the initial pose.
    """

    error = smooth_max_normalized_pose_error(
        planar_error,
        height_error,
        rotation_error,
        planar_threshold=planar_scale,
        height_threshold=height_scale,
        rotation_threshold=rotation_scale,
        temperature=temperature,
    )
    return 1.0 - torch.exp(-0.5 * torch.square(error))


def reference_relative_pose_improvement(
    reference_cost: torch.Tensor,
    current_cost: torch.Tensor,
    *,
    normalization_cost: float = 0.25,
) -> torch.Tensor:
    """Bound signed pose improvement relative to a contact-time baseline.

    A zero reward at the reference state avoids the negative reward jump that
    otherwise makes first contact less attractive.  Improvements are positive
    and regressions are negative; clipping keeps rare impacts from dominating
    PPO updates.
    """

    if reference_cost.shape != current_cost.shape:
        raise ValueError("reference and current costs must have matching shapes")
    if normalization_cost <= 0.0:
        raise ValueError("normalization cost must be positive")
    return torch.clamp(
        (reference_cost - current_cost) / float(normalization_cost),
        min=-1.0,
        max=1.0,
    )


def planar_pose_success(
    current_position: torch.Tensor,
    current_quaternion: torch.Tensor,
    goal_pose: torch.Tensor,
    *,
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Evaluate DAPL's planar-position and full-orientation success rule.

    Quaternions use the Isaac ``[w, x, y, z]`` convention.  Normalizing both
    inputs makes the metric robust to small simulator drift, and taking the
    absolute inner product handles the equivalent ``q`` and ``-q`` forms.
    """

    if current_position.shape[-1] != 3:
        raise ValueError("current_position must have three coordinates")
    if current_quaternion.shape[-1] != 4:
        raise ValueError("current_quaternion must have four coordinates")
    if goal_pose.shape[-1] != 7:
        raise ValueError("goal_pose must contain position and wxyz quaternion")
    if current_position.shape[:-1] != current_quaternion.shape[:-1]:
        raise ValueError("current position and quaternion batch shapes must match")
    if current_position.shape[:-1] != goal_pose.shape[:-1]:
        raise ValueError("current and goal batch shapes must match")
    if position_threshold <= 0.0 or rotation_threshold <= 0.0:
        raise ValueError("success thresholds must be positive")

    planar_distance = torch.linalg.vector_norm(
        goal_pose[..., :2] - current_position[..., :2], dim=-1
    )
    current_quaternion = functional.normalize(current_quaternion, dim=-1)
    goal_quaternion = functional.normalize(goal_pose[..., 3:7], dim=-1)
    quaternion_dot = torch.sum(current_quaternion * goal_quaternion, dim=-1)
    angular_distance = 2.0 * torch.acos(
        torch.clamp(torch.abs(quaternion_dot), max=1.0)
    )
    return (planar_distance < position_threshold) & (
        angular_distance < rotation_threshold
    )


def support_aware_pose_success(
    current_position: torch.Tensor,
    current_quaternion: torch.Tensor,
    goal_pose: torch.Tensor,
    *,
    planar_position_threshold: float = 0.02,
    height_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Evaluate a same-support-face pose goal without ignoring tipping.

    Planar displacement is the primary pushing objective.  Height is checked
    separately, while the geodesic quaternion distance jointly constrains yaw,
    roll, and pitch.  This is more interpretable than a single XYZ radius for
    tabletop pushing and prevents a tipped object from passing an XY-only goal.
    """

    if current_position.shape[-1] != 3:
        raise ValueError("current_position must have three coordinates")
    if current_quaternion.shape[-1] != 4:
        raise ValueError("current_quaternion must have four coordinates")
    if goal_pose.shape[-1] != 7:
        raise ValueError("goal_pose must contain position and wxyz quaternion")
    if current_position.shape[:-1] != current_quaternion.shape[:-1]:
        raise ValueError("current position and quaternion batch shapes must match")
    if current_position.shape[:-1] != goal_pose.shape[:-1]:
        raise ValueError("current and goal batch shapes must match")
    if (
        planar_position_threshold <= 0.0
        or height_threshold <= 0.0
        or rotation_threshold <= 0.0
    ):
        raise ValueError("success thresholds must be positive")

    planar_distance = torch.linalg.vector_norm(
        goal_pose[..., :2] - current_position[..., :2], dim=-1
    )
    height_error = torch.abs(goal_pose[..., 2] - current_position[..., 2])
    current_quaternion = functional.normalize(current_quaternion, dim=-1)
    goal_quaternion = functional.normalize(goal_pose[..., 3:7], dim=-1)
    quaternion_dot = torch.sum(current_quaternion * goal_quaternion, dim=-1)
    angular_distance = 2.0 * torch.acos(
        torch.clamp(torch.abs(quaternion_dot), max=1.0)
    )
    return (
        (planar_distance < planar_position_threshold)
        & (height_error < height_threshold)
        & (angular_distance < rotation_threshold)
    )


def update_consecutive_success_count(
    previous_count: torch.Tensor, pose_is_valid: torch.Tensor
) -> torch.Tensor:
    """Increment consecutive valid-pose counts and reset broken streaks."""

    if previous_count.shape != pose_is_valid.shape:
        raise ValueError("success count and validity tensors must have matching shapes")
    if previous_count.dtype == torch.bool or previous_count.is_floating_point():
        raise ValueError("success count must use an integer dtype")
    return torch.where(
        pose_is_valid,
        previous_count + 1,
        torch.zeros_like(previous_count),
    )
