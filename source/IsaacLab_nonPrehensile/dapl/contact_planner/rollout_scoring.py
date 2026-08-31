"""Joint-pose scoring for physics-rollout contact selection.

The score is expressed in units of the task thresholds.  An exact penalty is
applied to every violated predicate, so a small reduction in the current worst
error cannot silently push another pose component outside its acceptance set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicsRolloutScoringConfig:
    """Safety and task contract used to rank restored simulator rollouts."""

    planar_threshold_m: float = 0.020
    height_threshold_m: float = 0.010
    rotation_threshold_rad: float = 0.100
    predicate_violation_weight: float = 1.0
    mean_ratio_weight: float = 0.25
    minimum_cost_improvement: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "planar_threshold_m",
            "height_threshold_m",
            "rotation_threshold_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.predicate_violation_weight < 0.0:
            raise ValueError("predicate_violation_weight must be non-negative")
        if self.mean_ratio_weight < 0.0:
            raise ValueError("mean_ratio_weight must be non-negative")
        if self.minimum_cost_improvement < 0.0:
            raise ValueError("minimum_cost_improvement must be non-negative")


def joint_threshold_cost(
    planar_error: torch.Tensor,
    height_error: torch.Tensor,
    rotation_error: torch.Tensor,
    cfg: PhysicsRolloutScoringConfig,
) -> torch.Tensor:
    """Return an exact-penalty cost for the strict joint pose predicate.

    The sum term charges every violated threshold and the maximum term keeps
    pressure on the largest violation.  Once all predicates are satisfied,
    only the low-weight mean term remains to rank feasible poses.
    """

    if not (
        planar_error.shape == height_error.shape == rotation_error.shape
    ):
        raise ValueError("pose error tensors must have identical shapes")
    ratios = torch.stack(
        (
            planar_error / cfg.planar_threshold_m,
            height_error / cfg.height_threshold_m,
            rotation_error / cfg.rotation_threshold_rad,
        ),
        dim=-1,
    )
    if not torch.isfinite(ratios).all():
        raise ValueError("pose errors must be finite")
    if torch.any(ratios < 0.0):
        raise ValueError("pose errors must be non-negative")
    violation = torch.clamp(ratios - 1.0, min=0.0)
    violated_predicates = (ratios >= 1.0).sum(dim=-1).to(ratios.dtype)
    return (
        cfg.predicate_violation_weight * violated_predicates
        + violation.sum(dim=-1)
        + violation.amax(dim=-1)
        + cfg.mean_ratio_weight * ratios.mean(dim=-1)
    )


def rank_physics_rollouts(
    *,
    current_planar_error: torch.Tensor,
    current_height_error: torch.Tensor,
    current_rotation_error: torch.Tensor,
    rollout_planar_error: torch.Tensor,
    rollout_height_error: torch.Tensor,
    rollout_rotation_error: torch.Tensor,
    enabled: torch.Tensor,
    legal_safe_contact: torch.Tensor,
    c1_violation: torch.Tensor,
    cfg: PhysicsRolloutScoringConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the safest improving rollout for each environment.

    Rollout tensors and masks have shape ``[B,K]``.  The returned tuple is
    ``(best_index[B], has_improving_candidate[B], scores[B,K])``.  Invalid or
    non-improving candidates receive an infinite score and ``best_index`` is
    ``-1`` when the executor must fail closed.
    """

    rollout_shape = rollout_planar_error.shape
    if len(rollout_shape) != 2:
        raise ValueError("rollout errors must have shape [B, K]")
    for name, value in (
        ("rollout_height_error", rollout_height_error),
        ("rollout_rotation_error", rollout_rotation_error),
        ("enabled", enabled),
        ("legal_safe_contact", legal_safe_contact),
        ("c1_violation", c1_violation),
    ):
        if value.shape != rollout_shape:
            raise ValueError(f"{name} must match rollout error shape")
    batch_size = rollout_shape[0]
    for name, value in (
        ("current_planar_error", current_planar_error),
        ("current_height_error", current_height_error),
        ("current_rotation_error", current_rotation_error),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")

    current_cost = joint_threshold_cost(
        current_planar_error,
        current_height_error,
        current_rotation_error,
        cfg,
    )
    rollout_cost = joint_threshold_cost(
        rollout_planar_error,
        rollout_height_error,
        rollout_rotation_error,
        cfg,
    )
    legal = enabled.bool() & legal_safe_contact.bool() & ~c1_violation.bool()
    improving = rollout_cost <= (
        current_cost[:, None] - cfg.minimum_cost_improvement
    )
    valid = legal & improving
    scores = torch.where(
        valid,
        rollout_cost,
        torch.full_like(rollout_cost, torch.inf),
    )
    best_score, best_index = scores.min(dim=1)
    has_candidate = torch.isfinite(best_score)
    best_index = torch.where(
        has_candidate, best_index, torch.full_like(best_index, -1)
    )
    return best_index, has_candidate, scores


def rank_physics_rollout_pairs(
    *,
    current_cost: torch.Tensor,
    one_step_cost: torch.Tensor,
    pair_cost: torch.Tensor,
    legal_safe_contact: torch.Tensor,
    minimum_cost_improvement: float,
    intermediate_cost_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the first action of a two-step shooting horizon.

    ``pair_cost[b, i, j]`` is the predicted terminal cost after first action
    ``i`` and local effect ``j``.  Both effects must have passed independent
    C1/contact rollouts.  The returned first/second indices are ``-1`` when no
    pair improves on the current state.  A soft intermediate penalty avoids
    selecting unnecessarily disruptive first actions while still permitting
    the non-monotonic maneuver that a one-step controller cannot express.
    """

    if pair_cost.ndim != 3 or pair_cost.shape[1] != pair_cost.shape[2]:
        raise ValueError("pair_cost must have shape [B, K, K]")
    batch_size, candidate_count, _ = pair_cost.shape
    if current_cost.shape != (batch_size,):
        raise ValueError("current_cost must have shape [B]")
    expected = (batch_size, candidate_count)
    if one_step_cost.shape != expected:
        raise ValueError("one_step_cost must have shape [B, K]")
    if legal_safe_contact.shape != expected:
        raise ValueError("legal_safe_contact must have shape [B, K]")
    if minimum_cost_improvement < 0.0:
        raise ValueError("minimum_cost_improvement must be non-negative")
    if intermediate_cost_weight < 0.0:
        raise ValueError("intermediate_cost_weight must be non-negative")
    if not torch.isfinite(current_cost).all():
        raise ValueError("current_cost must be finite")

    legal = legal_safe_contact.bool()
    legal_pair = legal[:, :, None] & legal[:, None, :]
    improving_terminal = pair_cost <= (
        current_cost[:, None, None] - minimum_cost_improvement
    )
    intermediate_increase = torch.clamp(
        one_step_cost - current_cost[:, None], min=0.0
    )
    shooting_score = pair_cost + (
        intermediate_cost_weight * intermediate_increase[:, :, None]
    )
    shooting_score = torch.where(
        legal_pair & improving_terminal,
        shooting_score,
        torch.full_like(shooting_score, torch.inf),
    )
    flat_score = shooting_score.reshape(batch_size, -1)
    best_score, flat_index = flat_score.min(dim=1)
    has_pair = torch.isfinite(best_score)
    first_index = torch.div(flat_index, candidate_count, rounding_mode="floor")
    second_index = flat_index.remainder(candidate_count)
    invalid = torch.full_like(first_index, -1)
    first_index = torch.where(has_pair, first_index, invalid)
    second_index = torch.where(has_pair, second_index, invalid)
    return first_index, second_index, has_pair, shooting_score
