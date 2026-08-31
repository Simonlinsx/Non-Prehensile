from __future__ import annotations

import pytest
import torch

from dapl.contact_planner import (
    PhysicsRolloutScoringConfig,
    joint_threshold_cost,
    rank_physics_rollout_pairs,
    rank_physics_rollouts,
)


def test_joint_threshold_cost_penalizes_normalized_threshold_violation() -> None:
    cfg = PhysicsRolloutScoringConfig(
        predicate_violation_weight=0.0, mean_ratio_weight=0.0
    )
    score = joint_threshold_cost(
        torch.tensor([0.010]),
        torch.tensor([0.002]),
        torch.tensor([0.200]),
        cfg,
    )
    assert score.item() == pytest.approx(2.0)


def test_ranking_does_not_trade_new_rotation_violation_for_millimetres_of_xy() -> None:
    cfg = PhysicsRolloutScoringConfig(minimum_cost_improvement=0.0)
    best, valid, _ = rank_physics_rollouts(
        current_planar_error=torch.tensor([0.059]),
        current_height_error=torch.tensor([0.001]),
        current_rotation_error=torch.tensor([0.008]),
        rollout_planar_error=torch.tensor([[0.0375, 0.0383]]),
        rollout_height_error=torch.tensor([[0.001, 0.001]]),
        rollout_rotation_error=torch.tensor([[0.1213, 0.0932]]),
        enabled=torch.tensor([[True, True]]),
        legal_safe_contact=torch.tensor([[True, True]]),
        c1_violation=torch.tensor([[False, False]]),
        cfg=cfg,
    )
    assert bool(valid[0])
    assert best.item() == 1


def test_ranking_preserves_good_xy_while_reducing_rotation() -> None:
    cfg = PhysicsRolloutScoringConfig(minimum_cost_improvement=0.0)
    best, valid, scores = rank_physics_rollouts(
        current_planar_error=torch.tensor([0.010]),
        current_height_error=torch.tensor([0.002]),
        current_rotation_error=torch.tensor([0.180]),
        rollout_planar_error=torch.tensor([[0.035, 0.014]]),
        rollout_height_error=torch.tensor([[0.002, 0.002]]),
        rollout_rotation_error=torch.tensor([[0.060, 0.090]]),
        enabled=torch.tensor([[True, True]]),
        legal_safe_contact=torch.tensor([[True, True]]),
        c1_violation=torch.tensor([[False, False]]),
        cfg=cfg,
    )
    assert bool(valid[0])
    assert best.item() == 1
    assert scores[0, 1] < scores[0, 0]


def test_c1_violation_and_contact_miss_are_never_selected() -> None:
    cfg = PhysicsRolloutScoringConfig(minimum_cost_improvement=0.0)
    best, valid, scores = rank_physics_rollouts(
        current_planar_error=torch.tensor([0.080]),
        current_height_error=torch.tensor([0.002]),
        current_rotation_error=torch.tensor([0.200]),
        rollout_planar_error=torch.tensor([[0.001, 0.002]]),
        rollout_height_error=torch.tensor([[0.001, 0.001]]),
        rollout_rotation_error=torch.tensor([[0.001, 0.002]]),
        enabled=torch.tensor([[True, True]]),
        legal_safe_contact=torch.tensor([[True, False]]),
        c1_violation=torch.tensor([[True, False]]),
        cfg=cfg,
    )
    assert not bool(valid[0])
    assert best.item() == -1
    assert torch.isinf(scores).all()


def test_non_improving_rollouts_fail_closed() -> None:
    cfg = PhysicsRolloutScoringConfig(minimum_cost_improvement=0.02)
    best, valid, _ = rank_physics_rollouts(
        current_planar_error=torch.tensor([0.020]),
        current_height_error=torch.tensor([0.005]),
        current_rotation_error=torch.tensor([0.100]),
        rollout_planar_error=torch.tensor([[0.021, 0.020]]),
        rollout_height_error=torch.tensor([[0.005, 0.005]]),
        rollout_rotation_error=torch.tensor([[0.100, 0.101]]),
        enabled=torch.tensor([[True, True]]),
        legal_safe_contact=torch.tensor([[True, True]]),
        c1_violation=torch.tensor([[False, False]]),
        cfg=cfg,
    )
    assert not bool(valid[0])
    assert best.item() == -1


def test_shape_validation_rejects_mismatched_masks() -> None:
    cfg = PhysicsRolloutScoringConfig()
    with pytest.raises(ValueError, match="enabled"):
        rank_physics_rollouts(
            current_planar_error=torch.zeros(1),
            current_height_error=torch.zeros(1),
            current_rotation_error=torch.zeros(1),
            rollout_planar_error=torch.zeros(1, 2),
            rollout_height_error=torch.zeros(1, 2),
            rollout_rotation_error=torch.zeros(1, 2),
            enabled=torch.ones(1, 1, dtype=torch.bool),
            legal_safe_contact=torch.ones(1, 2, dtype=torch.bool),
            c1_violation=torch.zeros(1, 2, dtype=torch.bool),
            cfg=cfg,
        )


def test_two_step_fallback_can_select_a_non_monotonic_first_action() -> None:
    first, second, valid, scores = rank_physics_rollout_pairs(
        current_cost=torch.tensor([10.0]),
        one_step_cost=torch.tensor([[11.0, 9.5]]),
        pair_cost=torch.tensor([[[7.0, 8.0], [8.5, 9.0]]]),
        legal_safe_contact=torch.tensor([[True, True]]),
        minimum_cost_improvement=0.02,
        intermediate_cost_weight=0.25,
    )
    assert bool(valid[0])
    assert first.item() == 0
    assert second.item() == 0
    assert scores[0, 0, 0] < scores[0, 1, 0]


def test_two_step_fallback_rejects_pairs_with_an_illegal_effect() -> None:
    first, second, valid, _ = rank_physics_rollout_pairs(
        current_cost=torch.tensor([3.0]),
        one_step_cost=torch.tensor([[4.0, 4.0]]),
        pair_cost=torch.tensor([[[1.0, 1.0], [1.0, 1.0]]]),
        legal_safe_contact=torch.tensor([[False, False]]),
        minimum_cost_improvement=0.02,
    )
    assert not bool(valid[0])
    assert first.item() == -1
    assert second.item() == -1
