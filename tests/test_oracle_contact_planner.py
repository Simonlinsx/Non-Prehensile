from __future__ import annotations

import math

import pytest
import torch

from dapl.contact_planner import (
    OracleContactPlannerConfig,
    OraclePlanningScene,
    OracleSafeContactPlanner,
    horizontal_push_frame,
)


def _scene(
    *, blocked: bool = False, goal_y: float = 0.0, yaw_error: float = 0.0
) -> OraclePlanningScene:
    # A 12-cm handle followed by a compact functional head.  The hand cache is
    # expressed in the accepted horizontal convention where local +Z is the
    # pushing direction.
    handle_x = torch.linspace(-0.06, 0.04, 12).repeat_interleave(2)
    handle = torch.stack(
        (
            handle_x,
            torch.tensor((-0.006, 0.006)).repeat(12),
            torch.full((24,), 0.10),
        ),
        dim=1,
    )
    head = torch.tensor(
        [
            [0.075, -0.025, 0.085],
            [0.075, 0.025, 0.085],
            [0.100, -0.025, 0.115],
            [0.100, 0.025, 0.115],
        ]
    )
    points = torch.cat((handle, head), dim=0).unsqueeze(0)
    safe = torch.cat((torch.ones(24), torch.zeros(4))).unsqueeze(0)
    protected = torch.cat((torch.zeros(24), torch.ones(4))).unsqueeze(0)
    if blocked:
        # Mark all geometry non-safe.  A planner must fail closed rather than
        # silently falling back to a functional point.
        safe.zero_()
    hand_local = torch.tensor(
        [
            [0.000, -0.010, 0.000],
            [0.000, 0.010, 0.000],
            [0.000, -0.010, 0.020],
            [0.000, 0.010, 0.020],
        ]
    )
    return OraclePlanningScene(
        target_points=points,
        safe_scores=safe,
        protected_scores=protected,
        target_position=torch.tensor([[0.0, 0.0, 0.10]]),
        goal_position=torch.tensor([[0.08, goal_y, 0.10]]),
        tcp_position=torch.tensor([[-0.18, 0.0, 0.10]]),
        hand_points_local=hand_local,
        yaw_error=torch.tensor([yaw_error]),
    )


def test_horizontal_push_frame_aligns_local_forward_axis() -> None:
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    rotation = horizontal_push_frame(direction)
    assert torch.allclose(rotation[:, :, 2], torch.nn.functional.normalize(direction, dim=1))
    identity = rotation.transpose(1, 2) @ rotation
    assert torch.allclose(identity, torch.eye(3).expand_as(identity), atol=1.0e-6)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(2), atol=1.0e-6)


def test_planner_selects_legal_trailing_safe_contact() -> None:
    cfg = OracleContactPlannerConfig(
        contact_distance_m=0.002,
        forbidden_clearance_m=0.012,
        approach_clearance_m=0.008,
        precontact_standoff_m=0.040,
        output_candidates=4,
    )
    candidates = OracleSafeContactPlanner(cfg).plan(_scene())
    assert bool(candidates.any_valid[0])
    best = 0
    assert candidates.target_point_index[0, best] < 24
    assert torch.isfinite(candidates.contact_point[0, best]).all()
    assert candidates.safe_distance[0, best] <= cfg.contact_distance_m
    assert candidates.forbidden_clearance[0, best] > cfg.forbidden_clearance_m
    assert candidates.approach_clearance[0, best] > cfg.approach_clearance_m
    assert candidates.contact_tcp[0, best, 0] < 0.0
    assert torch.allclose(
        candidates.push_direction[0, best],
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1.0e-6,
    )
    assert candidates.push_tcp[0, best, 0] > candidates.contact_tcp[0, best, 0]


def test_planner_rotates_push_frame_with_goal_direction() -> None:
    candidates = OracleSafeContactPlanner().plan(_scene(goal_y=0.08))
    assert bool(candidates.any_valid[0])
    expected = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0.0])
    assert torch.allclose(candidates.push_direction[0, 0], expected, atol=1.0e-6)
    assert torch.allclose(candidates.hand_rotation[0, 0, :, 2], expected, atol=1.0e-6)


def test_planner_can_preserve_live_tcp_rotation() -> None:
    scene = _scene(goal_y=0.08)
    live_rotation = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    scene.tcp_rotation = live_rotation
    candidates = OracleSafeContactPlanner().plan(scene)
    assert bool(candidates.any_valid[0])
    assert torch.allclose(candidates.hand_rotation[0, 0], live_rotation[0])
    # Contact access follows the local surface normal while object motion
    # follows the independently optimized push direction.
    alignment = torch.dot(
        candidates.approach_direction[0, 0],
        candidates.push_direction[0, 0],
    )
    assert alignment < 0.99


def test_planner_rejects_hand_geometry_below_support_plane() -> None:
    scene = _scene()
    # The synthetic handle is only 15 mm above its lowest object point.  A
    # deliberately larger required support gap must make every contact fail
    # closed instead of clipping the hand through the table.
    cfg = OracleContactPlannerConfig(support_clearance_m=0.020)
    candidates = OracleSafeContactPlanner(cfg).plan(scene)
    assert not bool(candidates.any_valid[0])


def test_planner_fails_closed_without_safe_points() -> None:
    candidates = OracleSafeContactPlanner().plan(_scene(blocked=True))
    assert not bool(candidates.any_valid[0])
    assert torch.all(candidates.target_point_index == -1)
    assert torch.isnan(candidates.contact_tcp).all()


def test_short_horizon_push_is_bounded_and_predictive() -> None:
    cfg = OracleContactPlannerConfig(
        minimum_push_distance_m=0.01,
        maximum_push_distance_m=0.03,
        translation_efficiency=0.8,
        push_overshoot_m=0.0,
    )
    candidates = OracleSafeContactPlanner(cfg).plan(_scene())
    assert candidates.push_distance[0, 0].item() == pytest.approx(0.03)
    assert candidates.predicted_planar_error[0, 0].item() == pytest.approx(0.056)


def test_yaw_error_selects_contact_with_correct_moment_sign() -> None:
    positive = OracleSafeContactPlanner().plan(_scene(yaw_error=0.15))
    negative = OracleSafeContactPlanner().plan(_scene(yaw_error=-0.15))
    assert positive.contact_moment_arm[0, 0] > 0.0
    assert negative.contact_moment_arm[0, 0] < 0.0
    assert positive.predicted_yaw_error[0, 0] < 0.15
    assert negative.predicted_yaw_error[0, 0] < 0.15


def test_scene_validation_rejects_invalid_probabilities() -> None:
    scene = _scene()
    scene.safe_scores[0, 0] = 1.1
    with pytest.raises(ValueError, match="safe_scores"):
        OracleSafeContactPlanner().plan(scene)
