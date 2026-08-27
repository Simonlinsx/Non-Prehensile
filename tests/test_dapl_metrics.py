from __future__ import annotations

import unittest

import torch

from dapl.metrics import (
    axis_aligned_bounding_box_keypoints,
    bounded_joint_pose_tracking_cost,
    bounded_linear_distance_score,
    clearance_conditioned_route_scale,
    clearance_log_barrier,
    componentwise_progress_during_contact,
    dapl_combined_pose_error,
    dapl_multiscale_pose_score,
    dapl_tanh_proximity_reward,
    distance_progress_during_contact,
    discounted_potential_shaping,
    discounted_score_potential_shaping,
    dywa_exponential_keypoint_potential,
    flip_relative_goal_yaw_in_actor_observation,
    gate_navigation_at_legal_contact,
    goal_swept_semantic_point_index,
    lexicographic_route_potential,
    near_goal_motion_cost,
    normalized_clearance_violation,
    normalized_contact_distance_excess,
    normalized_distance_progress,
    potential_consistent_progress,
    planar_pose_success,
    positive_distance_progress_during_contact,
    positive_reference_relative_component_improvement,
    positive_reference_relative_error_improvement,
    positive_reference_relative_pareto_pose_improvement,
    positive_reference_relative_score,
    reference_relative_pose_improvement,
    rigid_body_ring_route_aabb_clearance,
    route_conditioned_alignment,
    sampled_segment_minimum_clearance,
    semantic_clearance_recovery_direction,
    semantic_ring_route_candidates,
    semantic_route_vector_field,
    semantic_tangential_recovery_direction,
    signed_reference_relative_error_improvement,
    signed_yaw_contact_moment_score,
    smooth_max_normalized_pose_error,
    support_aware_pose_success,
    update_route_detour_commitment,
    update_consecutive_success_count,
    wrench_aware_contact_support_score,
    weighted_componentwise_pose_progress,
    yaw_compatible_safe_point_mask,
)


class DAPLTaskMetricsTest(unittest.TestCase):
    def test_dapl_reward_primitives_match_reported_formulas(self) -> None:
        distance = torch.tensor([0.0, 0.1, 0.2])
        reward = dapl_tanh_proximity_reward(
            distance, standard_deviation=0.1
        )
        torch.testing.assert_close(reward, 1.0 - torch.tanh(distance / 0.1))
        pose = dapl_combined_pose_error(
            torch.tensor([0.05, 0.10]), torch.tensor([0.5, 1.0])
        )
        torch.testing.assert_close(pose, torch.tensor([0.15, 0.30]))

        multiscale = dapl_multiscale_pose_score(
            torch.tensor([0.05, 0.10]),
            torch.tensor([0.5, 1.0]),
        )
        expected_error = torch.tensor([0.15, 0.30])
        expected_score = (
            5.0 * (1.0 - torch.tanh(expected_error / 0.6))
            + 16.0 * (1.0 - torch.tanh(expected_error / 0.3))
        )
        torch.testing.assert_close(multiscale, expected_score)

        # Subtracting the reset score removes stationary payoff while keeping
        # persistent signed credit for better/worse current poses.
        relative = multiscale - multiscale[0]
        torch.testing.assert_close(relative[0], torch.tensor(0.0))
        self.assertLess(float(relative[1]), 0.0)

        positive = positive_reference_relative_score(
            multiscale[1:], multiscale[:1]
        )
        self.assertGreater(float(positive[0]), 0.0)
        torch.testing.assert_close(
            positive_reference_relative_score(multiscale, multiscale),
            torch.zeros_like(multiscale),
        )
        torch.testing.assert_close(
            positive_reference_relative_score(multiscale[:1], multiscale[1:]),
            torch.zeros(1),
        )

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            positive_reference_relative_score(multiscale, multiscale[:1])

    def test_positive_relative_joint_error_is_bounded_and_zero_on_regression(self) -> None:
        improvement = positive_reference_relative_error_improvement(
            torch.tensor([20.0, 10.0, 0.5, 0.0]),
            torch.tensor([18.0, 12.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            improvement,
            torch.tensor([0.1, 0.0, 0.5, 0.0]),
        )
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            positive_reference_relative_error_improvement(
                torch.ones(2), torch.ones(1)
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            positive_reference_relative_error_improvement(
                torch.ones(1),
                torch.zeros(1),
                reference_error_floor=0.0,
            )

    def test_signed_relative_joint_error_penalizes_regression(self) -> None:
        improvement = signed_reference_relative_error_improvement(
            torch.tensor([20.0, 10.0, 0.5, 0.0]),
            torch.tensor([18.0, 12.0, 0.0, 2.0]),
        )
        torch.testing.assert_close(
            improvement,
            torch.tensor([0.1, -0.2, 0.5, -1.0]),
        )

        leaky = signed_reference_relative_error_improvement(
            torch.tensor([20.0, 10.0, 0.5, 0.0]),
            torch.tensor([18.0, 12.0, 0.0, 2.0]),
            regression_scale=0.25,
        )
        torch.testing.assert_close(
            leaky,
            torch.tensor([0.1, -0.05, 0.5, -0.25]),
        )
        with self.assertRaisesRegex(ValueError, "regression_scale"):
            signed_reference_relative_error_improvement(
                torch.ones(1), torch.ones(1), regression_scale=1.01
            )

    def test_contact_relative_pose_improvement_has_no_entry_penalty(self) -> None:
        improvement = reference_relative_pose_improvement(
            torch.tensor([0.6, 0.6, 0.6]),
            torch.tensor([0.6, 0.3, 0.9]),
            normalization_cost=0.3,
        )
        torch.testing.assert_close(improvement, torch.tensor([0.0, 1.0, -1.0]))

    def test_joint_pose_cost_penalizes_wrong_yaw_without_relaxing_xy(self) -> None:
        planar = torch.tensor([0.08, 0.08, 0.01, 0.00])
        height = torch.tensor([0.001, 0.001, 0.001, 0.00])
        rotation = torch.tensor([0.15, 0.50, 0.05, 0.00])
        cost = bounded_joint_pose_tracking_cost(
            planar,
            height,
            rotation,
            planar_scale=0.08,
            height_scale=0.01,
            rotation_scale=0.20,
            temperature=0.25,
        )
        self.assertGreater(float(cost[1]), float(cost[0]) + 0.50)
        self.assertGreater(float(cost[0]), float(cost[2]))
        self.assertGreater(float(cost[2]), float(cost[3]))
        torch.testing.assert_close(cost[3], torch.tensor(0.0))

    def test_linear_distance_score_keeps_far_field_gradient(self) -> None:
        distance = torch.tensor([0.0, 0.07, 0.25, 0.50, 0.75])
        score = bounded_linear_distance_score(distance, maximum_distance=0.50)
        torch.testing.assert_close(
            score, torch.tensor([1.0, 0.86, 0.50, 0.0, 0.0])
        )

    def test_contact_distance_cost_is_zero_without_a_contact_cliff(self) -> None:
        distance = torch.tensor([0.0, 0.008, 0.016, 0.108, 0.208])
        cost = normalized_contact_distance_excess(
            distance,
            contact_distance=0.008,
            normalization_distance=0.10,
        )
        torch.testing.assert_close(
            cost, torch.tensor([0.0, 0.0, 0.08, 1.0, 1.0])
        )

    def test_clearance_violation_matches_contact_boundary(self) -> None:
        cost = normalized_clearance_violation(
            torch.tensor([0.005, 0.010, 0.015, 0.020, 0.030, torch.inf]),
            contact_distance=0.010,
            activation_distance=0.020,
        )
        torch.testing.assert_close(
            cost, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        )

        with self.assertRaisesRegex(ValueError, "exceed"):
            normalized_clearance_violation(
                torch.tensor([0.01]),
                contact_distance=0.01,
                activation_distance=0.01,
            )

    def test_clearance_log_barrier_is_strong_finite_and_zero_when_free(self) -> None:
        cost = clearance_log_barrier(
            torch.tensor([0.005, 0.010, 0.015, 0.020, torch.inf]),
            contact_distance=0.010,
            activation_distance=0.020,
            minimum_free_fraction=0.10,
        )
        torch.testing.assert_close(
            cost,
            torch.tensor(
                [
                    -torch.log(torch.tensor(0.10)),
                    -torch.log(torch.tensor(0.10)),
                    -torch.log(torch.tensor(0.50)),
                    0.0,
                    0.0,
                ]
            ),
        )

        with self.assertRaisesRegex(ValueError, "minimum_free_fraction"):
            clearance_log_barrier(
                torch.tensor([0.02]),
                contact_distance=0.01,
                activation_distance=0.02,
                minimum_free_fraction=0.0,
            )

    def test_normalized_distance_progress_is_signed_and_clipped(self) -> None:
        previous = torch.tensor([0.10, 0.10, 0.10, 0.10])
        current = torch.tensor([0.08, 0.11, 0.02, 0.20])
        progress = normalized_distance_progress(
            previous, current, normalization_distance=0.02
        )
        torch.testing.assert_close(progress, torch.tensor([1.0, -0.5, 1.0, -1.0]))

    def test_weighted_component_progress_has_no_smooth_max_dead_zone(self) -> None:
        previous = torch.tensor(
            [
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
            ]
        )
        current = torch.tensor(
            [
                [0.29, 0.00, 1.50],
                [0.31, 0.00, 1.45],
                [0.30, 0.00, 1.50],
            ]
        )
        reward = weighted_componentwise_pose_progress(
            previous,
            current,
            normalization_scales=(0.01, 0.005, 0.05),
            component_weights=(20.0, 4.0, 8.0),
        )
        # XY improvement is visible even while rotation is the larger error.
        # A 1-cm XY regression cannot be compensated by a 0.05-rad rotation
        # improvement under the accepted forward-teacher weighting.
        torch.testing.assert_close(reward, torch.tensor([20.0, -12.0, 0.0]))

        with self.assertRaisesRegex(ValueError, "one weight"):
            weighted_componentwise_pose_progress(
                previous,
                current,
                normalization_scales=(0.01, 0.005, 0.05),
                component_weights=(1.0, 1.0),
            )

    def test_positive_component_improvement_preserves_contact_exploration(self) -> None:
        reference = torch.tensor(
            [
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
            ]
        )
        current = torch.tensor(
            [
                [0.27, 0.00, 1.60],
                [0.33, 0.00, 1.35],
                [0.30, 0.00, 1.50],
            ]
        )
        reward = positive_reference_relative_component_improvement(
            reference,
            current,
            reference_error_floors=(0.02, 0.01, 0.10),
            component_weights=(20.0, 4.0, 8.0),
        )
        # XY improvement remains visible despite rotation regression, and the
        # converse is also true. Regressions themselves produce no negative
        # contact-entry incentive; a stationary pose remains exactly zero.
        torch.testing.assert_close(
            reward,
            torch.tensor([0.0625, 0.0250, 0.0]),
        )
        self.assertTrue(bool(torch.all((reward >= 0.0) & (reward <= 1.0))))

        with self.assertRaisesRegex(ValueError, "one reference-error floor"):
            positive_reference_relative_component_improvement(
                reference,
                current,
                reference_error_floors=(0.02, 0.01),
                component_weights=(20.0, 4.0, 8.0),
            )

    def test_pareto_pose_improvement_forbids_component_compensation(self) -> None:
        reference = torch.tensor(
            [
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
                [0.30, 0.00, 1.50],
            ]
        )
        current = torch.tensor(
            [
                [0.27, 0.002, 1.35],  # both improve; support remains valid
                [0.27, 0.002, 1.60],  # rotation regression
                [0.33, 0.002, 1.35],  # planar regression
                [0.27, 0.012, 1.35],  # leaves the strict support-height band
                [0.30, 0.000, 1.50],  # stationary reset pose
            ]
        )
        reward = positive_reference_relative_pareto_pose_improvement(
            reference,
            current,
            reference_planar_error_floor_m=0.02,
            reference_rotation_error_floor_rad=0.10,
            support_height_tolerance_m=0.01,
        )
        torch.testing.assert_close(
            reward,
            torch.tensor([0.10, 0.0, 0.0, 0.0, 0.0]),
        )

        with self.assertRaisesRegex(ValueError, "final components"):
            positive_reference_relative_pareto_pose_improvement(
                reference[:, :2],
                current[:, :2],
            )

    def test_potential_consistent_progress_cannot_reward_a_closed_loop(self) -> None:
        potential = torch.tensor([2.0, 1.0, 0.5, 2.0])
        alignment = torch.tensor([0.0, 1.0, 1.0])
        reward = potential_consistent_progress(
            potential[:-1],
            potential[1:],
            alignment,
            potential_scale=1.0,
            descent_gate_floor=0.25,
        )

        # The bounded scalar-potential differences telescope to zero over the
        # loop.  Alignment may only attenuate descent, while the complete
        # ascent cost remains, so the filtered loop return must be negative.
        self.assertLess(float(reward.sum()), 0.0)
        self.assertGreater(float(reward[1]), 0.0)
        self.assertLess(float(reward[2]), 0.0)

    def test_potential_consistent_progress_preserves_aligned_descent(self) -> None:
        previous = torch.tensor([2.0, 2.0, 1.0])
        current = torch.tensor([1.0, 1.0, 2.0])
        alignment = torch.tensor([1.0, -1.0, 1.0])
        reward = potential_consistent_progress(
            previous,
            current,
            alignment,
            potential_scale=1.0,
            descent_gate_floor=0.25,
        )

        bounded_delta = previous / (previous + 1.0) - current / (current + 1.0)
        torch.testing.assert_close(reward[0], bounded_delta[0])
        torch.testing.assert_close(reward[1], 0.25 * bounded_delta[1])
        torch.testing.assert_close(reward[2], bounded_delta[2])

    def test_lexicographic_route_potential_never_buys_extra_clearance(self) -> None:
        route_length = torch.tensor(
            [[0.40, 0.20, 0.10], [0.40, 0.30, 0.20]]
        )
        route_clearance = torch.tensor(
            [[0.50, 0.011, 0.010], [0.009, 0.008, 0.009]]
        )
        potential, selected, has_legal = lexicographic_route_potential(
            route_length,
            route_clearance,
            required_clearance=0.010,
            length_scale=0.20,
            violation_scale=0.010,
        )

        # Once routes are legal, the shortest one wins even if another route
        # has vastly more clearance.  With no legal route, maximum clearance
        # wins and length breaks an exact clearance tie.
        torch.testing.assert_close(selected, torch.tensor([2, 2]))
        torch.testing.assert_close(has_legal, torch.tensor([True, False]))
        self.assertLess(float(potential[0]), 1.0)
        self.assertGreaterEqual(float(potential[1]), 1.0)

    def test_discounted_potential_shaping_does_not_prefer_a_cycle(self) -> None:
        gamma = 0.99
        cycle_cost = torch.tensor([1.0, 0.20, 0.70, 1.0])
        stationary_cost = torch.ones_like(cycle_cost)
        cycle_reward = discounted_potential_shaping(
            cycle_cost[:-1], cycle_cost[1:], discount_factor=gamma
        )
        stationary_reward = discounted_potential_shaping(
            stationary_cost[:-1], stationary_cost[1:], discount_factor=gamma
        )
        discounts = gamma ** torch.arange(3, dtype=cycle_reward.dtype)

        # Both trajectories start and end at the same state after the same
        # horizon.  Intermediate descent followed by retreat has no advantage.
        torch.testing.assert_close(
            torch.sum(discounts * cycle_reward),
            torch.sum(discounts * stationary_reward),
        )

    def test_dywa_keypoint_potential_is_joint_and_discounted(self) -> None:
        distances = torch.tensor(
            [[0.30, 0.10, 0.20], [0.20, 0.05, 0.10]], dtype=torch.float32
        )
        potential = dywa_exponential_keypoint_potential(distances)
        expected = 0.302 * torch.pow(0.995, 243.12 * distances).mean(dim=-1)
        torch.testing.assert_close(potential, expected)
        self.assertGreater(float(potential[1]), float(potential[0]))
        reach_potential = dywa_exponential_keypoint_potential(
            distances, amplitude=0.0604
        )
        torch.testing.assert_close(reach_potential, 0.2 * potential)

        gamma = 0.99
        score_path = torch.tensor([0.10, 0.25, 0.18, 0.10])
        stationary = torch.full_like(score_path, 0.10)
        path_reward = discounted_score_potential_shaping(
            score_path[:-1], score_path[1:], discount_factor=gamma
        )
        stationary_reward = discounted_score_potential_shaping(
            stationary[:-1], stationary[1:], discount_factor=gamma
        )
        discounts = gamma ** torch.arange(3, dtype=path_reward.dtype)
        torch.testing.assert_close(
            torch.sum(discounts * path_reward),
            torch.sum(discounts * stationary_reward),
        )

    def test_axis_aligned_bounding_box_keypoints_cover_all_corners(self) -> None:
        points = torch.tensor(
            [
                [[-2.0, 1.0, 0.0], [3.0, -4.0, 5.0], [1.0, 2.0, -1.0]],
                [[0.0, 2.0, 4.0], [2.0, 6.0, 8.0], [1.0, 3.0, 5.0]],
            ]
        )
        corners = axis_aligned_bounding_box_keypoints(points)
        self.assertEqual(corners.shape, (2, 8, 3))
        torch.testing.assert_close(corners.amin(dim=1), points.amin(dim=1))
        torch.testing.assert_close(corners.amax(dim=1), points.amax(dim=1))
        self.assertEqual(torch.unique(corners[0], dim=0).shape[0], 8)

    def test_legal_contact_disables_only_precontact_navigation(self) -> None:
        cost, distance = gate_navigation_at_legal_contact(
            torch.tensor([0.8, 0.4, 0.2]),
            torch.tensor([0.12, 0.03, 0.01]),
            torch.tensor([False, True, False]),
        )
        torch.testing.assert_close(cost, torch.tensor([0.8, 0.0, 0.2]))
        torch.testing.assert_close(distance, torch.tensor([0.12, 0.0, 0.01]))

        with self.assertRaisesRegex(ValueError, "boolean"):
            gate_navigation_at_legal_contact(
                torch.ones(1), torch.ones(1), torch.ones(1)
            )

    def test_wrench_aware_contact_score_selects_yaw_compatible_side(self) -> None:
        # All three points are equally far behind a +X push.  A positive yaw
        # requires positive (r x f)_z and therefore selects negative Y; a
        # negative yaw selects the mirrored point.
        points = torch.tensor(
            [
                [[-0.10, -0.05], [-0.10, 0.00], [-0.10, 0.05]],
                [[-0.10, -0.05], [-0.10, 0.00], [-0.10, 0.05]],
            ]
        )
        score = wrench_aware_contact_support_score(
            points,
            torch.tensor([[0.10, 0.00], [0.10, 0.00]]),
            torch.tensor([0.50, -0.50]),
            yaw_moment_weight=1.0,
            yaw_activation_rad=0.10,
        )
        torch.testing.assert_close(score.argmax(dim=1), torch.tensor([0, 2]))

        translation_only = wrench_aware_contact_support_score(
            points[:1],
            torch.tensor([[0.10, 0.00]]),
            torch.tensor([0.50]),
            yaw_moment_weight=0.0,
            yaw_activation_rad=0.10,
        )
        torch.testing.assert_close(
            translation_only,
            torch.full_like(translation_only, 0.10),
        )

    def test_signed_yaw_contact_moment_is_bilateral_and_separable(self) -> None:
        points = torch.tensor(
            [
                [[-0.10, -0.05], [-0.10, 0.00], [-0.10, 0.05]],
                [[-0.10, -0.05], [-0.10, 0.00], [-0.10, 0.05]],
            ]
        )
        score = signed_yaw_contact_moment_score(
            points,
            torch.tensor([[0.10, 0.00], [0.10, 0.00]]),
            torch.tensor([0.50, -0.50]),
            yaw_activation_rad=0.10,
        )
        torch.testing.assert_close(score.argmax(dim=1), torch.tensor([0, 2]))
        torch.testing.assert_close(score[0], torch.flip(score[1], dims=(0,)))
        self.assertGreater(float(score[0, 0]), 0.0)
        self.assertGreater(float(score[1, 2]), 0.0)

        zero_yaw = signed_yaw_contact_moment_score(
            points[:1],
            torch.tensor([[0.10, 0.00]]),
            torch.zeros(1),
            yaw_activation_rad=0.10,
        )
        torch.testing.assert_close(zero_yaw, torch.zeros_like(zero_yaw))

    def test_yaw_positive_halfspace_is_broad_with_finite_fallback(self) -> None:
        score = torch.tensor(
            [[-0.010, 0.000, 0.003, 0.020], [-0.010, -0.004, -0.003, -0.020]]
        )
        safe = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        selected, fallback = yaw_compatible_safe_point_mask(
            score,
            safe,
            selection_mode="positive_halfspace",
            near_best_band_m=0.002,
            minimum_compatibility_m=0.002,
        )
        self.assertEqual(
            selected.tolist(),
            [[False, False, True, True], [False, True, True, False]],
        )
        self.assertEqual(fallback.tolist(), [False, True])

        near_best, near_best_fallback = yaw_compatible_safe_point_mask(
            score[:1],
            safe[:1],
            selection_mode="near_best",
            near_best_band_m=0.018,
        )
        self.assertEqual(near_best.tolist(), [[False, False, True, True]])
        self.assertFalse(bool(near_best_fallback[0]))

    def test_counterfactual_flips_only_recoverable_goal_yaw(self) -> None:
        observation = torch.arange(2 * 45, dtype=torch.float32).reshape(2, 45)
        rel_goal_start = 30
        counterfactual = flip_relative_goal_yaw_in_actor_observation(
            observation, rel_goal_start=rel_goal_start
        )
        expected = observation.clone()
        expected[:, rel_goal_start + 4] *= -1.0
        expected[:, rel_goal_start + 6] *= -1.0
        torch.testing.assert_close(counterfactual, expected)
        torch.testing.assert_close(
            observation,
            torch.arange(2 * 45, dtype=torch.float32).reshape(2, 45),
        )

        with self.assertRaisesRegex(ValueError, "relative goal"):
            flip_relative_goal_yaw_in_actor_observation(
                torch.zeros(1, 8), rel_goal_start=0
            )

    def test_sampled_corridor_clearance_detects_and_masks_obstacles(self) -> None:
        start = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        end = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        obstacles = torch.tensor(
            [
                [[0.5, 0.0, 0.0], [0.5, 2.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 3.0, 0.0]],
            ]
        )
        clearance = sampled_segment_minimum_clearance(
            start,
            end,
            obstacles,
            num_samples=5,
            start_fraction=0.1,
            end_fraction=0.9,
        )
        torch.testing.assert_close(clearance, torch.tensor([0.0, 1.0]))

        masked = sampled_segment_minimum_clearance(
            start[:1],
            end[:1],
            obstacles[:1],
            obstacle_mask=torch.tensor([[False, True]]),
            num_samples=5,
            start_fraction=0.1,
            end_fraction=0.9,
        )
        torch.testing.assert_close(masked, torch.tensor([2.0]))

        with self.assertRaisesRegex(ValueError, "fractions"):
            sampled_segment_minimum_clearance(
                start,
                end,
                obstacles,
                start_fraction=0.9,
                end_fraction=0.9,
            )

    def test_semantic_ring_routes_offer_a_legal_detour_without_a_waypoint_label(self) -> None:
        start = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        end = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        obstacles = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]],
            ]
        )
        mask = torch.tensor([[True, True], [False, False]])
        length, clearance = semantic_ring_route_candidates(
            start,
            end,
            obstacles,
            obstacle_mask=mask,
            body_radius=0.10,
            contact_clearance=0.05,
            detour_margin=0.10,
            num_candidates=8,
            num_segment_samples=9,
            obstacle_sample_count=2,
        )

        self.assertEqual(tuple(length.shape), (2, 9))
        self.assertLess(float(clearance[0, 0]), 0.05)
        legal_detour = clearance[0, 1:] >= 0.05
        self.assertTrue(bool(legal_detour.any()))
        self.assertTrue(bool(torch.all(length[0, 1:][legal_detour] > length[0, 0])))
        self.assertTrue(bool(torch.isinf(clearance[1]).all()))
        torch.testing.assert_close(length[1, 0], torch.tensor(2.0))

    def test_goal_swept_semantic_point_selects_blocked_protected_path(self) -> None:
        start = torch.tensor(
            [[[0.0, -0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]]
        )
        end = start + torch.tensor([[[1.0, 0.0, 0.0]]])
        obstacle = torch.tensor(
            [[[0.48, 0.01, 0.0], [0.52, 0.01, 0.0]]]
        )
        selected, distance = goal_swept_semantic_point_index(
            start,
            end,
            obstacle,
            point_mask=torch.tensor([[True, True, False]]),
            num_samples=5,
        )
        self.assertEqual(int(selected[0]), 1)
        self.assertLess(float(distance[0]), 0.03)

    def test_rigid_body_route_screen_rejects_wrong_point_detour_side(self) -> None:
        # The critical point alone can pass either side of the blocker.  A
        # second protected point makes the lower route illegal for the rigid
        # body, so the full-cloud screen must retain only the upper detour.
        start = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.16, 0.0]]]
        )
        end = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.16, 0.0]]]
        )
        obstacle = torch.tensor(
            [[[0.45, -0.08, -0.01], [0.55, 0.08, 0.01]]]
        )
        first_edge = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.5, 0.22, 0.0], [0.5, -0.22, 0.0]]]
        )
        clearance = rigid_body_ring_route_aabb_clearance(
            start,
            end,
            obstacle,
            first_edge,
            start[:, 0],
            end[:, 0],
            point_mask=torch.tensor([[True, True]]),
            num_segment_samples=5,
        )
        self.assertEqual(tuple(clearance.shape), (1, 3))
        self.assertGreater(float(clearance[0, 1]), 0.02)
        self.assertEqual(float(clearance[0, 2]), 0.0)

    def test_semantic_vector_field_is_lateral_only_when_direct_route_is_blocked(self) -> None:
        start = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        end = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        obstacles = torch.tensor(
            [
                [[0.0, -0.02, 0.0], [0.0, 0.02, 0.0]],
                [[0.0, -0.02, 0.0], [0.0, 0.02, 0.0]],
            ]
        )
        direction, detour, direct_clearance, selected_clearance = (
            semantic_route_vector_field(
                start,
                end,
                obstacles,
                obstacle_mask=torch.tensor([[True, True], [False, False]]),
                body_radius=0.10,
                contact_clearance=0.05,
                detour_margin=0.10,
                num_candidates=8,
                num_segment_samples=9,
                obstacle_sample_count=2,
            )
        )

        self.assertTrue(bool(detour[0]))
        self.assertLess(float(direct_clearance[0]), 0.05)
        self.assertGreaterEqual(float(selected_clearance[0]), 0.05)
        self.assertGreater(abs(float(direction[0, 1])), 0.05)
        self.assertFalse(bool(detour[1]))
        torch.testing.assert_close(direction[1], torch.tensor([1.0, 0.0, 0.0]))

    def test_semantic_vector_field_recovers_outward_when_all_routes_are_illegal(self) -> None:
        start = torch.tensor([[-0.02, 0.0, 0.0]])
        end = torch.tensor([[1.0, 0.0, 0.0]])
        obstacles = torch.tensor([[[0.0, -0.01, 0.0], [0.0, 0.01, 0.0]]])

        recovery, raw_clearance = semantic_clearance_recovery_direction(
            start,
            obstacles,
            safety_radius=0.15,
        )
        self.assertLess(float(recovery[0, 0]), -0.99)
        torch.testing.assert_close(raw_clearance, torch.tensor([0.02236068]))

        direction, detour, _, selected_clearance = semantic_route_vector_field(
            start,
            end,
            obstacles,
            body_radius=0.10,
            contact_clearance=0.05,
            detour_margin=0.10,
            num_candidates=8,
            num_segment_samples=9,
            obstacle_sample_count=2,
            recover_illegal_route=True,
        )
        self.assertTrue(bool(detour[0]))
        self.assertLess(float(selected_clearance[0]), 0.05)
        self.assertGreater(float(torch.sum(direction[0] * recovery[0])), 0.25)
        self.assertGreater(float(torch.linalg.vector_norm(direction[0, 1:])), 0.25)

    def test_semantic_vector_field_recovers_before_start_enters_inflation(self) -> None:
        # A semantic ring surrounds the start at 20 cm.  The controlled point
        # is still outside the 15 cm safety radius, but every direct/ring edge
        # crosses the obstacle shell.  This is the exact structural case that
        # must not fall back to the illegal direct edge.
        angles = torch.arange(8) * (2.0 * torch.pi / 8.0)
        obstacles = torch.stack(
            (
                0.20 * torch.cos(angles),
                0.20 * torch.sin(angles),
                torch.zeros_like(angles),
            ),
            dim=-1,
        ).unsqueeze(0)
        start = torch.tensor([[0.0, 0.0, 0.0]])
        end = torch.tensor([[1.0, 0.0, 0.0]])

        recovery, raw_clearance = semantic_clearance_recovery_direction(
            start,
            obstacles,
            safety_radius=0.15,
        )
        self.assertGreater(float(raw_clearance[0]), 0.15)
        torch.testing.assert_close(recovery, torch.tensor([[0.0, 0.0, 1.0]]))

        direction, detour, _, selected_clearance = semantic_route_vector_field(
            start,
            end,
            obstacles,
            body_radius=0.10,
            contact_clearance=0.05,
            detour_margin=0.10,
            num_candidates=8,
            num_segment_samples=9,
            obstacle_sample_count=8,
            recover_illegal_route=True,
        )
        self.assertTrue(bool(detour[0]))
        self.assertLess(float(selected_clearance[0]), 0.05)
        self.assertGreater(float(torch.sum(direction[0] * recovery[0])), 0.25)
        self.assertGreater(float(torch.linalg.vector_norm(direction[0, :2])), 0.25)

    def test_tangential_recovery_cannot_cancel_outward_clearance(self) -> None:
        start = torch.tensor([[0.0, 0.0, 0.0]])
        end = torch.tensor([[1.0, 0.0, 0.0]])
        outward = torch.tensor([[-1.0, 0.0, 0.0]])
        first_edges = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]
        )
        direction = semantic_tangential_recovery_direction(
            start,
            end,
            outward,
            torch.tensor([[0.0, 0.004, 0.004]]),
            first_edges,
            contact_clearance=0.010,
        )

        # The selected sign is deterministic, but the invariant is sign-free:
        # recovery must both move outward and travel along the boundary.
        self.assertGreater(float(torch.sum(direction[0] * outward[0])), 0.40)
        self.assertGreater(abs(float(direction[0, 1])), 0.80)
        torch.testing.assert_close(
            torch.linalg.vector_norm(direction, dim=1), torch.ones(1)
        )

    def test_tangential_recovery_restores_a_legal_route_around_open_wall(self) -> None:
        wall_y = torch.linspace(-0.10, 0.10, 21)
        obstacles = torch.stack(
            (torch.zeros_like(wall_y), wall_y, torch.zeros_like(wall_y)), dim=-1
        ).unsqueeze(0)
        start = torch.tensor([[-0.02, 0.0, 0.0]])
        end = torch.tensor([[0.30, 0.0, 0.0]])

        became_legal = False
        for _ in range(20):
            direction, _, _, selected_clearance = semantic_route_vector_field(
                start,
                end,
                obstacles,
                body_radius=0.10,
                contact_clearance=0.05,
                detour_margin=0.10,
                num_candidates=12,
                num_segment_samples=9,
                obstacle_sample_count=21,
                recover_illegal_route=True,
            )
            if bool(selected_clearance[0] >= 0.05):
                became_legal = True
                break
            start = start + 0.02 * direction

        self.assertTrue(became_legal)
        # It must have gone around the wall, not through its direct x corridor.
        self.assertGreater(float(start[0, 1]), 0.15)

    def test_route_conditioned_alignment_preserves_direct_progress_at_matched_scale(self) -> None:
        alignment = torch.tensor([1.0, -0.5, 0.25, -1.0])
        detour = torch.tensor([True, True, False, False])
        reward = route_conditioned_alignment(
            alignment, detour, direct_route_scale=0.15
        )
        torch.testing.assert_close(
            reward, torch.tensor([1.0, -0.5, 0.0375, -0.15])
        )
        torch.testing.assert_close(
            route_conditioned_alignment(alignment, detour),
            torch.tensor([1.0, -0.5, 0.0, 0.0]),
        )

        with self.assertRaisesRegex(ValueError, "shapes"):
            route_conditioned_alignment(
                alignment, detour[:2], direct_route_scale=0.15
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            route_conditioned_alignment(
                alignment, detour, direct_route_scale=1.1
            )

    def test_route_detour_commitment_does_not_drop_at_direct_transition(self) -> None:
        commitment = torch.tensor([False, True, False])
        commitment = update_route_detour_commitment(
            commitment, torch.tensor([True, False, False])
        )
        self.assertEqual(commitment.tolist(), [True, True, False])
        commitment = update_route_detour_commitment(
            commitment, torch.tensor([False, False, True])
        )
        self.assertEqual(commitment.tolist(), [True, True, True])

        with self.assertRaisesRegex(ValueError, "shapes"):
            update_route_detour_commitment(
                commitment, torch.tensor([True, False])
            )

    def test_clearance_conditioned_route_scale_is_continuous_and_recoverable(self) -> None:
        scale = clearance_conditioned_route_scale(
            torch.tensor([0.0, 0.010, 0.025, 0.040, 0.080, torch.inf]),
            contact_clearance=0.010,
            activation_clearance=0.040,
            direct_route_scale=0.15,
        )
        torch.testing.assert_close(
            scale, torch.tensor([1.0, 1.0, 0.575, 0.15, 0.15, 0.15])
        )

        with self.assertRaisesRegex(ValueError, "exceed"):
            clearance_conditioned_route_scale(
                torch.tensor([0.02]),
                contact_clearance=0.02,
                activation_clearance=0.02,
                direct_route_scale=0.15,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            clearance_conditioned_route_scale(
                torch.tensor([0.02]),
                contact_clearance=0.01,
                activation_clearance=0.04,
                direct_route_scale=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            clearance_conditioned_route_scale(
                torch.tensor([-0.01]),
                contact_clearance=0.01,
                activation_clearance=0.04,
                direct_route_scale=0.15,
            )

    def test_contact_gated_progress_rewards_only_positive_valid_pushes(self) -> None:
        reward = positive_distance_progress_during_contact(
            torch.tensor([0.08, 0.08, 0.08, 0.08]),
            torch.tensor([0.07, 0.09, 0.07, 0.06]),
            torch.tensor([True, True, False, True]),
            normalization_distance=0.01,
        )
        torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0, 1.0]))

        with self.assertRaisesRegex(ValueError, "shapes"):
            positive_distance_progress_during_contact(
                torch.tensor([0.08]),
                torch.tensor([0.07]),
                torch.tensor([True, False]),
                normalization_distance=0.01,
            )

    def test_contact_gated_signed_progress_penalizes_wrong_way_pushes(self) -> None:
        reward = distance_progress_during_contact(
            torch.tensor([0.08, 0.08, 0.08]),
            torch.tensor([0.07, 0.09, 0.07]),
            torch.tensor([True, True, False]),
            normalization_distance=0.01,
        )
        torch.testing.assert_close(reward, torch.tensor([1.0, -1.0, 0.0]))

    def test_componentwise_progress_preserves_independent_pose_signals(self) -> None:
        progress = componentwise_progress_during_contact(
            torch.tensor([[0.08, 0.010, 0.20], [0.08, 0.010, 0.20]]),
            torch.tensor([[0.07, 0.009, 0.21], [0.07, 0.009, 0.10]]),
            torch.tensor([True, False]),
            normalization_scales=(0.01, 0.005, 0.05),
        )
        torch.testing.assert_close(
            progress,
            torch.tensor([[1.0, 0.2, -0.2], [0.0, 0.0, 0.0]]),
        )

        with self.assertRaisesRegex(ValueError, "one normalization"):
            componentwise_progress_during_contact(
                torch.zeros(1, 3),
                torch.zeros(1, 3),
                torch.ones(1, dtype=torch.bool),
                normalization_scales=(1.0, 1.0),
            )

    def test_near_goal_motion_cost_is_gated_and_normalized(self) -> None:
        cost = near_goal_motion_cost(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            torch.tensor([0.0, 0.03, 0.03, 0.03]),
            torch.tensor([0.30, 0.30, 0.30, 0.30]),
            activation_pose_error=2.0,
            linear_speed_scale=0.03,
            angular_speed_scale=0.30,
        )
        torch.testing.assert_close(cost, torch.tensor([0.5, 0.5, 0.0, 0.0]))

    def test_joint_pose_error_is_zero_at_goal_and_one_at_thresholds(self) -> None:
        joint_error = smooth_max_normalized_pose_error(
            torch.tensor([0.0, 0.02]),
            torch.tensor([0.0, 0.01]),
            torch.tensor([0.0, 0.10]),
            planar_threshold=0.02,
            height_threshold=0.01,
            rotation_threshold=0.10,
        )
        torch.testing.assert_close(joint_error, torch.tensor([0.0, 1.0]))

    def test_joint_pose_error_tracks_the_worst_normalized_component(self) -> None:
        joint_error = smooth_max_normalized_pose_error(
            torch.tensor([0.04, 0.01]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.20]),
            planar_threshold=0.02,
            height_threshold=0.01,
            rotation_threshold=0.10,
            temperature=0.10,
        )
        self.assertTrue(torch.all(joint_error > 1.8))

    def test_planar_position_ignores_height_but_requires_orientation(self) -> None:
        current_position = torch.tensor(
            [[0.02, 0.01, 2.0], [0.02, 0.01, 0.0], [0.06, 0.0, 0.0]]
        )
        current_quaternion = torch.tensor(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
        )
        goal_pose = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

        result = planar_pose_success(
            current_position,
            current_quaternion,
            goal_pose,
            position_threshold=0.05,
            rotation_threshold=0.1,
        )

        self.assertEqual(result.tolist(), [True, False, False])

    def test_rejects_non_positive_thresholds(self) -> None:
        position = torch.zeros(1, 3)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        goal = torch.cat((position, quaternion), dim=-1)
        with self.assertRaisesRegex(ValueError, "thresholds"):
            planar_pose_success(
                position,
                quaternion,
                goal,
                position_threshold=0.0,
            )

    def test_support_aware_goal_requires_xy_height_and_full_rotation(self) -> None:
        half_angle = torch.tensor(0.04)
        valid_yaw = torch.tensor(
            [torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)]
        )
        tipped = torch.tensor(
            [torch.cos(torch.tensor(0.10)), torch.sin(torch.tensor(0.10)), 0.0, 0.0]
        )
        current_position = torch.tensor(
            [
                [0.019, 0.0, 0.009],
                [0.021, 0.0, 0.0],
                [0.0, 0.0, 0.011],
                [0.0, 0.0, 0.0],
            ]
        )
        current_quaternion = torch.stack(
            (
                valid_yaw,
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                tipped,
            )
        )
        goal_pose = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        ).repeat(4, 1)

        result = support_aware_pose_success(
            current_position,
            current_quaternion,
            goal_pose,
            planar_position_threshold=0.02,
            height_threshold=0.01,
            rotation_threshold=0.1,
        )

        self.assertEqual(result.tolist(), [True, False, False, False])

    def test_consecutive_success_count_resets_broken_streaks(self) -> None:
        count = torch.zeros(3, dtype=torch.long)
        count = update_consecutive_success_count(
            count, torch.tensor([True, True, False])
        )
        count = update_consecutive_success_count(
            count, torch.tensor([True, False, True])
        )
        self.assertEqual(count.tolist(), [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
