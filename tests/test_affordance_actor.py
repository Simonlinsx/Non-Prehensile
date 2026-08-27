from __future__ import annotations

import unittest

import torch

from rsl_rl.modules.actor_critic_affordance import ActorCriticAffordance


class AffordanceActorCriticTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = ActorCriticAffordance(4146, 4146, 7)
        self.model.eval()
        self.observations = torch.randn(3, 4146)

    def test_actor_and_critic_shapes(self) -> None:
        self.assertEqual(self.model.act_inference(self.observations).shape, (3, 7))
        self.assertEqual(self.model.evaluate(self.observations).shape, (3, 1))
        actions = self.model.act(self.observations)
        self.assertEqual(actions.shape, (3, 7))
        self.assertEqual(self.model.get_actions_log_prob(actions).shape, (3,))

    def test_each_point_keeps_xyz_and_semantics_together(self) -> None:
        target, obstacles, state = self.model._split_observations(self.observations)
        self.assertEqual(target.shape, (3, 512, 5))
        self.assertEqual(obstacles.shape, (3, 512, 3))
        self.assertEqual(state.shape, (3, 50))

    def test_point_order_is_irrelevant(self) -> None:
        target, obstacles, state = self.model._split_observations(self.observations)
        target = target[:, torch.randperm(512)]
        obstacles = obstacles[:, torch.randperm(512)]
        permuted = torch.cat((target.flatten(1), obstacles.flatten(1), state), dim=1)
        original_action = self.model.act_inference(self.observations)
        permuted_action = self.model.act_inference(permuted)
        torch.testing.assert_close(original_action, permuted_action, rtol=1.0e-5, atol=1.0e-6)

    def test_relative_goal_changes_per_point_attention_weights(self) -> None:
        """The global goal must query point tokens, not only bypass pooling."""

        model = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
        )
        target = torch.randn(1, 512, 5).repeat(2, 1, 1)
        state = torch.zeros(2, 45)
        # State layout: hand(9), robot(14), previous action(7), rel_goal(9),
        # target twist(6).  Change only the recoverable relative goal.
        state[1, 30:33] = torch.tensor((1.0, -0.5, 0.25))
        state[1, 33:39] = torch.tensor((0.0, 1.0, 0.0, -1.0, 0.0, 0.0))

        target_tokens = model.target_pointnet(target)
        query = model.state_encoder(state).unsqueeze(1)
        _, weights = model.target_attention(
            query,
            target_tokens,
            target_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.assertEqual(tuple(weights.shape), (2, 4, 1, 512))
        self.assertGreater(float((weights[0] - weights[1]).abs().max()), 1.0e-7)

    def test_multi_query_policy_keeps_external_observation_contract(self) -> None:
        model = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
            attention_queries=16,
        )
        actor_observations = torch.randn(2, 4141)
        critic_observations = torch.randn(2, 4146)
        target, _, state = model._split_observations(actor_observations)
        target_tokens = model.target_pointnet(target)
        state_queries = model.state_encoder(state).reshape(2, 16, 64)
        attended, weights = model.target_attention(
            state_queries,
            target_tokens,
            target_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.assertEqual(tuple(attended.shape), (2, 16, 64))
        self.assertEqual(tuple(weights.shape), (2, 4, 16, 512))
        self.assertEqual(tuple(model.act_inference(actor_observations).shape), (2, 7))
        self.assertEqual(tuple(model.evaluate(critic_observations).shape), (2, 1))

    def test_attention_query_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention_queries"):
            ActorCriticAffordance(4146, 4146, 7, attention_queries=0)

    def test_cartesian_action_contract_changes_only_state_and_action_dims(self) -> None:
        model = ActorCriticAffordance(
            4140,
            4145,
            6,
            environment_state_dim=44,
            critic_environment_state_dim=49,
        )
        self.assertEqual(tuple(model.act_inference(torch.randn(2, 4140)).shape), (2, 6))
        self.assertEqual(tuple(model.evaluate(torch.randn(2, 4145)).shape), (2, 1))

    def test_wrong_observation_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "4146"):
            ActorCriticAffordance(4145, 4145, 7)

    def test_teacher_contract_removes_only_five_physics_scalars(self) -> None:
        model = ActorCriticAffordance(
            4141,
            4141,
            7,
            environment_state_dim=45,
        )
        observations = torch.randn(2, 4141)
        target, obstacles, state = model._split_observations(observations)
        self.assertEqual(target.shape, (2, 512, 5))
        self.assertEqual(obstacles.shape, (2, 512, 3))
        self.assertEqual(state.shape, (2, 45))
        self.assertEqual(model.act_inference(observations).shape, (2, 7))

    def test_asymmetric_teacher_critic_has_an_independent_encoder(self) -> None:
        model = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
        )
        actor_observations = torch.randn(2, 4141)
        critic_observations = torch.randn(2, 4146)
        self.assertEqual(model.act_inference(actor_observations).shape, (2, 7))
        self.assertEqual(model.evaluate(critic_observations).shape, (2, 1))
        self.assertIsNot(model.target_pointnet, model.critic_target_pointnet)
        self.assertIsNot(model.feature_fusion, model.critic_feature_fusion)

    def test_asymmetric_contract_rejects_wrong_critic_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "critic expected 4146"):
            ActorCriticAffordance(
                4141,
                4141,
                7,
                environment_state_dim=45,
                critic_environment_state_dim=50,
            )

    def test_legacy_actor_checkpoint_loads_into_asymmetric_model(self) -> None:
        legacy = ActorCriticAffordance(
            4141,
            4141,
            7,
            environment_state_dim=45,
        )
        asymmetric = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
        )
        actor_observations = torch.randn(2, 4141)
        asymmetric.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(
            asymmetric.act_inference(actor_observations),
            legacy.act_inference(actor_observations),
        )

    def test_resume_noise_can_be_capped_independently_of_checkpoint_parameter(self) -> None:
        model = ActorCriticAffordance(
            4146,
            4146,
            7,
            init_noise_std=0.35,
            noise_std_type="log",
            max_noise_std=0.05,
        )
        model.act(self.observations)
        torch.testing.assert_close(
            model.action_std,
            torch.full_like(model.action_std, 0.05),
        )

    def test_relation_encoder_load_is_exactly_behavior_preserving(self) -> None:
        baseline = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
        )
        relation = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
            use_relation_features=True,
        )
        actor_observations = torch.randn(2, 4141)
        relation.load_state_dict(baseline.state_dict(), strict=True)
        torch.testing.assert_close(
            relation.act_inference(actor_observations),
            baseline.act_inference(actor_observations),
        )

        relation.zero_grad(set_to_none=True)
        relation.act_inference(actor_observations).sum().backward()
        self.assertIsNotNone(relation.target_relation_pointnet[-1].weight.grad)
        self.assertGreater(
            float(relation.target_relation_pointnet[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_relation_inputs_recover_hand_and_goal_conditioned_side(self) -> None:
        target = torch.zeros(1, 512, 5)
        target[0, :, 0] = torch.linspace(0.4, 0.6, 512)
        obstacles = torch.zeros(1, 512, 3)
        state = torch.zeros(1, 45)
        # Normalized hand position zero maps to metric [0.5, 0.0, 0.15].
        # Positive normalized rel-goal X maps to a +X displacement.
        state[0, 30] = 1.0
        target_relation, _, obstacle_valid = ActorCriticAffordance._relation_inputs(
            target, obstacles, state
        )
        torch.testing.assert_close(
            target_relation[0, 0, :3], torch.tensor([-0.25, 0.0, -0.375])
        )
        self.assertGreater(float(target_relation[0, 0, -1]), 0.0)
        self.assertLess(float(target_relation[0, -1, -1]), 0.0)
        self.assertFalse(bool(obstacle_valid[0]))

    def test_wrench_relation_selects_opposite_sides_for_opposite_yaw(self) -> None:
        target = torch.zeros(2, 512, 5)
        target[:, :, 0] = -0.10
        target[:, :, 1] = torch.linspace(-0.05, 0.05, 512)
        obstacles = torch.zeros(2, 512, 3)
        state = torch.zeros(2, 45)
        state[:, 30] = 1.0  # +X translation goal.
        yaw = torch.tensor([0.50, -0.50])
        state[:, 33] = torch.cos(yaw)
        state[:, 36] = torch.sin(yaw)

        relation, _, _ = ActorCriticAffordance._relation_inputs(
            target,
            obstacles,
            state,
            use_wrench_relation_features=True,
            yaw_moment_weight=1.0,
            yaw_activation_rad=0.10,
        )
        support = relation[..., -1]
        self.assertEqual(int(support[0].argmax()), 0)
        self.assertEqual(int(support[1].argmax()), 511)

        translation_only, _, _ = ActorCriticAffordance._relation_inputs(
            target,
            obstacles,
            state,
            use_wrench_relation_features=False,
        )
        torch.testing.assert_close(
            translation_only[0, :, -1],
            torch.zeros_like(translation_only[0, :, -1]),
        )

    def test_separate_wrench_relations_preserve_translation_and_yaw(self) -> None:
        target = torch.zeros(2, 512, 5)
        target[:, :, 0] = torch.linspace(-0.10, 0.10, 512)
        target[:, :, 1] = torch.linspace(-0.05, 0.05, 512)
        obstacles = torch.zeros(2, 512, 3)
        state = torch.zeros(2, 45)
        state[:, 30] = 1.0
        yaw = torch.tensor([0.30, -0.30])
        state[:, 33] = torch.cos(yaw)
        state[:, 36] = torch.sin(yaw)

        relation, _, _ = ActorCriticAffordance._relation_inputs(
            target,
            obstacles,
            state,
            use_wrench_relation_features=True,
            separate_wrench_relation_features=True,
            yaw_moment_weight=1.5,
            yaw_activation_rad=0.10,
        )
        self.assertEqual(tuple(relation.shape), (2, 512, 8))
        # Translation support is yaw-invariant, while the signed moment
        # channel mirrors exactly when the requested yaw changes sign.
        torch.testing.assert_close(relation[0, :, -2], relation[1, :, -2])
        torch.testing.assert_close(relation[0, :, -1], -relation[1, :, -1])

    def test_from_scratch_relation_branch_can_start_nonzero(self) -> None:
        model = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
            use_relation_features=True,
            use_wrench_relation_features=True,
            separate_wrench_relation_features=True,
            zero_initialize_relation_output=False,
        )
        final = next(
            layer
            for layer in reversed(model.target_relation_pointnet)
            if isinstance(layer, torch.nn.Linear)
        )
        self.assertGreater(float(final.weight.abs().sum()), 0.0)

    def test_protected_obstacle_relation_is_signed_and_masks_empty_clutter(self) -> None:
        target = torch.zeros(2, 512, 5)
        # A compact protected endpoint centred at x=0.10 m.
        target[:, :16, 0] = 0.10
        target[:, :16, 4] = 1.0
        obstacles = torch.zeros(2, 512, 3)
        obstacles[0, :, 0] = 0.14

        relation, valid = (
            ActorCriticAffordance._protected_obstacle_relation_inputs(
                target, obstacles
            )
        )
        self.assertEqual(tuple(relation.shape), (2, 512, 4))
        torch.testing.assert_close(
            relation[0, 0], torch.tensor((0.20, 0.0, 0.0, 0.20))
        )
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[1]))

    def test_protected_obstacle_residual_load_is_behavior_preserving(self) -> None:
        baseline = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
            use_relation_features=True,
            use_wrench_relation_features=True,
            separate_wrench_relation_features=True,
            zero_initialize_relation_output=False,
        )
        enhanced = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
            use_relation_features=True,
            use_wrench_relation_features=True,
            separate_wrench_relation_features=True,
            zero_initialize_relation_output=False,
            use_protected_obstacle_relation_features=True,
            zero_initialize_protected_obstacle_relation_output=True,
        )
        observations = torch.randn(2, 4141)
        enhanced.load_state_dict(baseline.state_dict(), strict=True)
        torch.testing.assert_close(
            enhanced.act_inference(observations),
            baseline.act_inference(observations),
        )

        enhanced.zero_grad(set_to_none=True)
        enhanced.act_inference(observations).sum().backward()
        final = next(
            layer
            for layer in reversed(enhanced.protected_obstacle_relation_pointnet)
            if isinstance(layer, torch.nn.Linear)
        )
        self.assertIsNotNone(final.weight.grad)
        self.assertGreater(float(final.weight.grad.abs().sum()), 0.0)

    def test_protected_obstacle_adapter_freezes_only_base_actor(self) -> None:
        model = ActorCriticAffordance(
            4141,
            4146,
            7,
            environment_state_dim=45,
            critic_environment_state_dim=50,
            use_relation_features=True,
            use_wrench_relation_features=True,
            separate_wrench_relation_features=True,
            use_protected_obstacle_relation_features=True,
            freeze_base_actor_for_protected_obstacle_transfer=True,
        )
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        frozen = {
            name for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("protected_obstacle_relation_pointnet.")
                or name.startswith("critic")
                for name in trainable
            )
        )
        self.assertIn("actor.0.weight", frozen)
        self.assertIn("target_pointnet.0.weight", frozen)
        self.assertIn("std", frozen)
        self.assertIn(
            "protected_obstacle_relation_pointnet.2.weight", trainable
        )
        self.assertIn("critic.0.weight", trainable)


if __name__ == "__main__":
    unittest.main()
