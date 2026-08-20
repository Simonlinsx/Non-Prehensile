from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from dapl.models import (
    DAPLFeatureNormalizer,
    DAPLSemanticPatchTokenizerConfig,
    DAPLWorldModel,
    DAPLWorldModelConfig,
    DAPLWorldModelLoss,
    DAPLWorldModelLossConfig,
)


class DAPLWorldModelTest(unittest.TestCase):
    @staticmethod
    def small_config() -> DAPLWorldModelConfig:
        return DAPLWorldModelConfig(
            tokenizer=DAPLSemanticPatchTokenizerConfig(
                token_dim=32,
                target_patches=2,
                obstacle_patches=2,
                end_effector_patches=2,
                neighbors=4,
            ),
            encoder_depth=2,
            attention_heads=4,
        )

    def test_default_architecture_matches_supplement(self) -> None:
        config = DAPLWorldModelConfig()
        self.assertEqual(config.tokenizer.total_patches, 40)
        self.assertEqual(config.tokenizer.token_dim, 128)
        self.assertEqual(config.encoder_depth, 12)
        self.assertEqual(config.attention_heads, 8)
        self.assertEqual(config.action_dim, 3)

    def test_forward_loss_shapes_and_gradients(self) -> None:
        torch.manual_seed(13)
        scene = torch.randn(2, 1280, 7)
        future = torch.randn(2, 1280, 7)
        flow = torch.randn(2, 3)
        model = DAPLWorldModel(self.small_config())

        prediction = model(scene, flow)
        losses = DAPLWorldModelLoss()(prediction, future)

        self.assertEqual(prediction.position.shape, (2, 1280, 3))
        self.assertEqual(prediction.velocity.shape, (2, 1280, 3))
        self.assertEqual(prediction.dynamics_tokens.shape, (2, 6, 32))
        self.assertEqual(prediction.point_features.shape, (2, 1280, 32))
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        gradient_count = sum(
            parameter.grad is not None for parameter in model.parameters()
        )
        self.assertGreater(gradient_count, 0)

    def test_exact_prediction_has_zero_loss(self) -> None:
        torch.manual_seed(19)
        future = torch.randn(1, 1280, 7)
        model = DAPLWorldModel(self.small_config())
        prediction = model(torch.randn_like(future), torch.zeros(1, 3))
        exact = replace(
            prediction,
            position=future[..., :3],
            velocity=future[..., 4:7],
        )
        losses = DAPLWorldModelLoss()(exact, future)
        self.assertEqual(losses.total.item(), 0.0)

    def test_variance_regularizer_detects_velocity_collapse(self) -> None:
        torch.manual_seed(23)
        future = torch.zeros(1, 1280, 7)
        future[..., 4:7] = torch.randn(1, 1280, 3)
        model = DAPLWorldModel(self.small_config())
        prediction = model(torch.zeros_like(future), torch.zeros(1, 3))
        collapsed = replace(
            prediction,
            position=future[..., :3],
            velocity=torch.zeros_like(future[..., 4:7]),
        )
        losses = DAPLWorldModelLoss(
            DAPLWorldModelLossConfig(
                position_weight=0.0, velocity_weight=0.0, variance_weight=100.0
            )
        )(collapsed, future)
        self.assertGreater(losses.variance.item(), 0.0)
        torch.testing.assert_close(losses.total, 100.0 * losses.variance)

    def test_running_feature_normalizer_round_trip(self) -> None:
        first = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
        second = first + 10.0
        normalizer = DAPLFeatureNormalizer()
        normalizer.update(first)
        normalizer.update(second)
        normalized = normalizer.normalize(first)
        restored = normalizer.denormalize(normalized)
        torch.testing.assert_close(restored, first)
        self.assertEqual(normalizer.count.item(), 12.0)


if __name__ == "__main__":
    unittest.main()
