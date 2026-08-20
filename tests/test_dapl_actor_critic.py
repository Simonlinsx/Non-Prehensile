from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from dapl.models import (
    DAPLFeatureNormalizer,
    DAPLSemanticPatchTokenizerConfig,
    DAPLWorldModel,
    DAPLWorldModelConfig,
)
from rsl_rl.modules import ActorCriticDAPL


class DAPLActorCriticTest(unittest.TestCase):
    @staticmethod
    def _checkpoint(path: Path) -> None:
        torch.manual_seed(31)
        config = DAPLWorldModelConfig(
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
        model = DAPLWorldModel(config)
        normalizer = DAPLFeatureNormalizer()
        normalizer.update(torch.randn(2, 1280, 7))
        torch.save(
            {
                "schema_version": 1,
                "step": 123,
                "model_config": asdict(config),
                "model": model.state_dict(),
                "normalizer": normalizer.state_dict(),
            },
            path,
        )

    def test_frozen_encoder_policy_shapes_and_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "world_model.pt"
            self._checkpoint(checkpoint)
            policy = ActorCriticDAPL(
                9004,
                9004,
                7,
                world_model_checkpoint_path=str(checkpoint),
                policy_attention_heads=4,
                fusion_hidden_dims=[32, 16, 8],
                actor_hidden_dims=[4],
                critic_hidden_dims=[4],
            )
            observations = torch.randn(2, 9004)

            actions = policy.act(observations)
            values = policy.evaluate(observations)
            log_probability = policy.get_actions_log_prob(actions)

            self.assertEqual(actions.shape, (2, 7))
            self.assertEqual(values.shape, (2, 1))
            self.assertEqual(log_probability.shape, (2,))
            self.assertEqual(policy.pretrained_world_model_step.item(), 123)
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in policy.dynamics_encoder.parameters()
                )
            )
            policy.train()
            self.assertFalse(policy.dynamics_encoder.training)

    def test_rejects_observation_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "world_model.pt"
            self._checkpoint(checkpoint)
            with self.assertRaisesRegex(ValueError, "dimension 9004"):
                ActorCriticDAPL(
                    1580,
                    1580,
                    7,
                    world_model_checkpoint_path=str(checkpoint),
                    policy_attention_heads=4,
                )


if __name__ == "__main__":
    unittest.main()
