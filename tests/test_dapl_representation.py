from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from dapl.representation import (
    DAPLSceneTensorBuilder,
    DAPLSceneTensorConfig,
    PhysicalSceneBatch,
    SceneComponent,
)


class DAPLSceneTensorBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DAPLSceneTensorConfig(
            target_points=2,
            obstacle_points=3,
            end_effector_points=1,
            canonical_object_points=4,
        )
        self.builder = DAPLSceneTensorBuilder(self.config, validate_values=True)

    @staticmethod
    def batch() -> PhysicalSceneBatch:
        return PhysicalSceneBatch(
            target_points=torch.tensor([[[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]]),
            target_mass=torch.tensor([2.0]),
            target_velocity=torch.tensor([[1.0, 2.0, 3.0]]),
            obstacle_points=torch.tensor(
                [[
                    [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
                ]]
            ),
            obstacle_masses=torch.tensor([[4.0, 8.0]]),
            obstacle_velocities=torch.tensor([[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
            end_effector_points=torch.tensor([[[9.0, 0.0, 0.0]]]),
            end_effector_mass=torch.tensor([3.0]),
            end_effector_velocity=torch.tensor([[-1.0, 0.0, 1.0]]),
        )

    def test_default_paper_shape(self) -> None:
        config = DAPLSceneTensorConfig()
        self.assertEqual(config.total_points, 1280)
        self.assertEqual(config.feature_dim, 7)

        batch = PhysicalSceneBatch(
            target_points=torch.zeros(2, 512, 3),
            target_mass=torch.tensor([1.0, 2.0]),
            target_velocity=torch.zeros(2, 3),
            obstacle_points=torch.ones(2, 2, 512, 3),
            obstacle_masses=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            obstacle_velocities=torch.zeros(2, 2, 3),
            end_effector_points=torch.zeros(2, 256, 3),
            end_effector_mass=torch.tensor([3.0, 4.0]),
            end_effector_velocity=torch.zeros(2, 3),
        )
        result = DAPLSceneTensorBuilder(config)(batch)
        self.assertEqual(result.features.shape, (2, 1280, 7))

    def test_nearest_selection_feature_order_and_point_mass(self) -> None:
        result = self.builder(self.batch())

        self.assertEqual(result.features.shape, (1, 6, 7))
        self.assertEqual(result.obstacle_source_indices.tolist(), [[0, 1, 2]])
        torch.testing.assert_close(result.target[0, :, 3], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(result.obstacles[0, :, 3], torch.tensor([1.0, 1.0, 1.0]))
        torch.testing.assert_close(result.end_effector[0, :, 3], torch.tensor([3.0]))
        torch.testing.assert_close(
            result.obstacles[0, :, 4:],
            torch.tensor([[4.0, 5.0, 6.0]]).expand(3, -1),
        )
        self.assertEqual(
            result.component_ids.tolist(),
            [
                SceneComponent.TARGET,
                SceneComponent.TARGET,
                SceneComponent.OBSTACLE,
                SceneComponent.OBSTACLE,
                SceneComponent.OBSTACLE,
                SceneComponent.END_EFFECTOR,
            ],
        )

    def test_selection_can_be_reused_for_future_frame_alignment(self) -> None:
        current = self.builder(self.batch())
        future_batch = self.batch()
        future_batch.obstacle_points[0, 0, :, 0] += 100.0
        future_batch.obstacle_points[0, 1, :, 0] -= 9.95

        independently_selected = self.builder(future_batch)
        aligned = self.builder(
            future_batch, obstacle_source_indices=current.obstacle_source_indices
        )

        self.assertNotEqual(
            independently_selected.obstacle_source_indices.tolist(),
            current.obstacle_source_indices.tolist(),
        )
        self.assertEqual(
            aligned.obstacle_source_indices.tolist(),
            current.obstacle_source_indices.tolist(),
        )
        self.assertTrue(torch.all(aligned.obstacles[..., 0] > 100.0))

    def test_padded_obstacles_are_never_selected(self) -> None:
        batch = replace(self.batch(), obstacle_mask=torch.tensor([[False, True]]))
        result = self.builder(batch)
        self.assertEqual(result.obstacle_object_indices.tolist(), [[1, 1, 1]])
        torch.testing.assert_close(result.obstacles[0, :, 3], torch.tensor([2.0, 2.0, 2.0]))

    def test_insufficient_valid_obstacle_points_is_rejected(self) -> None:
        batch = replace(self.batch(), obstacle_mask=torch.tensor([[False, False]]))
        with self.assertRaisesRegex(ValueError, "at least 3"):
            self.builder(batch)

    def test_negative_mass_is_rejected_in_validation_mode(self) -> None:
        batch = replace(self.batch(), target_mass=torch.tensor([-1.0]))
        with self.assertRaisesRegex(ValueError, "masses"):
            self.builder(batch)


if __name__ == "__main__":
    unittest.main()
