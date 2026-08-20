from __future__ import annotations

import unittest

import torch

from dapl.metrics import planar_pose_success


class DAPLTaskMetricsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
