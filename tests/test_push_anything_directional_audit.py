from __future__ import annotations

import unittest

from scripts.analyze_push_anything_directional_failures import (
    classify,
    longest_joint_gate_run,
)


class PushAnythingDirectionalAuditTest(unittest.TestCase):
    def test_joint_gate_requires_position_and_rotation_together(self) -> None:
        rows = [
            {"position_error_m": "0.01", "rotation_error_rad": "0.20"},
            {"position_error_m": "0.03", "rotation_error_rad": "0.05"},
            {"position_error_m": "0.01", "rotation_error_rad": "0.05"},
            {"position_error_m": "0.01", "rotation_error_rad": "0.05"},
            {"position_error_m": "0.03", "rotation_error_rad": "0.05"},
        ]
        self.assertEqual(longest_joint_gate_run(rows, 0.02, 0.1), 2)

    def test_classification_separates_guard_deadlock_and_near_gate(self) -> None:
        common = {
            "accepted": False,
            "best_position_error_m": 0.035,
        }
        self.assertEqual(
            classify({**common, "previous_target_semantic_rejections": 200}),
            "semantic_guard_reposition_deadlock",
        )
        self.assertEqual(
            classify({**common, "previous_target_semantic_rejections": 0}),
            "near_position_gate_or_pose_coupling",
        )


if __name__ == "__main__":
    unittest.main()
