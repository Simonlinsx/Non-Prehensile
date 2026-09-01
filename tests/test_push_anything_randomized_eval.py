from __future__ import annotations

from collections import Counter
import math
import unittest

from scripts.evaluate_push_anything_c1_randomized import summarize
from scripts.generate_push_anything_c1_eval_manifest import build_scenes


class PushAnythingRandomizedEvalTest(unittest.TestCase):
    def test_manifest_is_deterministic_balanced_and_front_hemisphere(self) -> None:
        scenes = build_scenes(50, 20260901)
        self.assertEqual(scenes, build_scenes(50, 20260901))
        self.assertEqual(len(scenes), 50)
        self.assertEqual(len({scene["scene_id"] for scene in scenes}), 50)
        self.assertEqual(len({scene["sampling_seed"] for scene in scenes}), 50)

        directions = [float(scene["goal_direction_deg"]) for scene in scenes]
        self.assertGreaterEqual(min(directions), -90.0)
        self.assertLessEqual(max(directions), 90.0)
        self.assertLess(min(directions), -85.0)
        self.assertGreater(max(directions), 85.0)
        self.assertEqual(
            Counter(scene["goal_distance_m"] for scene in scenes),
            Counter({0.06: 10, 0.07: 10, 0.08: 10, 0.09: 10, 0.10: 10}),
        )
        self.assertEqual(
            Counter(scene["goal_yaw_deg"] for scene in scenes),
            Counter({-10.0: 10, -5.0: 10, 0.0: 10, 5.0: 10, 10.0: 10}),
        )
        for scene in scenes:
            direction = math.radians(float(scene["goal_direction_deg"]))
            expected = [
                scene["initial_xy_m"][0]
                + scene["goal_distance_m"] * math.cos(direction),
                scene["initial_xy_m"][1]
                + scene["goal_distance_m"] * math.sin(direction),
            ]
            self.assertAlmostEqual(scene["goal_xy_m"][0], expected[0], places=6)
            self.assertAlmostEqual(scene["goal_xy_m"][1], expected[1], places=6)

    def test_summary_keeps_geometry_c1_and_joint_separate(self) -> None:
        results = [
            {
                "scene_id": "scene000",
                "goal_direction_deg": -80.0,
                "geometry_pass": True,
                "c1_pass": False,
                "accepted": False,
            },
            {
                "scene_id": "scene001",
                "goal_direction_deg": 20.0,
                "geometry_pass": True,
                "c1_pass": True,
                "accepted": True,
            },
        ]
        summary = summarize(results, total_scenes=50)
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["remaining"], 48)
        self.assertEqual(summary["geometry_successes"], 2)
        self.assertEqual(summary["c1_successes"], 1)
        self.assertEqual(summary["joint_successes"], 1)
        self.assertEqual(summary["joint_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
