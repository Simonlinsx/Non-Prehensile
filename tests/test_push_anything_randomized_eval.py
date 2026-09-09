from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_push_anything_c1_randomized import result_from_artifacts, summarize
from scripts.generate_push_anything_c1_eval_manifest import build_scenes


class PushAnythingRandomizedEvalTest(unittest.TestCase):
    def test_manifest_is_deterministic_balanced_and_outward_hemisphere(self) -> None:
        scenes = build_scenes(50, 20260901)
        self.assertEqual(scenes, build_scenes(50, 20260901))
        self.assertEqual(len(scenes), 50)
        self.assertEqual(len({scene["scene_id"] for scene in scenes}), 50)
        self.assertEqual(len({scene["sampling_seed"] for scene in scenes}), 50)

        directions = [
            float(scene["goal_direction_relative_deg"]) for scene in scenes
        ]
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
            self.assertEqual(
                scene["goal_direction_frame"], "robot_base_to_initial_target"
            )
            direction = math.radians(float(scene["goal_direction_deg"]))
            expected = [
                scene["initial_xy_m"][0]
                + scene["goal_distance_m"] * math.cos(direction),
                scene["initial_xy_m"][1]
                + scene["goal_distance_m"] * math.sin(direction),
            ]
            self.assertAlmostEqual(scene["goal_xy_m"][0], expected[0], places=6)
            self.assertAlmostEqual(scene["goal_xy_m"][1], expected[1], places=6)
            displacement = [
                scene["goal_xy_m"][axis] - scene["initial_xy_m"][axis]
                for axis in range(2)
            ]
            radial = [
                scene["initial_xy_m"][axis] - scene["robot_base_xy_m"][axis]
                for axis in range(2)
            ]
            self.assertGreater(sum(a * b for a, b in zip(displacement, radial)), 0.0)

    def test_manifest_supports_nominal_front120_distribution(self) -> None:
        scenes = build_scenes(50, 20260902, relative_direction_limit_deg=60.0)
        directions = [
            float(scene["goal_direction_relative_deg"]) for scene in scenes
        ]
        self.assertEqual(min(directions), -58.8)
        self.assertEqual(max(directions), 58.8)
        self.assertTrue(
            all(scene["goal_direction_relative_limit_deg"] == 60.0 for scene in scenes)
        )

    def test_random_initial_yaw_manifest_keeps_relative_goal_yaw_contract(self):
        scenes = build_scenes(50, 20260909, randomize_initial_yaw=True)
        self.assertEqual(scenes, build_scenes(50, 20260909, randomize_initial_yaw=True))
        self.assertEqual(len({s['initial_yaw_deg'] for s in scenes}), 50)
        self.assertLess(min(s['initial_yaw_deg'] for s in scenes), -170)
        self.assertGreater(max(s['initial_yaw_deg'] for s in scenes), 170)
        for s in scenes:
            self.assertAlmostEqual((s['goal_yaw_deg'] - s['initial_yaw_deg'] + 180) % 360 - 180,
                                   s['goal_yaw_delta_deg'])

    def test_summary_keeps_geometry_c1_and_joint_separate(self) -> None:
        results = [
            {
                "scene_id": "scene000",
                "goal_direction_deg": -80.0,
                "evaluable": True,
                "geometry_pass": True,
                "c1_pass": False,
                "accepted": False,
                "legal_safe_contact_rows": 0,
                "c1_violation_rows": 2,
            },
            {
                "scene_id": "scene001",
                "goal_direction_deg": 20.0,
                "evaluable": True,
                "geometry_pass": True,
                "c1_pass": True,
                "accepted": True,
                "legal_safe_contact_rows": 7,
                "c1_violation_rows": 0,
            },
        ]
        summary = summarize(results, total_scenes=50)
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["evaluable"], 2)
        self.assertEqual(summary["infrastructure_failures"], 0)
        self.assertEqual(summary["remaining"], 48)
        self.assertEqual(summary["geometry_successes"], 2)
        self.assertEqual(summary["c1_successes"], 1)
        self.assertEqual(summary["joint_successes"], 1)
        self.assertEqual(summary["joint_success_rate"], 0.5)
        self.assertEqual(summary["legal_contact_scenes"], 1)
        self.assertEqual(summary["no_legal_contact_scenes"], 1)
        self.assertEqual(summary["c1_violation_scenes"], 1)
        self.assertEqual(summary["by_direction_deg"]["[-90,-45)"]["geometry_pass"], 1)
        self.assertEqual(summary["by_direction_deg"]["[0,45)"]["c1_pass"], 1)

    def test_summary_excludes_infrastructure_failures_from_rates(self) -> None:
        results = [
            {
                "scene_id": "scene000",
                "goal_direction_deg": -60.0,
                "evaluable": True,
                "geometry_pass": True,
                "c1_pass": True,
                "accepted": True,
                "legal_safe_contact_rows": 5,
                "c1_violation_rows": 0,
            },
            {
                "scene_id": "scene001",
                "goal_direction_deg": 60.0,
                "evaluable": False,
                "geometry_pass": False,
                "c1_pass": False,
                "accepted": False,
                "legal_safe_contact_rows": None,
                "c1_violation_rows": None,
            },
        ]
        summary = summarize(results, total_scenes=2)
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["evaluable"], 1)
        self.assertEqual(summary["infrastructure_failures"], 1)
        self.assertEqual(summary["joint_success_rate"], 1.0)
        self.assertEqual(summary["failed_scene_ids"], [])
        self.assertEqual(summary["infrastructure_failure_scene_ids"], ["scene001"])
        self.assertEqual(summary["by_direction_deg"]["[45,90]"]["attempted"], 1)
        self.assertEqual(summary["by_direction_deg"]["[45,90]"]["evaluable"], 0)

    def test_child_process_crash_is_not_counted_as_algorithm_failure(self) -> None:
        scene = {
            "scene_id": "scene003",
            "initial_xy_m": [0.4, 0.19],
            "goal_xy_m": [0.43, 0.09],
            "goal_distance_m": 0.1,
            "goal_direction_deg": -73.8,
            "goal_yaw_deg": 0.0,
            "sampling_seed": 17,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "acceptance.json").write_text(
                json.dumps({"accepted": False}), encoding="utf-8"
            )
            (run_dir / "c1_semantic_audit.json").write_text(
                json.dumps({"c1_pass": True}), encoding="utf-8"
            )
            (run_dir / "joint_acceptance.json").write_text(
                json.dumps({"accepted": False}), encoding="utf-8"
            )
            (run_dir / "runner.log").write_text(
                "WARNING: child process 123 exited before monitoring completed.\n",
                encoding="utf-8",
            )
            result = result_from_artifacts(scene, run_dir)
        self.assertFalse(result["evaluable"])
        self.assertTrue(result["child_process_crash"])
        self.assertEqual(result["failure_kind"], "infrastructure")


if __name__ == "__main__":
    unittest.main()
