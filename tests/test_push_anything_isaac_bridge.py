from __future__ import annotations

import ast
import bisect
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dapl.contact_planner import (
    execution_rotation_guard_requires_retreat,
    load_c3_joint_trajectory,
    measured_contact_is_yaw_recovery,
    measured_yaw_brake_requested,
    measured_yaw_regression_brake_requested,
    quaternion_multiply_wxyz,
    resample_c3_joint_trajectory,
    stage_to_isaac_scene,
)
from dapl.scene import load_scene_manifest
from dapl.contact_planner.isaac_bridge import minimum_reference_height


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl"


def load_relay_trajectory_evaluator():
    source = REPO_ROOT / "third_party/push_anything/online_bridge_relay.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_trajectory"
    )
    namespace = {"bisect": bisect}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["evaluate_trajectory"]


def load_relay_object_trajectory_selector():
    source = REPO_ROOT / "third_party/push_anything/online_bridge_relay.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"trajectory_by_name", "target_object_trajectory"}
    ]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["target_object_trajectory"]


class PushAnythingIsaacBridgeTest(unittest.TestCase):
    def test_finger_height_floor_uses_rotated_lowest_vertex(self):
        vertices = [(-.01, -.02, -.045), (.01, .02, .008)]
        self.assertAlmostEqual(minimum_reference_height(vertices, (1, 0, 0, 0), -.029), .018)
        self.assertAlmostEqual(minimum_reference_height(vertices, (0, 2, 0, 0), -.029), -.019)
        half = math.sqrt(.5)
        self.assertAlmostEqual(minimum_reference_height(vertices, (half, 0, half, 0), -.029), -.017)

    def test_finger_height_floor_rejects_invalid_geometry(self):
        for vertices, quaternion, clearance in [([], (1, 0, 0, 0), .002),
                ([(0, 0, float("nan"))], (1, 0, 0, 0), .002),
                ([(0, 0, 0)], (0, 0, 0, 0), .002),
                ([(0, 0, 0)], (1, 0, 0, 0), -.001)]:
            with self.assertRaises(ValueError):
                minimum_reference_height(vertices, quaternion, -.029, clearance)

    def setUp(self) -> None:
        self.template = next(load_scene_manifest(TEMPLATE))
        self.stage = {
            "schema": "nonprehensile.push_anything_stage.v1",
            "asset_name": "DOMINO_020_hammer_safe",
            "initial_xy_m": [0.41, 0.19],
            "goal_xy_m": [0.44177933498176825, 0.25237045669318575],
            "root_height_m": -0.01606310028273897,
            "goal_yaw_deg": -5.0,
            "support_quaternion_wxyz": [
                -0.4937799140314683,
                0.5013379099086276,
                0.506216296031207,
                0.4985847553024161,
            ],
        }

    def test_stage_preserves_support_and_maps_goal_yaw(self) -> None:
        scene = stage_to_isaac_scene(self.stage, self.template, scene_id="scene000")
        task = scene.tasks[0]
        self.assertEqual(len(scene.objects), 4)
        self.assertEqual(scene.split, "eval")
        self.assertAlmostEqual(task.initial_pose[2], 0.01293689971726103)
        self.assertAlmostEqual(task.planar_displacement, 0.07)
        self.assertNotEqual(task.initial_pose[3:], self.template.target_object.pose[3:])
        for actual, wanted in zip(
            task.initial_pose[3:], self.stage["support_quaternion_wxyz"]
        ):
            self.assertAlmostEqual(actual, wanted)
        yaw = math.radians(-5.0)
        expected = quaternion_multiply_wxyz(
            (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
            self.stage["support_quaternion_wxyz"],
        )
        for actual, wanted in zip(task.goal_pose[3:], expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertAlmostEqual(scene.target_object.mass_kg, 0.05)
        self.assertAlmostEqual(scene.target_object.dynamic_friction, 0.3)

    def test_random_initial_yaw_rotates_body_without_rotating_support_geometry(self):
        self.stage.update(initial_yaw_deg=120.0, goal_yaw_deg=130.0)
        scene = stage_to_isaac_scene(self.stage, self.template, scene_id="yaw")
        initial = scene.tasks[0].initial_pose[3:]
        goal = scene.tasks[0].goal_pose[3:]
        dot = abs(sum(a*b for a, b in zip(initial, goal)))
        self.assertAlmostEqual(2*math.acos(min(1.0, dot)), math.radians(10), places=7)
        support = self.stage["support_quaternion_wxyz"]
        restored = quaternion_multiply_wxyz((.5, 0, 0, -math.sqrt(3)/2), initial)
        for a, b in zip(restored, support): self.assertAlmostEqual(a, b)

    def test_load_and_resample_joint_trajectory(self) -> None:
        fieldnames = [
            "franka_state_utime",
            *(f"franka_q_{index}" for index in range(7)),
            "ee_x_m",
            "ee_y_m",
            "ee_z_m",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for timestamp, value in ((100_000, 0.0), (200_000, 1.0), (300_000, 2.0)):
                    row = {"franka_state_utime": timestamp}
                    row.update({f"franka_q_{index}": value for index in range(7)})
                    row.update({"ee_x_m": value, "ee_y_m": 0.0, "ee_z_m": 0.1})
                    writer.writerow(row)
            samples = load_c3_joint_trajectory(path)
            for actual, expected in zip(
                (item.time_s for item in samples), (0.0, 0.1, 0.2)
            ):
                self.assertAlmostEqual(actual, expected)
            resampled = resample_c3_joint_trajectory(samples, 0.05)
            self.assertEqual(len(resampled), 5)
            self.assertAlmostEqual(resampled[1].q[0], 0.5)
            self.assertAlmostEqual(resampled[-1].q[-1], 2.0)

    def test_online_relay_uses_right_derivative_at_first_knot(self) -> None:
        evaluate_trajectory = load_relay_trajectory_evaluator()
        trajectory = type("Trajectory", (), {
            "time_vec": [1.0, 1.1, 1.2],
            "datapoints": [[0.0, 0.02, 0.03], [1.0, 1.0, 1.0]],
        })()

        value, velocity = evaluate_trajectory(trajectory, 1.0)

        self.assertEqual(value, (0.0, 1.0))
        self.assertAlmostEqual(velocity[0], 0.2)
        self.assertAlmostEqual(velocity[1], 0.0)

    def test_online_relay_reads_upstream_indexed_target_object_plan(self) -> None:
        select_target = load_relay_object_trajectory_selector()
        indexed = SimpleNamespace(
            trajectory_name="object_position_target_0", num_points=11)
        legacy = SimpleNamespace(
            trajectory_name="object_position_target", num_points=11)
        message = SimpleNamespace(saved_traj=SimpleNamespace(
            trajectories=[legacy, indexed]))

        self.assertIs(
            select_target(message, "object_position_target"), indexed)

    def test_measured_yaw_recovery_requires_progress_and_inward_rate(self) -> None:
        common = {
            "activation_threshold_rad": 0.05,
            "minimum_progress_rad": 0.002,
            "minimum_rate_rad_s": 0.005,
        }
        self.assertTrue(measured_contact_is_yaw_recovery(
            start_signed_error_rad=0.09,
            current_signed_error_rad=0.087,
            current_yaw_rate_rad_s=-0.03,
            **common,
        ))
        self.assertTrue(measured_contact_is_yaw_recovery(
            start_signed_error_rad=-0.09,
            current_signed_error_rad=-0.087,
            current_yaw_rate_rad_s=0.03,
            **common,
        ))
        self.assertFalse(measured_contact_is_yaw_recovery(
            start_signed_error_rad=0.09,
            current_signed_error_rad=0.087,
            current_yaw_rate_rad_s=0.03,
            **common,
        ))
        self.assertFalse(measured_contact_is_yaw_recovery(
            start_signed_error_rad=0.09,
            current_signed_error_rad=0.089,
            current_yaw_rate_rad_s=-0.03,
            **common,
        ))
        self.assertFalse(measured_contact_is_yaw_recovery(
            start_signed_error_rad=0.04,
            current_signed_error_rad=0.03,
            current_yaw_rate_rad_s=-0.03,
            **common,
        ))

    def test_measured_yaw_brake_predicts_inner_band_entry(self) -> None:
        self.assertTrue(measured_yaw_brake_requested(
            start_signed_error_rad=0.09,
            current_signed_error_rad=0.06,
            current_yaw_rate_rad_s=-0.20,
            recovery_latched=True,
            exit_threshold_rad=0.0525,
            lookahead_s=0.05,
        ))
        self.assertFalse(measured_yaw_brake_requested(
            start_signed_error_rad=0.09,
            current_signed_error_rad=0.06,
            current_yaw_rate_rad_s=0.20,
            recovery_latched=True,
            exit_threshold_rad=0.0525,
            lookahead_s=0.05,
        ))

    def test_measured_yaw_brake_requires_latched_corrective_motion(self) -> None:
        self.assertFalse(measured_yaw_brake_requested(
            start_signed_error_rad=0.09,
            current_signed_error_rad=0.04,
            current_yaw_rate_rad_s=-0.20,
            recovery_latched=False,
            exit_threshold_rad=0.0525,
            lookahead_s=0.05,
        ))
        self.assertTrue(measured_yaw_brake_requested(
            start_signed_error_rad=0.09,
            current_signed_error_rad=-0.01,
            current_yaw_rate_rad_s=-0.05,
            recovery_latched=True,
            exit_threshold_rad=0.0525,
            lookahead_s=0.05,
        ))

    def test_measured_yaw_regression_brake_is_sign_symmetric(self) -> None:
        for current_error in (0.026, -0.026):
            self.assertTrue(measured_yaw_regression_brake_requested(
                current_signed_error_rad=current_error,
                best_abs_error_rad=0.020,
                regression_tolerance_rad=0.005,
            ))
        self.assertFalse(measured_yaw_regression_brake_requested(
            current_signed_error_rad=-0.024,
            best_abs_error_rad=0.020,
            regression_tolerance_rad=0.005,
        ))

    def test_rotation_guard_requires_planner_recovery_outside_band(self) -> None:
        common = {
            "limit_rad": 0.1,
            "lookahead_s": 0.05,
            "recovery_outward_rate_tolerance_rad_s": 0.005,
        }
        self.assertTrue(execution_rotation_guard_requires_retreat(
            signed_error_rad=0.11,
            yaw_rate_rad_s=0.0,
            planner_marks_yaw_recovery=False,
            **common,
        ))
        self.assertFalse(execution_rotation_guard_requires_retreat(
            signed_error_rad=0.11,
            yaw_rate_rad_s=-0.02,
            planner_marks_yaw_recovery=True,
            **common,
        ))
        self.assertTrue(execution_rotation_guard_requires_retreat(
            signed_error_rad=-0.11,
            yaw_rate_rad_s=-0.02,
            planner_marks_yaw_recovery=True,
            **common,
        ))

    def test_rotation_guard_keeps_lookahead_inside_band(self) -> None:
        self.assertTrue(execution_rotation_guard_requires_retreat(
            signed_error_rad=0.095,
            yaw_rate_rad_s=0.2,
            limit_rad=0.1,
            lookahead_s=0.05,
            recovery_outward_rate_tolerance_rad_s=0.005,
            planner_marks_yaw_recovery=False,
        ))
        self.assertFalse(execution_rotation_guard_requires_retreat(
            signed_error_rad=0.095,
            yaw_rate_rad_s=-0.2,
            limit_rad=0.1,
            lookahead_s=0.05,
            recovery_outward_rate_tolerance_rad_s=0.005,
            planner_marks_yaw_recovery=False,
        ))

    def test_rejects_wrong_asset(self) -> None:
        stage = json.loads(json.dumps(self.stage))
        stage["asset_name"] = "not-the-hammer"
        with self.assertRaises(ValueError):
            stage_to_isaac_scene(stage, self.template, scene_id="bad")

    def test_rejects_missing_support_frame(self) -> None:
        stage = json.loads(json.dumps(self.stage))
        del stage["support_quaternion_wxyz"]
        with self.assertRaisesRegex(ValueError, "support_quaternion_wxyz"):
            stage_to_isaac_scene(stage, self.template, scene_id="bad-frame")


if __name__ == "__main__":
    unittest.main()
