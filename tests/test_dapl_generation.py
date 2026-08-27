from __future__ import annotations

import math
from pathlib import Path
import unittest

from dapl.generation import (
    ClutterAsset,
    ClutterGenerationConfig,
    StablePose,
    generate_clutter_scenes,
)
from dapl.scene import ClutterTrack, load_scene_manifest
from scripts.generate_domino_affordance_manifest import (
    generation_config_for_settings,
)


def asset(asset_id: str, *, large: bool) -> ClutterAsset:
    half_extent = 0.045 if large else 0.025
    return ClutterAsset(
        asset_id=asset_id,
        scale=(1.0, 1.0, 1.0),
        mass_kg=0.5 if large else 0.15,
        stable_poses=(
            StablePose(
                quaternion=(1.0, 0.0, 0.0, 0.0),
                support_height=half_extent,
                footprint=(-half_extent, -half_extent, half_extent, half_extent),
            ),
        ),
        is_large=large,
    )


CATALOG = tuple(
    [asset(f"large-{index}", large=True) for index in range(8)]
    + [asset(f"small-{index}", large=False) for index in range(10)]
)


class ClutterGenerationTest(unittest.TestCase):
    def test_default_config_is_the_reported_dapl_task_distribution(self) -> None:
        cfg = ClutterGenerationConfig()
        self.assertEqual(cfg.target_x_offset_range, (-0.15, 0.15))
        self.assertEqual(cfg.target_y_offset_range, (-0.30, 0.30))
        self.assertEqual(cfg.tasks_per_scene, 16)
        self.assertEqual(cfg.minimum_planar_displacement, 0.15)
        self.assertEqual(cfg.goal_xy_sampling, "independent")
        self.assertFalse(cfg.preserve_target_support_pose)
        self.assertIsNone(cfg.maximum_goal_yaw_delta)

    def test_planar_push_setting_changes_only_the_support_pose_contract(self) -> None:
        paper = generation_config_for_settings("dapl-paper", 3)
        planar = generation_config_for_settings("dapl-planar-push", 3)
        self.assertFalse(paper.preserve_target_support_pose)
        self.assertTrue(planar.preserve_target_support_pose)
        for field_name in paper.__dataclass_fields__:
            if field_name == "preserve_target_support_pose":
                continue
            self.assertEqual(getattr(planar, field_name), getattr(paper, field_name))

    def test_dywa_arm_div_setting_uses_center_ray_xy_contract(self) -> None:
        cfg = generation_config_for_settings("dywa-arm-div-planar-push", 3)
        self.assertEqual(cfg.target_x_offset_range, (-0.19, 0.19))
        self.assertEqual(cfg.target_y_offset_range, (-0.2375, 0.2375))
        self.assertEqual(cfg.minimum_planar_displacement, 0.055)
        self.assertEqual(cfg.goal_xy_sampling, "center_ray")
        self.assertTrue(cfg.preserve_target_support_pose)

        scene = generate_clutter_scenes(
            CATALOG,
            track="sparse",
            split="eval",
            scene_count=1,
            seed=71,
            target_asset_ids=("large-0",),
            config=cfg,
        )[0]
        for task in scene.tasks:
            initial_dx = cfg.table_center[0] - task.initial_pose[0]
            initial_dy = cfg.table_center[1] - task.initial_pose[1]
            goal_dx = task.goal_pose[0] - task.initial_pose[0]
            goal_dy = task.goal_pose[1] - task.initial_pose[1]
            radial_distance = math.hypot(initial_dx, initial_dy)
            self.assertGreaterEqual(task.planar_displacement, 0.055 - 1.0e-9)
            self.assertLessEqual(task.planar_displacement, radial_distance + 1.0e-9)
            self.assertAlmostEqual(initial_dx * goal_dy - initial_dy * goal_dx, 0.0, places=7)
            self.assertGreater(initial_dx * goal_dx + initial_dy * goal_dy, 0.0)
            self.assertEqual(task.initial_pose[2], task.goal_pose[2])

    def test_checked_in_hammer_proof_manifest_is_controlled_joint_pose(self) -> None:
        manifest = (
            Path(__file__).resolve().parents[1]
            / "data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl"
        )
        scenes = tuple(load_scene_manifest(manifest))
        self.assertEqual(len(scenes), 128)
        for scene in scenes:
            self.assertEqual(scene.target_object.asset_id, "020_hammer:0")
            self.assertEqual(len(scene.tasks), 1)
            task = scene.tasks[0]
            self.assertAlmostEqual(task.planar_displacement, 0.08, places=7)
            self.assertEqual(task.initial_pose[2], task.goal_pose[2])
            self.assertAlmostEqual(task.initial_pose[2], 0.0129368997, places=7)
            quaternion_dot = abs(
                sum(
                    left * right
                    for left, right in zip(task.initial_pose[3:], task.goal_pose[3:])
                )
            )
            rotation_distance = 2.0 * math.acos(min(1.0, quaternion_dot))
            self.assertAlmostEqual(rotation_distance, 0.15, places=7)

    def test_randomized_hammer_manifest_keeps_one_stable_support_face(self) -> None:
        manifest = (
            Path(__file__).resolve().parents[1]
            / "data/manifests/domino_hammer_joint_pose_randomized_128_v4_stable.jsonl"
        )
        scenes = tuple(load_scene_manifest(manifest))
        self.assertEqual(len(scenes), 128)
        initial_signatures = set()
        for scene in scenes:
            task = scene.tasks[0]
            initial_signatures.add(tuple(round(value, 7) for value in task.initial_pose))
            self.assertAlmostEqual(task.initial_pose[2], 0.0129368997, places=7)
            self.assertEqual(task.initial_pose[2], task.goal_pose[2])
            self.assertGreaterEqual(task.initial_pose[0], 0.465)
            self.assertLessEqual(task.initial_pose[0], 0.495)
            self.assertGreaterEqual(task.initial_pose[1], -0.020)
            self.assertLessEqual(task.initial_pose[1], 0.020)
            self.assertGreaterEqual(scene.target_object.static_friction, 0.70)
            self.assertLessEqual(scene.target_object.static_friction, 0.90)
            self.assertEqual(
                scene.target_object.static_friction,
                scene.target_object.dynamic_friction,
            )
            quaternion_dot = min(
                1.0,
                abs(
                    sum(
                        left * right
                        for left, right in zip(
                            task.initial_pose[3:], task.goal_pose[3:]
                        )
                    )
                ),
            )
            rotation_distance = 2.0 * math.acos(quaternion_dot)
            self.assertGreaterEqual(rotation_distance, 0.10)
            self.assertLessEqual(rotation_distance, 0.20)
        self.assertEqual(len(initial_signatures), 128)

    def test_all_tracks_follow_paper_counts_and_task_contract(self) -> None:
        for track in ClutterTrack:
            with self.subTest(track=track):
                scene = generate_clutter_scenes(
                    CATALOG, track=track, split="eval", scene_count=1, seed=41
                )[0]
                scene.validate_paper_contract()
                self.assertEqual(len(scene.objects), track.object_count)
                self.assertEqual(
                    sum(item.instance_id.startswith("large-") for item in scene.obstacle_objects),
                    track.large_obstacle_count,
                )
                self.assertEqual(
                    sum(item.instance_id.startswith("small-") for item in scene.obstacle_objects),
                    track.small_obstacle_count,
                )
                self.assertTrue(all(task.target_instance_id == "target" for task in scene.tasks))

    def test_generation_is_deterministic_per_scene_index(self) -> None:
        first = generate_clutter_scenes(
            CATALOG, track="sparse", split="train", scene_count=3, seed=123
        )
        repeated = generate_clutter_scenes(
            CATALOG, track="sparse", split="train", scene_count=3, seed=123
        )
        prefix = generate_clutter_scenes(
            CATALOG, track="sparse", split="train", scene_count=1, seed=123
        )
        self.assertEqual(first, repeated)
        self.assertEqual(first[:1], prefix)

    def test_target_roots_stay_inside_paper_central_region(self) -> None:
        cfg = ClutterGenerationConfig(table_center=(0.5, 0.0))
        scene = generate_clutter_scenes(
            CATALOG,
            track="moderate",
            split="eval",
            scene_count=1,
            seed=9,
            config=cfg,
        )[0]
        for task in scene.tasks:
            for pose in (task.initial_pose, task.goal_pose):
                self.assertGreaterEqual(pose[0], 0.35)
                self.assertLessEqual(pose[0], 0.65)
                self.assertGreaterEqual(pose[1], -0.30)
                self.assertLessEqual(pose[1], 0.30)

    def test_target_catalog_can_be_restricted_for_semantic_tasks(self) -> None:
        scene = generate_clutter_scenes(
            CATALOG,
            track="sparse",
            split="train",
            scene_count=1,
            seed=3,
            target_asset_ids=("large-0",),
            config=ClutterGenerationConfig(preserve_target_support_pose=True),
        )[0]
        self.assertEqual(scene.target_object.asset_id, "large-0")
        self.assertTrue(
            all(task.initial_pose[2] == task.goal_pose[2] for task in scene.tasks)
        )
        with self.assertRaisesRegex(ValueError, "absent from the catalog"):
            generate_clutter_scenes(
                CATALOG,
                track="sparse",
                split="train",
                scene_count=1,
                seed=3,
                target_asset_ids=("missing",),
            )

    def test_xy_curriculum_keeps_support_and_yaw_fixed(self) -> None:
        scene = generate_clutter_scenes(
            CATALOG,
            track="sparse",
            split="train",
            scene_count=1,
            seed=17,
            target_asset_ids=("large-0",),
            config=ClutterGenerationConfig(
                tasks_per_scene=4,
                minimum_planar_displacement=0.08,
                preserve_target_support_pose=True,
                maximum_goal_yaw_delta=0.0,
            ),
        )[0]
        for task in scene.tasks:
            self.assertEqual(task.initial_pose[2], task.goal_pose[2])
            initial_quat = task.initial_pose[3:]
            goal_quat = task.goal_pose[3:]
            dot = abs(sum(a * b for a, b in zip(initial_quat, goal_quat)))
            self.assertTrue(math.isclose(dot, 1.0, abs_tol=1.0e-7))

    def test_curriculum_yaw_delta_is_bounded(self) -> None:
        maximum_delta = 0.4
        scene = generate_clutter_scenes(
            CATALOG,
            track="sparse",
            split="train",
            scene_count=1,
            seed=23,
            target_asset_ids=("large-0",),
            config=ClutterGenerationConfig(
                tasks_per_scene=8,
                minimum_planar_displacement=0.08,
                preserve_target_support_pose=True,
                maximum_goal_yaw_delta=maximum_delta,
            ),
        )[0]
        for task in scene.tasks:
            initial_quat = task.initial_pose[3:]
            goal_quat = task.goal_pose[3:]
            dot = min(1.0, abs(sum(a * b for a, b in zip(initial_quat, goal_quat))))
            angle = 2.0 * math.acos(dot)
            self.assertLessEqual(angle, maximum_delta + 1.0e-7)

    def test_exact_obstacle_cohort_has_stable_slot_assignment(self) -> None:
        fixed_catalog = (
            asset("target", large=False),
            asset("large", large=True),
            asset("small-a", large=False),
            asset("small-b", large=False),
        )
        scenes = generate_clutter_scenes(
            fixed_catalog,
            track="sparse",
            split="train",
            scene_count=8,
            seed=31,
            target_asset_ids=("target",),
            config=ClutterGenerationConfig(tasks_per_scene=1),
        )
        slot_assets = tuple(
            tuple(item.asset_id for item in scene.obstacle_objects) for scene in scenes
        )
        self.assertTrue(all(items == slot_assets[0] for items in slot_assets))


if __name__ == "__main__":
    unittest.main()
