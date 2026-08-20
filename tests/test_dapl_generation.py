from __future__ import annotations

import unittest

from dapl.generation import (
    ClutterAsset,
    ClutterGenerationConfig,
    StablePose,
    generate_clutter_scenes,
)
from dapl.scene import ClutterTrack


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


if __name__ == "__main__":
    unittest.main()
