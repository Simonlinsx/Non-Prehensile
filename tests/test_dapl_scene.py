from __future__ import annotations

import tempfile
import unittest

from dapl.scene import (
    ClutterScene,
    ClutterTrack,
    ManipulationTask,
    SceneObject,
    load_scene_manifest,
    write_scene_manifest,
)


IDENTITY_POSE = (0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0)


def make_sparse_scene() -> ClutterScene:
    objects = tuple(
        SceneObject(
            instance_id=f"object-{index}",
            asset_id=f"asset-{index}",
            pose=IDENTITY_POSE,
            scale=(0.2, 0.2, 0.2),
            mass_kg=0.1 + index,
        )
        for index in range(4)
    )
    tasks = tuple(
        ManipulationTask(
            task_id=f"task-{index}",
            target_instance_id="object-0",
            initial_pose=IDENTITY_POSE,
            goal_pose=(0.15, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0),
        )
        for index in range(16)
    )
    return ClutterScene(
        scene_id="sparse-0000",
        split="train",
        track=ClutterTrack.SPARSE,
        objects=objects,
        tasks=tasks,
    )


class ClutterSceneTest(unittest.TestCase):
    def test_round_trip_jsonl_and_paper_contract(self) -> None:
        scene = make_sparse_scene()
        scene.validate_paper_contract()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/scenes.jsonl"
            write_scene_manifest(path, [scene])
            loaded = list(load_scene_manifest(path))
        self.assertEqual(loaded, [scene])

    def test_track_controls_object_count(self) -> None:
        scene = make_sparse_scene()
        with self.assertRaisesRegex(ValueError, "require 8 objects"):
            ClutterScene(
                scene_id=scene.scene_id,
                split=scene.split,
                track=ClutterTrack.MODERATE,
                objects=scene.objects,
                tasks=scene.tasks,
            )

    def test_unknown_task_target_is_rejected(self) -> None:
        scene = make_sparse_scene()
        bad_task = ManipulationTask(
            task_id="bad",
            target_instance_id="missing",
            initial_pose=IDENTITY_POSE,
            goal_pose=(0.2, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "unknown target"):
            ClutterScene(
                scene_id=scene.scene_id,
                split=scene.split,
                track=scene.track,
                objects=scene.objects,
                tasks=(bad_task,),
            )

    def test_quaternion_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "quaternion"):
            SceneObject(
                instance_id="object",
                asset_id="asset",
                pose=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                mass_kg=1.0,
            )


if __name__ == "__main__":
    unittest.main()
