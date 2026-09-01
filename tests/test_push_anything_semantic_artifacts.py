from __future__ import annotations

import json
from pathlib import Path
import unittest


class PushAnythingSemanticArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.semantic_dir = (
            Path(__file__).resolve().parents[1]
            / "data/push_anything_semantics/020_hammer_0"
        )
        cls.manifest = json.loads(
            (cls.semantic_dir / "semantic_mesh_manifest.json").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def obj_face_count(path: Path) -> int:
        with path.open(encoding="utf-8") as stream:
            return sum(line.startswith("f ") for line in stream)

    def test_partition_is_disjoint_and_complete(self) -> None:
        indices = self.manifest["source_face_indices"]
        safe = set(indices["safe"])
        protected = set(indices["protected"])
        neutral = set(indices["neutral"])
        self.assertFalse(safe & protected)
        self.assertFalse(safe & neutral)
        self.assertFalse(protected & neutral)
        self.assertEqual(
            len(safe | protected | neutral), self.manifest["valid_face_count"]
        )
        self.assertEqual(set(indices["unsafe"]), protected | neutral)
        self.assertLessEqual(set(indices["safe_guarded"]), safe)

    def test_meshes_match_manifest_counts(self) -> None:
        expected = {
            "safe": self.manifest["counts"]["safe"],
            "protected": self.manifest["counts"]["protected"],
            "neutral": self.manifest["counts"]["neutral"],
            "unsafe": len(self.manifest["source_face_indices"]["unsafe"]),
            "safe_guarded": self.manifest["guarded_safe_face_count"],
        }
        for semantic_class, count in expected.items():
            path = self.semantic_dir / self.manifest["meshes"][semantic_class]
            self.assertEqual(self.obj_face_count(path), count)
        self.assertEqual(
            self.obj_face_count(self.semantic_dir / self.manifest["physical_mesh"]),
            self.manifest["valid_face_count"],
        )

    def test_manifest_is_portable_and_fail_closed(self) -> None:
        self.assertFalse(Path(self.manifest["source_collision_mesh"]).is_absolute())
        self.assertGreater(self.manifest["guarded_safe_face_count"], 0)
        self.assertGreater(self.manifest["counts"]["protected"], 0)
        self.assertGreaterEqual(
            self.manifest["guarded_safe_min_sample_distance_m"],
            self.manifest["safe_boundary_clearance_m"],
        )
        self.assertGreaterEqual(
            self.manifest["guarded_safe_min_center_distance_m"],
            self.manifest["sampler_center_clearance_m"],
        )


if __name__ == "__main__":
    unittest.main()
