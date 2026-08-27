from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from dapl.domino import (
    DominoAffordanceAnnotation,
    DominoDataPaths,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
    parse_domino_asset_id,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def matrix_at(x: float) -> list[list[float]]:
    matrix = [row[:] for row in IDENTITY]
    matrix[0][3] = x
    return matrix


class DominoAffordanceTest(unittest.TestCase):
    def make_fixture(self, directory: str) -> DominoDataPaths:
        root = Path(directory)
        model = root / "assets" / "objects" / "099_fixture"
        (model / "collision").mkdir(parents=True)
        (model / "visual").mkdir()
        (model / "collision" / "base0.glb").touch()
        (model / "visual" / "base0.glb").touch()
        annotation = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0.5, 0.0, 0.0],
            "extents": [1.0, 0.1, 0.1],
            "contact_points_pose": [matrix_at(0.0)],
            "functional_matrix": [matrix_at(1.0)],
            "contact_points_discription": ["handle"],
            "functional_point_discription": ["head"],
        }
        (model / "model_data0.json").write_text(json.dumps(annotation))
        return DominoDataPaths.resolve(root, root / "usd")

    def test_asset_id_rejects_path_traversal(self) -> None:
        self.assertEqual(parse_domino_asset_id("020_hammer:0"), ("020_hammer", 0))
        for invalid in ("../020_hammer:0", "020_hammer/0", "hammer:0", "020_hammer:-1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_domino_asset_id(invalid)

    def test_paths_and_historical_description_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            asset = paths.require_source_asset("099_fixture:0")
            annotation = load_domino_affordance_annotation(asset)
            self.assertEqual(annotation.contact_anchors[0].description, "handle")
            self.assertEqual(annotation.functional_anchors[0].description, "head")
            self.assertEqual(asset.usd, Path(directory) / "usd/099_fixture/base0.usd")
            with self.assertRaisesRegex(FileNotFoundError, "prepare_domino"):
                paths.require_sim_asset("099_fixture:0")

    def test_sparse_anchors_become_aligned_safe_and_protected_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            annotation = load_domino_affordance_annotation(
                paths.require_source_asset("099_fixture:0")
            )
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
            )
            features = domino_point_affordance_features(
                points,
                annotation,
                safe_radius_m=0.1,
                protected_radius_m=0.1,
            )
            self.assertEqual(features.shape, (3, 2))
            self.assertGreater(features[0, 0].item(), 0.99)
            self.assertLess(features[0, 1].item(), 1.0e-5)
            self.assertGreater(features[2, 1].item(), 0.99)
            self.assertLess(features[2, 0].item(), 1.0e-5)

    def test_hammer_uses_complete_canonical_part_masks(self) -> None:
        annotation = DominoAffordanceAnnotation(
            asset_id="020_hammer:0",
            scale=(0.079, 0.079, 0.079),
            center=(0.0, 0.0, 0.0),
            extents=(0.4, 2.26, 1.73),
            contact_anchors=(),
            functional_anchors=(),
        )
        # Main handle, transition band, central head, and claw respectively.
        points = torch.tensor(
            [
                [0.0, -0.80, 0.00],
                [0.0, 0.20, 0.00],
                [0.0, 0.50, 0.00],
                [0.0, 0.70, 0.00],
                [0.0, 0.50, -0.30],
            ]
        )
        features = domino_point_affordance_features(points, annotation)
        expected = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        torch.testing.assert_close(features, expected)


if __name__ == "__main__":
    unittest.main()
