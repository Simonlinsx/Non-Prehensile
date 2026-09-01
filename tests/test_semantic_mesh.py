from __future__ import annotations

import unittest

import torch

from dapl.contact_planner.semantic_mesh import (
    FaceSemantic,
    partition_mesh_faces,
    triangle_semantic_samples,
)
from dapl.domino import DominoAffordanceAnnotation


class SemanticMeshTest(unittest.TestCase):
    @staticmethod
    def hammer_annotation() -> DominoAffordanceAnnotation:
        return DominoAffordanceAnnotation(
            asset_id="020_hammer:0",
            scale=(0.079, 0.079, 0.079),
            center=(0.0, 0.0, 0.0),
            extents=(0.4, 2.26, 1.73),
            contact_anchors=(),
            functional_anchors=(),
        )

    def test_conservative_partition_and_boundary_precedence(self) -> None:
        # Four disconnected triangles: safe, neutral, protected, and one
        # mixed boundary triangle whose protected vertex must dominate.
        vertices = torch.tensor(
            [
                [-0.1, 0.10, 0.00],
                [0.1, 0.10, 0.00],
                [0.0, 0.20, 0.10],
                [-0.1, 0.48, 0.00],
                [0.1, 0.48, 0.00],
                [0.0, 0.54, 0.10],
                [-0.1, 0.70, 0.00],
                [0.1, 0.70, 0.00],
                [0.0, 0.80, 0.10],
                [-0.1, 0.30, 0.00],
                [0.1, 0.30, 0.00],
                [0.0, 0.70, 0.10],
            ],
            dtype=torch.float32,
        )
        faces = torch.arange(12, dtype=torch.long).reshape(4, 3)
        result = partition_mesh_faces(vertices, faces, self.hammer_annotation())
        expected = torch.tensor(
            [
                int(FaceSemantic.SAFE),
                int(FaceSemantic.NEUTRAL),
                int(FaceSemantic.PROTECTED),
                int(FaceSemantic.PROTECTED),
            ]
        )
        torch.testing.assert_close(result.labels, expected)
        self.assertEqual(
            result.counts(), {"neutral": 1, "safe": 1, "protected": 2}
        )
        self.assertEqual(result.samples_per_face, 7)

    def test_samples_include_vertices_edges_and_centroid(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        samples = triangle_semantic_samples(
            vertices, torch.tensor([[0, 1, 2]], dtype=torch.long)
        )
        self.assertEqual(samples.shape, (1, 7, 3))
        torch.testing.assert_close(samples[0, -1], torch.tensor([2 / 3, 2 / 3, 0.0]))
        torch.testing.assert_close(samples[0, 3], torch.tensor([1.0, 0.0, 0.0]))

    def test_invalid_mesh_and_thresholds_fail_closed(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        annotation = self.hammer_annotation()
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            partition_mesh_faces(
                vertices, torch.tensor([[0, 1, 3]]), annotation
            )
        with self.assertRaisesRegex(ValueError, "safe_threshold"):
            partition_mesh_faces(
                vertices,
                torch.tensor([[0, 1, 2]]),
                annotation,
                safe_threshold=1.1,
            )

    def test_small_non_degenerate_triangle_is_not_float32_epsilon_rejected(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0], [0.0, 1.0e-4, 0.0]],
            dtype=torch.float32,
        )
        samples = triangle_semantic_samples(
            vertices, torch.tensor([[0, 1, 2]], dtype=torch.long)
        )
        self.assertEqual(samples.shape, (1, 7, 3))


if __name__ == "__main__":
    unittest.main()
