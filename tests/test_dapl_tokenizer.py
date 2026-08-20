from __future__ import annotations

import unittest

import torch

from dapl.models import (
    DAPLSemanticPatchTokenizer,
    DAPLSemanticPatchTokenizerConfig,
    farthest_point_indices,
)
from dapl.representation import SceneComponent


class DAPLSemanticPatchTokenizerTest(unittest.TestCase):
    def test_paper_shape_and_semantic_index_ranges(self) -> None:
        torch.manual_seed(7)
        scene = torch.randn(2, 1280, 7, requires_grad=True)
        tokenizer = DAPLSemanticPatchTokenizer()

        result = tokenizer(scene)

        self.assertEqual(result.tokens.shape, (2, 40, 128))
        self.assertEqual(result.centers.shape, (2, 40, 7))
        self.assertEqual(result.center_indices.shape, (2, 40))
        self.assertEqual(result.neighbor_indices.shape, (2, 40, 32))
        self.assertEqual(
            result.component_ids.tolist(),
            [SceneComponent.TARGET] * 16
            + [SceneComponent.OBSTACLE] * 16
            + [SceneComponent.END_EFFECTOR] * 8,
        )
        self.assertTrue(torch.all(result.neighbor_indices[:, :16] < 512))
        self.assertTrue(
            torch.all(
                (result.neighbor_indices[:, 16:32] >= 512)
                & (result.neighbor_indices[:, 16:32] < 1024)
            )
        )
        self.assertTrue(torch.all(result.neighbor_indices[:, 32:] >= 1024))

        result.tokens.square().mean().backward()
        self.assertIsNotNone(scene.grad)
        self.assertGreater(torch.count_nonzero(scene.grad).item(), 0)

    def test_grouping_is_deterministic_and_centers_are_neighbors(self) -> None:
        xyz = torch.stack(
            (torch.arange(64, dtype=torch.float32), torch.zeros(64), torch.zeros(64)),
            dim=-1,
        ).unsqueeze(0)
        first = farthest_point_indices(xyz, 8)
        second = farthest_point_indices(xyz, 8)
        torch.testing.assert_close(first, second)
        self.assertEqual(first[0, :3].tolist(), [0, 63, 31])

        torch.manual_seed(11)
        scene = torch.randn(1, 1280, 7)
        output = DAPLSemanticPatchTokenizer()(scene)
        self.assertTrue(
            torch.all(output.neighbor_indices[:, :, 0] == output.center_indices)
        )

    def test_config_rejects_invalid_neighborhood(self) -> None:
        with self.assertRaisesRegex(ValueError, "neighbors"):
            DAPLSemanticPatchTokenizerConfig(neighbors=300)


if __name__ == "__main__":
    unittest.main()
