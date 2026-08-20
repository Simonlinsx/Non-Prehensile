from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dapl.embodiment import load_dapl_hand_points


class DAPLEmbodimentTest(unittest.TestCase):
    def test_loads_finite_released_shape_as_float32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hand_merged.npy"
            expected = np.arange(256 * 3, dtype=np.float64).reshape(256, 3)
            np.save(path, expected)

            actual = load_dapl_hand_points(path)

            self.assertEqual(actual.shape, (256, 3))
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, torch.from_numpy(expected).float())

    def test_rejects_wrong_shape_and_non_finite_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hand_merged.npy"
            np.save(path, np.zeros((128, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_dapl_hand_points(path)

            points = np.zeros((256, 3), dtype=np.float32)
            points[0, 0] = np.nan
            np.save(path, points)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_dapl_hand_points(path)


if __name__ == "__main__":
    unittest.main()
