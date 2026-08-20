from __future__ import annotations

import unittest

from dapl.catalog import _rotation_matrix_to_quaternion


class CatalogTest(unittest.TestCase):
    def test_rotation_matrix_conversion_uses_wxyz(self) -> None:
        identity = _rotation_matrix_to_quaternion(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        self.assertEqual(identity, (1.0, 0.0, 0.0, 0.0))

        half_turn_z = _rotation_matrix_to_quaternion(
            ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
        )
        self.assertAlmostEqual(abs(half_turn_z[3]), 1.0)
        self.assertAlmostEqual(half_turn_z[0], 0.0)


if __name__ == "__main__":
    unittest.main()
