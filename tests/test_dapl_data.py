from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from dapl.data import DAPLDataPaths


class DAPLDataPathsTest(unittest.TestCase):
    def test_resolves_public_dataset_layout(self) -> None:
        paths = DAPLDataPaths.resolve("/tmp/dapl-fixture")
        asset = paths.asset("000074a334c541878360457c672b6c2e")

        self.assertEqual(
            asset.flattened_usd,
            Path("/tmp/dapl-fixture/flattened_usds/000074a334c541878360457c672b6c2e/"
                 "_000074a334c541878360457c672b6c2e.usd"),
        )
        self.assertEqual(
            asset.collision_mesh.name,
            "000074a334c541878360457c672b6c2e_geometry.obj",
        )
        self.assertEqual(paths.hand_points.name, "hand_merged.npy")

    def test_environment_resolution_has_no_machine_specific_fallback(self) -> None:
        root = DAPLDataPaths.resolve(environ={"DAPL_DATA_ROOT": "/tmp/from-env"})
        self.assertEqual(root.root, Path("/tmp/from-env"))
        with self.assertRaisesRegex(ValueError, "DAPL_DATA_ROOT"):
            DAPLDataPaths.resolve(environ={})

    def test_rejects_path_traversal_asset_id(self) -> None:
        paths = DAPLDataPaths.resolve("/tmp/dapl-fixture")
        with self.assertRaises(ValueError):
            paths.asset("../asset")

    def test_require_asset_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = DAPLDataPaths.resolve(directory)
            with self.assertRaisesRegex(FileNotFoundError, "asset-a"):
                paths.require_asset("asset-a")


if __name__ == "__main__":
    unittest.main()
