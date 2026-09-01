from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_push_anything_c1_acceptance import combine_acceptance


class PushAnythingAcceptanceTest(unittest.TestCase):
    def write_evidence(self, run_dir: Path, geometry: bool, c1: bool) -> None:
        (run_dir / "acceptance.json").write_text(
            json.dumps(
                {
                    "accepted": geometry,
                    "position_threshold_m": 0.02,
                    "rotation_threshold_rad": 0.1,
                    "dwell_messages_required": 5,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "c1_semantic_audit.json").write_text(
            json.dumps(
                {
                    "c1_pass": c1,
                    "legal_safe_contact_rows": 7,
                    "protected_contact_rows": 0,
                    "neutral_contact_rows": 0,
                    "c1_violation_rows": 0,
                }
            ),
            encoding="utf-8",
        )

    def test_requires_geometry_and_semantic_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.write_evidence(run_dir, geometry=True, c1=True)
            self.assertTrue(combine_acceptance(run_dir)["accepted"])

            self.write_evidence(run_dir, geometry=True, c1=False)
            self.assertFalse(combine_acceptance(run_dir)["accepted"])

    def test_missing_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                combine_acceptance(Path(directory))


if __name__ == "__main__":
    unittest.main()
