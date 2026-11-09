from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CyclicPipelineScriptTests(unittest.TestCase):
    def test_script_exposes_only_the_small_common_cli(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "run_cyclic_pipeline.py"),
                "--help",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for argument in (
            "--config",
            "--run-id",
            "--output-root",
            "--dry-run",
            "--execute",
            "--log-level",
        ):
            self.assertIn(argument, result.stdout)
        self.assertNotIn("--v-min", result.stdout)
        self.assertNotIn("--baseline-window-size-cycles", result.stdout)


if __name__ == "__main__":
    unittest.main()
