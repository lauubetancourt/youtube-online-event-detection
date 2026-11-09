from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CyclicStageScriptTests(unittest.TestCase):
    def test_compatibility_scripts_remain_executable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        scripts = (
            "run_cyclic_ingestion_simulation.py",
            "run_cyclic_ingestion_orchestrator.py",
            "run_cyclic_stateful_adapter.py",
            "run_cyclic_detection_connector.py",
            "run_cyclic_daily_signals.py",
            "run_daily_frequency_baseline.py",
        )

        for script_name in scripts:
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(project_root / "scripts" / script_name),
                        "--help",
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--config-file", result.stdout)


if __name__ == "__main__":
    unittest.main()
