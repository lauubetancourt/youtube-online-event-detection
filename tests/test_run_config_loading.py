from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.configuration import load_run_config, run_config_from_mapping


class RunConfigLoadingTests(unittest.TestCase):
    def _write_config(self, directory: Path, payload: dict) -> Path:
        path = directory / "run.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_json_and_keeps_component_defaults_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(
                Path(tmp),
                {
                    "identity": {"run_id": "run_from_json"},
                    "simulation": {
                        "ingestion": {
                            "input_path": "prepared/comments.parquet",
                            "output_dir": "outputs/cyclic",
                        }
                    },
                },
            )

            run = load_run_config(path)

        self.assertEqual(run.identity.run_id, "run_from_json")
        self.assertEqual(
            run.simulation.ingestion.input_path,
            "prepared/comments.parquet",
        )
        self.assertEqual(run.simulation.ingestion.analysis_window_size_days, 3)
        self.assertTrue(run.simulation.ingestion.dry_run)

    def test_applies_explicit_overrides_after_file_values(self) -> None:
        payload = {
            "identity": {"run_id": "run_file"},
            "simulation": {
                "ingestion": {
                    "input_path": "prepared/file.parquet",
                    "output_dir": "outputs/cyclic",
                    "dry_run": True,
                },
                "orchestration": {
                    "simulation_dir": "outputs/file",
                    "update_cycle_state": True,
                },
            },
        }

        run = run_config_from_mapping(
            payload,
            overrides={
                "identity": {"run_id": "run_override"},
                "simulation": {
                    "ingestion": {
                        "input_path": None,
                        "dry_run": False,
                    },
                    "orchestration": {"update_cycle_state": False},
                },
            },
        )

        self.assertEqual(run.identity.run_id, "run_override")
        self.assertEqual(
            run.simulation.ingestion.input_path,
            "prepared/file.parquet",
        )
        self.assertFalse(run.simulation.ingestion.dry_run)
        self.assertFalse(run.simulation.orchestration.update_cycle_state)

    def test_loads_an_execution_with_only_one_applicable_section(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "run_detection"},
                "detection": {
                    "daily_frequency": {
                        "simulation_dir": "outputs/cyclic",
                        "min_count": 25,
                    }
                },
            }
        )

        self.assertIsNone(run.simulation)
        self.assertIsNone(run.signals)
        self.assertEqual(run.detection.daily_frequency.min_count, 25)

    def test_rejects_unknown_keys_at_each_composition_level(self) -> None:
        invalid_payloads = [
            {
                "identity": {"run_id": "run_unknown"},
                "detection": {"daily_frequency": {}},
                "unexpected": {},
            },
            {
                "identity": {"run_id": "run_unknown"},
                "simulation": {"unexpected": {}},
            },
            {
                "identity": {"run_id": "run_unknown"},
                "detection": {"daily_frequency": {"unknown_threshold": 1}},
            },
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    run_config_from_mapping(payload)

    def test_rejects_invalid_json_types_without_coercion(self) -> None:
        with self.assertRaisesRegex(TypeError, "dry_run has invalid type str"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "run_invalid_type"},
                    "simulation": {
                        "ingestion": {
                            "input_path": "prepared/comments.parquet",
                            "output_dir": "outputs/cyclic",
                            "dry_run": "false",
                        }
                    },
                }
            )

        with self.assertRaisesRegex(TypeError, "simulation must be a JSON object"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "run_invalid_section"},
                    "simulation": [],
                }
            )

    def test_reports_missing_directory_and_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_run_config(directory / "missing.json")
            with self.assertRaisesRegex(ValueError, "must be a file"):
                load_run_config(directory)

            malformed = directory / "malformed.json"
            malformed.write_text('{"identity": ', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                load_run_config(malformed)


if __name__ == "__main__":
    unittest.main()
