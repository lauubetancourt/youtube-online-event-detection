from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from youtube_pipeline.configuration import (
    DataConfig,
    RunConfig,
    RunIdentityConfig,
    canonical_run_config_json,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.data_extraction import ExtractionConfig


class YouTubeAcquisitionConfigurationTests(unittest.TestCase):
    def test_run_config_reuses_existing_extraction_config(self) -> None:
        extraction = ExtractionConfig(
            query="synthetic query",
            data_root="outputs/data",
            save_legacy_csv=False,
        )

        run = RunConfig(
            identity=RunIdentityConfig(run_id="youtube_acquisition"),
            data=DataConfig(youtube_api=extraction),
        )

        self.assertIs(run.data.youtube_api, extraction)
        self.assertIsNone(run.simulation)
        self.assertIsNone(run.signals)
        self.assertIsNone(run.detection)
        self.assertIsNone(run.rag)

    def test_loader_is_strict_and_keeps_extraction_defaults_authoritative(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "youtube_from_mapping"},
                "data": {
                    "youtube_api": {
                        "query": "configured query",
                        "data_root": "acquisition_data",
                        "save_legacy_csv": False,
                    }
                },
            }
        )

        extraction = run.data.youtube_api
        self.assertIsInstance(extraction, ExtractionConfig)
        self.assertEqual(extraction.query, "configured query")
        self.assertEqual(extraction.data_root, "acquisition_data")
        self.assertFalse(extraction.save_legacy_csv)
        self.assertEqual(extraction.request_timeout_seconds, 30.0)
        self.assertEqual(extraction.retry_attempts, 3)

        with self.assertRaisesRegex(ValueError, "Unknown data.youtube_api fields"):
            run_config_from_mapping(
                {
                    "identity": {"run_id": "youtube_unknown"},
                    "data": {
                        "youtube_api": {
                            "unknown_acquisition_parameter": True,
                        }
                    },
                }
            )

    def test_resolves_acquisition_paths_without_requiring_them_to_exist(self) -> None:
        run = run_config_from_mapping(
            {
                "identity": {"run_id": "youtube_paths"},
                "data": {
                    "youtube_api": {
                        "data_root": "local/data",
                        "metadata_path": "local/manifests/extraction.json",
                    }
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            resolved = resolve_run_config(run, base_dir=base)

        extraction = resolved.config.data.youtube_api
        self.assertEqual(extraction.data_root, str(base / "local/data"))
        self.assertEqual(
            extraction.metadata_path,
            str(base / "local/manifests/extraction.json"),
        )
        mapping = json.loads(resolved.canonical_json)
        self.assertEqual(mapping["data"]["youtube_api"]["data_root"], "local/data")
        self.assertEqual(
            mapping["data"]["youtube_api"]["metadata_path"],
            "local/manifests/extraction.json",
        )

    def test_composition_does_not_duplicate_extraction_parameters(self) -> None:
        extraction_parameters = {field.name for field in fields(ExtractionConfig)}

        self.assertTrue(
            {field.name for field in fields(DataConfig)}.isdisjoint(
                extraction_parameters
            )
        )
        self.assertTrue(
            {field.name for field in fields(RunConfig)}.isdisjoint(
                extraction_parameters
            )
        )

    def test_absent_data_section_does_not_change_existing_canonical_config(self) -> None:
        from youtube_pipeline.cyclic_ingestion import CyclicIngestionConfig
        from youtube_pipeline.configuration import SimulationConfig

        run = RunConfig(
            identity=RunIdentityConfig(run_id="existing_run"),
            simulation=SimulationConfig(
                ingestion=CyclicIngestionConfig(
                    input_path="prepared/comments.parquet",
                    output_dir="outputs/cyclic",
                )
            ),
        )

        self.assertNotIn('"data"', canonical_run_config_json(run))


if __name__ == "__main__":
    unittest.main()
