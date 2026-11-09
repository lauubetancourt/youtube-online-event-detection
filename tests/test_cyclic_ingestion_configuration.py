from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import youtube_pipeline.cyclic_ingestion as cyclic_ingestion_module
from youtube_pipeline.configuration import resolve_run_config, run_config_from_mapping
from youtube_pipeline.cyclic_ingestion import (
    CyclicIngestionConfig,
    build_cyclic_ingestion_dry_run,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_DIR,
    load_legacy_cyclic_ingestion_config,
    resolve_cyclic_ingestion_config,
)


class CyclicIngestionConfigurationTests(unittest.TestCase):
    def test_domain_config_requires_paths_and_contains_no_historical_defaults(self) -> None:
        with self.assertRaises(TypeError):
            CyclicIngestionConfig()

        source = inspect.getsource(cyclic_ingestion_module)
        self.assertNotIn(LEGACY_INPUT_PATH, source)
        self.assertNotIn(LEGACY_OUTPUT_DIR, source)
        self.assertNotIn("import argparse", source)

    def test_common_resolver_supplies_only_component_config_to_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset = base / "prepared/comments.parquet"
            output = base / "outputs/cyclic"
            dataset.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "comment_id": "c1",
                        "video_id": "v1",
                        "event_time_utc": "2026-06-01T12:00:00Z",
                    },
                    {
                        "comment_id": "c2",
                        "video_id": "v1",
                        "event_time_utc": "2026-06-02T12:00:00Z",
                    },
                ]
            ).to_parquet(dataset, index=False)
            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_ingestion_resolved"},
                    "simulation": {
                        "ingestion": {
                            "input_path": "prepared/comments.parquet",
                            "output_dir": "outputs/cyclic",
                            "collection_start_date_local": "2026-06-01",
                            "collection_end_date_local": "2026-06-02",
                            "simulation_run_id": "sim_ingestion_resolved",
                        }
                    },
                }
            )
            component = resolve_run_config(run, base_dir=base).config.simulation.ingestion

            summary = build_cyclic_ingestion_dry_run(component)
            legacy_output = base / "outputs/legacy"
            legacy_component = load_legacy_cyclic_ingestion_config(
                None,
                overrides={
                    "input_path": dataset,
                    "output_dir": legacy_output,
                    "collection_start_date_local": "2026-06-01",
                    "collection_end_date_local": "2026-06-02",
                    "simulation_run_id": "sim_ingestion_resolved",
                },
            )
            legacy_summary = build_cyclic_ingestion_dry_run(legacy_component)

            self.assertIsInstance(component, CyclicIngestionConfig)
            self.assertEqual(component.input_path, dataset.resolve(strict=False))
            self.assertEqual(component.output_dir, output.resolve(strict=False))
            self.assertEqual(summary["simulation_run_id"], "sim_ingestion_resolved")
            self.assertEqual(summary["cycles_total"], 2)
            self.assertEqual(summary["unique_comment_count"], 2)
            self.assertEqual(summary["future_leak_count"], 0)
            comparable_keys = {
                "simulation_run_id",
                "cycles_total",
                "input_rows",
                "valid_timestamp_rows",
                "unique_comment_count",
                "duplicate_row_count",
                "invalid_timestamp_count",
                "future_leak_count",
                "total_new_comments",
                "max_analysis_comment_count",
            }
            self.assertEqual(
                {key: summary[key] for key in comparable_keys},
                {key: legacy_summary[key] for key in comparable_keys},
            )
            self.assertEqual(
                (output / "cycle_manifest.jsonl").read_text(encoding="utf-8"),
                (legacy_output / "cycle_manifest.jsonl").read_text(encoding="utf-8"),
            )

    def test_legacy_loader_keeps_old_defaults_outside_domain(self) -> None:
        config = load_legacy_cyclic_ingestion_config(None)

        self.assertEqual(config.input_path, LEGACY_INPUT_PATH)
        self.assertEqual(config.output_dir, LEGACY_OUTPUT_DIR)

    def test_entrypoint_accepts_run_profile_and_explicit_legacy_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            profile = base / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "run_profile"},
                        "simulation": {
                            "ingestion": {
                                "input_path": "profile/input.parquet",
                                "output_dir": "profile/output",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            from_profile = resolve_cyclic_ingestion_config(
                config_file=profile,
                base_dir=base,
            )
            from_legacy = resolve_cyclic_ingestion_config(
                config_file=None,
                overrides={
                    "input_path": "legacy/input.parquet",
                    "output_dir": "legacy/output",
                },
                base_dir=base,
            )

        self.assertEqual(
            from_profile.input_path,
            (base / "profile/input.parquet").resolve(strict=False),
        )
        self.assertEqual(
            from_profile.output_dir,
            (base / "profile/output").resolve(strict=False),
        )
        self.assertEqual(
            from_legacy.input_path,
            (base / "legacy/input.parquet").resolve(strict=False),
        )
        self.assertEqual(
            from_legacy.output_dir,
            (base / "legacy/output").resolve(strict=False),
        )


if __name__ == "__main__":
    unittest.main()
