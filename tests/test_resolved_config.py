from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.configuration import (
    DetectionConfig,
    RunConfig,
    RunIdentityConfig,
    SimulationConfig,
    canonical_run_config_json,
    resolve_run_config,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.cyclic_ingestion import CyclicIngestionConfig
from youtube_pipeline.daily_frequency_baseline import DailyFrequencyBaselineConfig


def _config(input_path: str | Path, *, min_count: int = 500) -> RunConfig:
    return RunConfig(
        identity=RunIdentityConfig(run_id="run_resolved"),
        simulation=SimulationConfig(
            ingestion=CyclicIngestionConfig(
                input_path=input_path,
                output_dir="outputs/cyclic",
            )
        ),
        detection=DetectionConfig(
            daily_frequency=DailyFrequencyBaselineConfig(
                simulation_dir="outputs/cyclic",
                min_count=min_count,
            )
        ),
    )


class ResolvedConfigTests(unittest.TestCase):
    def test_canonical_mapping_contains_effective_component_defaults(self) -> None:
        mapping = run_config_to_mapping(_config("data/comments.parquet"))

        ingestion = mapping["simulation"]["ingestion"]
        detection = mapping["detection"]["daily_frequency"]
        self.assertEqual(ingestion["analysis_window_size_days"], 3)
        self.assertTrue(ingestion["dry_run"])
        self.assertEqual(detection["baseline_window_size_cycles"], 3)
        self.assertEqual(detection["min_count"], 500)

    def test_canonical_json_is_deterministic_and_hash_is_sha256(self) -> None:
        config = _config("data/../data/comments.parquet")

        first = canonical_run_config_json(config)
        second = canonical_run_config_json(config)
        digest = run_config_hash(config)

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["identity"]["run_id"], "run_resolved")
        self.assertEqual(
            json.loads(first)["simulation"]["ingestion"]["input_path"],
            "data/comments.parquet",
        )
        self.assertEqual(digest, hashlib.sha256(first.encode("utf-8")).hexdigest())
        self.assertEqual(len(digest), 64)

    def test_resolved_hash_is_independent_of_workspace_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = resolve_run_config(
                _config("data/comments.parquet"),
                base_dir=first_tmp,
            )
            second = resolve_run_config(
                _config("data/comments.parquet"),
                base_dir=second_tmp,
            )

        self.assertNotEqual(
            first.config.simulation.ingestion.input_path,
            second.config.simulation.ingestion.input_path,
        )
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertNotIn(first_tmp, first.canonical_json)
        self.assertNotIn(second_tmp, second.canonical_json)

    def test_methodological_change_changes_config_hash(self) -> None:
        first = run_config_hash(_config("data/comments.parquet", min_count=500))
        second = run_config_hash(_config("data/comments.parquet", min_count=501))

        self.assertNotEqual(first, second)

    def test_absent_rag_section_preserves_pre_rag_config_identity(self) -> None:
        config = _config("data/comments.parquet")

        canonical = canonical_run_config_json(config)

        self.assertNotIn('"rag"', canonical)
        self.assertEqual(
            run_config_hash(config),
            "718c14a4a7a9f896a9208345afe4a1692f4b189dcaf8d2d6fd1313135568f0b8",
        )


if __name__ == "__main__":
    unittest.main()
