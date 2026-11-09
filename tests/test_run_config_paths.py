from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.configuration import (
    DetectionConfig,
    RunConfig,
    RunIdentityConfig,
    SignalsConfig,
    SimulationConfig,
    resolve_run_config_paths,
)
from youtube_pipeline.cyclic_daily_signals import CyclicDailySignalConfig
from youtube_pipeline.cyclic_detection_connector import CyclicDetectionConnectorConfig
from youtube_pipeline.cyclic_ingestion import CyclicIngestionConfig
from youtube_pipeline.cyclic_orchestration import CyclicOrchestratorConfig
from youtube_pipeline.cyclic_stateful_adapter import CyclicStatefulAdapterConfig
from youtube_pipeline.daily_frequency_baseline import DailyFrequencyBaselineConfig


class RunConfigPathTests(unittest.TestCase):
    def test_resolves_known_paths_without_mutating_source_config(self) -> None:
        ingestion = CyclicIngestionConfig(
            input_path="data/comments.parquet",
            output_dir="outputs/cyclic",
        )
        orchestration = CyclicOrchestratorConfig(simulation_dir="outputs/cyclic")
        adapter = CyclicStatefulAdapterConfig(simulation_dir="outputs/cyclic")
        daily = CyclicDailySignalConfig(
            simulation_dir="outputs/cyclic",
            canonical_dataset_path="data/comments.parquet",
            output_dir=None,
        )
        baseline = DailyFrequencyBaselineConfig(
            simulation_dir="outputs/cyclic",
            output_dir="outputs/detection",
        )
        connector = CyclicDetectionConnectorConfig(
            simulation_dir="outputs/cyclic",
            canonical_dataset_path="data/comments.parquet",
        )
        source = RunConfig(
            identity=RunIdentityConfig(run_id="run_paths"),
            simulation=SimulationConfig(
                ingestion=ingestion,
                orchestration=orchestration,
                stateful_adapter=adapter,
            ),
            signals=SignalsConfig(daily=daily),
            detection=DetectionConfig(
                connector=connector,
                daily_frequency=baseline,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            resolved = resolve_run_config_paths(source, base_dir=base)

            self.assertEqual(
                resolved.simulation.ingestion.input_path,
                base / "data/comments.parquet",
            )
            self.assertEqual(
                resolved.simulation.orchestration.simulation_dir,
                base / "outputs/cyclic",
            )
            self.assertEqual(
                resolved.simulation.stateful_adapter.simulation_dir,
                base / "outputs/cyclic",
            )
            self.assertEqual(
                resolved.signals.daily.canonical_dataset_path,
                base / "data/comments.parquet",
            )
            self.assertIsNone(resolved.signals.daily.output_dir)
            self.assertEqual(
                resolved.detection.connector.canonical_dataset_path,
                base / "data/comments.parquet",
            )
            self.assertEqual(
                resolved.detection.daily_frequency.output_dir,
                base / "outputs/detection",
            )

        self.assertEqual(ingestion.input_path, "data/comments.parquet")
        self.assertEqual(orchestration.simulation_dir, "outputs/cyclic")
        self.assertIsNone(daily.output_dir)
        self.assertIsNot(resolved.simulation.ingestion, ingestion)

    def test_keeps_absolute_paths_and_does_not_require_them_to_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            external = base.parent / "not-created" / "comments.parquet"
            source = RunConfig(
                identity=RunIdentityConfig(run_id="run_absolute"),
                simulation=SimulationConfig(
                    ingestion=CyclicIngestionConfig(
                        input_path=external,
                        output_dir=base / "outputs/cyclic",
                    )
                ),
            )

            resolved = resolve_run_config_paths(source, base_dir=base)

        self.assertEqual(resolved.simulation.ingestion.input_path, external)


if __name__ == "__main__":
    unittest.main()
