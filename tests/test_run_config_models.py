from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields

from youtube_pipeline.configuration import (
    DetectionConfig,
    RunConfig,
    RunIdentityConfig,
    SignalsConfig,
    SimulationConfig,
)
from youtube_pipeline.cyclic_daily_signals import CyclicDailySignalConfig
from youtube_pipeline.cyclic_detection_connector import CyclicDetectionConnectorConfig
from youtube_pipeline.cyclic_ingestion import CyclicIngestionConfig
from youtube_pipeline.cyclic_orchestration import CyclicOrchestratorConfig
from youtube_pipeline.cyclic_stateful_adapter import CyclicStatefulAdapterConfig
from youtube_pipeline.daily_frequency_baseline import DailyFrequencyBaselineConfig


class RunConfigModelTests(unittest.TestCase):
    def test_builds_minimum_run_from_existing_component_config(self) -> None:
        ingestion = CyclicIngestionConfig(
            input_path="prepared/comments.parquet",
            output_dir="outputs/cyclic",
            simulation_run_id="sim_minimum",
        )

        run = RunConfig(
            identity=RunIdentityConfig(run_id="run_minimum"),
            simulation=SimulationConfig(ingestion=ingestion),
        )

        self.assertIs(run.simulation.ingestion, ingestion)
        self.assertIsNone(run.signals)
        self.assertIsNone(run.detection)

    def test_composes_existing_cyclic_signal_and_detection_configs(self) -> None:
        ingestion = CyclicIngestionConfig(
            input_path="prepared/comments.parquet",
            output_dir="outputs/cyclic",
            simulation_run_id="sim_composed",
        )
        orchestration = CyclicOrchestratorConfig(simulation_dir="outputs/cyclic")
        adapter = CyclicStatefulAdapterConfig(simulation_dir="outputs/cyclic")
        daily_signal = CyclicDailySignalConfig(
            simulation_dir="outputs/cyclic",
            canonical_dataset_path="prepared/comments.parquet",
        )
        baseline = DailyFrequencyBaselineConfig(
            simulation_dir="outputs/cyclic",
            min_count=7,
        )
        connector = CyclicDetectionConnectorConfig(
            simulation_dir="outputs/cyclic",
            canonical_dataset_path="prepared/comments.parquet",
        )

        run = RunConfig(
            identity=RunIdentityConfig(run_id="run_composed"),
            simulation=SimulationConfig(
                ingestion=ingestion,
                orchestration=orchestration,
                stateful_adapter=adapter,
            ),
            signals=SignalsConfig(daily=daily_signal),
            detection=DetectionConfig(
                connector=connector,
                daily_frequency=baseline,
            ),
        )

        self.assertIs(run.simulation.ingestion, ingestion)
        self.assertIs(run.simulation.orchestration, orchestration)
        self.assertIs(run.simulation.stateful_adapter, adapter)
        self.assertIs(run.signals.daily, daily_signal)
        self.assertIs(run.detection.connector, connector)
        self.assertIs(run.detection.daily_frequency, baseline)
        self.assertEqual(run.detection.daily_frequency.min_count, 7)

    def test_supports_a_run_with_only_one_applicable_section(self) -> None:
        baseline = DailyFrequencyBaselineConfig(
            simulation_dir="outputs/cyclic",
            min_count=5,
        )

        run = RunConfig(
            identity=RunIdentityConfig(run_id="run_detection_only"),
            detection=DetectionConfig(daily_frequency=baseline),
        )

        self.assertIsNone(run.simulation)
        self.assertIsNone(run.signals)
        self.assertIs(run.detection.daily_frequency, baseline)

    def test_new_composition_models_are_immutable(self) -> None:
        identity = RunIdentityConfig(run_id="run_frozen")
        simulation = SimulationConfig(
            ingestion=CyclicIngestionConfig(
                input_path="prepared/comments.parquet",
                output_dir="outputs/cyclic",
            )
        )
        run = RunConfig(identity=identity, simulation=simulation)

        with self.assertRaises(FrozenInstanceError):
            identity.run_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            simulation.ingestion = None
        with self.assertRaises(FrozenInstanceError):
            run.simulation = None

    def test_rejects_empty_or_mistyped_structural_composition(self) -> None:
        identity = RunIdentityConfig(run_id="run_invalid")

        with self.assertRaisesRegex(ValueError, "at least one stage"):
            SimulationConfig()
        with self.assertRaisesRegex(ValueError, "at least one signal"):
            SignalsConfig()
        with self.assertRaisesRegex(ValueError, "at least one detector"):
            DetectionConfig()
        with self.assertRaisesRegex(ValueError, "at least one execution section"):
            RunConfig(identity=identity)
        with self.assertRaisesRegex(TypeError, "CyclicIngestionConfig"):
            SimulationConfig(ingestion="not-a-config")

    def test_composition_does_not_duplicate_component_parameters(self) -> None:
        composition_models = [
            RunConfig,
            SimulationConfig,
            SignalsConfig,
            DetectionConfig,
        ]
        component_parameter_names = {
            "input_path",
            "output_dir",
            "simulation_dir",
            "analysis_window_size_days",
            "xiao_signal_name",
            "baseline_window_size_cycles",
            "k_multiplier",
            "min_count",
            "min_delta",
            "min_pct_change",
            "cooldown_cycles",
            "v_min",
            "sensitivity_threshold",
        }

        for model in composition_models:
            with self.subTest(model=model.__name__):
                model_fields = {field.name for field in fields(model)}
                self.assertTrue(model_fields.isdisjoint(component_parameter_names))


if __name__ == "__main__":
    unittest.main()
