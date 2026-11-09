from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import youtube_pipeline.cyclic_detection_connector as connector_module
import youtube_pipeline.daily_frequency_baseline as baseline_module
from youtube_pipeline.cyclic_detection_connector import (
    CyclicDetectionConnectorConfig,
)
from youtube_pipeline.daily_frequency_baseline import (
    DailyFrequencyBaselineConfig,
)
from youtube_pipeline.entrypoints.cyclic_detection_connector import (
    load_legacy_detection_connector_config,
    resolve_cyclic_detection_config,
    resolve_cyclic_detection_connector_config,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_DIR,
)
from youtube_pipeline.entrypoints.daily_frequency_baseline import (
    load_legacy_baseline_config,
    resolve_daily_frequency_baseline_config,
)


class DetectionConfigurationTests(unittest.TestCase):
    def test_domain_configs_require_paths_and_hide_historical_defaults(self) -> None:
        with self.assertRaises(TypeError):
            CyclicDetectionConnectorConfig()
        with self.assertRaises(TypeError):
            DailyFrequencyBaselineConfig()

        for module in (connector_module, baseline_module):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn(LEGACY_INPUT_PATH, source)
                self.assertNotIn(LEGACY_OUTPUT_DIR, source)
                self.assertNotIn("import argparse", source)

    def test_legacy_defaults_remain_only_in_entrypoint_layer(self) -> None:
        connector = load_legacy_detection_connector_config(None)
        baseline = load_legacy_baseline_config(None)

        self.assertEqual(connector.simulation_dir, LEGACY_OUTPUT_DIR)
        self.assertEqual(connector.canonical_dataset_path, LEGACY_INPUT_PATH)
        self.assertEqual(baseline.simulation_dir, LEGACY_OUTPUT_DIR)
        self.assertEqual(baseline.cooldown_cycles, 0)

    def test_current_profile_resolves_both_detection_components(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        profile = repository_root / "configs/compatibility/cyclic_current.json"
        connector = resolve_cyclic_detection_connector_config(
            config_file=profile,
            base_dir=repository_root,
        )
        baseline = resolve_daily_frequency_baseline_config(
            config_file=profile,
            base_dir=repository_root,
        )

        expected_simulation = (repository_root / LEGACY_OUTPUT_DIR).resolve(strict=False)
        self.assertEqual(connector.simulation_dir, expected_simulation)
        self.assertEqual(
            connector.canonical_dataset_path,
            (repository_root / LEGACY_INPUT_PATH).resolve(strict=False),
        )
        self.assertEqual(connector.mode, "detection_dry_run")
        self.assertEqual(baseline.simulation_dir, expected_simulation)
        self.assertEqual(baseline.cooldown_cycles, 0)
        self.assertEqual(baseline.min_count, 500)

        detection = resolve_cyclic_detection_config(
            config_file=profile,
            base_dir=repository_root,
        )
        self.assertEqual(detection.xiao_ema.window_size, "120s")
        self.assertEqual(detection.xiao_ema.slide_interval, "30s")
        self.assertEqual(detection.xiao_ema.slow_window, "10min")
        self.assertEqual(detection.xiao_ema.sensitivity_threshold, 1.5)
        self.assertEqual(detection.xiao_ema.v_min, 46)
        self.assertEqual(detection.xiao_ema.cooldown, "3min")


if __name__ == "__main__":
    unittest.main()
