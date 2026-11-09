from __future__ import annotations

import json
import unittest
from pathlib import Path

from youtube_pipeline.configuration import load_run_config, resolve_run_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPOSITORY_ROOT / "configs/compatibility/cyclic_current.json"
LEGACY_SIMULATION_DIR = Path(
    "experiments/xiao/media/log_3/cyclic_ingestion_simulation"
)


class CurrentCyclicCompatibilityProfileTests(unittest.TestCase):
    def test_profile_describes_current_cyclic_signal_detection_flow(self) -> None:
        run = load_run_config(PROFILE_PATH)

        self.assertEqual(run.identity.run_id, "compatibility_cyclic_current")
        self.assertIsNotNone(run.simulation)
        self.assertIsNotNone(run.signals)
        self.assertIsNotNone(run.detection)
        self.assertEqual(run.artifacts.run_mode, "development")
        self.assertEqual(run.artifacts.trace_level, "minimal")

        ingestion = run.simulation.ingestion
        self.assertEqual(ingestion.simulation_run_id, "sim_42fc5b0f114b")
        self.assertEqual(ingestion.input_path, "data/gold/clean_comments.parquet")
        self.assertEqual(ingestion.output_dir, str(LEGACY_SIMULATION_DIR))
        self.assertEqual(ingestion.analysis_window_size_days, 3)
        self.assertEqual(ingestion.timezone, "America/Bogota")
        self.assertEqual(ingestion.canonical_timezone, "UTC")
        self.assertEqual(ingestion.cycle_frequency, "daily")
        self.assertTrue(ingestion.dry_run)

        self.assertEqual(
            run.simulation.orchestration.simulation_dir,
            str(LEGACY_SIMULATION_DIR),
        )
        self.assertFalse(run.simulation.orchestration.run_detection)
        self.assertEqual(
            run.simulation.stateful_adapter.simulation_dir,
            str(LEGACY_SIMULATION_DIR),
        )

        signals = run.signals.daily
        self.assertEqual(signals.mode, "signals_dry_run")
        self.assertEqual(signals.xiao_signal_name, "active_window_comment_count")
        self.assertEqual(
            signals.canonical_dataset_path,
            "data/gold/clean_comments.parquet",
        )
        self.assertFalse(signals.run_xiao)
        self.assertFalse(signals.run_detection)
        self.assertFalse(signals.run_rag)

        detector = run.detection.daily_frequency
        connector = run.detection.connector
        xiao = run.detection.xiao_ema
        self.assertEqual(connector.mode, "detection_dry_run")
        self.assertEqual(connector.max_cycles, 5)
        self.assertEqual(
            connector.canonical_dataset_path,
            "data/gold/clean_comments.parquet",
        )
        self.assertFalse(connector.run_detection)
        self.assertFalse(connector.run_rag)
        self.assertEqual(xiao.window_size, "120s")
        self.assertEqual(xiao.slide_interval, "30s")
        self.assertEqual(xiao.slow_window, "10min")
        self.assertEqual(xiao.sensitivity_threshold, 1.5)
        self.assertEqual(xiao.v_min, 46)
        self.assertEqual(xiao.cooldown, "3min")
        self.assertEqual(xiao.warmup_windows, 10)
        self.assertEqual(xiao.extreme_volume_multiplier, 2.0)
        self.assertEqual(detector.signal_name, "new_comment_count")
        self.assertEqual(detector.baseline_window_size_cycles, 3)
        self.assertEqual(detector.k_multiplier, 2.0)
        self.assertEqual(detector.min_count, 500)
        self.assertEqual(detector.min_delta, 250.0)
        self.assertEqual(detector.min_pct_change, 0.5)
        self.assertEqual(detector.warmup_cycles, 3)
        self.assertEqual(detector.cooldown_cycles, 0)
        self.assertEqual(
            detector.cooldown_policy,
            "disabled_for_daily_detection",
        )
        self.assertEqual(
            detector.output_dir,
            str(LEGACY_SIMULATION_DIR / "daily_frequency_baseline_cooldown_0"),
        )

        ingestion.validate()
        run.simulation.orchestration.validate_c2_scope()
        run.simulation.stateful_adapter.validate_c3_scope()
        signals.validate_c5_scope()
        connector.validate_c4_scope()
        detector.validate()

    def test_profile_contains_only_the_first_migration_block(self) -> None:
        payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            set(payload),
            {"identity", "simulation", "signals", "detection", "artifacts"},
        )
        self.assertEqual(
            payload["artifacts"],
            {"run_mode": "development", "trace_level": "minimal"},
        )
        self.assertNotIn("rag", payload)
        self.assertNotIn("acquisition", payload)
        self.assertNotIn("retrospective", payload)
        self.assertNotIn("polarization", payload)

    def test_profile_resolves_from_repository_root_with_stable_identity(self) -> None:
        first = resolve_run_config(
            load_run_config(PROFILE_PATH),
            base_dir=REPOSITORY_ROOT,
        )
        second = resolve_run_config(
            load_run_config(PROFILE_PATH),
            base_dir=REPOSITORY_ROOT,
        )

        self.assertEqual(
            first.config.simulation.ingestion.input_path,
            REPOSITORY_ROOT / "data/gold/clean_comments.parquet",
        )
        self.assertEqual(
            first.config.simulation.ingestion.output_dir,
            REPOSITORY_ROOT / LEGACY_SIMULATION_DIR,
        )
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.config_hash, second.config_hash)


if __name__ == "__main__":
    unittest.main()
