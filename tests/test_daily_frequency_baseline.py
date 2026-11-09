from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.configuration import resolve_run_config, run_config_from_mapping
from youtube_pipeline.daily_frequency_baseline import (
    CONFIGURED_COOLDOWN_POLICY,
    DEFAULT_COOLDOWN_POLICY,
    DailyFrequencyBaselineConfig,
    run_daily_frequency_baseline,
)
from youtube_pipeline.entrypoints.daily_frequency_baseline import (
    load_legacy_baseline_config,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _signal_row(index: int, value: int, *, cycle_id: str | None = None) -> dict:
    cid = cycle_id or f"cyc_{index}"
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cid,
        "cycle_index": index,
        "signal_date": f"2026-06-{index:02d}",
        "observation_time_utc": f"2026-06-{index + 1:02d}T05:00:00Z",
        "analysis_window_start_utc": f"2026-06-{max(1, index - 2):02d}T05:00:00Z",
        "analysis_window_end_utc": f"2026-06-{index + 1:02d}T05:00:00Z",
        "data_cutoff_utc": f"2026-06-{index + 1:02d}T05:00:00Z",
        "new_comment_count": value,
        "active_window_comment_count": value + 100,
        "delta_active_window_comment_count": None if index == 1 else 100,
        "pct_change_active_window_comment_count": None if index == 1 else 0.5,
        "active_video_count": 1,
        "comment_ids_hash": f"hash_{cid}",
        "join_status": "passed",
        "temporal_status": "passed",
        "schema_status": "passed",
    }


class DailyFrequencyBaselineTests(unittest.TestCase):
    def _write_artifacts(self, simulation_dir: Path, values: list[int]) -> None:
        simulation_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            simulation_dir / "cycle_signal_manifest.json",
            {
                "simulation_run_id": "sim_test",
                "stage": "C-5",
                "mode": "signals_dry_run",
                "execution_guards": {
                    "run_detection": False,
                    "run_rag": False,
                },
            },
        )
        rows = [_signal_row(index, value) for index, value in enumerate(values, start=1)]
        _write_jsonl(simulation_dir / "cycle_signal_series.jsonl", rows)
        _write_jsonl(
            simulation_dir / "cycle_signal_quality_report.jsonl",
            [
                {
                    "cycle_id": row["cycle_id"],
                    "cycle_index": row["cycle_index"],
                    "quality_status": "passed",
                }
                for row in rows
            ],
        )

    def test_detects_increase_over_baseline_after_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir, [100, 100, 300])

            summary = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    baseline_window_size_cycles=2,
                    warmup_cycles=2,
                    k_multiplier=2.0,
                    min_count=100,
                    min_delta=50,
                    min_pct_change=0.5,
                )
            )

            self.assertEqual(summary["events_detected"], 1)
            events = [
                json.loads(line)
                for line in (simulation_dir / "cycle_daily_frequency_events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(events[0]["cycle_id"], "cyc_3")
            self.assertEqual(events[0]["baseline_window_cycle_ids"], ["cyc_1", "cyc_2"])
            self.assertEqual(events[0]["signal_value"], 300.0)

    def test_warmup_prevents_detection_before_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir, [1000, 2000])

            summary = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    baseline_window_size_cycles=3,
                    warmup_cycles=3,
                    min_count=100,
                    min_delta=1,
                    min_pct_change=0.1,
                )
            )

            self.assertEqual(summary["events_detected"], 0)
            self.assertEqual(summary["warmup_cycles"], 2)

    def test_no_detection_when_support_is_below_min_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir, [10, 10, 100])

            summary = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    baseline_window_size_cycles=2,
                    warmup_cycles=2,
                    k_multiplier=2.0,
                    min_count=500,
                    min_delta=1,
                    min_pct_change=0.1,
                )
            )

            self.assertEqual(summary["events_detected"], 0)

    def test_no_detection_when_delta_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir, [100, 1000, 300])

            summary = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    baseline_window_size_cycles=2,
                    warmup_cycles=2,
                    k_multiplier=0.1,
                    min_count=100,
                    min_delta=1,
                    min_pct_change=0.0,
                )
            )

            self.assertEqual(summary["events_detected"], 0)

    def test_cooldown_blocks_adjacent_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir, [100, 100, 300, 1000])

            summary = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    baseline_window_size_cycles=2,
                    warmup_cycles=2,
                    k_multiplier=1.1,
                    min_count=100,
                    min_delta=50,
                    min_pct_change=0.1,
                    cooldown_cycles=1,
                )
            )

            self.assertEqual(summary["events_detected"], 1)
            scores = [
                json.loads(line)
                for line in (simulation_dir / "cycle_daily_frequency_scores.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(scores[3]["cooldown_status"], "active")
            self.assertFalse(scores[3]["trigger_candidate"])

    def test_pct_change_previous_zero_is_undefined_and_blocks_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir, [0, 1000])

            summary = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    baseline_window_size_cycles=1,
                    warmup_cycles=1,
                    k_multiplier=0.1,
                    min_count=100,
                    min_delta=1,
                    min_pct_change=0.1,
                    use_pct_change=True,
                )
            )

            self.assertEqual(summary["events_detected"], 0)
            scores = [
                json.loads(line)
                for line in (simulation_dir / "cycle_daily_frequency_scores.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(scores[1]["pct_change_status"], "undefined_previous_zero")
            self.assertFalse(scores[1]["condition_results"]["min_pct_change"])

    def test_rejects_external_execution_flags(self) -> None:
        for flag in [
            "run_rag",
            "run_llm",
            "run_serper",
            "use_embeddings",
            "use_vectorstore",
        ]:
            with self.assertRaises(ValueError):
                DailyFrequencyBaselineConfig(
                    simulation_dir="outputs/cyclic",
                    **{flag: True},
                ).validate()

    def test_defaults_disable_daily_cooldown(self) -> None:
        config = DailyFrequencyBaselineConfig(simulation_dir="outputs/cyclic")

        self.assertEqual(config.cooldown_cycles, 0)
        self.assertEqual(config.cooldown_policy, DEFAULT_COOLDOWN_POLICY)

    def test_positive_cooldown_is_still_supported_as_optional_config(self) -> None:
        config = DailyFrequencyBaselineConfig(
            simulation_dir="outputs/cyclic",
            cooldown_cycles=1,
        )

        self.assertEqual(config.cooldown_cycles, 1)
        self.assertEqual(config.cooldown_policy, CONFIGURED_COOLDOWN_POLICY)

    def test_common_resolver_matches_legacy_baseline_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            common_dir = base / "common"
            legacy_dir = base / "legacy"
            self._write_artifacts(common_dir, [100, 100, 300])
            self._write_artifacts(legacy_dir, [100, 100, 300])
            parameters = {
                "baseline_window_size_cycles": 2,
                "warmup_cycles": 2,
                "k_multiplier": 2.0,
                "min_count": 100,
                "min_delta": 50,
                "min_pct_change": 0.5,
            }
            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_daily_baseline"},
                    "detection": {
                        "daily_frequency": {
                            "simulation_dir": "common",
                            **parameters,
                        }
                    },
                }
            )
            common_config = resolve_run_config(
                run,
                base_dir=base,
            ).config.detection.daily_frequency
            legacy_config = load_legacy_baseline_config(
                None,
                overrides={"simulation_dir": legacy_dir, **parameters},
            )

            common_summary = run_daily_frequency_baseline(common_config)
            legacy_summary = run_daily_frequency_baseline(legacy_config)

            for key in (
                "detector_name",
                "cycles_evaluated",
                "warmup_cycles",
                "evaluable_cycles",
                "cooldown_cycles",
                "cooldown_policy",
                "events_detected",
            ):
                with self.subTest(key=key):
                    self.assertEqual(common_summary[key], legacy_summary[key])
            self.assertEqual(
                (common_dir / "cycle_daily_frequency_scores.jsonl").read_text(
                    encoding="utf-8"
                ),
                (legacy_dir / "cycle_daily_frequency_scores.jsonl").read_text(
                    encoding="utf-8"
                ),
            )


if __name__ == "__main__":
    unittest.main()
