from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import youtube_pipeline.cyclic_daily_signals as daily_signals_module
from youtube_pipeline.configuration import resolve_run_config, run_config_from_mapping
from youtube_pipeline.cyclic_daily_signals import (
    CyclicDailySignalConfig,
    run_cyclic_daily_signals,
)
from youtube_pipeline.entrypoints.cyclic_daily_signals import (
    load_legacy_daily_signal_config,
    resolve_cyclic_daily_signal_config,
)
from youtube_pipeline.entrypoints.cyclic_ingestion import (
    LEGACY_INPUT_PATH,
    LEGACY_OUTPUT_DIR,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _cycle(cycle_id: str, index: int, *, start_day: int, end_day: int) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle_id,
        "cycle_index": index,
        "cycle_run_at_local": f"2026-06-{end_day:02d}T00:00:00-0500",
        "cycle_run_at_utc": f"2026-06-{end_day:02d}T05:00:00Z",
        "collection_window_start_local": f"2026-06-{end_day - 1:02d}T00:00:00-0500",
        "collection_window_end_local": f"2026-06-{end_day:02d}T00:00:00-0500",
        "collection_window_start_utc": f"2026-06-{end_day - 1:02d}T05:00:00Z",
        "collection_window_end_utc": f"2026-06-{end_day:02d}T05:00:00Z",
        "analysis_window_start_local": f"2026-06-{start_day:02d}T00:00:00-0500",
        "analysis_window_end_local": f"2026-06-{end_day:02d}T00:00:00-0500",
        "analysis_window_start_utc": f"2026-06-{start_day:02d}T05:00:00Z",
        "analysis_window_end_utc": f"2026-06-{end_day:02d}T05:00:00Z",
        "analysis_window_size_days": 3,
        "data_cutoff_local": f"2026-06-{end_day:02d}T00:00:00-0500",
        "data_cutoff_utc": f"2026-06-{end_day:02d}T05:00:00Z",
        "timezone": "America/Bogota",
        "canonical_timezone": "UTC",
        "simulation_mode": "cyclic_ingestion_simulation",
        "rag_mode": "sidecars_only",
        "cycle_status": "dry_run_partitioned",
    }


def _monitoring_input(cycle: dict, active_count: int, new_count: int) -> dict:
    return {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": cycle["cycle_index"],
        "cycle_run_at_utc": cycle["cycle_run_at_utc"],
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "input_kind": "monitoring_prepared_not_executed",
        "active_window_comment_count": active_count,
        "new_comment_count": new_count,
        "exited_window_comment_count": 0,
        "cumulative_comment_count": active_count,
        "active_video_count": 1,
        "active_comment_ids_hash": "hash",
        "new_comment_ids_hash": "hash",
        "inventory_ref": "cycle_window_inventory.csv",
        "run_monitoring": False,
    }


def _window_row(
    *,
    cycle: dict,
    comment_id: str,
    event_time_utc: str,
    active: bool = True,
    new: bool = True,
) -> dict:
    return {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": cycle["cycle_index"],
        "comment_id": comment_id,
        "video_id": "video_a",
        "event_time_utc": event_time_utc,
        "first_seen_cycle_id": cycle["cycle_id"] if new else "cyc_1",
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "window_membership_role": "active_new"
        if active and new
        else "active_existing"
        if active
        else "exited_window",
        "is_new_in_cycle": new,
        "is_active_in_window": active,
        "is_accumulated_by_cycle": True,
        "exited_window": not active,
    }


class CyclicDailySignalsTests(unittest.TestCase):
    def _write_artifacts(self, simulation_dir: Path, *, future_leak: bool = False) -> Path:
        simulation_dir.mkdir(parents=True, exist_ok=True)
        cycles = [
            _cycle("cyc_1", 1, start_day=8, end_day=11),
            _cycle("cyc_2", 2, start_day=9, end_day=12),
        ]
        _write_json(
            simulation_dir / "cycle_adapter_manifest.json",
            {
                "simulation_run_id": "sim_test",
                "simulation_mode": "cyclic_ingestion_simulation",
                "adapter_stage": "C-3",
                "adapter_mode": "stateful",
                "interval_policy": "semi_open_daily_bounds_start_inclusive_end_exclusive",
                "execution_guards": {
                    "run_monitoring": False,
                    "run_detection": False,
                    "run_rag": False,
                },
            },
        )
        _write_jsonl(simulation_dir / "cycle_manifest.jsonl", cycles)
        _write_jsonl(
            simulation_dir / "cycle_monitoring_inputs.jsonl",
            [
                _monitoring_input(cycles[0], active_count=2, new_count=2),
                _monitoring_input(cycles[1], active_count=3, new_count=2),
            ],
        )
        c1_time = "2026-06-08T12:00:00Z"
        c2_time = "2026-06-10T12:00:00Z"
        c3_time = "2026-06-11T12:00:00Z"
        c4_time = "2026-06-11T13:00:00Z"
        if future_leak:
            c4_time = cycles[1]["data_cutoff_utc"]
        rows = [
            _window_row(cycle=cycles[0], comment_id="c1", event_time_utc=c1_time),
            _window_row(cycle=cycles[0], comment_id="c2", event_time_utc=c2_time),
            _window_row(cycle=cycles[1], comment_id="c1", event_time_utc=c1_time, active=False, new=False),
            _window_row(cycle=cycles[1], comment_id="c2", event_time_utc=c2_time, new=False),
            _window_row(cycle=cycles[1], comment_id="c3", event_time_utc=c3_time),
            _window_row(cycle=cycles[1], comment_id="c4", event_time_utc=c4_time),
        ]
        pd.DataFrame(rows).to_csv(simulation_dir / "cycle_window_inventory.csv", index=False)
        return simulation_dir

    def _write_gold(self, path: Path, *, minimal: bool = False, future_leak: bool = False) -> None:
        c4_time = "2026-06-11T13:00:00Z"
        if future_leak:
            c4_time = "2026-06-12T05:00:00Z"
        rows = [
            {
                "comment_id": "c1",
                "video_id": "video_a",
                "event_time_utc": "2026-06-08T12:00:00Z",
                "text": "uno",
            },
            {
                "comment_id": "c2",
                "video_id": "video_a",
                "event_time_utc": "2026-06-10T12:00:00Z",
                "text": "dos",
            },
            {
                "comment_id": "c3",
                "video_id": "video_a",
                "event_time_utc": "2026-06-11T12:00:00Z",
                "text": "tres",
            },
            {
                "comment_id": "c4",
                "video_id": "video_a",
                "event_time_utc": c4_time,
                "text": "cuatro",
            },
        ]
        if not minimal:
            for i, row in enumerate(rows):
                row.update(
                    {
                        "author_id": f"a{i}",
                        "is_reply": i % 2 == 0,
                        "emoji_count": i,
                        "exclamation_count": i + 1,
                        "question_count": 0,
                        "caps_ratio": 0.1 * i,
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_signals_dry_run_builds_one_observation_per_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulation_dir = self._write_artifacts(base / "cyclic")
            gold_path = base / "gold.csv"
            self._write_gold(gold_path)

            summary = run_cyclic_daily_signals(
                CyclicDailySignalConfig(
                    simulation_dir=simulation_dir,
                    canonical_dataset_path=gold_path,
                )
            )

            self.assertEqual(summary["mode"], "signals_dry_run")
            self.assertEqual(summary["processed_cycle_count"], 2)
            self.assertEqual(summary["xiao_execution_status"], "not_executed")
            self.assertFalse((simulation_dir / "cycle_xiao_state.json").exists())
            self.assertFalse((simulation_dir / "cycle_daily_events.jsonl").exists())
            signal_rows = [
                json.loads(line)
                for line in (simulation_dir / "cycle_signal_series.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(signal_rows), 2)
            self.assertEqual(signal_rows[0]["active_window_comment_count"], 2)
            self.assertEqual(signal_rows[1]["active_window_comment_count"], 3)
            self.assertIsNone(signal_rows[0]["delta_active_window_comment_count"])
            self.assertEqual(signal_rows[1]["delta_active_window_comment_count"], 1)
            self.assertEqual(signal_rows[1]["exited_window_comment_count"], 1)
            self.assertEqual(signal_rows[0]["reply_count"], 1)
            xiao_inputs = [
                json.loads(line)
                for line in (simulation_dir / "cycle_xiao_inputs.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(xiao_inputs[0]["xiao_signal_name"], "active_window_comment_count")
            self.assertEqual(xiao_inputs[1]["xiao_signal_value"], 3)

    def test_optional_unavailable_signals_do_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulation_dir = self._write_artifacts(base / "cyclic")
            gold_path = base / "gold.csv"
            self._write_gold(gold_path, minimal=True)

            summary = run_cyclic_daily_signals(
                CyclicDailySignalConfig(
                    simulation_dir=simulation_dir,
                    canonical_dataset_path=gold_path,
                )
            )

            self.assertIn("unique_author_count", summary["unavailable_signal_names"])
            quality = [
                json.loads(line)
                for line in (simulation_dir / "cycle_signal_quality_report.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                all(
                    row["quality_status"] == "passed_with_unavailable_optional_signals"
                    for row in quality
                )
            )

    def test_signals_dry_run_rejects_future_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulation_dir = self._write_artifacts(base / "cyclic", future_leak=True)
            gold_path = base / "gold.csv"
            self._write_gold(gold_path, future_leak=True)

            with self.assertRaisesRegex(ValueError, "join/temporal validation failed"):
                run_cyclic_daily_signals(
                    CyclicDailySignalConfig(
                        simulation_dir=simulation_dir,
                        canonical_dataset_path=gold_path,
                    )
                )

    def test_signals_dry_run_rejects_execution_flags(self) -> None:
        for flag in [
            "run_xiao",
            "run_detection",
            "run_rag",
            "run_llm",
            "run_serper",
            "use_embeddings",
            "use_vectorstore",
        ]:
            with self.assertRaises(ValueError):
                CyclicDailySignalConfig(
                    simulation_dir="outputs/cyclic",
                    canonical_dataset_path="prepared/comments.parquet",
                    **{flag: True},
                ).validate_c5_scope()

    def test_common_resolver_matches_legacy_signal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            common_dir = self._write_artifacts(base / "common")
            legacy_dir = self._write_artifacts(base / "legacy")
            gold_path = base / "gold.csv"
            self._write_gold(gold_path)
            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_daily_signals"},
                    "signals": {
                        "daily": {
                            "simulation_dir": "common",
                            "canonical_dataset_path": "gold.csv",
                        }
                    },
                }
            )
            common_config = resolve_run_config(run, base_dir=base).config.signals.daily
            legacy_config = load_legacy_daily_signal_config(
                None,
                overrides={
                    "simulation_dir": legacy_dir,
                    "canonical_dataset_path": gold_path,
                },
            )

            common_summary = run_cyclic_daily_signals(common_config)
            legacy_summary = run_cyclic_daily_signals(legacy_config)

            comparable_keys = {
                "simulation_run_id",
                "mode",
                "processed_cycle_count",
                "failed_quality_count",
                "xiao_execution_status",
                "rag_execution_status",
                "xiao_signal_name",
                "unavailable_signal_names",
            }
            self.assertEqual(
                {key: common_summary[key] for key in comparable_keys},
                {key: legacy_summary[key] for key in comparable_keys},
            )
            for artifact in (
                "cycle_signal_series.jsonl",
                "cycle_signal_quality_report.jsonl",
                "cycle_xiao_inputs.jsonl",
            ):
                with self.subTest(artifact=artifact):
                    self.assertEqual(
                        (common_dir / artifact).read_text(encoding="utf-8"),
                        (legacy_dir / artifact).read_text(encoding="utf-8"),
                    )

    def test_paths_and_argparse_are_outside_signal_domain(self) -> None:
        with self.assertRaises(TypeError):
            CyclicDailySignalConfig()
        source = inspect.getsource(daily_signals_module)
        self.assertNotIn(LEGACY_INPUT_PATH, source)
        self.assertNotIn(LEGACY_OUTPUT_DIR, source)
        self.assertNotIn("import argparse", source)

        legacy = load_legacy_daily_signal_config(None)
        self.assertEqual(legacy.simulation_dir, LEGACY_OUTPUT_DIR)
        self.assertEqual(legacy.canonical_dataset_path, LEGACY_INPUT_PATH)

    def test_current_profile_resolves_daily_signal_component(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = resolve_cyclic_daily_signal_config(
            config_file=repository_root / "configs/compatibility/cyclic_current.json",
            base_dir=repository_root,
        )

        self.assertEqual(
            config.simulation_dir,
            (repository_root / LEGACY_OUTPUT_DIR).resolve(strict=False),
        )
        self.assertEqual(
            config.canonical_dataset_path,
            (repository_root / LEGACY_INPUT_PATH).resolve(strict=False),
        )
        self.assertEqual(config.xiao_signal_name, "active_window_comment_count")


if __name__ == "__main__":
    unittest.main()
