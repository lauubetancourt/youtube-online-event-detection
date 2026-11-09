from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from youtube_pipeline.configuration import resolve_run_config, run_config_from_mapping
from youtube_pipeline.daily_rag_sidecars import (
    DailyRagSidecarBuildConfig,
    write_daily_rag_sidecar_artifacts_from_config,
)
from youtube_pipeline.entrypoints.daily_rag_sidecars import (
    resolve_daily_rag_sidecar_config,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _event_row() -> dict:
    return {
        "daily_event_id": "dfe_test",
        "cycle_id": "cyc_2",
        "cycle_index": 2,
        "detector_name": "daily_frequency_baseline",
        "signal_name": "new_comment_count",
        "signal_value": 2.0,
        "baseline_mean": 1.0,
        "ratio_to_baseline": 2.0,
        "delta_value": 1.0,
        "pct_change_value": 1.0,
        "threshold_value": 2.0,
        "trigger_reason": "test trigger",
        "analysis_window_start_utc": "2026-06-01T00:00:00Z",
        "analysis_window_end_utc": "2026-06-04T00:00:00Z",
        "data_cutoff_utc": "2026-06-04T00:00:00Z",
    }


def _comments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comment_id": "c1",
                "video_id": "v1",
                "event_time_utc": "2026-06-01T10:00:00Z",
                "text": "prior context",
                "text_clean": "prior context",
                "is_reply": False,
                "reply_to_comment_id": None,
            },
            {
                "comment_id": "c2",
                "video_id": "v1",
                "event_time_utc": "2026-06-03T10:00:00Z",
                "text": "new alert evidence",
                "text_clean": "new alert evidence",
                "is_reply": False,
                "reply_to_comment_id": None,
            },
            {
                "comment_id": "c3",
                "video_id": "v2",
                "event_time_utc": "2026-06-03T11:00:00Z",
                "text": "reply alert evidence",
                "text_clean": "reply alert evidence",
                "is_reply": True,
                "reply_to_comment_id": "c4",
            },
            {
                "comment_id": "c4",
                "video_id": "v2",
                "event_time_utc": "2026-06-02T09:00:00Z",
                "text": "thread root",
                "text_clean": "thread root",
                "is_reply": False,
                "reply_to_comment_id": None,
            },
        ]
    )


def _window_inventory(*, future_leak: bool = False, missing_gold: bool = False) -> pd.DataFrame:
    event_time = "2026-06-04T00:00:00Z" if future_leak else "2026-06-03T10:00:00Z"
    rows = [
        {
            "simulation_run_id": "sim_test",
            "cycle_id": "cyc_2",
            "cycle_index": 2,
            "comment_id": "c1",
            "video_id": "v1",
            "event_time_utc": "2026-06-01T10:00:00Z",
            "first_seen_cycle_id": "cyc_1",
            "analysis_window_start_utc": "2026-06-01T00:00:00Z",
            "analysis_window_end_utc": "2026-06-04T00:00:00Z",
            "data_cutoff_utc": "2026-06-04T00:00:00Z",
            "is_new_in_cycle": False,
            "is_active_in_window": True,
        },
        {
            "simulation_run_id": "sim_test",
            "cycle_id": "cyc_2",
            "cycle_index": 2,
            "comment_id": "c_missing" if missing_gold else "c2",
            "video_id": "v1",
            "event_time_utc": event_time,
            "first_seen_cycle_id": "cyc_2",
            "analysis_window_start_utc": "2026-06-01T00:00:00Z",
            "analysis_window_end_utc": "2026-06-04T00:00:00Z",
            "data_cutoff_utc": "2026-06-04T00:00:00Z",
            "is_new_in_cycle": True,
            "is_active_in_window": True,
        },
        {
            "simulation_run_id": "sim_test",
            "cycle_id": "cyc_2",
            "cycle_index": 2,
            "comment_id": "c3",
            "video_id": "v2",
            "event_time_utc": "2026-06-03T11:00:00Z",
            "first_seen_cycle_id": "cyc_2",
            "analysis_window_start_utc": "2026-06-01T00:00:00Z",
            "analysis_window_end_utc": "2026-06-04T00:00:00Z",
            "data_cutoff_utc": "2026-06-04T00:00:00Z",
            "is_new_in_cycle": True,
            "is_active_in_window": True,
        },
        {
            "simulation_run_id": "sim_test",
            "cycle_id": "cyc_2",
            "cycle_index": 2,
            "comment_id": "c4",
            "video_id": "v2",
            "event_time_utc": "2026-06-02T09:00:00Z",
            "first_seen_cycle_id": "cyc_1",
            "analysis_window_start_utc": "2026-06-01T00:00:00Z",
            "analysis_window_end_utc": "2026-06-04T00:00:00Z",
            "data_cutoff_utc": "2026-06-04T00:00:00Z",
            "is_new_in_cycle": False,
            "is_active_in_window": True,
        },
    ]
    return pd.DataFrame(rows)


class DailyRagSidecarTests(unittest.TestCase):
    def _write_inputs(
        self,
        root: Path,
        *,
        future_leak: bool = False,
        missing_gold: bool = False,
    ) -> DailyRagSidecarBuildConfig:
        daily_events_path = root / "events.jsonl"
        comments_path = root / "comments.csv"
        window_path = root / "window.csv"
        output_dir = root / "out"
        _write_jsonl(daily_events_path, [_event_row()])
        _comments().to_csv(comments_path, index=False)
        _window_inventory(future_leak=future_leak, missing_gold=missing_gold).to_csv(
            window_path,
            index=False,
        )
        return DailyRagSidecarBuildConfig(
            daily_events_path=str(daily_events_path),
            comments_path=str(comments_path),
            cycle_window_inventory_path=str(window_path),
            output_dir=str(output_dir),
            daily_scores_path=None,
            daily_detector_manifest_path=None,
            cycle_signal_series_path=None,
            cycle_stateful_context_path=None,
            max_comments_per_context_unit=2,
        )

    def test_builds_daily_packages_and_separates_alert_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_inputs(Path(tmp))
            summary = write_daily_rag_sidecar_artifacts_from_config(config)

            self.assertEqual(summary["events_processed"], 1)
            self.assertEqual(summary["alert_evidence_comments"], 2)
            self.assertEqual(summary["validation_context_comments"], 4)
            inventory = pd.read_csv(Path(config.output_dir) / "daily_event_comment_inventory.csv")
            self.assertEqual(int(inventory["is_alert_evidence"].sum()), 2)
            self.assertEqual(int(inventory["is_validation_context"].sum()), 4)
            self.assertIn("alert_evidence", set(inventory["temporal_role"]))
            self.assertIn("validation_context_prior", set(inventory["temporal_role"]))

    def test_context_units_cover_all_inventory_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_inputs(Path(tmp))
            write_daily_rag_sidecar_artifacts_from_config(config)

            inventory = pd.read_csv(Path(config.output_dir) / "daily_event_comment_inventory.csv")
            context_map = pd.read_csv(Path(config.output_dir) / "daily_context_unit_comment_map.csv")
            units = [
                json.loads(line)
                for line in (Path(config.output_dir) / "daily_rag_context_units.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            inv_pairs = set(zip(inventory["daily_event_id"], inventory["comment_id"]))
            ctx_pairs = set(zip(context_map["daily_event_id"], context_map["comment_id"]))
            self.assertEqual(inv_pairs, ctx_pairs)
            self.assertTrue(all("context_role" in unit for unit in units))
            self.assertTrue(all("temporal_scope" in unit for unit in units))
            self.assertTrue(all(unit["context_role"] != "mixed_unit" for unit in units))

    def test_future_leak_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_inputs(Path(tmp), future_leak=True)

            with self.assertRaises(ValueError):
                write_daily_rag_sidecar_artifacts_from_config(config)

    def test_missing_gold_comment_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_inputs(Path(tmp), missing_gold=True)

            with self.assertRaises(ValueError):
                write_daily_rag_sidecar_artifacts_from_config(config)

    def test_rejects_external_execution_flags(self) -> None:
        for flag in [
            "run_llm",
            "run_serper",
            "use_embeddings",
            "use_vectorstore",
            "run_g1",
            "run_g2",
        ]:
            with self.assertRaises(ValueError):
                DailyRagSidecarBuildConfig(
                    daily_events_path="events.jsonl",
                    output_dir="outputs/sidecars",
                    comments_path="comments.parquet",
                    cycle_window_inventory_path="window.csv",
                    **{flag: True},
                ).validate()

    def test_common_resolver_preserves_legacy_sidecar_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inputs = self._write_inputs(base)
            common_output = base / "common"
            legacy_output = base / "legacy"
            component_payload = {
                "daily_events_path": inputs.daily_events_path,
                "output_dir": common_output,
                "comments_path": inputs.comments_path,
                "cycle_window_inventory_path": inputs.cycle_window_inventory_path,
                "daily_scores_path": None,
                "daily_detector_manifest_path": None,
                "cycle_signal_series_path": None,
                "cycle_stateful_context_path": None,
                "max_comments_per_context_unit": 2,
                "run_id": "run_sidecars_compatibility",
            }
            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_sidecars_compatibility"},
                    "rag": {"daily_sidecars": component_payload},
                }
            )
            common = resolve_run_config(
                run,
                base_dir=base,
            ).config.rag.daily_sidecars
            legacy = resolve_daily_rag_sidecar_config(
                config_file=None,
                overrides={
                    **component_payload,
                    "output_dir": legacy_output,
                    "run_id": "run_sidecars_compatibility",
                },
                base_dir=base,
            )

            common_summary = write_daily_rag_sidecar_artifacts_from_config(common)
            legacy_summary = write_daily_rag_sidecar_artifacts_from_config(legacy)

            comparable_keys = {
                "run_id",
                "artifact_version",
                "events_processed",
                "alert_evidence_comments",
                "validation_context_comments",
                "validation_status",
                "future_leak_count",
            }
            self.assertEqual(
                {key: common_summary[key] for key in comparable_keys},
                {key: legacy_summary[key] for key in comparable_keys},
            )
            for artifact in (
                "daily_event_comment_inventory.csv",
                "daily_event_video_map.csv",
                "daily_event_thread_map.csv",
                "daily_context_unit_comment_map.csv",
            ):
                with self.subTest(artifact=artifact):
                    assert_frame_equal(
                        pd.read_csv(common_output / artifact),
                        pd.read_csv(legacy_output / artifact),
                    )


if __name__ == "__main__":
    unittest.main()
