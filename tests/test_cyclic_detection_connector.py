from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import pandas as pd

from youtube_pipeline.configuration import resolve_run_config, run_config_from_mapping
from youtube_pipeline.cyclic_detection_connector import (
    CyclicDetectionConnectorConfig,
    run_cyclic_detection_connector,
)
from youtube_pipeline.entrypoints.cyclic_detection_connector import (
    load_legacy_detection_connector_config,
)
from youtube_pipeline.detectors import XiaoEMAConfig, create_detector


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _cycle_input(cycle_id: str, index: int, active_count: int, overlap: int = 0) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle_id,
        "cycle_index": index,
        "cycle_run_at_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "analysis_window_start_utc": f"2026-06-{index + 7:02d}T05:00:00Z",
        "analysis_window_end_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "data_cutoff_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "input_kind": "detection_prepared_not_executed",
        "active_window_comment_count": active_count,
        "new_comment_count": active_count,
        "exited_window_comment_count": 0,
        "cumulative_comment_count": active_count,
        "overlap_with_previous_cycle_count": overlap,
        "active_comment_ids_hash": f"hash_{cycle_id}",
        "overlap_comment_ids_hash": f"overlap_{cycle_id}",
        "decision_state_stub_ref": "cycle_stateful_context.json#decision_state_stub",
        "cooldown_state_stub_ref": "cycle_stateful_context.json#cooldown_state_stub",
        "emitted_event_registry_stub_ref": (
            "cycle_stateful_context.json#emitted_event_registry_stub"
        ),
        "inventory_ref": "cycle_window_inventory.csv",
        "run_detection": False,
    }


def _monitoring_input(cycle_id: str, index: int, active_count: int) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle_id,
        "cycle_index": index,
        "cycle_run_at_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "analysis_window_start_utc": f"2026-06-{index + 7:02d}T05:00:00Z",
        "analysis_window_end_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "data_cutoff_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "input_kind": "monitoring_prepared_not_executed",
        "active_window_comment_count": active_count,
        "new_comment_count": active_count,
        "exited_window_comment_count": 0,
        "cumulative_comment_count": active_count,
        "active_video_count": 1 if active_count else 0,
        "active_comment_ids_hash": f"hash_{cycle_id}",
        "new_comment_ids_hash": f"new_{cycle_id}",
        "inventory_ref": "cycle_window_inventory.csv",
        "run_monitoring": False,
    }


def _window_row(
    *,
    cycle_id: str,
    index: int,
    comment_id: str,
    day: int,
    active: bool = True,
) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle_id,
        "cycle_index": index,
        "comment_id": comment_id,
        "video_id": "video_a",
        "event_time_utc": f"2026-06-{day:02d}T12:00:00+00:00",
        "first_seen_cycle_id": cycle_id,
        "analysis_window_start_utc": f"2026-06-{index + 7:02d}T05:00:00Z",
        "analysis_window_end_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "data_cutoff_utc": f"2026-06-{index + 10:02d}T05:00:00Z",
        "window_membership_role": "active_new" if active else "exited_window",
        "is_new_in_cycle": True,
        "is_active_in_window": active,
        "is_accumulated_by_cycle": True,
        "exited_window": not active,
    }


class CyclicDetectionConnectorTests(unittest.TestCase):
    def _write_artifacts(self, simulation_dir: Path) -> None:
        simulation_dir.mkdir(parents=True, exist_ok=True)
        detection_inputs = [
            _cycle_input("cyc_1", 1, 2),
            _cycle_input("cyc_2", 2, 1, overlap=1),
            _cycle_input("cyc_empty", 3, 0),
        ]
        monitoring_inputs = [
            _monitoring_input("cyc_1", 1, 2),
            _monitoring_input("cyc_2", 2, 1),
            _monitoring_input("cyc_empty", 3, 0),
        ]
        window_rows = [
            _window_row(cycle_id="cyc_1", index=1, comment_id="c1", day=9),
            _window_row(cycle_id="cyc_1", index=1, comment_id="c2", day=9),
            _window_row(cycle_id="cyc_2", index=2, comment_id="c2", day=9),
        ]
        _write_json(
            simulation_dir / "cycle_adapter_manifest.json",
            {
                "simulation_run_id": "sim_test",
                "adapter_stage": "C-3",
                "adapter_mode": "stateful",
                "execution_guards": {
                    "run_monitoring": False,
                    "run_detection": False,
                    "run_rag": False,
                },
            },
        )
        _write_json(
            simulation_dir / "cycle_stateful_context.json",
            {
                "simulation_run_id": "sim_test",
                "adapter_mode": "stateful",
                "seen_comment_count": 2,
            },
        )
        _write_jsonl(simulation_dir / "cycle_detection_inputs.jsonl", detection_inputs)
        _write_jsonl(simulation_dir / "cycle_monitoring_inputs.jsonl", monitoring_inputs)
        pd.DataFrame(window_rows).to_csv(
            simulation_dir / "cycle_window_inventory.csv",
            index=False,
        )

    def _write_gold(self, path: Path, *, duplicate: bool = False) -> None:
        rows = [
            {
                "comment_id": "c1",
                "video_id": "video_a",
                "event_time_utc": "2026-06-09T12:00:00Z",
                "text": "primer comentario",
                "author_id": "a1",
                "emoji_count": 0,
                "exclamation_count": 1,
                "question_count": 0,
            },
            {
                "comment_id": "c2",
                "video_id": "video_a",
                "event_time_utc": "2026-06-09T12:00:00Z",
                "text": "segundo comentario",
                "author_id": "a2",
                "emoji_count": 1,
                "exclamation_count": 0,
                "question_count": 1,
            },
        ]
        if duplicate:
            rows.append(
                {
                    "comment_id": "c2",
                    "video_id": "video_a",
                    "event_time_utc": "2026-06-09T12:01:00Z",
                    "text": "duplicado",
                    "author_id": "a3",
                    "emoji_count": 0,
                    "exclamation_count": 0,
                    "question_count": 0,
                }
            )
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_detection_dry_run_writes_contracts_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir)
            summary = run_cyclic_detection_connector(
                CyclicDetectionConnectorConfig(
                    simulation_dir=simulation_dir,
                    canonical_dataset_path="prepared/comments.parquet",
                    max_cycles=2,
                )
            )

            self.assertEqual(summary["mode"], "detection_dry_run")
            self.assertEqual(summary["processed_cycle_count"], 2)
            self.assertEqual(summary["pending_cycle_count"], 1)
            self.assertEqual(summary["events_detected_count"], 0)
            detection_outputs = [
                json.loads(line)
                for line in (simulation_dir / "cycle_detection_outputs.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(detection_outputs), 2)
            self.assertTrue(all(row["detection_status"] == "prepared_not_executed" for row in detection_outputs))
            self.assertTrue(all(row["run_detection"] is False for row in detection_outputs))
            self.assertNotIn("comment_ids", detection_outputs[0])
            state = json.loads((simulation_dir / "cycle_detector_state.json").read_text())
            self.assertTrue(state["stateful"])
            self.assertEqual(state["pending_cycle_ids"], ["cyc_empty"])
            registry = (simulation_dir / "cycle_event_registry.jsonl").read_text()
            self.assertEqual(registry, "")

    def test_empty_cycle_is_prepared_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir)
            run_cyclic_detection_connector(
                CyclicDetectionConnectorConfig(
                    simulation_dir=simulation_dir,
                    canonical_dataset_path="prepared/comments.parquet",
                    max_cycles=3,
                )
            )
            quality = [
                json.loads(line)
                for line in (simulation_dir / "cycle_detection_quality_report.jsonl")
                .read_text()
                .splitlines()
            ]
            empty = [row for row in quality if row["cycle_id"] == "cyc_empty"][0]
            self.assertEqual(empty["quality_status"], "passed")
            self.assertEqual(empty["checks"]["future_leak_count"], 0)

    def test_c4_rejects_forbidden_execution_flags(self) -> None:
        for flag in ["run_monitoring", "run_detection", "run_rag"]:
            with self.assertRaises(ValueError):
                CyclicDetectionConnectorConfig(
                    simulation_dir="outputs/cyclic",
                    canonical_dataset_path="prepared/comments.parquet",
                    **{flag: True},
                ).validate_c4_scope()

    def test_c4_rejects_debug_full_rows_without_approval(self) -> None:
        with self.assertRaises(ValueError):
            CyclicDetectionConnectorConfig(
                simulation_dir="outputs/cyclic",
                canonical_dataset_path="prepared/comments.parquet",
                mode="detection_smoke_test",
                debug_full_rows=True,
            ).validate_c4_scope()

    def test_detection_smoke_test_joins_gold_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulation_dir = base / "cyclic"
            gold_path = base / "clean_comments.parquet"
            self._write_artifacts(simulation_dir)
            self._write_gold(gold_path)

            summary = run_cyclic_detection_connector(
                CyclicDetectionConnectorConfig(
                    simulation_dir=simulation_dir,
                    mode="detection_smoke_test",
                    max_cycles=2,
                    canonical_dataset_path=gold_path,
                )
            )

            output_dir = simulation_dir / "detection_smoke_test"
            self.assertEqual(summary["mode"], "detection_smoke_test")
            self.assertEqual(summary["processed_cycle_count"], 2)
            self.assertTrue(summary["gold_comment_id_unique"])
            self.assertFalse(summary["full_rows_written"])
            self.assertTrue((output_dir / "cycle_smoke_test_manifest.json").exists())
            self.assertFalse((output_dir / "debug_cycle_materialized_rows.parquet").exists())

            join_reports = [
                json.loads(line)
                for line in (output_dir / "cycle_smoke_test_join_report.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(join_reports), 2)
            self.assertTrue(all(row["join_status"] == "passed" for row in join_reports))
            self.assertTrue(
                all(
                    row["joined_comment_count"] == row["active_window_comment_count"]
                    for row in join_reports
                )
            )
            self.assertTrue(all(row["missing_comment_id_count"] == 0 for row in join_reports))
            self.assertTrue(all(row["extra_joined_comment_count"] == 0 for row in join_reports))
            monitoring = [
                json.loads(line)
                for line in (output_dir / "cycle_monitoring_outputs.jsonl")
                .read_text()
                .splitlines()
            ]
            detection = [
                json.loads(line)
                for line in (output_dir / "cycle_detection_outputs.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                all(row["monitoring_status"] == "executed_smoke_test" for row in monitoring)
            )
            self.assertTrue(
                all(row["detection_status"] == "executed_smoke_test" for row in detection)
            )
            self.assertNotIn("comment_ids", monitoring[0])
            self.assertNotIn("comment_ids", detection[0])

    def test_detection_smoke_test_receives_explicit_xiao_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulation_dir = base / "cyclic"
            gold_path = base / "clean_comments.parquet"
            self._write_artifacts(simulation_dir)
            self._write_gold(gold_path)
            xiao_config = XiaoEMAConfig()

            with patch(
                "youtube_pipeline.cyclic_detection_connector.create_detector",
                wraps=create_detector,
            ) as detector_factory:
                run_cyclic_detection_connector(
                    CyclicDetectionConnectorConfig(
                        simulation_dir=simulation_dir,
                        mode="detection_smoke_test",
                        max_cycles=2,
                        canonical_dataset_path=gold_path,
                    ),
                    xiao_config=xiao_config,
                )

            detector_factory.assert_called_once_with(
                config=xiao_config,
                log_fn=ANY,
            )

    def test_detection_smoke_test_rejects_duplicate_gold_comment_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulation_dir = base / "cyclic"
            gold_path = base / "clean_comments.parquet"
            self._write_artifacts(simulation_dir)
            self._write_gold(gold_path, duplicate=True)

            with self.assertRaisesRegex(ValueError, "comment_id must be unique"):
                run_cyclic_detection_connector(
                    CyclicDetectionConnectorConfig(
                        simulation_dir=simulation_dir,
                        mode="detection_smoke_test",
                        max_cycles=2,
                        canonical_dataset_path=gold_path,
                    )
                )

    def test_detection_smoke_test_rejects_manual_execution_flags(self) -> None:
        for flag in ["run_monitoring", "run_detection", "run_rag"]:
            with self.assertRaises(ValueError):
                CyclicDetectionConnectorConfig(
                    simulation_dir="outputs/cyclic",
                    canonical_dataset_path="prepared/comments.parquet",
                    mode="detection_smoke_test",
                    **{flag: True},
                ).validate_c4_scope()

    def test_common_resolver_matches_legacy_dry_run_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            common_dir = base / "common"
            legacy_dir = base / "legacy"
            self._write_artifacts(common_dir)
            self._write_artifacts(legacy_dir)
            run = run_config_from_mapping(
                {
                    "identity": {"run_id": "run_detection_connector"},
                    "detection": {
                        "connector": {
                            "simulation_dir": "common",
                            "canonical_dataset_path": "prepared/comments.parquet",
                            "max_cycles": 2,
                        }
                    },
                }
            )
            common_config = resolve_run_config(
                run,
                base_dir=base,
            ).config.detection.connector
            legacy_config = load_legacy_detection_connector_config(
                None,
                overrides={
                    "simulation_dir": legacy_dir,
                    "canonical_dataset_path": "prepared/comments.parquet",
                    "max_cycles": 2,
                },
            )

            common_summary = run_cyclic_detection_connector(common_config)
            legacy_summary = run_cyclic_detection_connector(legacy_config)

            comparable_keys = {
                "simulation_run_id",
                "mode",
                "processed_cycle_count",
                "pending_cycle_count",
                "failed_quality_count",
                "events_detected_count",
            }
            self.assertEqual(
                {key: common_summary[key] for key in comparable_keys},
                {key: legacy_summary[key] for key in comparable_keys},
            )
            for artifact in (
                "cycle_monitoring_outputs.jsonl",
                "cycle_detection_outputs.jsonl",
                "cycle_detection_quality_report.jsonl",
                "cycle_event_registry.jsonl",
            ):
                with self.subTest(artifact=artifact):
                    self.assertEqual(
                        (common_dir / artifact).read_text(encoding="utf-8"),
                        (legacy_dir / artifact).read_text(encoding="utf-8"),
                    )


if __name__ == "__main__":
    unittest.main()
