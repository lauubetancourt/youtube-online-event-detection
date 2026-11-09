from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.cyclic_ingestion import INTERVAL_POLICY
from youtube_pipeline.cyclic_stateful_adapter import (
    CyclicStatefulAdapterConfig,
    run_cyclic_stateful_adapter,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _cycle(cycle_id: str, index: int, start_day: int, active_count: int) -> dict:
    start = f"2026-06-{start_day:02d}"
    end = f"2026-06-{start_day + 1:02d}"
    analysis_start_day = start_day - 2
    analysis_start = f"2026-06-{analysis_start_day:02d}"
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle_id,
        "cycle_index": index,
        "execution_order": index,
        "cycle_run_at_local": f"{end}T00:00:00-0500",
        "cycle_run_at_utc": f"{end}T05:00:00Z",
        "collection_window_start_utc": f"{start}T05:00:00Z",
        "collection_window_end_utc": f"{end}T05:00:00Z",
        "analysis_window_start_utc": f"{analysis_start}T05:00:00Z",
        "analysis_window_end_utc": f"{end}T05:00:00Z",
        "data_cutoff_utc": f"{end}T05:00:00Z",
        "new_comment_count": 1,
        "cumulative_comment_count": index,
        "analysis_comment_count": active_count,
        "future_leak_count": 0,
        "initial_status": "pending",
        "ready_status": "ready",
        "final_status": "completed_dry_run",
        "state_transitions": ["pending", "ready", "completed_dry_run"],
        "run_monitoring": False,
        "run_detection": False,
        "run_rag": False,
        "dry_run_only": True,
        "skip_reason": None,
    }


def _input_row(comment_id: str, cycle_id: str, index: int, day: int) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "source_row_id": index,
        "comment_id": comment_id,
        "video_id": "video_a",
        "event_time_utc": f"2026-06-{day:02d}T12:00:00+00:00",
        "event_time_local": f"2026-06-{day:02d}T07:00:00-05:00",
        "event_date_local": f"2026-06-{day:02d}",
        "assigned_cycle_id": cycle_id,
        "assigned_cycle_index": index,
        "first_seen_cycle_id": cycle_id,
        "first_seen_cycle_index": index,
        "is_new_in_cycle": True,
        "is_duplicate": False,
        "duplicate_occurrence_index": 0,
        "is_late_arrival": False,
        "late_arrival_status": "not_inferable_missing_ingestion_timestamp",
    }


def _processed_row(comment_id: str, cycle: dict, first_seen_cycle_id: str, day: int) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle["cycle_id"],
        "cycle_index": cycle["cycle_index"],
        "comment_id": comment_id,
        "video_id": "video_a",
        "event_time_utc": f"2026-06-{day:02d}T12:00:00+00:00",
        "event_time_local": f"2026-06-{day:02d}T07:00:00-05:00",
        "first_seen_cycle_id": first_seen_cycle_id,
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "included_in_analysis": True,
        "is_duplicate": False,
        "is_late_arrival": False,
    }


class CyclicStatefulAdapterTests(unittest.TestCase):
    def _write_artifacts(self, simulation_dir: Path) -> None:
        simulation_dir.mkdir(parents=True, exist_ok=True)
        cycles = [
            _cycle("cyc_1", 1, 10, 1),
            _cycle("cyc_2", 2, 11, 2),
            _cycle("cyc_3", 3, 12, 3),
            _cycle("cyc_4", 4, 13, 3),
        ]
        _write_json(
            simulation_dir / "online_simulation_manifest.json",
            {
                "simulation_run_id": "sim_test",
                "simulation_mode": "cyclic_ingestion_simulation",
                "interval_policy": INTERVAL_POLICY,
                "temporal_policy": {
                    "no_future_leakage_rule": "event_time_utc < data_cutoff_utc",
                    "collection_window_rule": (
                        "collection_window_start_utc <= event_time_utc < "
                        "collection_window_end_utc"
                    ),
                    "analysis_window_rule": (
                        "analysis_window_start_utc <= event_time_utc < "
                        "analysis_window_end_utc"
                    ),
                    "filtering_uses_utc": True,
                },
            },
        )
        _write_json(
            simulation_dir / "cycle_orchestration_manifest.json",
            {
                "simulation_run_id": "sim_test",
                "orchestration_status": "completed_dry_run",
                "execution_guards": {
                    "run_monitoring": False,
                    "run_detection": False,
                    "run_rag": False,
                },
            },
        )
        _write_jsonl(simulation_dir / "cycle_orchestration_plan.jsonl", cycles)
        _write_json(simulation_dir / "cycle_state.json", {"simulation_run_id": "sim_test"})
        pd.DataFrame(
            [
                _input_row("c1", "cyc_1", 1, 10),
                _input_row("c2", "cyc_2", 2, 11),
                _input_row("c3", "cyc_3", 3, 12),
                _input_row("c4", "cyc_4", 4, 13),
            ]
        ).to_csv(simulation_dir / "cycle_input_inventory.csv", index=False)
        processed = [
            _processed_row("c1", cycles[0], "cyc_1", 10),
            _processed_row("c1", cycles[1], "cyc_1", 10),
            _processed_row("c2", cycles[1], "cyc_2", 11),
            _processed_row("c1", cycles[2], "cyc_1", 10),
            _processed_row("c2", cycles[2], "cyc_2", 11),
            _processed_row("c3", cycles[2], "cyc_3", 12),
            _processed_row("c2", cycles[3], "cyc_2", 11),
            _processed_row("c3", cycles[3], "cyc_3", 12),
            _processed_row("c4", cycles[3], "cyc_4", 13),
        ]
        pd.DataFrame(processed).to_csv(
            simulation_dir / "cycle_processed_inventory.csv",
            index=False,
        )

    def test_stateful_adapter_builds_sliding_windows_and_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir)
            summary = run_cyclic_stateful_adapter(
                CyclicStatefulAdapterConfig(simulation_dir=simulation_dir)
            )

            self.assertEqual(summary["adapter_status"], "prepared")
            self.assertEqual(summary["cycles_total"], 4)
            self.assertEqual(summary["seen_comment_count"], 4)
            self.assertEqual(summary["cycles_with_window_overlap"], 3)
            window = pd.read_csv(simulation_dir / "cycle_window_inventory.csv")
            c1_active = window.loc[
                (window["comment_id"] == "c1")
                & (window["is_active_in_window"].astype(str).str.lower() == "true")
            ]
            self.assertEqual(len(c1_active), 3)
            c1_exit = window.loc[
                (window["comment_id"] == "c1")
                & (window["window_membership_role"] == "exited_window")
            ]
            self.assertEqual(len(c1_exit), 1)
            self.assertEqual(c1_exit.iloc[0]["cycle_id"], "cyc_4")
            context = json.loads((simulation_dir / "cycle_stateful_context.json").read_text())
            self.assertEqual(context["adapter_mode"], "stateful")
            self.assertEqual(context["new_comment_ids_by_cycle"]["cyc_1"], ["c1"])
            self.assertEqual(context["exited_window_comment_ids_by_cycle"]["cyc_4"], ["c1"])

    def test_detection_readiness_has_no_future_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic"
            self._write_artifacts(simulation_dir)
            run_cyclic_stateful_adapter(
                CyclicStatefulAdapterConfig(simulation_dir=simulation_dir)
            )
            rows = [
                json.loads(line)
                for line in (simulation_dir / "cycle_detection_readiness_report.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                all(row["readiness_status"] == "ready_for_future_monitoring_detection" for row in rows)
            )
            self.assertTrue(all(row["checks"]["future_leak_count"] == 0 for row in rows))

    def test_c3_rejects_execution_flags(self) -> None:
        for flag in ["run_monitoring", "run_detection", "run_rag"]:
            with self.assertRaises(ValueError):
                CyclicStatefulAdapterConfig(
                    simulation_dir="outputs/cyclic",
                    **{flag: True},
                ).validate_c3_scope()


if __name__ == "__main__":
    unittest.main()
