from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.cyclic_ingestion import INTERVAL_POLICY
from youtube_pipeline.cyclic_orchestration import (
    CyclicOrchestratorConfig,
    run_cyclic_orchestrator_dry_run,
    validate_cycle_contracts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _manifest(simulation_run_id: str = "sim_test") -> dict:
    return {
        "simulation_run_id": simulation_run_id,
        "simulation_mode": "cyclic_ingestion_simulation",
        "interval_policy": INTERVAL_POLICY,
        "temporal_policy": {
            "canonical_filter_field": "event_time_utc",
            "canonical_timezone": "UTC",
            "local_cycle_timezone": "America/Bogota",
            "no_future_leakage_rule": "event_time_utc < data_cutoff_utc",
            "collection_window_rule": (
                "collection_window_start_utc <= event_time_utc < "
                "collection_window_end_utc"
            ),
            "analysis_window_rule": (
                "analysis_window_start_utc <= event_time_utc < analysis_window_end_utc"
            ),
            "filtering_uses_utc": True,
        },
    }


def _cycle(
    *,
    cycle_id: str,
    cycle_index: int,
    day: str,
    analysis_count: int = 1,
    future_leak_count: int = 0,
) -> dict:
    return {
        "simulation_run_id": "sim_test",
        "cycle_id": cycle_id,
        "cycle_index": cycle_index,
        "cycle_run_at_local": f"{day}T00:00:00-0500",
        "cycle_run_at_utc": f"{day}T05:00:00Z",
        "collection_window_start_local": "2026-06-01T00:00:00-0500",
        "collection_window_end_local": "2026-06-02T00:00:00-0500",
        "collection_window_start_utc": "2026-06-01T05:00:00Z",
        "collection_window_end_utc": "2026-06-02T05:00:00Z",
        "analysis_window_start_local": "2026-05-30T00:00:00-0500",
        "analysis_window_end_local": "2026-06-02T00:00:00-0500",
        "analysis_window_start_utc": "2026-05-30T05:00:00Z",
        "analysis_window_end_utc": "2026-06-02T05:00:00Z",
        "analysis_window_size_days": 3,
        "data_cutoff_local": "2026-06-02T00:00:00-0500",
        "data_cutoff_utc": "2026-06-02T05:00:00Z",
        "timezone": "America/Bogota",
        "canonical_timezone": "UTC",
        "simulation_mode": "cyclic_ingestion_simulation",
        "rag_mode": "sidecars_only",
        "future_leak_count": future_leak_count,
        "new_comment_count": analysis_count,
        "cumulative_comment_count": analysis_count,
        "analysis_comment_count": analysis_count,
    }


class CyclicOrchestrationTests(unittest.TestCase):
    def _write_artifacts(self, simulation_dir: Path, cycles: list[dict]) -> None:
        simulation_dir.mkdir(parents=True, exist_ok=True)
        _write_json(simulation_dir / "online_simulation_manifest.json", _manifest())
        _write_jsonl(simulation_dir / "cycle_manifest.jsonl", cycles)
        _write_json(
            simulation_dir / "cycle_state.json",
            {
                "simulation_run_id": "sim_test",
                "existing_key": "preserved",
            },
        )

    def test_orchestrator_records_deterministic_dry_run_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic_ingestion_simulation"
            self._write_artifacts(
                simulation_dir,
                [
                    _cycle(cycle_id="cyc_a", cycle_index=1, day="2026-06-02"),
                    _cycle(
                        cycle_id="cyc_b",
                        cycle_index=2,
                        day="2026-06-03",
                        analysis_count=0,
                    ),
                ],
            )

            summary = run_cyclic_orchestrator_dry_run(
                CyclicOrchestratorConfig(simulation_dir=simulation_dir)
            )

            self.assertEqual(summary["orchestration_status"], "completed_dry_run")
            self.assertEqual(summary["cycles_total"], 2)
            self.assertEqual(summary["completed_dry_run_cycle_count"], 1)
            self.assertEqual(summary["skipped_no_comments_cycle_count"], 1)

            plan = [
                json.loads(line)
                for line in (simulation_dir / "cycle_orchestration_plan.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(plan[0]["state_transitions"], ["pending", "ready", "completed_dry_run"])
            self.assertEqual(plan[1]["state_transitions"], ["pending", "skipped_no_comments"])
            self.assertEqual([row["execution_order"] for row in plan], [1, 2])
            self.assertFalse(any(row["run_detection"] for row in plan))
            state = json.loads((simulation_dir / "cycle_state.json").read_text())
            self.assertEqual(state["existing_key"], "preserved")
            self.assertEqual(state["orchestration"]["status"], "completed_dry_run")

    def test_duplicate_cycle_ids_are_rejected(self) -> None:
        cycles = [
            _cycle(cycle_id="cyc_dup", cycle_index=1, day="2026-06-02"),
            _cycle(cycle_id="cyc_dup", cycle_index=2, day="2026-06-03"),
        ]
        errors = validate_cycle_contracts(manifest=_manifest(), cycles=cycles)
        self.assertTrue(any("Duplicate cycle_id" in error for error in errors))

    def test_out_of_order_cycles_are_rejected(self) -> None:
        cycles = [
            _cycle(cycle_id="cyc_b", cycle_index=2, day="2026-06-03"),
            _cycle(cycle_id="cyc_a", cycle_index=1, day="2026-06-02"),
        ]
        errors = validate_cycle_contracts(manifest=_manifest(), cycles=cycles)
        self.assertTrue(any("not ordered" in error for error in errors))

    def test_future_leak_count_blocks_orchestration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "cyclic_ingestion_simulation"
            self._write_artifacts(
                simulation_dir,
                [
                    _cycle(
                        cycle_id="cyc_bad",
                        cycle_index=1,
                        day="2026-06-02",
                        future_leak_count=1,
                    )
                ],
            )
            with self.assertRaises(ValueError):
                run_cyclic_orchestrator_dry_run(
                    CyclicOrchestratorConfig(simulation_dir=simulation_dir)
                )

    def test_c2_rejects_execution_flags(self) -> None:
        for flag in ["run_monitoring", "run_detection", "run_rag"]:
            kwargs = {"simulation_dir": "outputs/cyclic", flag: True}
            with self.assertRaises(ValueError):
                CyclicOrchestratorConfig(**kwargs).validate_c2_scope()


if __name__ == "__main__":
    unittest.main()
