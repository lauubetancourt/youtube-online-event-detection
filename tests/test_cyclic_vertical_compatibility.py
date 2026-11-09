from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.cyclic_daily_signals import (
    CyclicDailySignalConfig,
    run_cyclic_daily_signals,
)
from youtube_pipeline.cyclic_ingestion import (
    CyclicIngestionConfig,
    build_cyclic_ingestion_dry_run,
)
from youtube_pipeline.cyclic_orchestration import (
    CyclicOrchestratorConfig,
    run_cyclic_orchestrator_dry_run,
)
from youtube_pipeline.cyclic_stateful_adapter import (
    CyclicStatefulAdapterConfig,
    run_cyclic_stateful_adapter,
)
from youtube_pipeline.daily_frequency_baseline import (
    DailyFrequencyBaselineConfig,
    run_daily_frequency_baseline,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class CyclicVerticalCompatibilityTests(unittest.TestCase):
    def test_synthetic_cyclic_flow_preserves_cycles_signals_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_path = base / "comments.parquet"
            simulation_dir = base / "cyclic"
            comments = []
            for day, count in [(1, 2), (3, 6)]:
                for index in range(count):
                    comments.append(
                        {
                            "comment_id": f"c{day}_{index}",
                            "video_id": "video_1",
                            "event_time_utc": (
                                f"2026-06-{day:02d}T12:{index:02d}:00Z"
                            ),
                            "text": f"synthetic comment {day}-{index}",
                        }
                    )
            pd.DataFrame(comments).to_parquet(dataset_path, index=False)

            ingestion = build_cyclic_ingestion_dry_run(
                CyclicIngestionConfig(
                    input_path=dataset_path,
                    output_dir=simulation_dir,
                    collection_start_date_local="2026-06-01",
                    collection_end_date_local="2026-06-03",
                    analysis_window_size_days=2,
                    simulation_run_id="sim_vertical",
                )
            )
            orchestration = run_cyclic_orchestrator_dry_run(
                CyclicOrchestratorConfig(simulation_dir=simulation_dir)
            )
            adapter = run_cyclic_stateful_adapter(
                CyclicStatefulAdapterConfig(simulation_dir=simulation_dir)
            )
            signals = run_cyclic_daily_signals(
                CyclicDailySignalConfig(
                    simulation_dir=simulation_dir,
                    canonical_dataset_path=dataset_path,
                )
            )
            detection = run_daily_frequency_baseline(
                DailyFrequencyBaselineConfig(
                    simulation_dir=simulation_dir,
                    signal_name="active_window_comment_count",
                    baseline_window_size_cycles=2,
                    warmup_cycles=2,
                    k_multiplier=2.0,
                    min_count=5,
                    min_delta=3,
                    min_pct_change=1.0,
                )
            )

            self.assertEqual(ingestion["cycles_total"], 3)
            self.assertEqual(ingestion["future_leak_count"], 0)
            self.assertEqual(orchestration["orchestration_status"], "completed_dry_run")
            self.assertEqual(orchestration["completed_dry_run_cycle_count"], 3)
            self.assertEqual(adapter["adapter_status"], "prepared")
            self.assertEqual(adapter["readiness_failed"], 0)
            self.assertEqual(signals["processed_cycle_count"], 3)
            self.assertEqual(signals["failed_quality_count"], 0)
            self.assertEqual(detection["events_detected"], 1)

            cycles = _read_jsonl(simulation_dir / "cycle_manifest.jsonl")
            self.assertEqual(
                [cycle["cycle_id"] for cycle in cycles],
                [
                    "cyc_164f2e976f86",
                    "cyc_d8fa73c70901",
                    "cyc_0b120338344c",
                ],
            )
            self.assertEqual(
                [cycle["data_cutoff_utc"] for cycle in cycles],
                [
                    "2026-06-02T05:00:00Z",
                    "2026-06-03T05:00:00Z",
                    "2026-06-04T05:00:00Z",
                ],
            )

            signal_rows = _read_jsonl(simulation_dir / "cycle_signal_series.jsonl")
            self.assertEqual(
                [row["active_window_comment_count"] for row in signal_rows],
                [2, 2, 6],
            )
            self.assertEqual(
                [row["delta_active_window_comment_count"] for row in signal_rows],
                [None, 0, 4],
            )

            events = _read_jsonl(
                simulation_dir / "cycle_daily_frequency_events.jsonl"
            )
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["daily_event_id"], "dfe_74c4ee9f2cc1")
            self.assertEqual(events[0]["cycle_id"], "cyc_0b120338344c")
            self.assertEqual(events[0]["signal_value"], 6.0)
            self.assertEqual(events[0]["baseline_mean"], 2.0)
            self.assertEqual(events[0]["delta_value"], 4.0)
            self.assertEqual(events[0]["pct_change_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
