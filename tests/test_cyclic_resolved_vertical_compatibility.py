from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.configuration import load_run_config, resolve_run_config
from youtube_pipeline.cyclic_daily_signals import run_cyclic_daily_signals
from youtube_pipeline.cyclic_detection_connector import (
    run_cyclic_detection_connector,
)
from youtube_pipeline.cyclic_ingestion import build_cyclic_ingestion_dry_run
from youtube_pipeline.cyclic_orchestration import run_cyclic_orchestrator_dry_run
from youtube_pipeline.cyclic_stateful_adapter import run_cyclic_stateful_adapter
from youtube_pipeline.daily_frequency_baseline import run_daily_frequency_baseline


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class CyclicResolvedVerticalCompatibilityTests(unittest.TestCase):
    def test_one_resolved_config_preserves_the_migrated_vertical_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_path = base / "comments.parquet"
            simulation_dir = base / "cyclic"
            config_path = base / "run_config.json"

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

            config_path.write_text(
                json.dumps(
                    {
                        "identity": {"run_id": "run_resolved_vertical"},
                        "simulation": {
                            "ingestion": {
                                "input_path": "comments.parquet",
                                "output_dir": "cyclic",
                                "collection_start_date_local": "2026-06-01",
                                "collection_end_date_local": "2026-06-03",
                                "analysis_window_size_days": 2,
                                "simulation_run_id": "sim_vertical",
                            },
                            "orchestration": {"simulation_dir": "cyclic"},
                            "stateful_adapter": {"simulation_dir": "cyclic"},
                        },
                        "signals": {
                            "daily": {
                                "simulation_dir": "cyclic",
                                "canonical_dataset_path": "comments.parquet",
                            }
                        },
                        "detection": {
                            "connector": {
                                "simulation_dir": "cyclic",
                                "canonical_dataset_path": "comments.parquet",
                                "mode": "detection_dry_run",
                                "max_cycles": 3,
                            },
                            "xiao_ema": {},
                            "daily_frequency": {
                                "simulation_dir": "cyclic",
                                "signal_name": "active_window_comment_count",
                                "baseline_window_size_cycles": 2,
                                "warmup_cycles": 2,
                                "k_multiplier": 2.0,
                                "min_count": 5,
                                "min_delta": 3,
                                "min_pct_change": 1.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_run_config(config_path)
            resolved = resolve_run_config(loaded, base_dir=base)
            run_config = resolved.config

            self.assertEqual(run_config.identity.run_id, "run_resolved_vertical")
            self.assertEqual(len(resolved.config_hash), 64)
            self.assertIsNotNone(run_config.simulation)
            self.assertIsNotNone(run_config.signals)
            self.assertIsNotNone(run_config.detection)

            simulation = run_config.simulation
            signals_config = run_config.signals
            detection_config = run_config.detection
            assert simulation is not None
            assert signals_config is not None
            assert detection_config is not None
            assert simulation.ingestion is not None
            assert simulation.orchestration is not None
            assert simulation.stateful_adapter is not None
            assert signals_config.daily is not None
            assert detection_config.connector is not None
            assert detection_config.daily_frequency is not None

            self.assertEqual(simulation.ingestion.input_path, dataset_path.resolve())
            self.assertEqual(simulation.ingestion.output_dir, simulation_dir.resolve())
            self.assertEqual(
                signals_config.daily.simulation_dir,
                simulation_dir.resolve(),
            )
            self.assertEqual(
                detection_config.connector.simulation_dir,
                simulation_dir.resolve(),
            )

            ingestion = build_cyclic_ingestion_dry_run(simulation.ingestion)
            orchestration = run_cyclic_orchestrator_dry_run(
                simulation.orchestration
            )
            adapter = run_cyclic_stateful_adapter(simulation.stateful_adapter)
            connector = run_cyclic_detection_connector(detection_config.connector)
            signals = run_cyclic_daily_signals(signals_config.daily)
            detection = run_daily_frequency_baseline(
                detection_config.daily_frequency
            )

            self.assertEqual(ingestion["cycles_total"], 3)
            self.assertEqual(ingestion["future_leak_count"], 0)
            self.assertEqual(orchestration["orchestration_status"], "completed_dry_run")
            self.assertEqual(orchestration["completed_dry_run_cycle_count"], 3)
            self.assertEqual(adapter["adapter_status"], "prepared")
            self.assertEqual(adapter["readiness_failed"], 0)
            self.assertEqual(connector["mode"], "detection_dry_run")
            self.assertEqual(connector["processed_cycle_count"], 3)
            self.assertEqual(connector["events_detected_count"], 0)
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
