from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.activity_signals import (
    EventWindowCommentCountSignal,
    event_window_comment_count_definition,
)
from youtube_pipeline.cyclic_ingestion import (
    CyclicIngestionConfig,
    build_cyclic_ingestion_dry_run,
)


class _SignalRecordingProducer(EventWindowCommentCountSignal):
    """Test-only observer of the support retained at each reference tick."""

    def __init__(self) -> None:
        self.support_ids_by_tick: list[list[str]] = []
        super().__init__(
            definition=event_window_comment_count_definition(
                window="120s",
                cadence="30s",
                time_basis="event_time_utc",
            ),
            timestamp_column="event_time_utc",
        )

    def _build_observation(self, tick: pd.Timestamp):
        observation = super()._build_observation(tick)
        self.support_ids_by_tick.append(
            [item["comment_id"] for item in self._buffer]
        )
        return observation


class XiaoReferenceSignalSemanticsTests(unittest.TestCase):
    def test_reference_signal_preserves_ticks_boundaries_and_counts(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        signal = _SignalRecordingProducer()
        observations = []
        events = [
            ("a", 0),
            ("b", 30),
            ("c", 90),
            ("d", 120),
            ("e", 150),
            ("f-same-timestamp", 150),
            ("g", 180),
        ]

        for comment_id, seconds in events:
            signal.on_event(
                {
                    "comment_id": comment_id,
                    "event_time_utc": start + pd.Timedelta(seconds=seconds),
                    "text": comment_id,
                },
                on_observation=observations.append,
            )

        self.assertEqual(
            [pd.Timestamp(item.observation_time_utc) for item in observations],
            [start + pd.Timedelta(seconds=value) for value in (0, 30, 60, 90, 120, 150, 180)],
        )
        self.assertEqual(
            [item.value for item in observations],
            [1, 2, 2, 3, 4, 4, 5],
        )
        self.assertEqual(
            signal.support_ids_by_tick[4],
            ["a", "b", "c", "d"],
        )
        self.assertEqual(
            signal.support_ids_by_tick[5],
            ["b", "c", "d", "e"],
        )
        self.assertNotIn(
            "f-same-timestamp",
            signal.support_ids_by_tick[5],
        )
        self.assertIn(
            "f-same-timestamp",
            signal.support_ids_by_tick[6],
        )

    def test_first_tick_aligns_forward_and_gap_ticks_need_a_later_event(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        signal = _SignalRecordingProducer()
        observations = []

        signal.on_event(
            {
                "comment_id": "first",
                "event_time_utc": start + pd.Timedelta(seconds=5),
                "text": "first",
            },
            on_observation=observations.append,
        )
        self.assertEqual(observations, [])

        signal.on_event(
            {
                "comment_id": "later",
                "event_time_utc": start + pd.Timedelta(seconds=91),
                "text": "later",
            },
            on_observation=observations.append,
        )

        self.assertEqual(
            [pd.Timestamp(item.observation_time_utc) for item in observations],
            [start + pd.Timedelta(seconds=value) for value in (30, 60, 90)],
        )
        self.assertEqual(
            [item.value for item in observations],
            [1, 1, 1],
        )


class DailyReferenceSignalSemanticsTests(unittest.TestCase):
    def test_new_comment_count_uses_bogota_publication_day_half_open_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_path = base / "comments.parquet"
            output_dir = base / "cyclic"
            pd.DataFrame(
                [
                    {
                        "comment_id": "before-start",
                        "video_id": "v1",
                        "event_time_utc": "2026-06-01T04:59:59Z",
                    },
                    {
                        "comment_id": "at-start",
                        "video_id": "v1",
                        "event_time_utc": "2026-06-01T05:00:00Z",
                    },
                    {
                        "comment_id": "inside",
                        "video_id": "v2",
                        "event_time_utc": "2026-06-01T12:00:00Z",
                    },
                    {
                        "comment_id": "inside",
                        "video_id": "v2",
                        "event_time_utc": "2026-06-01T13:00:00Z",
                    },
                    {
                        "comment_id": "at-end",
                        "video_id": "v3",
                        "event_time_utc": "2026-06-02T05:00:00Z",
                    },
                ]
            ).to_parquet(input_path, index=False)

            build_cyclic_ingestion_dry_run(
                CyclicIngestionConfig(
                    input_path=input_path,
                    output_dir=output_dir,
                    timezone="America/Bogota",
                    collection_start_date_local="2026-05-31",
                    collection_end_date_local="2026-06-03",
                    simulation_run_id="sim_activity_signal_semantics",
                )
            )

            cycles = [
                json.loads(line)
                for line in (output_dir / "cycle_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_local_start = {
                row["collection_window_start_local"]: row for row in cycles
            }
            june_first = by_local_start["2026-06-01T00:00:00-0500"]

            self.assertEqual(
                june_first["collection_window_start_utc"],
                "2026-06-01T05:00:00Z",
            )
            self.assertEqual(
                june_first["collection_window_end_utc"],
                "2026-06-02T05:00:00Z",
            )
            self.assertEqual(june_first["new_comment_count"], 2)
            self.assertEqual(june_first["duplicate_row_count"], 1)
            self.assertEqual(june_first["future_leak_count"], 0)
            self.assertEqual(
                by_local_start["2026-05-31T00:00:00-0500"]["new_comment_count"],
                1,
            )
            self.assertEqual(
                by_local_start["2026-06-02T00:00:00-0500"]["new_comment_count"],
                1,
            )
            self.assertEqual(
                by_local_start["2026-06-03T00:00:00-0500"]["new_comment_count"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
