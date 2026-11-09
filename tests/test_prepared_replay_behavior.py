from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.run_pipeline import run_playback


class PreparedReplayBehaviorTests(unittest.TestCase):
    def test_prepared_replay_preserves_snapshot_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_path = base / "prepared_comments.parquet"
            output_path = base / "outputs" / "snapshots.csv"
            pd.DataFrame(
                [
                    {
                        "comment_id": "c3",
                        "author_id": "a2",
                        "video_id": "v2",
                        "event_time_utc": "2026-01-01T00:02:00Z",
                        "text": "third",
                    },
                    {
                        "comment_id": "c1",
                        "author_id": "a1",
                        "video_id": "v1",
                        "event_time_utc": "2026-01-01T00:00:00Z",
                        "text": "first",
                    },
                    {
                        "comment_id": "c2",
                        "author_id": "a1",
                        "video_id": "v1",
                        "event_time_utc": "2026-01-01T00:01:00Z",
                        "text": "second",
                    },
                ]
            ).to_parquet(input_path, index=False)

            result = run_playback(
                input_path=str(input_path),
                output_snapshots=str(output_path),
                window_size="1min",
                speed=120.0,
                max_sleep_seconds=0.0,
            )
            snapshots = pd.read_csv(output_path)

        self.assertEqual(result, output_path)
        self.assertEqual(
            list(snapshots.columns),
            [
                "window_start",
                "window_end",
                "size",
                "activity.volume",
                "activity.unique_authors",
                "activity.unique_videos",
            ],
        )
        self.assertEqual(snapshots["size"].tolist(), [1, 2, 2])
        self.assertEqual(snapshots["activity.volume"].tolist(), [1, 2, 2])
        self.assertEqual(snapshots["activity.unique_authors"].tolist(), [1, 1, 2])
        self.assertEqual(snapshots["activity.unique_videos"].tolist(), [1, 1, 2])
        self.assertEqual(
            snapshots["window_end"].tolist(),
            [
                "2026-01-01 00:00:00+00:00",
                "2026-01-01 00:01:00+00:00",
                "2026-01-01 00:02:00+00:00",
            ],
        )


if __name__ == "__main__":
    unittest.main()
