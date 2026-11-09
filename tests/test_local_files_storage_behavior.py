from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

import youtube_pipeline.run_pipeline as legacy_pipeline_module
import youtube_pipeline.storage as storage_module


class LocalFilesStorageBehaviorTests(unittest.TestCase):
    def test_storage_reads_local_tables_and_preserves_persistence_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            videos_path = base / "videos.csv"
            comments_path = base / "comments.parquet"
            data_root = base / "prepared"
            videos = pd.DataFrame(
                [{"video_id": "v1", "publishedAt": "2026-01-01T00:00:00Z"}]
            )
            comments = pd.DataFrame(
                [
                    {
                        "comment_id": "c1",
                        "video_id": "v1",
                        "published_at": "2026-01-02T00:00:00Z",
                    }
                ]
            )
            videos.to_csv(videos_path, index=False)
            comments.to_parquet(comments_path, index=False)
            expected = {
                "videos_rows": 1,
                "comments_rows": 1,
                "videos_jsonl": "prepared/bronze/videos.jsonl",
                "comments_jsonl": "prepared/bronze/comments.jsonl",
                "videos_parquet": "prepared/silver/videos",
                "comments_parquet": "prepared/silver/comments",
            }

            with patch.object(
                storage_module,
                "persist_batch_snapshot",
                return_value=expected,
            ) as persist:
                result = legacy_pipeline_module.run_storage(
                    str(videos_path),
                    str(comments_path),
                    str(data_root),
                )

        self.assertEqual(result, expected)
        persisted_videos = persist.call_args.args[0]
        persisted_comments = persist.call_args.args[1]
        assert_frame_equal(persisted_videos, videos)
        assert_frame_equal(persisted_comments, comments)
        self.assertEqual(persist.call_args.kwargs, {"data_root": str(data_root)})


if __name__ == "__main__":
    unittest.main()
