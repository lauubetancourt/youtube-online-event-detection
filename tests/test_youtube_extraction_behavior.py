from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import pandas as pd

import youtube_pipeline.data_extraction as extraction_module
from youtube_pipeline.data_extraction import ExtractionConfig, run_extraction_pipeline


class YouTubeExtractionBehaviorTests(unittest.TestCase):
    def test_pipeline_preserves_counts_persistence_and_metadata_contract(self) -> None:
        config = ExtractionConfig(
            query="synthetic query",
            data_root="synthetic_data",
            save_legacy_csv=False,
        )
        logger = logging.getLogger("test.youtube.extraction")
        videos = pd.DataFrame([{"video_id": "v1"}, {"video_id": "v2"}])
        comments = pd.DataFrame(
            [
                {"comment_id": "c1", "video_id": "v1"},
                {"comment_id": "c2", "video_id": "v2"},
                {"comment_id": "c3", "video_id": "v2"},
            ]
        )
        persisted = {
            "videos_jsonl": "synthetic_data/bronze/videos.jsonl",
            "comments_jsonl": "synthetic_data/bronze/comments.jsonl",
        }

        with (
            patch(
                "youtube_pipeline.entrypoints.youtube_extraction."
                "resolve_youtube_api_key",
                return_value="test-api-key",
            ),
            patch.object(extraction_module, "YouTubeClient") as client_type,
            patch.object(extraction_module, "search_videos", return_value=[]),
            patch.object(
                extraction_module,
                "build_videos_dataframe",
                return_value=videos,
            ),
            patch.object(
                extraction_module,
                "enrich_video_data_with_channels",
                return_value=videos,
            ),
            patch.object(extraction_module, "get_all_comments", return_value=[]),
            patch.object(
                extraction_module,
                "build_comments_dataframe",
                return_value=comments,
            ),
            patch.object(
                extraction_module,
                "persist_batch_snapshot",
                return_value=persisted,
            ) as persist,
            patch.object(
                extraction_module,
                "_write_run_metadata",
                return_value="synthetic_data/bronze/runs/run.json",
            ),
        ):
            summary = run_extraction_pipeline(config, logger)

        client_type.assert_called_once_with(
            api_key="test-api-key",
            config=config,
            logger=logger,
        )
        persist.assert_called_once()
        self.assertEqual(summary["videos_found"], 2)
        self.assertEqual(summary["comments_found"], 3)
        self.assertFalse(summary["quota_hit"])
        self.assertIsNone(summary["quota_stage"])
        self.assertEqual(summary["persisted"], persisted)
        self.assertEqual(
            summary["run_metadata"],
            "synthetic_data/bronze/runs/run.json",
        )
        self.assertNotIn("legacy_csv", summary)


if __name__ == "__main__":
    unittest.main()
