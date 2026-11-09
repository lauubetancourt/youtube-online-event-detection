from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from youtube_pipeline.run_pipeline import run_clean


class CleaningPipelineBehaviorTests(unittest.TestCase):
    def test_cleaning_preserves_normalization_and_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_path = base / "silver_comments.parquet"
            output_path = base / "gold" / "clean_comments.parquet"
            pd.DataFrame(
                [
                    {
                        "comment_id": "c1",
                        "video_id": "v1",
                        "author_id": "a1",
                        "text": "HOLA mundo feliz!!!",
                        "event_time_utc": "2026-01-02T03:04:05Z",
                        "is_reply": False,
                        "reply_to_comment_id": None,
                    }
                ]
            ).to_parquet(input_path, index=False)

            result = run_clean(
                input_path=str(input_path),
                output_path=str(output_path),
                raw_text_col="text",
                timestamp_col="published_at",
                keep_spam=True,
            )
            cleaned = pd.read_parquet(output_path)

        self.assertEqual(result, output_path)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.loc[0, "comment_id"], "c1")
        self.assertEqual(
            cleaned.loc[0, "text_clean"],
            "hola mundo feliz exclaim_token",
        )
        self.assertEqual(int(cleaned.loc[0, "exclamation_count"]), 3)
        self.assertEqual(int(cleaned.loc[0, "event_time_unix_s"]), 1767323045)
        self.assertEqual(
            int(cleaned.loc[0, "event_time_unix_ms"]),
            int(cleaned.loc[0, "event_time_unix_s"]),
        )
        self.assertFalse(bool(cleaned.loc[0, "is_probable_spam"]))


if __name__ == "__main__":
    unittest.main()
