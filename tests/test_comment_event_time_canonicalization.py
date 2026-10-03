from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from youtube_pipeline.storage import canonicalize_comment_event_time


class CommentEventTimeCanonicalizationTests(unittest.TestCase):
    def _comments(self, **timestamp_fields: object) -> pd.DataFrame:
        return pd.DataFrame([{
            "comment_id": "comment-1",
            "video_id": "video-1",
            "text": "texto",
            **timestamp_fields,
        }])

    def _assert_published_at_normalizes(
        self,
        value: str,
        expected: str = "2026-04-20T14:00:00Z",
    ) -> None:
        result = canonicalize_comment_event_time(
            self._comments(published_at=value)
        )
        self.assertEqual(result.loc[0, "event_time_utc"], pd.Timestamp(expected))
        self.assertEqual(result.loc[0, "published_at"], value)

    def test_explicit_z_normalizes_to_utc(self) -> None:
        self._assert_published_at_normalizes("2026-04-20T14:00:00Z")

    def test_explicit_zero_offset_normalizes_to_utc(self) -> None:
        self._assert_published_at_normalizes("2026-04-20T14:00:00+00:00")

    def test_nonzero_offset_is_converted_to_utc(self) -> None:
        self._assert_published_at_normalizes("2026-04-20T09:00:00-05:00")

    def test_naive_timestamp_is_rejected_with_row_context(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "comment_id='comment-1'.*field='published_at'.*2026-04-20 09:00:00",
        ):
            canonicalize_comment_event_time(
                self._comments(published_at="2026-04-20 09:00:00")
            )

    def test_invalid_timestamp_is_rejected_with_row_context(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "comment_id='comment-1'.*field='published_at'.*not-a-date",
        ):
            canonicalize_comment_event_time(
                self._comments(published_at="not-a-date")
            )

    def test_existing_canonical_field_is_authoritative_and_idempotent(self) -> None:
        source = self._comments(
            event_time_utc="2026-04-20T14:00:00+00:00",
            published_at="not-a-date",
        )
        first = canonicalize_comment_event_time(source)
        second = canonicalize_comment_event_time(first)
        self.assertEqual(
            first.loc[0, "event_time_utc"],
            pd.Timestamp("2026-04-20T14:00:00Z"),
        )
        self.assertEqual(first.loc[0, "published_at"], "not-a-date")
        assert_series_equal(first["event_time_utc"], second["event_time_utc"])

    def test_missing_both_timestamp_fields_is_explicit(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires 'event_time_utc' or source field 'published_at'",
        ):
            canonicalize_comment_event_time(self._comments())


if __name__ == "__main__":
    unittest.main()
