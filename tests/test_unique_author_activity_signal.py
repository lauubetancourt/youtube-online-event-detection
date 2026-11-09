from __future__ import annotations

import unittest
from functools import partial

import pandas as pd

from youtube_pipeline.activity_signals import (
    EventWindowActivitySignal,
    EventWindowCommentCountSignal,
    event_window_comment_count_definition,
    event_window_unique_author_count_definition,
    unique_author_count_measurement,
)


def _comment_signal() -> EventWindowCommentCountSignal:
    return EventWindowCommentCountSignal(
        definition=event_window_comment_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        ),
        timestamp_column="event_time_utc",
    )


def _author_signal() -> EventWindowActivitySignal:
    return EventWindowActivitySignal(
        definition=event_window_unique_author_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        ),
        timestamp_column="event_time_utc",
        measurement=partial(
            unique_author_count_measurement,
            author_column="author_id",
        ),
    )


def _collect(signal, events: list[dict]):
    observations = []
    for event in events:
        signal.on_event(event, on_observation=observations.append)
    return observations


class UniqueAuthorActivitySignalTests(unittest.TestCase):
    def test_repeated_authors_differ_from_comment_count_on_same_ticks(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        events = [
            {
                "comment_id": "c1",
                "author_id": "A",
                "event_time_utc": start,
            },
            {
                "comment_id": "c2",
                "author_id": "A",
                "event_time_utc": start + pd.Timedelta(seconds=10),
            },
            {
                "comment_id": "c3",
                "author_id": "B",
                "event_time_utc": start + pd.Timedelta(seconds=20),
            },
            {
                "comment_id": "c4",
                "author_id": "C",
                "event_time_utc": start + pd.Timedelta(seconds=40),
            },
            {
                "comment_id": "later",
                "author_id": "outside-tick-60",
                "event_time_utc": start + pd.Timedelta(seconds=61),
            },
        ]

        comments = _collect(_comment_signal(), events)
        authors = _collect(_author_signal(), events)
        comment_at_60 = next(
            item
            for item in comments
            if pd.Timestamp(item.observation_time_utc)
            == start + pd.Timedelta(seconds=60)
        )
        authors_at_60 = next(
            item
            for item in authors
            if pd.Timestamp(item.observation_time_utc)
            == start + pd.Timedelta(seconds=60)
        )

        self.assertEqual(comment_at_60.value, 4)
        self.assertEqual(authors_at_60.value, 3)
        self.assertEqual(comment_at_60.support_count, 4)
        self.assertEqual(authors_at_60.support_count, 4)
        self.assertEqual(
            [item.observation_time_utc for item in comments],
            [item.observation_time_utc for item in authors],
        )
        self.assertEqual(
            authors_at_60.signal.signal_id,
            "unique_author_count_event_window_120s_step_30s",
        )
        self.assertEqual(authors_at_60.signal.metric, "unique_authors")
        self.assertEqual(authors_at_60.signal.unit, "authors/window")

    def test_repeated_author_is_counted_once(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        events = [
            {
                "comment_id": f"c{index}",
                "author_id": author,
                "event_time_utc": start + pd.Timedelta(seconds=seconds),
            }
            for index, (author, seconds) in enumerate(
                [("A", 0), ("A", 10), ("A", 20), ("B", 30)],
                start=1,
            )
        ]

        observations = _collect(_author_signal(), events)
        at_30 = observations[-1]

        self.assertEqual(pd.Timestamp(at_30.observation_time_utc), start + pd.Timedelta(seconds=30))
        self.assertEqual(at_30.value, 2)
        self.assertEqual(at_30.support_count, 4)
        self.assertEqual(at_30.quality, "passed")

    def test_author_outside_closed_window_expires(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        events = [
            {
                "comment_id": "c-a",
                "author_id": "A",
                "event_time_utc": start,
            },
            {
                "comment_id": "c-b",
                "author_id": "B",
                "event_time_utc": start + pd.Timedelta(seconds=30),
            },
            {
                "comment_id": "c-c",
                "author_id": "C",
                "event_time_utc": start + pd.Timedelta(seconds=150),
            },
        ]

        observations = _collect(_author_signal(), events)
        at_120 = next(
            item
            for item in observations
            if pd.Timestamp(item.observation_time_utc)
            == start + pd.Timedelta(seconds=120)
        )
        at_150 = observations[-1]

        self.assertEqual(at_120.value, 2)
        self.assertEqual(at_150.value, 2)
        self.assertEqual(at_150.support_count, 2)

    def test_missing_authors_are_excluded_and_degrade_quality(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        events = [
            {
                "comment_id": "known-a",
                "author_id": "A",
                "event_time_utc": start,
            },
            {
                "comment_id": "missing",
                "author_id": None,
                "event_time_utc": start + pd.Timedelta(seconds=10),
            },
            {
                "comment_id": "blank",
                "author_id": "  ",
                "event_time_utc": start + pd.Timedelta(seconds=20),
            },
            {
                "comment_id": "known-b",
                "author_id": "B",
                "event_time_utc": start + pd.Timedelta(seconds=30),
            },
        ]

        observation = _collect(_author_signal(), events)[-1]

        self.assertEqual(observation.value, 2)
        self.assertEqual(observation.support_count, 4)
        self.assertEqual(observation.quality, "degraded_missing_author_id")

    def test_equal_timestamp_policy_matches_comment_count_signal(self) -> None:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        events = [
            {
                "comment_id": "first",
                "author_id": "A",
                "event_time_utc": start,
            },
            {
                "comment_id": "same-timestamp",
                "author_id": "B",
                "event_time_utc": start,
            },
            {
                "comment_id": "later",
                "author_id": "C",
                "event_time_utc": start + pd.Timedelta(seconds=30),
            },
        ]

        comments = _collect(_comment_signal(), events)
        authors = _collect(_author_signal(), events)

        self.assertEqual([item.value for item in comments], [1, 3])
        self.assertEqual([item.value for item in authors], [1, 3])
        self.assertEqual(
            [item.observation_time_utc for item in comments],
            [item.observation_time_utc for item in authors],
        )


if __name__ == "__main__":
    unittest.main()
