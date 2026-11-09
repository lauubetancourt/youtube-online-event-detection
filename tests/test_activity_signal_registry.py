from __future__ import annotations

import unittest

import pandas as pd

from youtube_pipeline.activity_signals import (
    ACTIVITY_SIGNAL_REGISTRY,
    COMMENT_COUNT_SIGNAL_ID,
    UNIQUE_AUTHOR_COUNT_SIGNAL_ID,
    ActivitySignalDefinition,
    EventWindowActivitySignal,
    EventWindowCommentCountSignal,
    create_activity_signal,
    event_window_comment_count_definition,
    event_window_unique_author_count_definition,
    get_activity_signal_ids,
)


class ActivitySignalRegistryTests(unittest.TestCase):
    def test_registry_exposes_only_the_two_supported_activity_signals(self) -> None:
        self.assertEqual(
            get_activity_signal_ids(),
            tuple(sorted((COMMENT_COUNT_SIGNAL_ID, UNIQUE_AUTHOR_COUNT_SIGNAL_ID))),
        )
        self.assertEqual(set(ACTIVITY_SIGNAL_REGISTRY), set(get_activity_signal_ids()))
        self.assertTrue(
            all(
                callable(factory)
                for factory in ACTIVITY_SIGNAL_REGISTRY.values()
            )
        )

    def test_comment_count_id_creates_the_compatibility_producer(self) -> None:
        definition = event_window_comment_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        )

        signal = create_activity_signal(definition)

        self.assertIsInstance(signal, EventWindowCommentCountSignal)
        self.assertIs(signal.definition, definition)

    def test_unique_author_id_creates_the_shared_event_window_producer(self) -> None:
        definition = event_window_unique_author_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        )

        signal = create_activity_signal(definition)
        observation = signal.on_event(
            {
                "comment_id": "c1",
                "author_id": "author-a",
                "event_time_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            }
        )[0]

        self.assertIsInstance(signal, EventWindowActivitySignal)
        self.assertNotIsInstance(signal, EventWindowCommentCountSignal)
        self.assertIs(signal.definition, definition)
        self.assertEqual(observation.value, 1)
        self.assertEqual(observation.support_count, 1)
        self.assertEqual(observation.quality, "passed")

    def test_factory_rejects_an_unknown_signal_id_explicitly(self) -> None:
        definition = ActivitySignalDefinition(
            signal_id="unknown_activity_signal",
            metric="unknown",
            source="prepared_comments",
            scope="selected_comment_stream",
            unit="unknown",
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
            timezone="UTC",
            interval_policy="closed",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown activity signal 'unknown_activity_signal'",
        ):
            create_activity_signal(definition)

    def test_registered_family_preserves_configured_window_and_cadence(self) -> None:
        definition = event_window_comment_count_definition(
            window="180s",
            cadence="60s",
            time_basis="event_time_utc",
        )

        signal = create_activity_signal(definition)

        self.assertEqual(
            signal.definition.signal_id,
            "comment_count_event_window_180s_step_60s",
        )
        self.assertEqual(signal.window, pd.Timedelta(seconds=180))
        self.assertEqual(signal.cadence, pd.Timedelta(seconds=60))

    def test_factory_creates_fresh_state_for_each_execution(self) -> None:
        definition = event_window_comment_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        )
        first = create_activity_signal(definition)
        second = create_activity_signal(definition)
        event = {
            "comment_id": "c1",
            "event_time_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
        }

        first_observation = first.on_event(event)[0]
        second_observation = second.on_event(event)[0]

        self.assertIsNot(first, second)
        self.assertEqual(first_observation.value, 1)
        self.assertEqual(second_observation.value, 1)


if __name__ == "__main__":
    unittest.main()
