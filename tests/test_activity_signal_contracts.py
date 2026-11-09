from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone

from youtube_pipeline.activity_signals import (
    ActivityObservation,
    ActivitySignalDefinition,
    CLOSED_INTERVAL,
    LEFT_CLOSED_RIGHT_OPEN_INTERVAL,
    event_window_comment_count_definition,
)


def _xiao_reference_signal() -> ActivitySignalDefinition:
    return ActivitySignalDefinition(
        signal_id="comment_count_event_window_120s_step_30s",
        metric="comment_count",
        source="prepared_comments",
        scope="selected_comment_stream",
        unit="comments",
        window="120s",
        cadence="30s",
        time_basis="event_time_utc",
        timezone="UTC",
        interval_policy=CLOSED_INTERVAL,
    )


def _daily_reference_signal() -> ActivitySignalDefinition:
    return ActivitySignalDefinition(
        signal_id="new_comment_count_local_day_daily",
        metric="unique_comment_count",
        source="prepared_comments",
        scope="simulation_corpus",
        unit="comments",
        window="local_calendar_day",
        cadence="daily",
        time_basis="event_time_utc",
        timezone="America/Bogota",
        interval_policy=LEFT_CLOSED_RIGHT_OPEN_INTERVAL,
    )


class ActivitySignalDefinitionTests(unittest.TestCase):
    def test_reference_signals_have_explicit_distinct_semantics(self) -> None:
        xiao = _xiao_reference_signal()
        daily = _daily_reference_signal()

        self.assertEqual(xiao.window, "120s")
        self.assertEqual(xiao.cadence, "30s")
        self.assertEqual(xiao.interval_policy, CLOSED_INTERVAL)
        self.assertEqual(daily.window, "local_calendar_day")
        self.assertEqual(daily.cadence, "daily")
        self.assertEqual(daily.timezone, "America/Bogota")
        self.assertEqual(
            daily.interval_policy,
            LEFT_CLOSED_RIGHT_OPEN_INTERVAL,
        )

    def test_signal_contract_is_immutable_and_detector_neutral(self) -> None:
        signal = _xiao_reference_signal()
        field_names = {field.name for field in fields(ActivitySignalDefinition)}

        with self.assertRaises(FrozenInstanceError):
            signal.window = "60s"  # type: ignore[misc]
        self.assertTrue(
            {
                "signal_id",
                "metric",
                "source",
                "scope",
                "unit",
                "window",
                "cadence",
                "time_basis",
                "timezone",
                "interval_policy",
            }.issubset(field_names)
        )
        self.assertTrue(
            {"detector", "v_min", "threshold", "warmup", "cooldown"}.isdisjoint(
                field_names
            )
        )

    def test_equivalent_durations_have_one_semantic_signal_id(self) -> None:
        seconds = event_window_comment_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        )
        minutes = event_window_comment_count_definition(
            window="2min",
            cadence="0.5min",
            time_basis="event_time_utc",
        )

        self.assertEqual(
            seconds.signal_id,
            "comment_count_event_window_120s_step_30s",
        )
        self.assertEqual(minutes.signal_id, seconds.signal_id)

    def test_signal_contract_rejects_ambiguous_identity_and_time_semantics(self) -> None:
        with self.assertRaises(ValueError):
            ActivitySignalDefinition(
                signal_id="Comment count",
                metric="comment_count",
                source="prepared_comments",
                scope="selected_comment_stream",
                unit="comments",
                window="120s",
                cadence="30s",
                time_basis="event_time_utc",
                timezone="UTC",
                interval_policy=CLOSED_INTERVAL,
            )
        with self.assertRaises(ValueError):
            ActivitySignalDefinition(
                signal_id="comment_count",
                metric="comment_count",
                source="prepared_comments",
                scope="selected_comment_stream",
                unit="comments",
                window="120s",
                cadence="30s",
                time_basis="event_time_utc",
                timezone="Not/A_Timezone",
                interval_policy=CLOSED_INTERVAL,
            )
        with self.assertRaises(ValueError):
            ActivitySignalDefinition(
                signal_id="comment_count",
                metric="comment_count",
                source="prepared_comments",
                scope="selected_comment_stream",
                unit="comments",
                window="120s",
                cadence="30s",
                time_basis="event_time_utc",
                timezone="UTC",
                interval_policy="unspecified",
            )


class ActivityObservationTests(unittest.TestCase):
    def test_observation_carries_value_support_quality_and_explicit_window(self) -> None:
        observation = ActivityObservation(
            signal=_xiao_reference_signal(),
            observation_time_utc=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
            window_start_utc=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            window_end_utc=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
            value=4,
            support_count=4,
            quality="passed",
        )

        self.assertEqual(observation.signal.signal_id, _xiao_reference_signal().signal_id)
        self.assertEqual(observation.value, 4)
        self.assertEqual(observation.support_count, 4)
        self.assertEqual(observation.quality, "passed")
        with self.assertRaises(FrozenInstanceError):
            observation.value = 5  # type: ignore[misc]

    def test_observation_rejects_future_window_and_non_utc_time(self) -> None:
        signal = _daily_reference_signal()
        with self.assertRaises(ValueError):
            ActivityObservation(
                signal=signal,
                observation_time_utc=datetime(2026, 6, 2, 4, 59, tzinfo=timezone.utc),
                window_start_utc=datetime(2026, 6, 1, 5, 0, tzinfo=timezone.utc),
                window_end_utc=datetime(2026, 6, 2, 5, 0, tzinfo=timezone.utc),
                value=2,
                support_count=2,
            )
        with self.assertRaises(ValueError):
            ActivityObservation(
                signal=signal,
                observation_time_utc=datetime(2026, 6, 2, 5, 0),
                window_start_utc=datetime(2026, 6, 1, 5, 0, tzinfo=timezone.utc),
                window_end_utc=datetime(2026, 6, 2, 5, 0, tzinfo=timezone.utc),
                value=2,
                support_count=2,
            )

    def test_observation_rejects_invalid_value_support_and_window(self) -> None:
        signal = _xiao_reference_signal()
        valid = {
            "signal": signal,
            "observation_time_utc": datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
            "window_start_utc": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            "window_end_utc": datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
            "value": 4,
            "support_count": 4,
        }

        with self.assertRaises(ValueError):
            ActivityObservation(**{**valid, "value": float("nan")})
        with self.assertRaises(ValueError):
            ActivityObservation(**{**valid, "support_count": -1})
        with self.assertRaises(ValueError):
            ActivityObservation(
                **{
                    **valid,
                    "window_start_utc": valid["window_end_utc"],
                }
            )


if __name__ == "__main__":
    unittest.main()
