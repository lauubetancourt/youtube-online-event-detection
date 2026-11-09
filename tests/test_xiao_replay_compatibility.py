from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    dispatch_activity_observation,
)
from youtube_pipeline.activity_signals import EventWindowCommentCountSignal
from youtube_pipeline.detectors import XiaoEMAConfig, XiaoEMATriggerDetector
from youtube_pipeline.replay import replay_events


class _RecordingSource:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


class _RecordingHook:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.finalized_at: pd.Timestamp | None = None

    def __call__(self, event: dict) -> None:
        self.events.append(event)

    def finalize(self, final_ts: pd.Timestamp | None) -> None:
        self.finalized_at = final_ts


class XiaoCompatibilityTests(unittest.TestCase):
    def test_xiao_config_is_the_single_default_authority(self) -> None:
        config = XiaoEMAConfig()
        detector = XiaoEMATriggerDetector(
            config=config,
            log_fn=lambda _message: None,
        )

        self.assertEqual(config.ts_col, "event_time_utc")
        self.assertEqual(config.text_col, "text")
        self.assertEqual(config.window_size, "120s")
        self.assertEqual(config.slide_interval, "30s")
        self.assertEqual(config.slow_window, "10min")
        self.assertEqual(config.sensitivity_threshold, 1.5)
        self.assertEqual(config.v_min, 46)
        self.assertEqual(config.cooldown, "3min")
        self.assertEqual(config.warmup_windows, 10)
        self.assertEqual(config.extreme_volume_multiplier, 2.0)
        self.assertIs(detector.config, config)
        self.assertEqual(detector.fast_span_steps, 4)
        self.assertEqual(detector.slow_span_steps, 20)
        self.assertEqual(detector.v_extreme, 92)
        self.assertEqual(
            detector.signal_definition.signal_id,
            "comment_count_event_window_120s_step_30s",
        )
        self.assertEqual(detector.signal_definition.metric, "comment_count")

    def test_default_xiao_detects_known_burst_without_changing_contract(self) -> None:
        detector = XiaoEMATriggerDetector(log_fn=lambda _message: None)
        start = pd.Timestamp("2026-01-01T00:00:00Z")

        for index in range(11):
            detector.on_event(
                {
                    "event_time_utc": start + pd.Timedelta(seconds=30 * index),
                    "text": f"steady-{index}",
                }
            )
        for index in range(50):
            detector.on_event(
                {
                    "event_time_utc": start + pd.Timedelta(seconds=330),
                    "text": f"burst-{index}",
                }
            )
        detector.on_event(
            {
                "event_time_utc": start + pd.Timedelta(seconds=390),
                "text": "after-trigger",
            }
        )
        detector.finalize(start + pd.Timedelta(minutes=10))

        self.assertEqual(detector.fast_span_steps, 4)
        self.assertEqual(detector.slow_span_steps, 20)
        self.assertEqual(detector.warmup_windows, 10)
        self.assertEqual(detector.windows_processed, 14)
        self.assertEqual(len(detector.completed_triggers), 1)

        trigger = detector.completed_triggers[0]
        self.assertEqual(trigger["trigger_time"], start + pd.Timedelta(minutes=6))
        self.assertEqual(trigger["cooldown_until"], start + pd.Timedelta(minutes=9))
        self.assertEqual(trigger["closed_at"], start + pd.Timedelta(minutes=10))
        self.assertEqual(trigger["volume"], 53)
        self.assertAlmostEqual(trigger["strength"], 2.9614987988866464)
        self.assertEqual(
            trigger["comments"],
            [
                {
                    "event_time_utc": start + pd.Timedelta(seconds=390),
                    "text": "after-trigger",
                }
            ],
        )

    def test_explicit_observation_route_preserves_xiao_decisions(self) -> None:
        event_detector = XiaoEMATriggerDetector(log_fn=lambda _message: None)
        observation_detector = XiaoEMATriggerDetector(log_fn=lambda _message: None)
        signal = EventWindowCommentCountSignal(
            definition=observation_detector.signal_definition,
            timestamp_column=observation_detector.config.ts_col,
        )
        route = ActivityDetectionRouteConfig(
            signal_id=signal.definition.signal_id,
            detector_id="xiao_ema",
        )
        detection_results = []
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        events = [
            {
                "event_time_utc": start + pd.Timedelta(seconds=30 * index),
                "text": f"steady-{index}",
            }
            for index in range(11)
        ]
        events.extend(
            {
                "event_time_utc": start + pd.Timedelta(seconds=330),
                "text": f"burst-{index}",
            }
            for index in range(50)
        )
        events.append(
            {
                "event_time_utc": start + pd.Timedelta(seconds=390),
                "text": "after-trigger",
            }
        )

        for event in events:
            event_detector.on_event(event)
            signal.on_event(
                event,
                on_observation=lambda observation: detection_results.append(
                    dispatch_activity_observation(
                        route=route,
                        observation=observation,
                        detectors={"xiao_ema": observation_detector},
                    )
                ),
            )
        event_detector.finalize(start + pd.Timedelta(minutes=10))
        observation_detector.finalize(start + pd.Timedelta(minutes=10))

        self.assertEqual(len(event_detector.completed_triggers), 1)
        self.assertEqual(len(observation_detector.completed_triggers), 1)
        event_trigger = event_detector.completed_triggers[0]
        observation_trigger = observation_detector.completed_triggers[0]
        for field in (
            "trigger_time",
            "cooldown_until",
            "closed_at",
            "volume",
            "strength",
        ):
            with self.subTest(field=field):
                self.assertEqual(event_trigger[field], observation_trigger[field])
        triggered_results = [item for item in detection_results if item.triggered]
        self.assertEqual(len(triggered_results), 1)
        neutral_result = triggered_results[0]
        self.assertEqual(
            pd.Timestamp(neutral_result.observation_time_utc),
            observation_trigger["trigger_time"],
        )
        self.assertEqual(neutral_result.detector_id, "xiao_ema")
        self.assertEqual(
            neutral_result.signal_id,
            "comment_count_event_window_120s_step_30s",
        )
        self.assertAlmostEqual(neutral_result.score, observation_trigger["strength"])
        self.assertEqual(
            neutral_result.detector_metadata["volume"],
            observation_trigger["volume"],
        )


class ReplayCompatibilityTests(unittest.TestCase):
    def test_replay_filters_orders_delays_and_finalizes_selected_events(self) -> None:
        source = _RecordingSource()
        hook = _RecordingHook()
        events = pd.DataFrame(
            [
                {"comment_id": "c4", "event_time_utc": "2026-01-01T00:01:30Z"},
                {"comment_id": "c2", "event_time_utc": "2026-01-01T00:00:30Z"},
                {"comment_id": "c1", "event_time_utc": "2026-01-01T00:00:00Z"},
                {"comment_id": "c3", "event_time_utc": "2026-01-01T00:01:00Z"},
            ]
        )
        events["event_time_utc"] = pd.to_datetime(events["event_time_utc"], utc=True)

        with (
            patch("youtube_pipeline.replay.require_streamz"),
            patch("youtube_pipeline.replay.time.sleep") as sleep,
        ):
            replay_events(
                source,
                events,
                speed=120.0,
                max_sleep_seconds=1.0,
                start="2026-01-01T00:00:30Z",
                end="2026-01-01T00:01:00Z",
                event_hooks=[hook],
            )

        self.assertEqual(
            [event["comment_id"] for event in source.events],
            ["c2", "c3"],
        )
        self.assertEqual(
            [event["comment_id"] for event in hook.events],
            ["c2", "c3"],
        )
        sleep.assert_called_once_with(0.25)
        self.assertEqual(
            hook.finalized_at,
            pd.Timestamp("2026-01-01T00:01:00Z"),
        )


if __name__ == "__main__":
    unittest.main()
