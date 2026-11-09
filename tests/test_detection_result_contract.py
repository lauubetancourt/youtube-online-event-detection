from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone

from youtube_pipeline.activity_detection import DetectionResult


class DetectionResultContractTests(unittest.TestCase):
    def test_minimum_result_is_detector_neutral_and_immutable(self) -> None:
        result = DetectionResult(
            detector_id="xiao_ema",
            signal_id="comment_count_event_window_120s_step_30s",
            observation_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            triggered=False,
            quality="passed",
        )

        self.assertIsNone(result.score)
        self.assertEqual(dict(result.detector_metadata), {})
        with self.assertRaises(FrozenInstanceError):
            result.triggered = True

    def test_required_fields_and_utc_time_are_validated(self) -> None:
        common = {
            "signal_id": "comment_count_event_window_120s_step_30s",
            "observation_time_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "triggered": False,
            "quality": "passed",
        }

        with self.assertRaises(TypeError):
            DetectionResult(**common)
        with self.assertRaises(ValueError):
            DetectionResult(detector_id="", **common)
        with self.assertRaises(ValueError):
            DetectionResult(
                detector_id="xiao_ema",
                **{
                    **common,
                    "observation_time_utc": datetime(2026, 1, 1),
                },
            )

    def test_specific_metadata_does_not_pollute_common_fields(self) -> None:
        result = DetectionResult(
            detector_id="xiao_ema",
            signal_id="comment_count_event_window_120s_step_30s",
            observation_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            triggered=True,
            quality="passed",
            score=2.5,
            detector_metadata={
                "ema_fast": 50.0,
                "ema_slow": 20.0,
                "conditions": {"volume_threshold": True},
            },
        )

        common_fields = {item.name for item in fields(DetectionResult)}
        self.assertNotIn("ema_fast", common_fields)
        self.assertEqual(result.detector_metadata["ema_fast"], 50.0)
        with self.assertRaises(TypeError):
            result.detector_metadata["ema_fast"] = 99.0
        with self.assertRaises(TypeError):
            result.detector_metadata["conditions"]["volume_threshold"] = False


if __name__ == "__main__":
    unittest.main()
