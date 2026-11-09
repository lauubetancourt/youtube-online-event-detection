from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from functools import partial

import pandas as pd

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    dispatch_activity_observation,
)
from youtube_pipeline.activity_signals import (
    EventWindowActivitySignal,
    event_window_unique_author_count_definition,
    unique_author_count_measurement,
)
from youtube_pipeline.configuration import (
    DetectionConfig,
    RunIdentityConfig,
    run_config_from_mapping,
    run_config_to_mapping,
)
from youtube_pipeline.detectors import (
    XiaoEMAConfig,
    XiaoEMATriggerDetector,
    create_detector,
)


COMMENT_SIGNAL_ID = "comment_count_event_window_120s_step_30s"
AUTHOR_SIGNAL_ID = "unique_author_count_event_window_120s_step_30s"


class ActivityDetectionCompositionTests(unittest.TestCase):
    def test_configuration_declares_comment_count_to_xiao_route(self) -> None:
        config = run_config_from_mapping(
            {
                "identity": {"run_id": "route_comment_xiao"},
                "detection": {
                    "activity_route": {
                        "signal_id": COMMENT_SIGNAL_ID,
                        "detector_id": "xiao_ema",
                    },
                    "xiao_ema": {},
                },
            }
        )

        self.assertEqual(config.identity, RunIdentityConfig("route_comment_xiao"))
        self.assertEqual(config.detection.activity_route.signal_id, COMMENT_SIGNAL_ID)
        self.assertEqual(config.detection.activity_route.detector_id, "xiao_ema")
        self.assertEqual(config.detection.xiao_ema, XiaoEMAConfig())
        self.assertEqual(
            run_config_to_mapping(config)["detection"]["activity_route"],
            {"signal_id": COMMENT_SIGNAL_ID, "detector_id": "xiao_ema"},
        )

    def test_configuration_can_declare_unique_authors_to_xiao_route(self) -> None:
        config = run_config_from_mapping(
            {
                "identity": {"run_id": "route_authors_xiao"},
                "detection": {
                    "activity_route": {
                        "signal_id": AUTHOR_SIGNAL_ID,
                        "detector_id": "xiao_ema",
                    },
                    "xiao_ema": {},
                },
            }
        )

        self.assertEqual(config.detection.activity_route.signal_id, AUTHOR_SIGNAL_ID)
        self.assertEqual(config.detection.activity_route.detector_id, "xiao_ema")

    def test_route_is_immutable_and_does_not_duplicate_detector_parameters(self) -> None:
        route = ActivityDetectionRouteConfig(
            signal_id=COMMENT_SIGNAL_ID,
            detector_id="xiao_ema",
        )

        with self.assertRaises(FrozenInstanceError):
            route.signal_id = AUTHOR_SIGNAL_ID
        self.assertFalse(hasattr(route, "v_min"))
        self.assertFalse(hasattr(route, "window_size"))
        self.assertFalse(hasattr(route, "sensitivity_threshold"))

    def test_route_requires_a_registered_and_configured_detector(self) -> None:
        unknown = ActivityDetectionRouteConfig(
            signal_id=COMMENT_SIGNAL_ID,
            detector_id="unknown_detector",
        )
        with self.assertRaisesRegex(ValueError, "Unknown activity detector"):
            DetectionConfig(activity_route=unknown, xiao_ema=XiaoEMAConfig())

        xiao_route = ActivityDetectionRouteConfig(
            signal_id=COMMENT_SIGNAL_ID,
            detector_id="xiao_ema",
        )
        with self.assertRaisesRegex(ValueError, "matching detector configuration"):
            DetectionConfig(activity_route=xiao_route)

    def test_unique_author_observation_reaches_xiao_without_source_rows(self) -> None:
        route = ActivityDetectionRouteConfig(
            signal_id=AUTHOR_SIGNAL_ID,
            detector_id="xiao_ema",
        )
        detector = create_detector(
            name=route.detector_id,
            config=XiaoEMAConfig(),
            log_fn=lambda _message: None,
        )
        signal = EventWindowActivitySignal(
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
        observations = signal.on_event(
            {
                "comment_id": "c1",
                "author_id": "author-a",
                "event_time_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            }
        )

        result = dispatch_activity_observation(
            route=route,
            observation=observations[0],
            detectors={"xiao_ema": detector},
        )

        self.assertEqual(detector.windows_processed, 1)
        self.assertEqual(detector.completed_triggers, [])
        self.assertEqual(result.detector_id, "xiao_ema")
        self.assertEqual(result.signal_id, AUTHOR_SIGNAL_ID)
        self.assertFalse(result.triggered)
        self.assertEqual(result.quality, "passed")
        self.assertFalse(result.detector_metadata["warmup_complete"])

    def test_signal_quality_is_propagated_without_detector_reinterpretation(self) -> None:
        route = ActivityDetectionRouteConfig(
            signal_id=AUTHOR_SIGNAL_ID,
            detector_id="xiao_ema",
        )
        detector = create_detector(
            name=route.detector_id,
            config=XiaoEMAConfig(),
            log_fn=lambda _message: None,
        )
        signal = EventWindowActivitySignal(
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
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        passed_observation = signal.on_event(
            {
                "comment_id": "known",
                "author_id": "author-a",
                "event_time_utc": start,
            }
        )[0]
        degraded_observation = signal.on_event(
            {
                "comment_id": "missing",
                "author_id": None,
                "event_time_utc": start + pd.Timedelta(seconds=30),
            }
        )[0]

        passed_result = dispatch_activity_observation(
            route=route,
            observation=passed_observation,
            detectors={"xiao_ema": detector},
        )
        degraded_result = dispatch_activity_observation(
            route=route,
            observation=degraded_observation,
            detectors={"xiao_ema": detector},
        )

        self.assertEqual(passed_observation.quality, "passed")
        self.assertEqual(passed_result.quality, passed_observation.quality)
        self.assertEqual(
            degraded_observation.quality,
            "degraded_missing_author_id",
        )
        self.assertEqual(
            degraded_result.quality,
            degraded_observation.quality,
        )
        self.assertFalse(degraded_result.detector_metadata["warmup_complete"])
        self.assertNotEqual(degraded_result.quality, "warmup")

    def test_dispatch_rejects_an_observation_from_another_signal(self) -> None:
        route = ActivityDetectionRouteConfig(
            signal_id=COMMENT_SIGNAL_ID,
            detector_id="xiao_ema",
        )
        detector = XiaoEMATriggerDetector(log_fn=lambda _message: None)
        signal = EventWindowActivitySignal(
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
        observation = signal.on_event(
            {
                "author_id": "author-a",
                "event_time_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            }
        )[0]

        with self.assertRaisesRegex(ValueError, "signal_id"):
            dispatch_activity_observation(
                route=route,
                observation=observation,
                detectors={"xiao_ema": detector},
            )


if __name__ == "__main__":
    unittest.main()
