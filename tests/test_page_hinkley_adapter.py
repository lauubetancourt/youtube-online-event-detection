from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    dispatch_activity_observation,
)
from youtube_pipeline.activity_signals import (
    ActivityObservation,
    ActivitySignalDefinition,
    CLOSED_INTERVAL,
)
from youtube_pipeline.configuration import (
    PageHinkleyConfig,
    run_config_from_mapping,
    run_config_to_mapping,
)
from youtube_pipeline.detectors import PageHinkleyAdapter, create_detector
from youtube_pipeline.event_candidates import (
    EventCandidateLineage,
    promote_detection_result,
)


SIGNAL = ActivitySignalDefinition(
    signal_id="synthetic_numeric_signal",
    metric="synthetic_numeric_value",
    source="synthetic_test_stream",
    scope="contract_test",
    unit="numeric_index",
    window="30s",
    cadence="30s",
    time_basis="event_time_utc",
    timezone="UTC",
    interval_policy=CLOSED_INTERVAL,
)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SYNTHETIC_CONFIG = PageHinkleyConfig(
    min_instances=5,
    delta=0.0,
    threshold=2.0,
    alpha=1.0,
    mode="up",
)


def _observation(
    value: int | float,
    index: int,
    *,
    quality: str = "passed",
) -> ActivityObservation:
    observation_time = START + timedelta(seconds=30 * index)
    return ActivityObservation(
        signal=SIGNAL,
        observation_time_utc=observation_time,
        window_start_utc=observation_time - timedelta(seconds=30),
        window_end_utc=observation_time,
        value=value,
        support_count=1,
        quality=quality,
    )


def _results(
    detector: PageHinkleyAdapter,
    values: list[int | float],
) -> list:
    return [
        detector.on_observation(_observation(value, index))
        for index, value in enumerate(values)
    ]


class PageHinkleyConfigTests(unittest.TestCase):
    def test_defaults_match_river_technical_defaults_and_are_immutable(self) -> None:
        config = PageHinkleyConfig()

        self.assertEqual(config.min_instances, 30)
        self.assertEqual(config.delta, 0.005)
        self.assertEqual(config.threshold, 50.0)
        self.assertEqual(config.alpha, 0.9999)
        self.assertEqual(config.mode, "both")
        with self.assertRaises(FrozenInstanceError):
            config.mode = "up"

    def test_custom_values_are_preserved(self) -> None:
        config = PageHinkleyConfig(
            min_instances=12,
            delta=0.25,
            threshold=8,
            alpha=0.95,
            mode="down",
        )

        self.assertEqual(config.min_instances, 12)
        self.assertEqual(config.delta, 0.25)
        self.assertEqual(config.threshold, 8.0)
        self.assertEqual(config.alpha, 0.95)
        self.assertEqual(config.mode, "down")

    def test_invalid_types_ranges_and_modes_are_rejected(self) -> None:
        invalid_cases = (
            ({"min_instances": True}, TypeError),
            ({"min_instances": 0}, ValueError),
            ({"delta": -0.1}, ValueError),
            ({"delta": float("nan")}, ValueError),
            ({"threshold": 0}, ValueError),
            ({"alpha": 0}, ValueError),
            ({"alpha": 1.1}, ValueError),
            ({"mode": 1}, TypeError),
            ({"mode": "sideways"}, ValueError),
        )

        for kwargs, error_type in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error_type):
                    PageHinkleyConfig(**kwargs)


class PageHinkleyAdapterTests(unittest.TestCase):
    def test_maps_observation_to_neutral_immutable_result(self) -> None:
        detector = PageHinkleyAdapter(config=SYNTHETIC_CONFIG)
        observation = _observation(-0.25, 0, quality="degraded_synthetic_input")

        result = detector.on_observation(observation)

        self.assertEqual(result.detector_id, "page_hinkley")
        self.assertEqual(result.signal_id, observation.signal.signal_id)
        self.assertEqual(result.observation_time_utc, observation.observation_time_utc)
        self.assertEqual(result.quality, observation.quality)
        self.assertFalse(result.triggered)
        self.assertIsNone(result.score)
        self.assertEqual(
            dict(result.detector_metadata),
            {
                "observations_processed": 1,
                "min_instances": 5,
                "mode": "up",
                "warmup_complete": False,
            },
        )
        json.dumps(dict(result.detector_metadata))
        with self.assertRaises(FrozenInstanceError):
            result.triggered = True
        with self.assertRaises(TypeError):
            result.detector_metadata["mode"] = "down"

    def test_stable_stream_does_not_trigger_immediately(self) -> None:
        results = _results(
            PageHinkleyAdapter(config=SYNTHETIC_CONFIG),
            [10.0] * 40,
        )

        self.assertFalse(any(result.triggered for result in results))

    def test_upward_change_eventually_triggers_in_up_mode(self) -> None:
        results = _results(
            PageHinkleyAdapter(config=SYNTHETIC_CONFIG),
            [1.0] * 10 + [10.0] * 20,
        )

        self.assertTrue(any(result.triggered for result in results))

    def test_up_mode_does_not_interpret_equivalent_drop_as_an_increase(self) -> None:
        results = _results(
            PageHinkleyAdapter(config=SYNTHETIC_CONFIG),
            [10.0] * 10 + [1.0] * 20,
        )

        self.assertFalse(any(result.triggered for result in results))

    def test_adapter_counter_follows_river_reset_on_update_after_drift(self) -> None:
        detector = PageHinkleyAdapter(config=SYNTHETIC_CONFIG)
        triggered_result = None
        for index, value in enumerate([1.0] * 10 + [10.0] * 20):
            result = detector.on_observation(_observation(value, index))
            if result.triggered:
                triggered_result = result
                break
        assert triggered_result is not None

        after_reset = detector.on_observation(_observation(10.0, 30))

        self.assertTrue(triggered_result.detector_metadata["warmup_complete"])
        self.assertEqual(after_reset.detector_metadata["observations_processed"], 1)
        self.assertFalse(after_reset.detector_metadata["warmup_complete"])
        self.assertFalse(after_reset.triggered)

    def test_two_fresh_instances_are_deterministic(self) -> None:
        values = [1.0] * 10 + [10.0] * 20 + [2.0] * 10

        first = [
            result.triggered
            for result in _results(
                PageHinkleyAdapter(config=SYNTHETIC_CONFIG),
                values,
            )
        ]
        second = [
            result.triggered
            for result in _results(
                PageHinkleyAdapter(config=SYNTHETIC_CONFIG),
                values,
            )
        ]

        self.assertEqual(first, second)

    def test_json_route_selects_page_hinkley_without_importing_river(self) -> None:
        run_config = run_config_from_mapping(
            {
                "identity": {"run_id": "page_hinkley_composition"},
                "detection": {
                    "activity_route": {
                        "signal_id": SIGNAL.signal_id,
                        "detector_id": "page_hinkley",
                    },
                    "page_hinkley": {
                        "min_instances": 5,
                        "delta": 0.0,
                        "threshold": 2.0,
                        "alpha": 1.0,
                        "mode": "up",
                    },
                },
            }
        )
        route = run_config.detection.activity_route
        config = run_config.detection.page_hinkley
        assert route is not None
        assert config is not None
        detector = create_detector(name=route.detector_id, config=config)

        result = dispatch_activity_observation(
            route=route,
            observation=_observation(1.0, 0),
            detectors={route.detector_id: detector},
        )

        self.assertEqual(result.detector_id, "page_hinkley")
        self.assertEqual(result.signal_id, SIGNAL.signal_id)
        self.assertEqual(
            run_config_to_mapping(run_config)["detection"]["page_hinkley"],
            {
                "min_instances": 5,
                "delta": 0.0,
                "threshold": 2.0,
                "alpha": 1.0,
                "mode": "up",
            },
        )

    def test_triggered_result_promotes_without_candidate_changes(self) -> None:
        detector = PageHinkleyAdapter(config=SYNTHETIC_CONFIG)
        observations = [
            _observation(value, index)
            for index, value in enumerate([1.0] * 10 + [10.0] * 20)
        ]
        triggered_pair = next(
            (observation, result)
            for observation in observations
            if (result := detector.on_observation(observation)).triggered
        )
        observation, result = triggered_pair

        candidate = promote_detection_result(
            candidate_id="evt_page_hinkley_synthetic",
            observation=observation,
            detection_result=result,
            lineage=EventCandidateLineage(
                run_id="run_page_hinkley_test",
                config_hash="sha256_synthetic",
                dataset_ref="synthetic:test_stream",
            ),
            lifecycle_state="point",
        )

        self.assertEqual(candidate.detector_id, "page_hinkley")
        self.assertIs(candidate.observation, observation)
        self.assertIs(candidate.detection_result, result)
        self.assertEqual(candidate.quality, observation.quality)


if __name__ == "__main__":
    unittest.main()
