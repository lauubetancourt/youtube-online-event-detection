from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

from youtube_pipeline.activity_detection import dispatch_activity_observation
from youtube_pipeline.activity_signals import (
    ActivityObservation,
    ActivitySignalDefinition,
    CLOSED_INTERVAL,
)
from youtube_pipeline.configuration import (
    ADWINConfig,
    load_run_config,
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.detectors import ADWINAdapter, create_detector, get_detector_names
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
    window="120s",
    cadence="30s",
    time_basis="event_time_utc",
    timezone="UTC",
    interval_policy=CLOSED_INTERVAL,
)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SYNTHETIC_CONFIG = ADWINConfig(
    delta=0.01,
    clock=1,
    max_buckets=5,
    min_window_length=5,
    grace_period=10,
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
        window_start_utc=observation_time - timedelta(seconds=120),
        window_end_utc=observation_time,
        value=value,
        support_count=1,
        quality=quality,
    )


def _results(
    detector: ADWINAdapter,
    values: list[int | float],
) -> list:
    return [
        detector.on_observation(_observation(value, index))
        for index, value in enumerate(values)
    ]


def _run_config_payload(*, delta: float = 0.01) -> dict:
    return {
        "identity": {"run_id": "adwin_composition"},
        "detection": {
            "activity_route": {
                "signal_id": SIGNAL.signal_id,
                "detector_id": "adwin",
            },
            "adwin": {
                "delta": delta,
                "clock": 1,
                "max_buckets": 5,
                "min_window_length": 5,
                "grace_period": 10,
            },
        },
    }


class ADWINConfigTests(unittest.TestCase):
    def test_defaults_match_river_technical_defaults_and_are_immutable(self) -> None:
        config = ADWINConfig()

        self.assertEqual(config.delta, 0.002)
        self.assertEqual(config.clock, 32)
        self.assertEqual(config.max_buckets, 5)
        self.assertEqual(config.min_window_length, 5)
        self.assertEqual(config.grace_period, 10)
        with self.assertRaises(FrozenInstanceError):
            config.delta = 0.01

    def test_custom_values_are_preserved(self) -> None:
        config = ADWINConfig(
            delta=0.05,
            clock=8,
            max_buckets=7,
            min_window_length=3,
            grace_period=0,
        )

        self.assertEqual(config.delta, 0.05)
        self.assertEqual(config.clock, 8)
        self.assertEqual(config.max_buckets, 7)
        self.assertEqual(config.min_window_length, 3)
        self.assertEqual(config.grace_period, 0)

    def test_invalid_types_and_ranges_are_rejected(self) -> None:
        invalid_cases = (
            ({"delta": True}, TypeError),
            ({"delta": 0}, ValueError),
            ({"delta": 1}, ValueError),
            ({"delta": float("nan")}, ValueError),
            ({"clock": True}, TypeError),
            ({"clock": 0}, ValueError),
            ({"max_buckets": 0}, ValueError),
            ({"min_window_length": 0}, ValueError),
            ({"grace_period": -1}, ValueError),
            ({"grace_period": 1.5}, TypeError),
        )

        for kwargs, error_type in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error_type):
                    ADWINConfig(**kwargs)

    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ADWIN config fields"):
            ADWINConfig.from_mapping({"bucket_history": True})

    def test_json_loading_and_canonical_serialization_include_all_parameters(self) -> None:
        payload = _run_config_payload()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "adwin.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            run_config = load_run_config(config_path)

        config = run_config.detection.adwin
        assert config is not None
        self.assertEqual(config, SYNTHETIC_CONFIG)
        self.assertEqual(
            run_config_to_mapping(run_config)["detection"]["adwin"],
            payload["detection"]["adwin"],
        )

    def test_adwin_parameter_change_changes_config_hash(self) -> None:
        first = run_config_from_mapping(_run_config_payload(delta=0.01))
        second = run_config_from_mapping(_run_config_payload(delta=0.02))

        self.assertNotEqual(run_config_hash(first), run_config_hash(second))


class ADWINAdapterTests(unittest.TestCase):
    def test_maps_observation_to_neutral_immutable_result(self) -> None:
        detector = ADWINAdapter(config=SYNTHETIC_CONFIG)
        observation = _observation(-0.25, 0, quality="degraded_synthetic_input")

        result = detector.on_observation(observation)

        self.assertEqual(result.detector_id, "adwin")
        self.assertEqual(result.signal_id, observation.signal.signal_id)
        self.assertEqual(result.observation_time_utc, observation.observation_time_utc)
        self.assertEqual(result.quality, observation.quality)
        self.assertFalse(result.triggered)
        self.assertIsNone(result.score)
        self.assertEqual(
            dict(result.detector_metadata),
            {
                "observations_processed": 1,
                "width": 1.0,
                "estimation": -0.25,
                "variance": 0.0,
            },
        )
        json.dumps(dict(result.detector_metadata))
        with self.assertRaises(FrozenInstanceError):
            result.triggered = True
        with self.assertRaises(TypeError):
            result.detector_metadata["width"] = 2

    def test_stable_stream_does_not_trigger_immediately(self) -> None:
        results = _results(
            ADWINAdapter(config=SYNTHETIC_CONFIG),
            [10.0] * 200,
        )

        self.assertFalse(any(result.triggered for result in results))

    def test_grace_period_defers_change_detection(self) -> None:
        config = ADWINConfig(
            delta=0.01,
            clock=1,
            max_buckets=5,
            min_window_length=5,
            grace_period=250,
        )
        results = _results(
            ADWINAdapter(config=config),
            [1.0] * 100 + [10.0] * 100,
        )

        self.assertFalse(any(result.triggered for result in results))

    def test_clear_mean_shift_eventually_triggers(self) -> None:
        results = _results(
            ADWINAdapter(config=SYNTHETIC_CONFIG),
            [1.0] * 100 + [10.0] * 100,
        )

        self.assertTrue(any(result.triggered for result in results))

    def test_public_width_contracts_when_drift_is_detected(self) -> None:
        results = _results(
            ADWINAdapter(config=SYNTHETIC_CONFIG),
            [1.0] * 100 + [10.0] * 100,
        )
        trigger_index = next(
            index for index, result in enumerate(results) if result.triggered
        )
        width_at_trigger = results[trigger_index].detector_metadata["width"]
        earlier_max_width = max(
            result.detector_metadata["width"] for result in results[:trigger_index]
        )

        self.assertLess(width_at_trigger, earlier_max_width)

    def test_next_update_follows_river_post_drift_reset(self) -> None:
        detector = ADWINAdapter(config=SYNTHETIC_CONFIG)
        triggered_result = None
        trigger_index = None
        for index, value in enumerate([1.0] * 100 + [10.0] * 100):
            result = detector.on_observation(_observation(value, index))
            if result.triggered:
                triggered_result = result
                trigger_index = index
                break
        assert triggered_result is not None
        assert trigger_index is not None

        after_reset = detector.on_observation(
            _observation(10.0, trigger_index + 1)
        )

        self.assertGreater(triggered_result.detector_metadata["width"], 1)
        self.assertEqual(after_reset.detector_metadata["observations_processed"], 1)
        self.assertEqual(after_reset.detector_metadata["width"], 1.0)
        self.assertFalse(after_reset.triggered)

    def test_two_fresh_instances_are_deterministic(self) -> None:
        values = [1.0] * 100 + [10.0] * 100 + [2.0] * 100

        first = [
            result.triggered
            for result in _results(ADWINAdapter(config=SYNTHETIC_CONFIG), values)
        ]
        second = [
            result.triggered
            for result in _results(ADWINAdapter(config=SYNTHETIC_CONFIG), values)
        ]

        self.assertEqual(first, second)

    def test_json_route_selects_adwin_without_importing_river(self) -> None:
        run_config = run_config_from_mapping(_run_config_payload())
        route = run_config.detection.activity_route
        config = run_config.detection.adwin
        assert route is not None
        assert config is not None
        detector = create_detector(name=route.detector_id, config=config)

        result = dispatch_activity_observation(
            route=route,
            observation=_observation(1.0, 0),
            detectors={route.detector_id: detector},
        )

        self.assertEqual(result.detector_id, "adwin")
        self.assertEqual(result.signal_id, SIGNAL.signal_id)

    def test_registry_keeps_all_neutral_detectors_available(self) -> None:
        detector_names = get_detector_names()

        self.assertIn("xiao_ema", detector_names)
        self.assertIn("page_hinkley", detector_names)
        self.assertIn("adwin", detector_names)

    def test_triggered_result_promotes_without_candidate_changes(self) -> None:
        detector = ADWINAdapter(config=SYNTHETIC_CONFIG)
        observations = [
            _observation(value, index)
            for index, value in enumerate([1.0] * 100 + [10.0] * 100)
        ]
        observation, result = next(
            (observation, result)
            for observation in observations
            if (result := detector.on_observation(observation)).triggered
        )

        candidate = promote_detection_result(
            candidate_id="evt_adwin_synthetic",
            observation=observation,
            detection_result=result,
            lineage=EventCandidateLineage(
                run_id="run_adwin_test",
                config_hash="sha256_synthetic",
                dataset_ref="synthetic:test_stream",
            ),
            lifecycle_state="point",
        )

        self.assertEqual(candidate.detector_id, "adwin")
        self.assertIs(candidate.observation, observation)
        self.assertIs(candidate.detection_result, result)
        self.assertEqual(candidate.quality, observation.quality)


if __name__ == "__main__":
    unittest.main()
