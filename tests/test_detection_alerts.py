from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    DetectionResult,
    dispatch_numeric_observation,
)
from youtube_pipeline.configuration import (
    canonical_run_config_json,
    run_config_from_mapping,
    run_config_hash,
)
from youtube_pipeline.detection_alerts import (
    ACTIVITY,
    POLARIZATION,
    DetectionGapPolicyConfig,
    detection_alert_from_result,
)
from youtube_pipeline.detectors import (
    ADWINAdapter,
    ADWINConfig,
    KSWINAdapter,
    KSWINConfig,
    PageHinkleyAdapter,
    PageHinkleyConfig,
)
from youtube_pipeline.signal_observations import (
    NO_CLASSIFIABLE_OPINIONS,
    NO_INPUT,
    VALUE_AVAILABLE,
    SignalObservation,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _result(*, triggered: bool, signal_id: str = "polarization_mec") -> DetectionResult:
    return DetectionResult(
        detector_id="kswin",
        signal_id=signal_id,
        observation_time_utc=START,
        triggered=triggered,
        quality="degraded",
    )


def _observation(
    value: float | None,
    second: int,
    *,
    status: str = VALUE_AVAILABLE,
) -> SignalObservation:
    return SignalObservation(
        signal_id="polarization_mec",
        observation_time_utc=START + timedelta(seconds=second),
        value=value,
        status=status,
        support_count=2 if status == VALUE_AVAILABLE else 0,
        quality="passed",
        metadata={},
    )


class DetectionAlertContractTests(unittest.TestCase):
    def test_only_triggered_results_create_deterministic_immutable_alerts(self) -> None:
        self.assertIsNone(detection_alert_from_result(
            _result(triggered=False),
            run_id="run-1",
            config_hash="hash-1",
            signal_family=POLARIZATION,
            support_count=2,
            data_cutoff_utc=START + timedelta(minutes=1),
            detection_artifact_ref="polarization_detection_results.jsonl",
        ))
        kwargs = {
            "run_id": "run-1",
            "config_hash": "hash-1",
            "signal_family": POLARIZATION,
            "support_count": 2,
            "data_cutoff_utc": START + timedelta(minutes=1),
            "detection_artifact_ref": "polarization_detection_results.jsonl",
        }
        first = detection_alert_from_result(_result(triggered=True), **kwargs)
        second = detection_alert_from_result(_result(triggered=True), **kwargs)
        assert first is not None and second is not None
        self.assertEqual(first, second)
        self.assertEqual(first.alert_time_utc, START)
        self.assertTrue(first.triggered)
        self.assertIn(first.detection_result_id, first.detection_result_ref)
        with self.assertRaises(FrozenInstanceError):
            first.quality = "changed"

    def test_activity_and_polarization_alerts_coexist_without_identity_collision(self) -> None:
        common = {
            "run_id": "run-1",
            "config_hash": "hash-1",
            "support_count": 3,
            "data_cutoff_utc": START + timedelta(minutes=1),
            "detection_artifact_ref": "detection_results.jsonl",
        }
        activity = detection_alert_from_result(
            _result(triggered=True, signal_id="comment_count_event_window_120s_step_30s"),
            signal_family=ACTIVITY,
            **common,
        )
        polarization = detection_alert_from_result(
            _result(triggered=True), signal_family=POLARIZATION, **common
        )
        assert activity is not None and polarization is not None
        self.assertNotEqual(activity.alert_id, polarization.alert_id)
        self.assertEqual(activity.signal_family, ACTIVITY)
        self.assertEqual(polarization.signal_family, POLARIZATION)

    def test_causality_family_and_support_are_validated(self) -> None:
        common = {
            "result": _result(triggered=True),
            "run_id": "run-1",
            "config_hash": "hash-1",
            "signal_family": POLARIZATION,
            "support_count": 2,
            "data_cutoff_utc": START + timedelta(minutes=1),
            "detection_artifact_ref": "results.jsonl",
        }
        with self.assertRaisesRegex(ValueError, "strictly before"):
            detection_alert_from_result(
                **{**common, "data_cutoff_utc": START}
            )
        with self.assertRaisesRegex(ValueError, "signal_family"):
            detection_alert_from_result(
                **{**common, "signal_family": "UNKNOWN"}
            )
        with self.assertRaisesRegex(ValueError, "support_count"):
            detection_alert_from_result(
                **{**common, "support_count": -1}
            )

    def test_explicit_gap_policy_is_typed_serialized_and_hash_visible(self) -> None:
        base = {
            "identity": {"run_id": "gap-config"},
            "signals": {"activity": {
                "signal_id": "comment_count_event_window_120s_step_30s",
                "metric": "comment_count",
                "source": "prepared_comments",
                "scope": "selected_comment_stream",
                "unit": "comments",
                "window": "120s",
                "cadence": "30s",
                "time_basis": "event_time_utc",
                "timezone": "UTC",
                "interval_policy": "closed",
            }},
            "detection": {
                "activity_route": {
                    "signal_id": "comment_count_event_window_120s_step_30s",
                    "detector_id": "page_hinkley",
                },
                "page_hinkley": {},
            },
        }
        without = run_config_from_mapping(base)
        with_policy_payload = {
            **base,
            "detection": {
                **base["detection"],
                "gap_policy": {"policy_id": "preserve_state_v1"},
            },
        }
        with_policy = run_config_from_mapping(with_policy_payload)
        self.assertIsInstance(
            with_policy.detection.gap_policy, DetectionGapPolicyConfig
        )
        self.assertIn('"gap_policy"', canonical_run_config_json(with_policy))
        self.assertNotEqual(run_config_hash(without), run_config_hash(with_policy))
        with self.assertRaisesRegex(ValueError, "policy_id"):
            DetectionGapPolicyConfig(policy_id="reset_after_any_gap")
        with self.assertRaisesRegex(ValueError, "Unknown detection gap policy"):
            run_config_from_mapping({
                **base,
                "detection": {
                    **base["detection"],
                    "gap_policy": {
                        "policy_id": "preserve_state_v1",
                        "gap_seconds": 60,
                    },
                },
            })

    def test_preserve_state_skips_all_gaps_for_each_detector(self) -> None:
        adapters = {
            "page_hinkley": PageHinkleyAdapter(config=PageHinkleyConfig()),
            "adwin": ADWINAdapter(config=ADWINConfig()),
            "kswin": KSWINAdapter(config=KSWINConfig(seed=42)),
        }
        for detector_id, detector in adapters.items():
            with self.subTest(detector_id=detector_id):
                route = ActivityDetectionRouteConfig(
                    signal_id="polarization_mec", detector_id=detector_id
                )
                results = []
                for observation in (
                    _observation(.3, 0),
                    _observation(None, 1, status=NO_INPUT),
                    _observation(None, 2, status=NO_CLASSIFIABLE_OPINIONS),
                    _observation(.7, 3),
                ):
                    result = dispatch_numeric_observation(
                        route=route,
                        observation=observation,
                        detectors={detector_id: detector},
                    )
                    if result is not None:
                        results.append(result)
                self.assertEqual(len(results), 2)
                self.assertEqual(
                    [result.detector_metadata["observations_processed"] for result in results],
                    [1, 2],
                )


if __name__ == "__main__":
    unittest.main()
