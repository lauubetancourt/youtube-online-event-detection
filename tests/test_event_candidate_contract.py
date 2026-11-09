from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone

from youtube_pipeline.activity_detection import DetectionResult
from youtube_pipeline.activity_signals import (
    ActivityObservation,
    event_window_comment_count_definition,
)
from youtube_pipeline.event_candidates import (
    EventCandidate,
    EventCandidateLineage,
    project_completed_xiao_trigger,
    promote_detection_result,
)


SIGNAL_ID = "comment_count_event_window_120s_step_30s"


def _observation(*, quality: str = "passed") -> ActivityObservation:
    end = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    return ActivityObservation(
        signal=event_window_comment_count_definition(
            window="120s",
            cadence="30s",
            time_basis="event_time_utc",
        ),
        observation_time_utc=end,
        window_start_utc=end - timedelta(seconds=120),
        window_end_utc=end,
        value=53,
        support_count=53,
        quality=quality,
    )


def _result(
    observation: ActivityObservation,
    *,
    triggered: bool = True,
    quality: str | None = None,
) -> DetectionResult:
    return DetectionResult(
        detector_id="xiao_ema",
        signal_id=observation.signal.signal_id,
        observation_time_utc=observation.observation_time_utc,
        triggered=triggered,
        quality=observation.quality if quality is None else quality,
        score=2.75,
        detector_metadata={
            "volume": 53,
            "strength": 2.75,
            "ema_fast": 44.0,
            "ema_slow": 16.0,
            "warmup_complete": True,
            "cooldown_active": True,
        },
    )


def _lineage() -> EventCandidateLineage:
    return EventCandidateLineage(
        run_id="run_test",
        config_hash="sha256_test_config",
        dataset_ref="prepared_dataset:test_fixture",
        run_manifest_ref="runs/run_test/run_manifest.json",
    )


class EventCandidateContractTests(unittest.TestCase):
    def test_triggered_result_is_promoted_without_duplicating_authorities(self) -> None:
        observation = _observation()
        result = _result(observation)

        candidate = promote_detection_result(
            candidate_id="evt_existing_formula",
            observation=observation,
            detection_result=result,
            lineage=_lineage(),
        )

        self.assertEqual(candidate.candidate_id, "evt_existing_formula")
        self.assertIs(candidate.observation, observation)
        self.assertIs(candidate.detection_result, result)
        self.assertEqual(candidate.detector_id, "xiao_ema")
        self.assertEqual(candidate.signal_id, SIGNAL_ID)
        self.assertEqual(candidate.quality, observation.quality)
        self.assertEqual(
            candidate.evidence_window_start_utc,
            observation.window_start_utc,
        )
        self.assertEqual(candidate.evidence_window_end_utc, observation.window_end_utc)

        candidate_fields = {item.name for item in fields(EventCandidate)}
        self.assertTrue(
            {
                "candidate_id",
                "observation",
                "detection_result",
                "lineage",
                "evidence_window_start_utc",
                "evidence_window_end_utc",
            }.issubset(candidate_fields)
        )
        self.assertTrue(
            {
                "quality",
                "score",
                "detector_metadata",
                "comment_ids",
                "video_ids",
                "context_units",
                "run_config",
            }.isdisjoint(candidate_fields)
        )
        lineage_fields = {item.name for item in fields(EventCandidateLineage)}
        self.assertEqual(
            lineage_fields,
            {"run_id", "config_hash", "dataset_ref", "run_manifest_ref"},
        )

    def test_candidate_and_lineage_are_immutable(self) -> None:
        observation = _observation()
        candidate = promote_detection_result(
            candidate_id="evt_immutable",
            observation=observation,
            detection_result=_result(observation),
            lineage=_lineage(),
        )

        with self.assertRaises(FrozenInstanceError):
            candidate.candidate_id = "evt_changed"
        with self.assertRaises(FrozenInstanceError):
            candidate.lineage.run_id = "run_changed"

    def test_non_triggered_or_inconsistent_results_cannot_be_promoted(self) -> None:
        observation = _observation()
        common = {
            "candidate_id": "evt_invalid",
            "observation": observation,
            "lineage": _lineage(),
        }

        with self.assertRaisesRegex(ValueError, "triggered"):
            promote_detection_result(
                **common,
                detection_result=_result(observation, triggered=False),
            )
        with self.assertRaisesRegex(ValueError, "quality"):
            promote_detection_result(
                **common,
                detection_result=_result(observation, quality="degraded_other"),
            )
        mismatched_signal = DetectionResult(
            detector_id="xiao_ema",
            signal_id="another_signal",
            observation_time_utc=observation.observation_time_utc,
            triggered=True,
            quality=observation.quality,
        )
        with self.assertRaisesRegex(ValueError, "signal_id"):
            promote_detection_result(
                **common,
                detection_result=mismatched_signal,
            )

    def test_quality_is_carried_without_candidate_or_detector_reinterpretation(self) -> None:
        observation = _observation(quality="degraded_missing_author_id")
        result = _result(observation)

        candidate = promote_detection_result(
            candidate_id="evt_degraded",
            observation=observation,
            detection_result=result,
            lineage=_lineage(),
        )

        self.assertEqual(candidate.quality, "degraded_missing_author_id")
        self.assertIs(candidate.observation, observation)
        self.assertIs(candidate.detection_result, result)

    def test_lifecycle_is_optional_and_closed_time_is_validated(self) -> None:
        observation = _observation()
        result = _result(observation)

        point_candidate = promote_detection_result(
            candidate_id="dfe_point",
            observation=observation,
            detection_result=result,
            lineage=_lineage(),
            lifecycle_state="point",
        )
        self.assertEqual(point_candidate.lifecycle_state, "point")
        self.assertIsNone(point_candidate.closed_at_utc)

        with self.assertRaisesRegex(ValueError, "lifecycle_state"):
            promote_detection_result(
                candidate_id="evt_invalid_lifecycle",
                observation=observation,
                detection_result=result,
                lineage=_lineage(),
                lifecycle_state="cooldown",
            )
        with self.assertRaisesRegex(ValueError, "closed_at_utc"):
            promote_detection_result(
                candidate_id="evt_invalid_close",
                observation=observation,
                detection_result=result,
                lineage=_lineage(),
                lifecycle_state="open",
                closed_at_utc=observation.observation_time_utc + timedelta(minutes=3),
            )

    def test_candidate_rejects_evidence_after_detection_time(self) -> None:
        observation = _observation()

        with self.assertRaisesRegex(ValueError, "must not extend"):
            promote_detection_result(
                candidate_id="evt_future_evidence",
                observation=observation,
                detection_result=_result(observation),
                lineage=_lineage(),
                evidence_window_end_utc=(
                    observation.observation_time_utc + timedelta(seconds=1)
                ),
            )

    def test_xiao_projection_preserves_completed_trigger_shape(self) -> None:
        observation = _observation()
        result = _result(observation)
        closed_at = observation.observation_time_utc + timedelta(minutes=4)
        cooldown_until = observation.observation_time_utc + timedelta(minutes=3)
        candidate = promote_detection_result(
            candidate_id="evt_existing_xiao",
            observation=observation,
            detection_result=result,
            lineage=_lineage(),
            lifecycle_state="closed",
            closed_at_utc=closed_at,
        )
        legacy_comments = (
            {
                "event_time_utc": observation.observation_time_utc
                + timedelta(seconds=30),
                "text": "after-trigger",
            },
        )

        projected = project_completed_xiao_trigger(
            candidate,
            cooldown_until_utc=cooldown_until,
            comments=legacy_comments,
        )

        self.assertEqual(
            projected,
            {
                "trigger_time": observation.observation_time_utc,
                "cooldown_until": cooldown_until,
                "volume": 53,
                "strength": 2.75,
                "comments": [dict(legacy_comments[0])],
                "closed_at": closed_at,
            },
        )
        self.assertNotIn("candidate_id", projected)
        self.assertNotIn("lineage", projected)


if __name__ == "__main__":
    unittest.main()
