from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from youtube_pipeline.association_candidates import (
    ASSOCIATION_TO_CANDIDATE_V1,
    READY_FOR_EVIDENCE,
    AssociatedEventCandidate,
    EventCandidateProjector,
    load_associated_event_candidates,
    project_detection_association,
    run_event_candidate_fusion,
)
from youtube_pipeline.detection_alerts import ACTIVITY, POLARIZATION, DetectionAlert
from youtube_pipeline.detection_association import (
    END_OF_STREAM,
    DetectionAssociationConfig,
    TemporalDetectionAssociator,
    load_detection_associations,
    run_temporal_detection_association,
)
from youtube_pipeline.event_candidates import EventCandidate


START = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def _alert(alert_id: str, minute: int, family: str) -> DetectionAlert:
    alert_time = START + timedelta(minutes=minute)
    return DetectionAlert(
        alert_id=alert_id,
        detection_result_id=f"result-{alert_id}",
        detection_result_ref=f"results.jsonl#result-{alert_id}",
        run_id="candidate-run",
        config_hash=f"{family.lower()}-config",
        signal_id=(
            "comment_count_event_window_test"
            if family == ACTIVITY
            else "polarization_mec"
        ),
        signal_family=family,
        detector_id="kswin",
        alert_time_utc=alert_time,
        data_cutoff_utc=alert_time + timedelta(seconds=1),
        triggered=True,
        quality="passed",
        support_count=3,
    )


def _association(
    association_type: str,
    *,
    tolerance: str = "5min",
    end_of_stream: bool = False,
):
    if association_type == ACTIVITY:
        alerts = [_alert("a1", 0, ACTIVITY)]
    elif association_type == POLARIZATION:
        alerts = [_alert("p1", 0, POLARIZATION)]
    else:
        alerts = [
            _alert("a1", 0, ACTIVITY),
            _alert("a2", 1, ACTIVITY),
            _alert("p1", 2, POLARIZATION),
            _alert("p2", 3, POLARIZATION),
        ]
    config = DetectionAssociationConfig(
        tolerance=tolerance,
        parameter_status="TEST_ONLY",
    )
    associator = TemporalDetectionAssociator(run_id="candidate-run", config=config)
    cutoff = (
        START + timedelta(minutes=4)
        if end_of_stream
        else START + timedelta(hours=1)
    )
    return associator.advance(
        alerts,
        data_cutoff_utc=cutoff,
        end_of_stream=True,
    )[0]


def _project(association, *, cutoff: datetime | None = None):
    return project_detection_association(
        association,
        config_hash="candidate-config",
        created_at_cutoff_utc=cutoff or START + timedelta(hours=2),
        input_policy_id="causal_minimal_v1",
    )


class AssociatedEventCandidateTests(unittest.TestCase):
    def test_candidate_types_map_directly_from_finalized_associations(self) -> None:
        for association_type in (
            ACTIVITY,
            POLARIZATION,
            "ACTIVITY_POLARIZATION",
        ):
            with self.subTest(association_type=association_type):
                association = _association(association_type)
                candidate = _project(association)
                self.assertIsNotNone(candidate)
                assert candidate is not None
                self.assertEqual(candidate.candidate_type, association_type)
                self.assertEqual(candidate.status, READY_FOR_EVIDENCE)
                self.assertEqual(
                    candidate.activity_alert_ids, association.activity_alert_ids
                )
                self.assertEqual(
                    candidate.polarization_alert_ids,
                    association.polarization_alert_ids,
                )

    def test_projection_copies_association_times_and_parameter_lineage(self) -> None:
        association = _association("ACTIVITY_POLARIZATION")
        candidate = _project(association)
        assert candidate is not None
        self.assertEqual(candidate.start_time_utc, association.start_time_utc)
        self.assertEqual(candidate.end_time_utc, association.end_time_utc)
        self.assertEqual(
            candidate.reference_time_utc, association.reference_time_utc
        )
        self.assertEqual(
            candidate.association_finalized_at_cutoff_utc,
            association.finalized_at_cutoff_utc,
        )
        self.assertEqual(candidate.association_policy_id, association.policy_id)
        self.assertEqual(candidate.association_tolerance, "5min")
        self.assertEqual(candidate.association_parameter_status, "TEST_ONLY")
        self.assertEqual(candidate.projection_policy_id, ASSOCIATION_TO_CANDIDATE_V1)
        self.assertNotIn(
            "evidence_window_start_utc",
            {item.name for item in fields(AssociatedEventCandidate)},
        )

    def test_candidate_identity_is_deterministic_and_not_cutoff_dependent(self) -> None:
        association = _association(ACTIVITY)
        first = _project(association, cutoff=START + timedelta(hours=2))
        second = _project(association, cutoff=START + timedelta(hours=3))
        assert first is not None and second is not None
        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertNotEqual(first.created_at_cutoff_utc, second.created_at_cutoff_utc)

    def test_candidate_type_invariants_reject_incompatible_alert_lists(self) -> None:
        joint = _project(_association("ACTIVITY_POLARIZATION"))
        assert joint is not None
        with self.assertRaisesRegex(ValueError, "candidate_type"):
            replace(joint, candidate_type=ACTIVITY)
        with self.assertRaisesRegex(ValueError, "candidate_type"):
            replace(
                joint,
                candidate_type=POLARIZATION,
                polarization_alert_ids=(),
            )

    def test_projection_requires_causal_finalization_and_ignores_pending(self) -> None:
        association = _association(ACTIVITY)
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            _project(
                association,
                cutoff=association.finalized_at_cutoff_utc - timedelta(seconds=1),
            )

        # A corrupted or future pending envelope cannot cross the projection guard.
        object.__setattr__(association, "status", "PENDING")
        self.assertIsNone(_project(association))

    def test_incremental_projection_is_one_shot_and_policy_is_fixed_per_run(self) -> None:
        projector = EventCandidateProjector(
            run_id="candidate-run",
            config_hash="candidate-config",
            input_policy_id="causal_minimal_v1",
        )
        association = _association(ACTIVITY)
        cutoff = START + timedelta(hours=2)
        self.assertIsNotNone(
            projector.promote(association, created_at_cutoff_utc=cutoff)
        )
        self.assertIsNone(
            projector.promote(
                association,
                created_at_cutoff_utc=cutoff + timedelta(minutes=1),
            )
        )
        self.assertEqual(
            projector.promoted_association_ids,
            frozenset({association.association_id}),
        )
        with self.assertRaisesRegex(ValueError, "remain fixed"):
            projector.promote(
                _association(POLARIZATION, tolerance="10min"),
                created_at_cutoff_utc=cutoff + timedelta(minutes=2),
            )

    def test_multi_alert_and_multi_video_lineage_are_preserved(self) -> None:
        association = _association("ACTIVITY_POLARIZATION")
        candidate = project_detection_association(
            association,
            config_hash="candidate-config",
            created_at_cutoff_utc=START + timedelta(hours=2),
            input_policy_id="causal_minimal_v1",
            association_ref="association.jsonl#joint",
            associated_video_ids=("video-b", "video-a"),
        )
        assert candidate is not None
        self.assertEqual(candidate.activity_alert_ids, ("a1", "a2"))
        self.assertEqual(candidate.polarization_alert_ids, ("p1", "p2"))
        self.assertEqual(candidate.associated_video_ids, ("video-a", "video-b"))
        self.assertEqual(candidate.association_ref, "association.jsonl#joint")

    def test_end_of_stream_association_is_a_valid_candidate(self) -> None:
        association = _association(POLARIZATION, end_of_stream=True)
        self.assertEqual(association.finalization_reason, END_OF_STREAM)
        candidate = _project(association)
        assert candidate is not None
        self.assertEqual(candidate.association_finalization_reason, END_OF_STREAM)

    def test_artifacts_round_trip_and_manifest_exclude_evidence_and_scores(self) -> None:
        associations = [
            _association(ACTIVITY),
            _association(POLARIZATION),
            _association("ACTIVITY_POLARIZATION"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = run_event_candidate_fusion(
                associations=associations,
                run_id="candidate-run",
                config_hash="candidate-config",
                created_at_cutoff_utc=START + timedelta(hours=2),
                output_dir=tmp,
                input_policy_id="causal_minimal_v1",
                association_artifact_ref="association/detection_associations.jsonl",
                associated_video_ids_by_association={
                    associations[2].association_id: ("video-a", "video-b")
                },
            )
            candidates = load_associated_event_candidates(artifacts.candidate_path)
            manifest = json.loads(artifacts.manifest_path.read_text())
        self.assertEqual(len(candidates), 3)
        self.assertEqual(artifacts.activity, 1)
        self.assertEqual(artifacts.polarization, 1)
        self.assertEqual(artifacts.joint, 1)
        self.assertEqual(manifest["association_policies"][0]["tolerance"], "5min")
        self.assertEqual(
            manifest["association_policies"][0]["parameter_status"],
            "TEST_ONLY",
        )
        self.assertFalse(manifest["candidate_projection"]["score_fusion"])
        self.assertFalse(manifest["downstream"]["evidence_retrieved"])
        self.assertFalse(manifest["downstream"]["rag_executed"])
        persisted_fields = candidates[0].to_mapping()
        self.assertNotIn("score", persisted_fields)
        self.assertNotIn("comments", persisted_fields)
        self.assertNotIn("evidence_window_start_utc", persisted_fields)

    def test_association_artifact_round_trip_supplies_candidate_source(self) -> None:
        alerts = [_alert("a1", 0, ACTIVITY), _alert("p1", 2, POLARIZATION)]
        with tempfile.TemporaryDirectory() as tmp:
            association_artifacts = run_temporal_detection_association(
                config=DetectionAssociationConfig(
                    tolerance="5min",
                    parameter_status="TEST_ONLY",
                ),
                alerts=alerts,
                run_id="candidate-run",
                config_hash="association-config",
                data_cutoff_utc=START + timedelta(hours=1),
                output_dir=Path(tmp) / "association",
                input_policy_id="causal_minimal_v1",
            )
            loaded = load_detection_associations(
                association_artifacts.association_path
            )
            candidate_artifacts = run_event_candidate_fusion(
                associations=loaded,
                run_id="candidate-run",
                config_hash="candidate-config",
                created_at_cutoff_utc=START + timedelta(hours=2),
                output_dir=Path(tmp) / "candidates",
                input_policy_id="causal_minimal_v1",
                association_artifact_ref=str(association_artifacts.association_path),
            )
            candidate = load_associated_event_candidates(
                candidate_artifacts.candidate_path
            )[0]
        self.assertEqual(candidate.association_id, loaded[0].association_id)
        self.assertIn(candidate.association_id, candidate.association_ref)
        self.assertEqual(
            set(candidate.activity_alert_ids + candidate.polarization_alert_ids),
            {"a1", "p1"},
        )

    def test_legacy_event_candidate_contract_remains_distinct(self) -> None:
        legacy_fields = {item.name for item in fields(EventCandidate)}
        neutral_fields = {item.name for item in fields(AssociatedEventCandidate)}
        self.assertIn("observation", legacy_fields)
        self.assertIn("detection_result", legacy_fields)
        self.assertIn("evidence_window_start_utc", legacy_fields)
        self.assertNotIn("observation", neutral_fields)
        self.assertNotIn("detection_result", neutral_fields)
        self.assertNotIn("evidence_window_start_utc", neutral_fields)


if __name__ == "__main__":
    unittest.main()
