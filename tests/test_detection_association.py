from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

from youtube_pipeline.configuration import (
    canonical_run_config_json,
    run_config_from_mapping,
    run_config_hash,
)
from youtube_pipeline.detection_alerts import (
    ACTIVITY,
    POLARIZATION,
    DetectionAlert,
    load_detection_alerts,
)
from youtube_pipeline.detection_association import (
    END_OF_STREAM,
    TOLERANCE_ELAPSED,
    DetectionAssociationConfig,
    TemporalDetectionAssociator,
    run_temporal_detection_association,
)


START = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def _alert(
    alert_id: str,
    minute: int,
    family: str,
    *,
    signal_id: str | None = None,
    detector_id: str = "kswin",
    config_hash: str = "branch-config",
    available_after_seconds: int = 1,
) -> DetectionAlert:
    alert_time = START + timedelta(minutes=minute)
    return DetectionAlert(
        alert_id=alert_id,
        detection_result_id=f"result-{alert_id}",
        detection_result_ref=f"results.jsonl#result-{alert_id}",
        run_id="association-run",
        config_hash=config_hash,
        signal_id=(
            signal_id
            or (
                "comment_count_event_window_120s_step_30s"
                if family == ACTIVITY
                else "polarization_mec"
            )
        ),
        signal_family=family,
        detector_id=detector_id,
        alert_time_utc=alert_time,
        data_cutoff_utc=alert_time + timedelta(seconds=available_after_seconds),
        triggered=True,
        quality="passed",
        support_count=3,
    )


def _test_config(tolerance: str = "5min") -> DetectionAssociationConfig:
    return DetectionAssociationConfig(
        tolerance=tolerance,
        parameter_status="TEST_ONLY",
    )


class TemporalDetectionAssociationTests(unittest.TestCase):
    def _finalize(self, alerts, *, tolerance: str = "5min"):
        associator = TemporalDetectionAssociator(
            run_id="association-run",
            config=_test_config(tolerance),
        )
        return associator.advance(
            alerts,
            data_cutoff_utc=START + timedelta(hours=1),
            end_of_stream=True,
        )

    def test_same_time_inside_and_exact_boundary_are_joint(self) -> None:
        for polarization_minute in (0, 4, 5):
            with self.subTest(polarization_minute=polarization_minute):
                associations = self._finalize([
                    _alert("a1", 0, ACTIVITY),
                    _alert("p1", polarization_minute, POLARIZATION),
                ])
                self.assertEqual(len(associations), 1)
                self.assertEqual(
                    associations[0].association_type,
                    "ACTIVITY_POLARIZATION",
                )
                self.assertEqual(associations[0].activity_alert_ids, ("a1",))
                self.assertEqual(associations[0].polarization_alert_ids, ("p1",))

    def test_outside_delta_produces_valid_single_family_associations(self) -> None:
        associations = self._finalize([
            _alert("a1", 0, ACTIVITY),
            _alert("p1", 6, POLARIZATION),
        ])
        self.assertEqual(
            [item.association_type for item in associations],
            [ACTIVITY, POLARIZATION],
        )
        self.assertEqual(sum(item.alert_count for item in associations), 2)

    def test_different_grids_associate_only_by_timestamp(self) -> None:
        associations = self._finalize([
            _alert(
                "a30s",
                1,
                ACTIVITY,
                signal_id="comment_count_window_test_step_30s",
            ),
            _alert(
                "p2m",
                4,
                POLARIZATION,
                signal_id="polarization_mec_window_test_step_2min",
            ),
        ])
        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0].association_type, "ACTIVITY_POLARIZATION")

    def test_multiple_alerts_per_family_share_one_bounded_cluster(self) -> None:
        associations = self._finalize([
            _alert("a2", 2, ACTIVITY, detector_id="adwin"),
            _alert("p2", 4, POLARIZATION, signal_id="polarization_emd_pol"),
            _alert("a1", 0, ACTIVITY, detector_id="page_hinkley"),
            _alert("p1", 3, POLARIZATION, signal_id="polarization_mec"),
        ])
        self.assertEqual(len(associations), 1)
        association = associations[0]
        self.assertEqual(association.activity_alert_ids, ("a1", "a2"))
        self.assertEqual(association.polarization_alert_ids, ("p1", "p2"))
        self.assertEqual(association.alert_count, 4)

    def test_chained_alert_does_not_extend_cluster_transitively(self) -> None:
        associations = self._finalize([
            _alert("a1", 0, ACTIVITY),
            _alert("p1", 4, POLARIZATION),
            _alert("a2", 8, ACTIVITY),
        ])
        self.assertEqual(len(associations), 2)
        self.assertEqual(
            associations[0].activity_alert_ids + associations[0].polarization_alert_ids,
            ("a1", "p1"),
        )
        self.assertEqual(associations[1].activity_alert_ids, ("a2",))

    def test_causal_buffer_waits_past_closed_boundary(self) -> None:
        associator = TemporalDetectionAssociator(
            run_id="association-run",
            config=_test_config(),
        )
        self.assertEqual(
            associator.advance(
                [_alert("a1", 0, ACTIVITY)],
                data_cutoff_utc=START + timedelta(minutes=1),
            ),
            (),
        )
        self.assertEqual(
            associator.advance(
                [_alert("p1", 4, POLARIZATION)],
                data_cutoff_utc=START + timedelta(minutes=5),
            ),
            (),
        )
        finalized = associator.advance(
            [],
            data_cutoff_utc=START + timedelta(minutes=5, seconds=1),
        )
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].finalization_reason, TOLERANCE_ELAPSED)
        self.assertEqual(finalized[0].reference_time_utc, START + timedelta(minutes=4))

    def test_end_of_stream_flushes_pending_alert_without_losing_it(self) -> None:
        associator = TemporalDetectionAssociator(
            run_id="association-run",
            config=_test_config(),
        )
        finalized = associator.advance(
            [_alert("p1", 0, POLARIZATION)],
            data_cutoff_utc=START + timedelta(minutes=1),
            end_of_stream=True,
        )
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].association_type, POLARIZATION)
        self.assertEqual(finalized[0].finalization_reason, END_OF_STREAM)
        self.assertEqual(associator.pending_count, 0)
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            associator.advance(
                [], data_cutoff_utc=START + timedelta(minutes=2)
            )

    def test_future_cross_run_duplicates_and_late_alerts_are_rejected(self) -> None:
        associator = TemporalDetectionAssociator(
            run_id="association-run",
            config=_test_config(),
        )
        future = _alert("future", 4, POLARIZATION)
        with self.assertRaisesRegex(ValueError, "not causally available"):
            associator.advance(
                [future], data_cutoff_utc=START + timedelta(minutes=4)
            )
        other_run_payload = _alert("other", 0, ACTIVITY).to_mapping()
        other_run_payload["run_id"] = "other-run"
        with self.assertRaisesRegex(ValueError, "run_id"):
            associator.advance(
                [DetectionAlert.from_mapping(other_run_payload)],
                data_cutoff_utc=START + timedelta(minutes=1),
            )

        associator.advance(
            [_alert("a1", 0, ACTIVITY)],
            data_cutoff_utc=START + timedelta(minutes=6),
        )
        with self.assertRaisesRegex(ValueError, "Late DetectionAlert"):
            associator.advance(
                [_alert("late", 4, POLARIZATION)],
                data_cutoff_utc=START + timedelta(minutes=7),
            )

    def test_identity_is_deterministic_and_contract_is_immutable(self) -> None:
        forward = self._finalize([
            _alert("a1", 0, ACTIVITY),
            _alert("p1", 4, POLARIZATION),
        ])[0]
        reverse = self._finalize([
            _alert("p1", 4, POLARIZATION),
            _alert("a1", 0, ACTIVITY),
        ])[0]
        self.assertEqual(forward.association_id, reverse.association_id)
        self.assertEqual(forward.start_time_utc, START)
        self.assertEqual(forward.end_time_utc, START + timedelta(minutes=4))
        with self.assertRaises(FrozenInstanceError):
            forward.association_type = ACTIVITY

    def test_config_loads_serializes_validates_and_changes_hash(self) -> None:
        without = run_config_from_mapping({
            "identity": {"run_id": "association-config"},
            "association": {
                "policy_id": "bounded_temporal_clusters_v1",
                "tolerance": "5min",
                "parameter_status": "TEST_ONLY",
            },
        })
        changed = run_config_from_mapping({
            "identity": {"run_id": "association-config"},
            "association": {
                "policy_id": "bounded_temporal_clusters_v1",
                "tolerance": "6min",
                "parameter_status": "TEST_ONLY",
            },
        })
        self.assertIsInstance(without.association, DetectionAssociationConfig)
        self.assertIn('"association"', canonical_run_config_json(without))
        self.assertNotEqual(run_config_hash(without), run_config_hash(changed))
        with self.assertRaisesRegex(ValueError, "Unknown association fields"):
            run_config_from_mapping({
                "identity": {"run_id": "association-config"},
                "association": {
                    "policy_id": "bounded_temporal_clusters_v1",
                    "tolerance": "5min",
                    "parameter_status": "TEST_ONLY",
                    "evidence_window": "10min",
                },
            })

    def test_artifact_round_trip_and_manifest_preserve_lineage_without_candidates(self) -> None:
        alerts = [
            _alert("a1", 0, ACTIVITY, config_hash="activity-config"),
            _alert("p1", 4, POLARIZATION, config_hash="polarization-config"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alert_path = root / "input_alerts.jsonl"
            alert_path.write_text(
                "".join(json.dumps(item.to_mapping()) + "\n" for item in alerts),
                encoding="utf-8",
            )
            loaded = load_detection_alerts(alert_path)
            artifacts = run_temporal_detection_association(
                config=_test_config(),
                alerts=loaded,
                run_id="association-run",
                config_hash="association-config",
                data_cutoff_utc=START + timedelta(hours=1),
                output_dir=root / "association",
                input_policy_id="causal_minimal_v1",
                input_alert_artifacts={"combined": str(alert_path)},
            )
            rows = [
                json.loads(line)
                for line in artifacts.association_path.read_text().splitlines()
                if line.strip()
            ]
            manifest = json.loads(artifacts.manifest_path.read_text())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["activity_alert_ids"], ["a1"])
        self.assertEqual(rows[0]["polarization_alert_ids"], ["p1"])
        self.assertNotIn("quality", rows[0])
        self.assertFalse(manifest["downstream"]["event_candidates_created"])
        self.assertEqual(manifest["counts"]["joint"], 1)
        self.assertEqual(
            manifest["input"]["source_config_hashes"],
            ["activity-config", "polarization-config"],
        )
        self.assertFalse(manifest["input"]["config_hash_equality_required"])


if __name__ == "__main__":
    unittest.main()
