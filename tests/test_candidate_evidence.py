from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from youtube_pipeline.association_candidates import (
    ASSOCIATION_TO_CANDIDATE_V1,
    READY_FOR_EVIDENCE,
    AssociatedEventCandidate,
)
from youtube_pipeline.candidate_evidence import (
    CANDIDATE_COMMENT_INVENTORY_FILE,
    DELAYED_ENRICHMENT,
    EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
    EVIDENCE_PENDING_HORIZON,
    EVIDENCE_READY,
    IMMEDIATE_CAUSAL,
    NO_EVIDENCE_FOUND,
    CandidateEvidencePolicyConfig,
    build_candidate_evidence,
    derive_candidate_evidence_scope,
)
from youtube_pipeline.configuration import (
    resolve_run_config,
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.rag_consumer import write_rag_consumer_artifacts


UTC = timezone.utc
REFERENCE = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def _candidate(
    candidate_type: str = "ACTIVITY_POLARIZATION",
    *,
    candidate_id: str = "candidate-1",
    videos: tuple[str, ...] = ("v1",),
    reference: datetime = REFERENCE,
    created: datetime | None = None,
) -> AssociatedEventCandidate:
    if candidate_type == "ACTIVITY":
        activity_ids, polarization_ids = ("a1",), ()
    elif candidate_type == "POLARIZATION":
        activity_ids, polarization_ids = (), ("p1",)
    else:
        activity_ids, polarization_ids = ("a1",), ("p1",)
    finalized = reference + timedelta(minutes=5)
    return AssociatedEventCandidate(
        candidate_id=candidate_id,
        run_id="evidence-run",
        config_hash="candidate-config",
        association_id=f"association-{candidate_id}",
        association_ref=f"associations.jsonl#association-{candidate_id}",
        projection_policy_id=ASSOCIATION_TO_CANDIDATE_V1,
        association_policy_id="bounded_temporal_clusters_v1",
        association_tolerance="5min",
        association_parameter_status="TEST_ONLY",
        candidate_type=candidate_type,
        activity_alert_ids=activity_ids,
        polarization_alert_ids=polarization_ids,
        start_time_utc=reference - timedelta(minutes=1),
        end_time_utc=reference,
        reference_time_utc=reference,
        association_finalized_at_cutoff_utc=finalized,
        created_at_cutoff_utc=created or finalized,
        association_finalization_reason="TOLERANCE_ELAPSED",
        status=READY_FOR_EVIDENCE,
        input_policy_id="causal_minimal_v1",
        associated_video_ids=videos,
    )


def _row(
    comment_id: str,
    when: datetime,
    text: str,
    *,
    video: str = "v1",
    author: str | None = "author",
    parent: str | None = None,
    status: str | None = None,
    spam: bool = False,
) -> dict[str, object]:
    return {
        "comment_id": comment_id,
        "video_id": video,
        "author_id": author,
        "event_time_utc": when.isoformat(),
        "text": text,
        "parent_comment_id": parent,
        "is_reply": parent is not None,
        "classification_status": status,
        "is_probable_spam": spam,
    }


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _config(
    source: Path,
    output: Path,
    *,
    mode: str = IMMEDIATE_CAUSAL,
    lookback: str = "2min",
    lookahead: str = "10min",
) -> CandidateEvidencePolicyConfig:
    return CandidateEvidencePolicyConfig(
        source_path=str(source),
        output_dir=str(output),
        lookback=lookback,
        lookahead=lookahead,
        mode=mode,
        parameter_status="TEST_ONLY",
        max_comments_per_context_unit=2,
    )


def _packages(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class CandidateEvidenceTests(unittest.TestCase):
    def test_config_is_strict_immutable_resolved_and_hash_visible(self) -> None:
        payload = {
            "identity": {"run_id": "evidence-config"},
            "rag": {
                "candidate_evidence": {
                    "source_path": "silver.csv",
                    "output_dir": "candidate-evidence",
                    "lookback": "2min",
                    "lookahead": "10min",
                    "mode": "immediate_causal",
                    "parameter_status": "TEST_ONLY",
                }
            },
        }
        config = run_config_from_mapping(payload)
        assert config.rag is not None
        evidence = config.rag.candidate_evidence
        assert evidence is not None
        with self.assertRaises(FrozenInstanceError):
            evidence.lookback = "3min"  # type: ignore[misc]
        serialized = run_config_to_mapping(config)
        self.assertEqual(
            serialized["rag"]["candidate_evidence"]["policy_id"],
            "candidate_evidence_v1",
        )
        changed = json.loads(json.dumps(payload))
        changed["rag"]["candidate_evidence"]["lookback"] = "3min"
        self.assertNotEqual(
            run_config_hash(config),
            run_config_hash(run_config_from_mapping(changed)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run_config(config, base_dir=tmp)
        assert resolved.config.rag is not None
        resolved_evidence = resolved.config.rag.candidate_evidence
        assert resolved_evidence is not None
        self.assertTrue(Path(resolved_evidence.source_path).is_absolute())
        with self.assertRaisesRegex(ValueError, "Unknown rag.candidate_evidence"):
            run_config_from_mapping({
                **payload,
                "rag": {"candidate_evidence": {
                    **payload["rag"]["candidate_evidence"],
                    "unknown": True,
                }},
            })

    def test_immediate_scope_uses_reference_anchor_and_explicit_causal_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root / "silver.csv", root / "out")
            scope = derive_candidate_evidence_scope(_candidate(), config)
        self.assertEqual(scope.anchor_time_utc, REFERENCE)
        self.assertEqual(
            scope.evidence_window_start_utc,
            REFERENCE - timedelta(minutes=2),
        )
        self.assertEqual(
            scope.requested_window_end_utc,
            REFERENCE + timedelta(minutes=10),
        )
        self.assertEqual(
            scope.evidence_window_end_utc,
            REFERENCE + timedelta(minutes=5),
        )
        self.assertEqual(scope.evidence_cutoff_utc, REFERENCE + timedelta(minutes=5))
        self.assertTrue(scope.window_clipped_by_causality)

    def test_delayed_enrichment_waits_for_strictly_closed_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(
                root / "silver.csv",
                root / "out",
                mode=DELAYED_ENRICHMENT,
                lookahead="10min",
            )
            candidate = _candidate()
            pending = derive_candidate_evidence_scope(
                candidate,
                config,
                current_data_cutoff_utc=REFERENCE + timedelta(minutes=10),
            )
            ready = derive_candidate_evidence_scope(
                candidate,
                config,
                current_data_cutoff_utc=REFERENCE + timedelta(
                    minutes=10, microseconds=1
                ),
            )
        self.assertEqual(pending.status, EVIDENCE_PENDING_HORIZON)
        self.assertEqual(ready.status, EVIDENCE_READY)
        self.assertFalse(ready.window_clipped_by_causality)

    def test_window_boundaries_future_exclusion_and_lookback_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            rows = [
                _row("at-start", REFERENCE - timedelta(minutes=2), "start"),
                _row("at-anchor", REFERENCE, "anchor"),
                _row("before-cutoff", REFERENCE + timedelta(minutes=4), "before"),
                _row("at-cutoff", REFERENCE + timedelta(minutes=5), "future"),
                _row("after-cutoff", REFERENCE + timedelta(minutes=6), "future"),
                _row("other-video", REFERENCE, "other", video="v2"),
            ]
            _write_source(source, rows)
            artifacts = build_candidate_evidence(
                candidates=[_candidate()],
                config=_config(source, root / "out"),
            )
            inventory = pd.read_csv(
                artifacts.output_dir / CANDIDATE_COMMENT_INVENTORY_FILE
            )
            lookback_artifacts = build_candidate_evidence(
                candidates=[_candidate(candidate_id="lookback")],
                config=_config(
                    source,
                    root / "lookback",
                    lookahead="0s",
                ),
            )
            lookback_inventory = pd.read_csv(
                lookback_artifacts.output_dir / CANDIDATE_COMMENT_INVENTORY_FILE
            )
        self.assertEqual(
            inventory["comment_id"].tolist(),
            ["at-start", "at-anchor", "before-cutoff"],
        )
        self.assertEqual(
            lookback_inventory["comment_id"].tolist(),
            ["at-start", "at-anchor"],
        )

    def test_multi_video_root_reply_and_parent_outside_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            _write_source(source, [
                _row("outside-parent", REFERENCE - timedelta(minutes=3), "parent"),
                _row("root", REFERENCE - timedelta(minutes=1), "root"),
                _row("reply", REFERENCE, "reply", parent="root"),
                _row("orphan", REFERENCE + timedelta(minutes=1), "orphan", parent="outside-parent"),
                _row("v2-comment", REFERENCE, "second video", video="v2"),
            ])
            artifacts = build_candidate_evidence(
                candidates=[_candidate(videos=("v1", "v2"))],
                config=_config(source, root / "out"),
            )
            inventory = pd.read_csv(
                artifacts.output_dir / CANDIDATE_COMMENT_INVENTORY_FILE
            )
            video_map = pd.read_csv(artifacts.output_dir / "candidate_video_map.csv")
        self.assertEqual(set(video_map["video_id"]), {"v1", "v2"})
        reply = inventory.set_index("comment_id").loc["reply"]
        orphan = inventory.set_index("comment_id").loc["orphan"]
        self.assertTrue(reply["parent_in_inventory"])
        self.assertFalse(orphan["parent_in_inventory"])
        self.assertEqual(orphan["parent_comment_id"], "outside-parent")
        self.assertNotIn("outside-parent", set(inventory["comment_id"]))

    def test_missing_video_lineage_and_empty_scope_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            _write_source(source, [_row("c1", REFERENCE, "text")])
            blocked = build_candidate_evidence(
                candidates=[_candidate(videos=())],
                config=_config(source, root / "blocked"),
            )
            empty = build_candidate_evidence(
                candidates=[_candidate(candidate_id="empty", videos=("missing",))],
                config=_config(source, root / "empty"),
            )
            blocked_package = _packages(blocked.package_path)[0]
            empty_package = _packages(empty.package_path)[0]
        self.assertEqual(
            blocked_package["evidence_status"],
            EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
        )
        self.assertEqual(empty_package["evidence_status"], NO_EVIDENCE_FOUND)

    def test_silver_canonical_source_retains_gold_filtered_semantics_and_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            duplicate = _row(
                "short",
                REFERENCE,
                "🔥",
                status="NOT_APPLICABLE",
                spam=True,
            )
            _write_source(source, [
                duplicate,
                duplicate,
                _row("uncertain", REFERENCE + timedelta(minutes=1), "¿?", status="UNCERTAIN"),
            ])
            artifacts = build_candidate_evidence(
                candidates=[_candidate()],
                config=_config(source, root / "out"),
            )
            inventory = pd.read_csv(
                artifacts.output_dir / CANDIDATE_COMMENT_INVENTORY_FILE
            )
            manifest = json.loads(artifacts.manifest_path.read_text())
        self.assertEqual(inventory["comment_id"].tolist(), ["short", "uncertain"])
        self.assertEqual(inventory.loc[0, "text"], "🔥")
        self.assertEqual(inventory.loc[0, "classification_status"], "NOT_APPLICABLE")
        self.assertEqual(manifest["source"]["exact_duplicates_suppressed"], 1)
        self.assertFalse(manifest["source"]["semantic_filtering"])

    def test_same_policy_applies_to_all_candidate_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            _write_source(source, [_row("c1", REFERENCE, "text")])
            candidates = [
                _candidate("ACTIVITY", candidate_id="a"),
                _candidate("POLARIZATION", candidate_id="p"),
                _candidate("ACTIVITY_POLARIZATION", candidate_id="ap"),
            ]
            artifacts = build_candidate_evidence(
                candidates=candidates,
                config=_config(source, root / "out"),
            )
            packages = _packages(artifacts.package_path)
        self.assertEqual({item["candidate_type"] for item in packages}, {
            "ACTIVITY", "POLARIZATION", "ACTIVITY_POLARIZATION",
        })
        self.assertEqual({item["evidence_policy"]["policy_id"] for item in packages}, {
            "candidate_evidence_v1"
        })

    def test_inventory_and_package_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            _write_source(source, [
                _row("b", REFERENCE, "b"),
                _row("a", REFERENCE, "a"),
            ])
            first = build_candidate_evidence(
                candidates=[_candidate()],
                config=_config(source, root / "first"),
            )
            second = build_candidate_evidence(
                candidates=[_candidate()],
                config=_config(source, root / "second"),
            )
            first_manifest = json.loads(first.manifest_path.read_text())
            second_manifest = json.loads(second.manifest_path.read_text())
            inventory = pd.read_csv(
                first.output_dir / CANDIDATE_COMMENT_INVENTORY_FILE
            )
            first_package_hash = hashlib.sha256(
                first.package_path.read_bytes()
            ).hexdigest()
            second_package_hash = hashlib.sha256(
                second.package_path.read_bytes()
            ).hexdigest()
        self.assertEqual(inventory["comment_id"].tolist(), ["a", "b"])
        self.assertEqual(
            first_manifest["artifact_hashes"]["candidate_evidence_packages_sha256"],
            second_manifest["artifact_hashes"]["candidate_evidence_packages_sha256"],
        )
        self.assertEqual(
            first_package_hash,
            second_package_hash,
        )

    def test_explicit_compatibility_view_is_readable_by_existing_rag_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "silver.csv"
            _write_source(source, [
                _row("root", REFERENCE - timedelta(minutes=1), "root"),
                _row("reply", REFERENCE, "reply", parent="root"),
            ])
            artifacts = build_candidate_evidence(
                candidates=[_candidate()],
                config=_config(source, root / "evidence"),
            )
            result = write_rag_consumer_artifacts(
                sidecars_dir=artifacts.rag_compatibility_dir,
                output_dir=root / "consumer",
                run_id="compatibility-consumer",
            )
            manifest = json.loads(artifacts.manifest_path.read_text())
        self.assertEqual(result["counts"]["event_count"], 1)
        self.assertTrue(result["coverage_checks"]["all_inventory_comments_have_context"])
        self.assertEqual(
            manifest["rag_compatibility"]["strategy"],
            "explicit_candidate_id_to_event_id_alias",
        )
        self.assertFalse(manifest["downstream"]["rag_consumer_executed"])
        self.assertFalse(manifest["downstream"]["g1_executed"])


if __name__ == "__main__":
    unittest.main()
