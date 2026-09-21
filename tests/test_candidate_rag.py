from __future__ import annotations

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
    DELAYED_ENRICHMENT,
    EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
    EVIDENCE_PENDING_HORIZON,
    IMMEDIATE_CAUSAL,
    NO_EVIDENCE_FOUND,
    CandidateEvidencePolicyConfig,
    build_candidate_evidence,
)
from youtube_pipeline.candidate_rag import (
    CANDIDATE_RAG_MANIFEST_FILE,
    CANDIDATE_VALIDATION_RESULTS_FILE,
    VALIDATED,
    CandidateRagValidationConfig,
    run_candidate_rag_validation,
)
from youtube_pipeline.configuration import (
    resolve_run_config,
    run_config_from_mapping,
    run_config_hash,
    run_config_to_mapping,
)
from youtube_pipeline.rag_generation_g1 import (
    RAG_GENERATION_MANIFEST_FILE as G1_MANIFEST,
    RAG_RAW_MODEL_RESPONSES_FILE as G1_RAW,
    RAG_VALIDATION_REPORTS_FILE as G1_REPORTS,
)
from youtube_pipeline.rag_generation_g2_hierarchical import (
    RAG_EVENT_VALIDATION_SUMMARY_FILE,
    RAG_GENERATION_MANIFEST_FILE as G2_MANIFEST,
    RAG_RAW_MODEL_RESPONSES_FILE as G2_RAW,
    RAG_VIDEO_EXTERNAL_EVIDENCE_FILE,
    RAG_VIDEO_NEWS_QUERIES_FILE,
    RAG_VIDEO_VALIDATION_REPORTS_FILE,
    _filter_external_evidence_by_cutoff,
)


UTC = timezone.utc
REFERENCE = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _candidate(
    candidate_type: str,
    *,
    candidate_id: str,
    videos: tuple[str, ...] = ("video-a", "video-b"),
) -> AssociatedEventCandidate:
    activity = ("alert-a",) if candidate_type != "POLARIZATION" else ()
    polarization = ("alert-p",) if candidate_type != "ACTIVITY" else ()
    finalized = REFERENCE + timedelta(minutes=5)
    return AssociatedEventCandidate(
        candidate_id=candidate_id,
        run_id="candidate-run",
        config_hash="candidate-config",
        association_id="association-" + candidate_id,
        association_ref="detection_associations.jsonl#" + candidate_id,
        projection_policy_id=ASSOCIATION_TO_CANDIDATE_V1,
        association_policy_id="bounded_temporal_clusters_v1",
        association_tolerance="5min",
        association_parameter_status="TEST_ONLY",
        candidate_type=candidate_type,
        activity_alert_ids=activity,
        polarization_alert_ids=polarization,
        start_time_utc=REFERENCE - timedelta(minutes=1),
        end_time_utc=REFERENCE,
        reference_time_utc=REFERENCE,
        association_finalized_at_cutoff_utc=finalized,
        created_at_cutoff_utc=finalized,
        association_finalization_reason="TOLERANCE_ELAPSED",
        status=READY_FOR_EVIDENCE,
        input_policy_id="causal_minimal_v1",
        associated_video_ids=videos,
    )


def _source(path: Path, *, comments: bool = True) -> None:
    rows = []
    if comments:
        rows = [
            {
                "comment_id": "a-root",
                "video_id": "video-a",
                "author_id": "author-a",
                "event_time_utc": (REFERENCE - timedelta(minutes=1)).isoformat(),
                "text": "Contexto interno A",
                "parent_comment_id": None,
                "is_reply": False,
            },
            {
                "comment_id": "b-root",
                "video_id": "video-b",
                "author_id": "author-b",
                "event_time_utc": REFERENCE.isoformat(),
                "text": "Contexto interno B",
                "parent_comment_id": None,
                "is_reply": False,
            },
            {
                "comment_id": "future",
                "video_id": "video-a",
                "author_id": "future-author",
                "event_time_utc": (REFERENCE + timedelta(minutes=6)).isoformat(),
                "text": "No debe llegar a RAG",
                "parent_comment_id": None,
                "is_reply": False,
            },
        ]
    pd.DataFrame(
        rows,
        columns=[
            "comment_id",
            "video_id",
            "author_id",
            "event_time_utc",
            "text",
            "parent_comment_id",
            "is_reply",
        ],
    ).to_csv(path, index=False)


def _persist_candidates(path: Path, candidates: list[AssociatedEventCandidate]) -> None:
    _write_jsonl(path, [candidate.to_mapping() for candidate in candidates])


class _MockRag:
    def __init__(self, case: unittest.TestCase) -> None:
        self.case = case
        self.g1_calls = 0
        self.g2_calls = 0

    def g1(self, config) -> dict:
        self.g1_calls += 1
        contexts = _read_jsonl(Path(config.consumer_dir) / "rag_context_payloads.jsonl")
        selected = next(item for item in contexts if item["event_id"] == config.event_id)
        comment_ids = [item["comment_id"] for item in selected["used_context_comments"]]
        self.case.assertEqual(set(comment_ids), {"a-root", "b-root"})
        self.case.assertNotIn("future", comment_ids)
        report = {
            "validation_id": "g1-validation",
            "event_id": config.event_id,
            "validation_status": "confirmed",
            "event_interpretation": "internal_community_reaction",
            "confidence_label": "medium",
            "cited_comment_ids": ["a-root"],
            "cited_context_unit_ids": [],
            "used_context_comment_ids": comment_ids,
            "limitations": [],
        }
        _write_jsonl(Path(config.output_dir) / G1_REPORTS, [report])
        _write_jsonl(
            Path(config.output_dir) / G1_RAW,
            [{"raw_response": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}],
        )
        _write_json(
            Path(config.output_dir) / G1_MANIFEST,
            {"run_id": "mock-g1", "model_name": config.model, "prompt_version": "rag_validation_prompt_v0.1"},
        )
        return {"run_id": "mock-g1", "validation_status": "confirmed"}

    def g2(self, config) -> dict:
        self.g2_calls += 1
        self.case.assertEqual(
            config.external_evidence_cutoff_utc,
            (REFERENCE + timedelta(minutes=5)).isoformat(),
        )
        video_reports = [
            {
                "video_validation_id": "val-a",
                "event_id": config.event_id,
                "video_id": "video-a",
                "validation_status": "partially_confirmed",
                "event_interpretation": "possible_external_event",
                "external_evidence_assessment": "supports",
            },
            {
                "video_validation_id": "val-b",
                "event_id": config.event_id,
                "video_id": "video-b",
                "validation_status": "insufficient_evidence",
                "event_interpretation": "internal_community_reaction",
                "external_evidence_assessment": "insufficient",
            },
        ]
        summary = {
            "event_summary_id": "summary",
            "event_id": config.event_id,
            "videos_evaluated": ["video-a", "video-b"],
            "overall_validation_status": "partially_confirmed",
            "overall_interpretation": "unclear",
            "agreement_between_videos": "mixed",
        }
        evidence = [{
            "external_evidence_id": "external-1",
            "event_id": config.event_id,
            "video_id": "video-a",
            "published_at": (REFERENCE - timedelta(hours=1)).isoformat(),
            "provider": "serper_news",
        }]
        output = Path(config.output_dir)
        _write_jsonl(output / RAG_VIDEO_VALIDATION_REPORTS_FILE, video_reports)
        _write_jsonl(output / RAG_EVENT_VALIDATION_SUMMARY_FILE, [summary])
        _write_jsonl(output / RAG_VIDEO_EXTERNAL_EVIDENCE_FILE, evidence)
        _write_jsonl(output / RAG_VIDEO_NEWS_QUERIES_FILE, [{"event_id": config.event_id}])
        _write_jsonl(
            output / G2_RAW,
            [{"raw_response": {"usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}}],
        )
        _write_json(
            output / G2_MANIFEST,
            {
                "run_id": "mock-g2",
                "query_call_count": 1,
                "validation_call_count": 2,
                "event_summary": summary,
            },
        )
        return {"run_id": "mock-g2", "event_summary": summary}


class CandidateRagTests(unittest.TestCase):
    def _prepare(
        self,
        root: Path,
        candidate: AssociatedEventCandidate,
        *,
        comments: bool = True,
        mode: str = IMMEDIATE_CAUSAL,
        current_cutoff: datetime | None = None,
    ) -> CandidateRagValidationConfig:
        source = root / "silver.csv"
        candidates = root / "event_candidates.jsonl"
        evidence = root / "evidence"
        _source(source, comments=comments)
        _persist_candidates(candidates, [candidate])
        build_candidate_evidence(
            candidates=[candidate],
            config=CandidateEvidencePolicyConfig(
                source_path=str(source),
                output_dir=str(evidence),
                lookback="2min",
                lookahead="10min",
                mode=mode,
                parameter_status="TEST_ONLY",
            ),
            current_data_cutoff_utc=current_cutoff,
        )
        return CandidateRagValidationConfig(
            candidates_path=str(candidates),
            evidence_dir=str(evidence),
            output_dir=str(root / "validation"),
            candidate_id=candidate.candidate_id,
        )

    def test_config_is_immutable_resolved_serialized_and_hashed(self) -> None:
        payload = {
            "identity": {"run_id": "candidate-rag"},
            "rag": {
                "candidate_validation": {
                    "candidates_path": "event_candidates.jsonl",
                    "evidence_dir": "evidence",
                    "output_dir": "validation",
                    "candidate_id": "candidate-1",
                }
            },
        }
        config = run_config_from_mapping(payload)
        assert config.rag is not None and config.rag.candidate_validation is not None
        with self.assertRaises(FrozenInstanceError):
            config.rag.candidate_validation.candidate_id = "other"  # type: ignore[misc]
        mapping = run_config_to_mapping(config)
        self.assertEqual(
            mapping["rag"]["candidate_validation"]["integration_policy_id"],
            "candidate_rag_integration_v1",
        )
        changed = json.loads(json.dumps(payload))
        changed["rag"]["candidate_validation"]["candidate_id"] = "candidate-2"
        self.assertNotEqual(
            run_config_hash(config),
            run_config_hash(run_config_from_mapping(changed)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run_config(config, base_dir=tmp)
            assert resolved.config.rag is not None
            candidate_config = resolved.config.rag.candidate_validation
            assert candidate_config is not None
            self.assertTrue(Path(candidate_config.candidates_path).is_absolute())

        legacy_g2 = run_config_from_mapping({
            "identity": {"run_id": "legacy-g2"},
            "rag": {
                "g2_hierarchical": {
                    "consumer_dir": "consumer",
                    "output_dir": "g2",
                }
            },
        })
        legacy_mapping = run_config_to_mapping(legacy_g2)
        self.assertNotIn(
            "external_evidence_cutoff_utc",
            legacy_mapping["rag"]["g2_hierarchical"],
        )

    def test_end_to_end_mocked_preserves_lineage_causality_and_multi_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _candidate("ACTIVITY_POLARIZATION", candidate_id="joint")
            config = self._prepare(root, candidate)
            mocked = _MockRag(self)
            artifacts = run_candidate_rag_validation(
                config,
                execute=True,
                g1_runner=mocked.g1,
                g2_runner=mocked.g2,
            )
            self.assertEqual(artifacts.status, VALIDATED)
            self.assertEqual(mocked.g1_calls, 1)
            self.assertEqual(mocked.g2_calls, 1)
            result = _read_jsonl(artifacts.validation_results_path)[0]
            self.assertEqual(result["candidate_id"], "joint")
            self.assertEqual(result["candidate_type"], "ACTIVITY_POLARIZATION")
            self.assertEqual(result["g1_result"]["compatibility_event_id"], "joint")
            self.assertEqual(result["g2_result"]["videos_evaluated"], ["video-a", "video-b"])
            self.assertEqual(result["inventory_comment_count"], 2)
            self.assertEqual(result["selected_comment_count"], 2)
            self.assertEqual(result["selected_fraction"], 1.0)
            self.assertEqual(result["internal_future_leak_count"], 0)
            self.assertEqual(result["external_future_leak_count"], 0)
            manifest = json.loads(artifacts.manifest_path.read_text())
            self.assertEqual(manifest["causality"]["future_leak_count"], 0)
            self.assertEqual(manifest["token_usage"]["total_tokens"], 45)
            self.assertEqual(manifest["counts"]["g2_video_results"], 2)

    def test_all_candidate_types_use_same_pipeline(self) -> None:
        for candidate_type in ("ACTIVITY", "POLARIZATION", "ACTIVITY_POLARIZATION"):
            with self.subTest(candidate_type=candidate_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                candidate = _candidate(candidate_type, candidate_id=candidate_type.lower())
                config = self._prepare(root, candidate)
                mocked = _MockRag(self)
                result = run_candidate_rag_validation(
                    config,
                    execute=True,
                    g1_runner=mocked.g1,
                    g2_runner=mocked.g2,
                )
                payload = _read_jsonl(result.validation_results_path)[0]
                self.assertEqual(payload["candidate_type"], candidate_type)
                self.assertEqual(mocked.g1_calls, 1)
                self.assertEqual(mocked.g2_calls, 1)

    def test_non_ready_evidence_statuses_do_not_execute_rag(self) -> None:
        cases = [
            (
                _candidate("ACTIVITY", candidate_id="empty"),
                False,
                IMMEDIATE_CAUSAL,
                None,
                NO_EVIDENCE_FOUND,
            ),
            (
                _candidate("ACTIVITY", candidate_id="missing", videos=()),
                True,
                IMMEDIATE_CAUSAL,
                None,
                EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
            ),
            (
                _candidate("ACTIVITY", candidate_id="pending"),
                True,
                DELAYED_ENRICHMENT,
                REFERENCE + timedelta(minutes=6),
                EVIDENCE_PENDING_HORIZON,
            ),
        ]
        for candidate, comments, mode, cutoff, expected in cases:
            with self.subTest(status=expected), tempfile.TemporaryDirectory() as tmp:
                config = self._prepare(
                    Path(tmp),
                    candidate,
                    comments=comments,
                    mode=mode,
                    current_cutoff=cutoff,
                )
                mocked = _MockRag(self)
                artifacts = run_candidate_rag_validation(
                    config,
                    execute=True,
                    g1_runner=mocked.g1,
                    g2_runner=mocked.g2,
                )
                self.assertEqual(artifacts.status, expected)
                self.assertEqual(mocked.g1_calls, 0)
                self.assertEqual(mocked.g2_calls, 0)
                result = _read_jsonl(artifacts.validation_results_path)[0]
                self.assertIsNone(result["g1_result"])
                self.assertIsNone(result["g2_result"])

    def test_explicit_execute_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._prepare(
                Path(tmp),
                _candidate("ACTIVITY", candidate_id="guard"),
            )
            with self.assertRaisesRegex(RuntimeError, "execute=True"):
                run_candidate_rag_validation(config)

    def test_external_causal_filter_excludes_future_and_unknown_dates(self) -> None:
        records = [
            {"published_at": "2026-01-01T09:00:00+00:00", "id": "past"},
            {"published_at": "2026-01-01T10:05:00+00:00", "id": "at"},
            {"published_at": "2026-01-02T10:00:00+00:00", "id": "future"},
            {"published_at": "2025-12-31", "id": "prior-day"},
            {"published_at": "2026-01-01", "id": "same-day-imprecise"},
            {"published_at": "not-a-date", "id": "unknown"},
        ]
        kept, audit = _filter_external_evidence_by_cutoff(
            records,
            "2026-01-01T10:05:00+00:00",
        )
        self.assertEqual([item["id"] for item in kept], ["past", "prior-day"])
        self.assertEqual(audit["excluded_at_or_after_cutoff"], 2)
        self.assertEqual(audit["excluded_unknown_published_at"], 2)
        legacy, legacy_audit = _filter_external_evidence_by_cutoff(records, None)
        self.assertEqual(legacy, records)
        self.assertEqual(legacy_audit["kept"], 6)


if __name__ == "__main__":
    unittest.main()
