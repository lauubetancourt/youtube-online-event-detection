from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from youtube_pipeline.candidate_review import (
    NOT_AVAILABLE,
    build_candidate_review,
    describe_validation_pair,
)
from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _candidate(candidate_id: str, time: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "run_id": "test_run",
        "candidate_type": "ACTIVITY",
        "reference_time_utc": time,
        "created_at_cutoff_utc": time,
        "associated_video_ids": [f"video-{candidate_id}"],
        "association_id": f"association-{candidate_id}",
        "association_policy_id": "policy-v1",
        "association_tolerance": "5min",
        "association_finalization_reason": "TOLERANCE_ELAPSED",
        "activity_alert_ids": [f"alert-{candidate_id}"],
        "polarization_alert_ids": [],
    }


def _complete_validation(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "validation_id": f"candval-{candidate_id}",
        "inventory_comment_count": 3,
        "selected_comment_count": 2,
        "selected_fraction": 2 / 3,
        "internal_future_leak_count": 0,
        "external_future_leak_count": 0,
        "final_synthesis": {
            "internal_validation_status": "confirmed",
            "external_validation_status": "insufficient_evidence",
            "validation_status": "partially_confirmed",
        },
        "g1_result": {
            "validation_id": f"g1-{candidate_id}",
            "validation_status": "confirmed",
            "confidence_label": "medium",
            "event_interpretation": "internal_community_reaction",
            "claim_summary": "Persisted claim summary.",
            "evidence_summary": "Persisted evidence summary.",
            "limitations": ["Persisted G1 limitation."],
            "cited_comment_ids": ["comment-1"],
            "cited_context_unit_ids": ["context-1"],
        },
        "g2_result": {
            "event_summary_id": f"g2-{candidate_id}",
            "overall_validation_status": "insufficient_evidence",
            "overall_interpretation": "unclear",
            "videos_evaluated": [f"video-{candidate_id}"],
            "videos_with_external_evidence": [],
            "videos_without_external_evidence": [f"video-{candidate_id}"],
            "skipped_video_ids": [],
            "failed_video_ids": [],
            "pending_video_ids": [],
            "next_batch_required": False,
            "summary_completeness": "complete",
            "event_batch_status": "complete",
            "agreement_between_videos": "insufficient",
            "global_limitations": ["Persisted G2 limitation."],
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_complete_candidate_and_g1_g2_pair(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [_candidate("candidate-a", "2026-01-01T00:00:00+00:00")],
    )
    _write_jsonl(
        tmp_path / "candidate_validation_results.jsonl",
        [_complete_validation("candidate-a")],
    )
    _write_jsonl(
        tmp_path / "candidate_evidence/candidate_evidence_packages.jsonl",
        [{"candidate_id": "candidate-a", "comment_count": 3, "context_unit_count": 1}],
    )
    artifacts = build_candidate_review(tmp_path)
    row = _read_csv(artifacts.csv_path)[0]
    assert row["internal_validation_status"] == CONFIRMED
    assert row["external_validation_status"] == INSUFFICIENT_EVIDENCE
    assert row["g1_claim_summary"] == "Persisted claim summary."
    assert row["candidate_validation_id"] == "candval-candidate-a"
    assert "Se confirmó un patrón conversacional interno" in row["human_interpretation"]
    assert "no se obtuvo evidencia externa suficiente" in row["human_interpretation"]


def test_legacy_statuses_are_normalized() -> None:
    assert (
        describe_validation_pair(CONFIRMED, INSUFFICIENT_EVIDENCE)
        == "Se confirmó un patrón conversacional interno en YouTube, pero no se obtuvo evidencia externa suficiente para corroborarlo."
    )


def test_partially_confirmed_legacy_artifacts_become_insufficient(
    tmp_path: Path,
) -> None:
    validation = _complete_validation("candidate-a")
    validation["final_synthesis"]["internal_validation_status"] = (
        "partially_confirmed"
    )
    validation["final_synthesis"]["external_validation_status"] = (
        "partially_confirmed"
    )
    validation["g1_result"]["validation_status"] = "partially_confirmed"
    validation["g2_result"]["overall_validation_status"] = "partially_confirmed"
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [_candidate("candidate-a", "2026-01-01T00:00:00+00:00")],
    )
    _write_jsonl(tmp_path / "candidate_validation_results.jsonl", [validation])
    row = _read_csv(build_candidate_review(tmp_path).csv_path)[0]
    assert row["internal_validation_status"] == INSUFFICIENT_EVIDENCE
    assert row["external_validation_status"] == INSUFFICIENT_EVIDENCE


def test_candidate_without_rag_is_reported(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [_candidate("candidate-a", "2026-01-01T00:00:00+00:00")],
    )
    artifacts = build_candidate_review(tmp_path)
    row = _read_csv(artifacts.csv_path)[0]
    assert row["internal_validation_status"] == ""
    assert row["external_validation_status"] == ""
    assert row["internal_execution_status"] == "NOT_EXECUTED"
    assert row["external_completeness_status"] == "PARTIAL"
    assert "NOT EVALUATED" in artifacts.markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("missing_field", "expected_g1", "expected_g2"),
    [
        ("g1_result", "", INSUFFICIENT_EVIDENCE),
        ("g2_result", CONFIRMED, ""),
    ],
)
def test_candidate_with_one_rag_dimension_missing(
    tmp_path: Path,
    missing_field: str,
    expected_g1: str,
    expected_g2: str,
) -> None:
    validation = _complete_validation("candidate-a")
    validation.pop("final_synthesis")
    validation.pop(missing_field)
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [_candidate("candidate-a", "2026-01-01T00:00:00+00:00")],
    )
    _write_jsonl(tmp_path / "candidate_validation_results.jsonl", [validation])
    row = _read_csv(build_candidate_review(tmp_path).csv_path)[0]
    assert row["internal_validation_status"] == expected_g1
    assert row["external_validation_status"] == expected_g2


def test_candidates_are_ordered_by_reference_time_then_id(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [
            _candidate("candidate-b", "2026-01-02T00:00:00+00:00"),
            _candidate("candidate-c", "2026-01-01T00:00:00+00:00"),
            _candidate("candidate-a", "2026-01-01T00:00:00+00:00"),
        ],
    )
    rows = _read_csv(build_candidate_review(tmp_path).csv_path)
    assert [row["candidate_id"] for row in rows] == [
        "candidate-a",
        "candidate-c",
        "candidate-b",
    ]
    assert [row["candidate_index"] for row in rows] == ["1", "2", "3"]


def test_csv_markdown_consistency_and_no_global_fusion(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [_candidate("candidate-a", "2026-01-01T00:00:00+00:00")],
    )
    _write_jsonl(
        tmp_path / "candidate_validation_results.jsonl",
        [_complete_validation("candidate-a")],
    )
    artifacts = build_candidate_review(tmp_path)
    row = _read_csv(artifacts.csv_path)[0]
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert row["candidate_id"] in markdown
    assert f"**Result:** {row['internal_validation_status']}" in markdown
    assert f"**Result:** {row['external_validation_status']}" in markdown
    assert "**Execution:** EXECUTED" in markdown
    assert "**Completeness:** COMPLETE" in markdown
    assert "overall_status" not in row
    assert "Overall:" not in markdown


def test_candidate_review_marks_technical_and_partial_stages_not_evaluated(
    tmp_path: Path,
) -> None:
    validation = _complete_validation("candidate-a")
    validation.pop("final_synthesis")
    validation["g1_result"].update(
        validation_status="insufficient_evidence",
        generation_error=True,
        limitations=["context_limit_exceeded"],
    )
    validation["g2_result"].update(
        summary_completeness="partial",
        event_batch_status="partial_pending_batch",
        next_batch_required=True,
        pending_video_ids=["video-pending"],
    )
    _write_jsonl(
        tmp_path / "event_candidates.jsonl",
        [_candidate("candidate-a", "2026-01-01T00:00:00+00:00")],
    )
    _write_jsonl(tmp_path / "candidate_validation_results.jsonl", [validation])

    artifacts = build_candidate_review(tmp_path)
    row = _read_csv(artifacts.csv_path)[0]
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert row["internal_validation_status"] == ""
    assert row["internal_execution_status"] == "FAILED"
    assert row["internal_completeness_status"] == "PARTIAL"
    assert row["internal_reason_code"] == "CONTEXT_LIMIT_EXCEEDED"
    assert row["external_validation_status"] == ""
    assert row["external_execution_status"] == "EXECUTED"
    assert row["external_completeness_status"] == "PARTIAL"
    assert row["external_reason_code"] == "PENDING_BATCHES"
    assert markdown.count("**Result:** NOT EVALUATED") == 2
    assert "External corroboration could not be fully evaluated." in markdown


def test_missing_required_candidates_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="event_candidates.jsonl"):
        build_candidate_review(tmp_path)
