from __future__ import annotations

import json
from pathlib import Path

import pytest

from youtube_pipeline.rag_generation_g1 import (
    _citation_verification as verify_g1,
    _compact_event_payload,
    _response_schema as g1_schema,
    _schema_errors as g1_schema_errors,
)
from youtube_pipeline.rag_generation_g2 import _schema_errors
from youtube_pipeline.rag_generation_g2_hierarchical import (
    _build_event_summary,
    _build_video_validation_payload,
    _query_response_schema,
    _verify_video_report,
    _video_validation_schema,
    has_sufficient_search_metadata,
)
from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_CONFIRMED,
    VALIDATION_STATUS_VALUES,
    normalize_validation_status,
)


def _context() -> dict:
    return {
        "post_trigger_context_used": False,
        "selected_context_units": [
            {
                "context_unit_id": "unit-1",
                "video_id": "video-1",
                "comment_ids": ["comment-1"],
                "context_text": "texto duplicado",
            }
        ],
        "used_context_comments": [
            {
                "comment_id": "comment-1",
                "video_id": "video-1",
                "event_time_utc": "2026-01-01T00:00:00Z",
                "text": "texto duplicado",
            }
        ],
    }


def _bundle() -> dict:
    context = _context()
    return {
        "event_id": "event-1",
        "video_id": "video-1",
        "video": {"video_id": "video-1", "title": "Título informativo"},
        "associated_context_unit_ids": ["unit-1"],
        "source_comment_ids": ["comment-1"],
        "context_units": context["selected_context_units"],
        "comments": context["used_context_comments"],
        "post_trigger_context_used": False,
    }


def _external() -> list[dict]:
    return [
        {
            "external_evidence_id": "external-1",
            "title": "Noticia relevante",
            "snippet": "Corrobora el fenómeno",
            "source": "Fuente",
            "link": "https://example.test/news",
            "publication_time_precision": "EXACT",
            "temporal_relation_to_cutoff": "CAUSALLY_AVAILABLE",
        }
    ]


def _g2_report(status: str, assessment: str, *, cite_external: bool) -> dict:
    return {
        "validation_status": status,
        "event_interpretation": "external_event" if cite_external else "unclear",
        "external_evidence_assessment": assessment,
        "cited_comment_ids": [],
        "cited_context_unit_ids": [],
        "cited_external_evidence_ids": ["external-1"] if cite_external else [],
    }


def test_new_schemas_allow_only_three_statuses_and_no_runtime_provenance() -> None:
    expected = set(VALIDATION_STATUS_VALUES)
    for schema in (g1_schema(), _video_validation_schema()):
        assert set(schema["properties"]["validation_status"]["enum"]) == expected
        for runtime_field in (
            "model_name",
            "prompt_version",
            "validated_at_utc",
        ):
            assert runtime_field not in schema["properties"]
    assert set(_query_response_schema()["properties"]) == {"query"}


def test_g1_three_state_citation_semantics() -> None:
    context = _context()
    for status in (CONFIRMED, NOT_CONFIRMED):
        verification = verify_g1(
            report={
                "validation_status": status,
                "cited_comment_ids": ["comment-1"],
                "cited_context_unit_ids": [],
            },
            context_payload=context,
        )
        assert verification["valid"]
    assert verify_g1(
        report={
            "validation_status": INSUFFICIENT_EVIDENCE,
            "cited_comment_ids": [],
            "cited_context_unit_ids": [],
        },
        context_payload=context,
    )["valid"]

    minimal = {
        key: (
            "confirmed"
            if key == "validation_status"
            else "unclear"
            if key == "event_interpretation"
            else "low"
            if key == "confidence_label"
            else []
            if key in {"cited_comment_ids", "cited_context_unit_ids", "limitations"}
            else ""
        )
        for key in g1_schema()["required"]
    }
    assert "invalid_validation_status" in g1_schema_errors(minimal)


def test_g2_decisive_states_require_external_evidence() -> None:
    bundle = _bundle()
    confirmed = _verify_video_report(
        report=_g2_report(CONFIRMED, "supports", cite_external=True),
        bundle=bundle,
        external_evidence=_external(),
    )
    assert confirmed["valid"]

    not_confirmed = _verify_video_report(
        report=_g2_report(NOT_CONFIRMED, "contradicts", cite_external=True),
        bundle=bundle,
        external_evidence=_external(),
    )
    assert not_confirmed["valid"]

    insufficient = _verify_video_report(
        report=_g2_report(
            INSUFFICIENT_EVIDENCE,
            "no_external_evidence",
            cite_external=False,
        ),
        bundle=bundle,
        external_evidence=[],
    )
    assert insufficient["valid"]

    invalid = _verify_video_report(
        report=_g2_report(CONFIRMED, "supports", cite_external=False),
        bundle=bundle,
        external_evidence=[],
    )
    assert not invalid["valid"]
    assert "no_external_evidence_requires_insufficient_evidence" in invalid["errors"]


def test_deterministic_g2_synthesis_never_creates_extra_statuses() -> None:
    batch = [
        {
            "video_id": "video-1",
            "video_batch_status": "processed",
        }
    ]
    external = [{"event_id": "event-1", "video_id": "video-1"}]
    for status, expected in (
        (CONFIRMED, CONFIRMED),
        (NOT_CONFIRMED, NOT_CONFIRMED),
        (INSUFFICIENT_EVIDENCE, INSUFFICIENT_EVIDENCE),
        ("partially_confirmed", INSUFFICIENT_EVIDENCE),
    ):
        summary = _build_event_summary(
            event_id="event-1",
            video_reports=[
                {
                    "video_validation_id": "validation-1",
                    "video_id": "video-1",
                    "validation_status": status,
                    "event_interpretation": "external_event",
                    "external_evidence_assessment": (
                        "supports" if expected == CONFIRMED else "contradicts"
                    ),
                }
            ],
            external_evidence=external,
            video_batch_records=batch,
            batch_id="batch-1",
            created_at_utc="2026-01-01T00:00:00Z",
        )
        assert summary["overall_validation_status"] == expected


def test_prompt_payloads_keep_text_once_and_runtime_allowlists_outside_model() -> None:
    compact_g1 = _compact_event_payload({"event_id": "event-1"}, _context())
    assert "context_text" not in compact_g1["selected_context_units"][0]
    assert compact_g1["used_context_comments"][0]["text"] == "texto duplicado"
    assert "used_context_comment_ids" not in compact_g1

    g2_payload = _build_video_validation_payload(
        validation_input={"event_id": "event-1"},
        bundle=_bundle(),
        query_record={
            "query_candidate_id": "query-1",
            "query_text": "Título informativo",
            "query_generation_mode": "DETERMINISTIC_METADATA",
            "metadata_fields_used": ["video_title"],
        },
        external_evidence=_external(),
    )
    assert "context_text" not in g2_payload["internal_evidence"]["context_units"][0]
    assert "source_comment_ids" not in g2_payload["internal_evidence"]
    assert "model_name" not in g2_payload["external_retrieval"]["query"]
    assert _schema_errors({"query": "Título informativo"}, _query_response_schema()) == []


def test_d1_c1_b_persisted_artifacts_are_legacy_readable_without_network() -> None:
    root = Path("outputs/causal_end_to_end/d1_c1_b/candidate_rag")
    candidate_dirs = sorted(root.glob("candidate_*"))
    if not candidate_dirs:
        pytest.skip("Persisted D1-C1-B pilot artifacts are not present.")

    processed_video_units = 0
    metadata_sufficient = 0
    legacy_statuses: list[str] = []
    for candidate_dir in candidate_dirs:
        query_path = candidate_dir / "g2_hierarchical/rag_video_news_queries.jsonl"
        input_path = candidate_dir / "rag_consumer/rag_validation_inputs.jsonl"
        g1_path = candidate_dir / "g1/rag_validation_reports.jsonl"
        if not query_path.exists():
            continue
        query_rows = [
            json.loads(line) for line in query_path.read_text().splitlines() if line
        ]
        processed_video_units += len(query_rows)
        inputs = [
            json.loads(line) for line in input_path.read_text().splitlines() if line
        ]
        event_id = candidate_dir.name
        selected = next(row for row in inputs if row.get("event_id") == event_id)
        videos = {
            row.get("video_id"): row for row in selected.get("associated_videos", [])
        }
        metadata_sufficient += sum(
            has_sufficient_search_metadata(videos[row["video_id"]])
            for row in query_rows
        )
        legacy_statuses.extend(
            json.loads(line)["validation_status"]
            for line in g1_path.read_text().splitlines()
            if line
        )

    assert processed_video_units == 26
    assert metadata_sufficient == 0
    assert all(
        normalize_validation_status(status) in VALIDATION_STATUS_VALUES
        for status in legacy_statuses
    )
