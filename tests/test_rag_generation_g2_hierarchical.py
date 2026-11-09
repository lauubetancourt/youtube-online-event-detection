from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.rag_generation_g2 import _schema_errors
from youtube_pipeline.rag_generation_g2_hierarchical import (
    DETERMINISTIC_METADATA,
    LLM_FALLBACK,
    QUERY_PROMPT_VERSION,
    RagG2HierarchicalConfig,
    _build_video_query_payload,
    _build_video_bundles,
    _classify_video_query,
    _deterministic_metadata_query,
    _query_candidate_id,
    _query_record_from_model,
    _query_response_schema,
    _select_video_batch,
    _verify_video_report,
    has_sufficient_search_metadata,
    plan_rag_g2_hierarchical_dry_run,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _validation_input(event_id: str = "evt_test") -> dict:
    return {
        "event_id": event_id,
        "trigger_time_utc": "2026-02-20T12:00:00+00:00",
        "window_start_utc": "2026-02-20T11:00:00+00:00",
        "window_end_utc": "2026-02-20T12:00:00+00:00",
        "associated_videos": [
            {
                "video_id": "video_a",
                "title": "Alpha entrevista canal",
                "channel_title": "Canal A",
                "inventory_comment_count": 2,
            },
            {
                "video_id": "video_b",
                "title": "Beta noticias",
                "channel_title": "Canal B",
                "inventory_comment_count": 3,
            },
            {
                "video_id": "video_c",
                "title": "Gamma declaraciones",
                "channel_title": "Canal C",
                "inventory_comment_count": 2,
            },
        ],
    }


def _context_payload(event_id: str = "evt_test") -> dict:
    comments = [
        {
            "comment_id": "a1",
            "context_unit_id": "unit_a1",
            "video_id": "video_a",
            "event_time_utc": "2026-02-20T11:20:00+00:00",
            "text": "Comentario alpha uno",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
        {
            "comment_id": "a2",
            "context_unit_id": "unit_a1",
            "video_id": "video_a",
            "event_time_utc": "2026-02-20T11:21:00+00:00",
            "text": "Comentario alpha dos",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
        {
            "comment_id": "b1",
            "context_unit_id": "unit_b1",
            "video_id": "video_b",
            "event_time_utc": "2026-02-20T11:10:00+00:00",
            "text": "Comentario beta uno",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
        {
            "comment_id": "b2",
            "context_unit_id": "unit_b1",
            "video_id": "video_b",
            "event_time_utc": "2026-02-20T11:11:00+00:00",
            "text": "Comentario beta dos",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
        {
            "comment_id": "b3",
            "context_unit_id": "unit_b1",
            "video_id": "video_b",
            "event_time_utc": "2026-02-20T11:12:00+00:00",
            "text": "Comentario beta tres",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
        {
            "comment_id": "c1",
            "context_unit_id": "unit_c1",
            "video_id": "video_c",
            "event_time_utc": "2026-02-20T11:00:00+00:00",
            "text": "Comentario gamma uno",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
        {
            "comment_id": "c2",
            "context_unit_id": "unit_c2",
            "video_id": "video_c",
            "event_time_utc": "2026-02-20T11:01:00+00:00",
            "text": "Comentario gamma dos",
            "temporal_role": "alert_evidence",
            "available_at_trigger": True,
            "relative_to_trigger": "before_trigger",
            "is_post_trigger_context": False,
        },
    ]
    return {
        "event_id": event_id,
        "selected_context_units": [
            {
                "context_unit_id": "unit_a1",
                "context_type": "video",
                "video_id": "video_a",
                "time_start_utc": "2026-02-20T11:20:00+00:00",
                "time_end_utc": "2026-02-20T11:21:00+00:00",
                "comment_count": 2,
                "context_text": "Alpha",
            },
            {
                "context_unit_id": "unit_b1",
                "context_type": "video",
                "video_id": "video_b",
                "time_start_utc": "2026-02-20T11:10:00+00:00",
                "time_end_utc": "2026-02-20T11:12:00+00:00",
                "comment_count": 3,
                "context_text": "Beta",
            },
            {
                "context_unit_id": "unit_c1",
                "context_type": "thread",
                "video_id": "video_c",
                "time_start_utc": "2026-02-20T11:00:00+00:00",
                "time_end_utc": "2026-02-20T11:00:00+00:00",
                "comment_count": 1,
                "context_text": "Gamma uno",
            },
            {
                "context_unit_id": "unit_c2",
                "context_type": "thread",
                "video_id": "video_c",
                "time_start_utc": "2026-02-20T11:01:00+00:00",
                "time_end_utc": "2026-02-20T11:01:00+00:00",
                "comment_count": 1,
                "context_text": "Gamma dos",
            },
        ],
        "used_context_comments": comments,
    }


class RagG2HierarchicalTests(unittest.TestCase):
    def test_video_batch_order_is_deterministic_and_pending_is_not_exclusion(self) -> None:
        validation_input = _validation_input()
        bundles = _build_video_bundles(
            validation_input=validation_input,
            context_payload=_context_payload(),
        )
        config = RagG2HierarchicalConfig(
            consumer_dir="unused",
            output_dir="unused",
            event_id="evt_test",
            max_videos_per_event_batch=2,
        )
        plan = _select_video_batch(
            validation_input=validation_input,
            bundles=bundles,
            config=config,
            batch_id="batch_test",
        )

        self.assertEqual(plan["ordered_video_ids"], ["video_b", "video_c", "video_a"])
        self.assertEqual(
            [bundle["video_id"] for bundle in plan["selected_bundles"]],
            ["video_b", "video_c"],
        )
        self.assertEqual(plan["pending_video_ids"], ["video_a"])
        pending_record = [
            record
            for record in plan["video_batch_records"]
            if record["video_id"] == "video_a"
        ][0]
        self.assertEqual(pending_record["video_batch_status"], "pending_batch")

    def test_query_schema_and_record_have_no_unused_claim_output(self) -> None:
        payload = {"query": "Beta noticias Canal B"}
        self.assertEqual(_schema_errors(payload, _query_response_schema()), [])

        validation_input = _validation_input()
        bundle = _build_video_bundles(
            validation_input=validation_input,
            context_payload=_context_payload(),
        )[1]
        query_payload = _build_video_query_payload(
            validation_input=validation_input,
            bundle=bundle,
        )
        query_status, query_limitations, verification = _classify_video_query(
            payload["query"], query_payload
        )
        self.assertIn(query_status, {"valid", "broad_but_valid"})

        record = _query_record_from_model(
            query_candidate_id=_query_candidate_id(
                "evt_test", "video_b", QUERY_PROMPT_VERSION, "gpt-5-mini"
            ),
            query_result=payload,
            query_status=query_status,
            query_limitations=query_limitations,
            query_scope_verification=verification,
            validation_input=validation_input,
            bundle=bundle,
            search_start="2026-02-19",
            search_end="2026-02-21",
            config=RagG2HierarchicalConfig(
                consumer_dir="unused",
                output_dir="unused",
                event_id="evt_test",
            ),
            generated_at_utc="2026-02-20T12:00:00+00:00",
        )
        self.assertEqual(record["query_generation_mode"], LLM_FALLBACK)
        self.assertEqual(record["query_text"], payload["query"])
        self.assertNotIn("claim_verification_query", record)

    def test_metadata_first_query_uses_title_with_or_without_channel(self) -> None:
        with_channel = {
            "title": "Entrevista a Iván Cepeda en Noticias Caracol",
            "channel_title": "Noticias Caracol",
            "published_at": "2026-04-20T10:00:00Z",
        }
        self.assertTrue(has_sufficient_search_metadata(with_channel))
        query, fields = _deterministic_metadata_query(with_channel)
        self.assertEqual(
            query,
            "Entrevista a Iván Cepeda en Noticias Caracol 2026-04-20",
        )
        self.assertEqual(fields, ["video_title", "video_date"])

        title_only = {"title": "Iván Cepeda debate presidencial"}
        self.assertTrue(has_sufficient_search_metadata(title_only))
        self.assertEqual(
            _deterministic_metadata_query(title_only),
            ("Iván Cepeda debate presidencial", ["video_title"]),
        )

    def test_generic_or_missing_title_requires_llm_fallback(self) -> None:
        for video in ({}, {"title": "Video"}, {"title": "En vivo"}):
            self.assertFalse(has_sufficient_search_metadata(video))

    def test_fallback_rejects_unsupported_event_inference(self) -> None:
        validation_input = _validation_input()
        bundle = _build_video_bundles(
            validation_input=validation_input,
            context_payload=_context_payload(),
        )[0]
        payload = _build_video_query_payload(
            validation_input=validation_input,
            bundle=bundle,
        )
        status, limitations, verification = _classify_video_query(
            "Alpha acto de campaña presidencial", payload
        )
        self.assertEqual(status, "invalid_invented_content")
        self.assertIn("query_contains_unsupported_event_inference", limitations)
        self.assertIn("acto de campaña", verification["unsupported_phrases"])

    def test_video_report_rejects_cross_video_citations(self) -> None:
        validation_input = _validation_input()
        bundles = _build_video_bundles(
            validation_input=validation_input,
            context_payload=_context_payload(),
        )
        video_a_bundle = [bundle for bundle in bundles if bundle["video_id"] == "video_a"][0]
        report = {
            "video_id": "video_a",
            "validation_status": "CONFIRMED",
            "event_interpretation": "internal_community_reaction",
            "cited_comment_ids": ["b1"],
            "cited_context_unit_ids": ["unit_b1"],
            "cited_external_evidence_ids": [],
            "used_context_comment_ids": ["a1", "b1"],
            "used_external_evidence_ids": [],
            "post_trigger_context_used": False,
        }

        verification = _verify_video_report(
            report=report,
            bundle=video_a_bundle,
            external_evidence=[],
        )

        self.assertFalse(verification["valid"])
        self.assertIn("invalid_cited_comment_ids_for_video", verification["errors"])
        self.assertIn("invalid_cited_context_unit_ids_for_video", verification["errors"])
        self.assertIn(
            "no_external_evidence_requires_insufficient_evidence",
            verification["errors"],
        )

    def test_dry_run_reads_fixtures_without_network_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            consumer_dir = root / "consumer"
            output_dir = root / "output"
            consumer_dir.mkdir()
            _write_jsonl(
                consumer_dir / "rag_validation_inputs.jsonl",
                [_validation_input()],
            )
            _write_jsonl(
                consumer_dir / "rag_context_payloads.jsonl",
                [_context_payload()],
            )
            (consumer_dir / "rag_consumer_manifest.json").write_text(
                json.dumps({"run_id": "consumer_fixture"}),
                encoding="utf-8",
            )

            plan = plan_rag_g2_hierarchical_dry_run(
                RagG2HierarchicalConfig(
                    consumer_dir=str(consumer_dir),
                    output_dir=str(output_dir),
                    max_videos_per_event_batch=2,
                )
            )

            self.assertTrue(plan["dry_run"])
            self.assertFalse(plan["network_calls_made"])
            self.assertFalse(plan["openai_calls_made"])
            self.assertFalse(plan["serper_calls_made"])
            self.assertFalse(plan["claim_verification_query_generated"])
            self.assertEqual(plan["summary"]["videos_total"], 3)
            self.assertEqual(plan["summary"]["videos_processable_in_current_batch"], 2)
            self.assertEqual(plan["summary"]["videos_pending"], 1)
            self.assertEqual(plan["summary"]["estimated_llm_calls"], 2)
            self.assertTrue(
                all(
                    row["query_generation"]["query_generation_mode"]
                    == DETERMINISTIC_METADATA
                    for row in plan["events"][0]["video_batch_plan"]
                    if row["video_batch_status"] == "processed"
                )
            )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
