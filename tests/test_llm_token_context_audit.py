from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.llm_token_context_audit import (
    GATE_AUDIT_FILE,
    MANIFEST_FILE,
    RAG_AUDIT_FILE,
    STANCE_AUDIT_FILE,
    SUMMARY_FILE,
    LlmTokenContextAuditConfig,
    run_llm_token_context_audit,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(value) for value in values) + ("\n" if values else ""),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _raw(event_id: str, record_type: str, total: int) -> dict:
    return {
        "event_id": event_id,
        "record_type": record_type,
        "raw_response": {
            "usage": {
                "input_tokens": total - 10,
                "output_tokens": 10,
                "total_tokens": total,
            }
        },
    }


class LlmTokenContextAuditTests(unittest.TestCase):
    def _fixture(self, root: Path) -> LlmTokenContextAuditConfig:
        stance = root / "stance"
        _write_jsonl(
            stance / "comment_stances.jsonl",
            [
                {
                    "comment_id": "root-reported",
                    "classification_status": "CLASSIFIED",
                    "attempts": 1,
                    "token_usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                    "token_usage_status": "REPORTED",
                },
                {
                    "comment_id": "reply-partial",
                    "parent_comment_id": "root-reported",
                    "classification_status": "UNCERTAIN",
                    "attempts": 2,
                    "token_usage": {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130},
                    "token_usage_status": "PARTIAL",
                },
                {
                    "comment_id": "root-unknown",
                    "classification_status": "NOT_APPLICABLE",
                    "attempts": 1,
                },
            ],
        )
        _write_json(
            stance / "classifier_manifest.json",
            {
                "classifier_run_id": "stance-run",
                "model": "gpt-4.1-2025-04-14",
                "prompt_id": "comment_stance_likert5",
                "prompt_version": "1",
                "prompt_artifact": {"system_instructions": "fixed system"},
                "proposition": {
                    "proposition_id": "topic-a",
                    "proposition_text": "Proposition",
                    "context": "Topic context",
                },
                "token_usage": {
                    "input_tokens": 320,
                    "output_tokens": 20,
                    "total_tokens": 340,
                    "reported_attempts": 2,
                    "provider_attempts": 3,
                },
                "inference_parameters": {"max_completion_tokens": 128},
            },
        )
        comments = root / "comments.csv"
        _write_csv(
            comments,
            [
                {"comment_id": "root-reported", "video_id": "v1", "text": "Root", "reply_to_comment_id": ""},
                {"comment_id": "reply-partial", "video_id": "v1", "text": "Reply", "reply_to_comment_id": "root-reported"},
                {"comment_id": "root-unknown", "video_id": "v2", "text": "Unknown", "reply_to_comment_id": ""},
            ],
            ["comment_id", "video_id", "text", "reply_to_comment_id"],
        )

        inventory = root / "event_comment_inventory.csv"
        inventory_fields = [
            "event_id", "comment_id", "video_id", "root_comment_id", "parent_comment_id",
            "is_reply", "event_time_utc", "trigger_time_utc", "text", "text_clean",
        ]
        _write_csv(
            inventory,
            [
                {"event_id": "c1", "comment_id": "a", "video_id": "v1", "root_comment_id": "a", "parent_comment_id": "", "is_reply": "False", "event_time_utc": "2026-01-01T00:00:00+00:00", "trigger_time_utc": "2026-01-01T00:01:00+00:00", "text": "a", "text_clean": "a"},
                {"event_id": "c1", "comment_id": "b", "video_id": "v2", "root_comment_id": "b", "parent_comment_id": "", "is_reply": "False", "event_time_utc": "2026-01-01T00:00:30+00:00", "trigger_time_utc": "2026-01-01T00:01:00+00:00", "text": "b", "text_clean": "b"},
            ],
            inventory_fields,
        )
        units = root / "rag_context_units.jsonl"
        _write_jsonl(
            units,
            [
                {"event_id": "c1", "context_unit_id": "u1", "video_id": "v1", "context_type": "thread", "comment_ids": ["a"], "comment_count": 1, "context_text": "a"},
                {"event_id": "c1", "context_unit_id": "u2", "video_id": "v2", "context_type": "thread", "comment_ids": ["b"], "comment_count": 1, "context_text": "b"},
            ],
        )
        mapping = root / "context_unit_comment_map.csv"
        _write_csv(
            mapping,
            [
                {"event_id": "c1", "comment_id": "a", "context_unit_id": "u1"},
                {"event_id": "c1", "comment_id": "b", "context_unit_id": "u2"},
            ],
            ["event_id", "comment_id", "context_unit_id"],
        )
        validation_inputs = root / "rag_validation_inputs.jsonl"
        _write_jsonl(validation_inputs, [{"event_id": "c1"}, {"event_id": "c2"}])
        payloads = root / "rag_context_payloads.jsonl"
        _write_jsonl(
            payloads,
            [
                {
                    "event_id": "c1",
                    "selected_context_units": [
                        {"context_unit_id": "u1", "video_id": "v1", "context_type": "thread", "comment_ids": ["a"], "context_text": "a"}
                    ],
                    "used_context_comments": [
                        {"comment_id": "a", "video_id": "v1", "root_comment_id": "a", "text": "a"}
                    ],
                    "post_trigger_context_used": False,
                },
                {"event_id": "c2", "selected_context_units": [], "used_context_comments": [], "post_trigger_context_used": False},
            ],
        )
        g1_raw = root / "g1_raw.jsonl"
        _write_jsonl(g1_raw, [_raw("c1", "g1", 100), _raw("c2", "g1", 200)])
        g1_reports = root / "g1_reports.jsonl"
        _write_jsonl(
            g1_reports,
            [
                {"event_id": "c1", "validation_status": "confirmed"},
                {"event_id": "c2", "validation_status": "not_confirmed"},
            ],
        )
        g1_manifest = root / "g1_manifest.json"
        _write_json(g1_manifest, {"model_name": "gpt-5-mini", "prompt_version": "g1-v1", "context_policy": {"max_approx_tokens": 16000}})
        g2_raw = root / "g2_raw.jsonl"
        _write_jsonl(
            g2_raw,
            [
                _raw("c1", "video_query_generation", 300),
                _raw("c1", "video_validation_generation", 400),
                _raw("c2", "video_query_generation", 500),
                _raw("c2", "video_validation_generation", 600),
            ],
        )
        g2_queries = root / "g2_queries.jsonl"
        _write_jsonl(
            g2_queries,
            [
                {"event_id": "c1", "video_id": "v1"},
                {"event_id": "c1", "video_id": "v2"},
                {"event_id": "c2", "video_id": "v3"},
            ],
        )
        g2_external = root / "g2_external.jsonl"
        _write_jsonl(
            g2_external,
            [
                {"event_id": "c1", "external_evidence_id": "e1", "video_id": "v1"},
                {"event_id": "c2", "external_evidence_id": "e2", "video_id": "v3"},
            ],
        )
        g2_reports = root / "g2_reports.jsonl"
        _write_jsonl(
            g2_reports,
            [
                {"event_id": "c1", "video_id": "v1", "external_evidence_assessment": "supports", "used_external_evidence_ids": ["e1"]},
                {"event_id": "c2", "video_id": "v3", "external_evidence_assessment": "supports", "used_external_evidence_ids": ["e2"]},
            ],
        )
        g2_manifest = root / "g2_manifest.json"
        _write_json(
            g2_manifest,
            {
                "query_model_name": "gpt-5-mini",
                "validation_model_name": "gpt-5-mini",
                "query_prompt_version": "query-v1",
                "validation_prompt_version": "validation-v1",
                "batch_limits": {"max_estimated_tokens_per_event_batch": 80000},
            },
        )
        return LlmTokenContextAuditConfig(
            output_dir=(root / "audit").as_posix(),
            stance_artifact_dir=stance.as_posix(),
            stance_comments_path=comments.as_posix(),
            event_comment_inventory_path=inventory.as_posix(),
            context_units_path=units.as_posix(),
            context_unit_comment_map_path=mapping.as_posix(),
            validation_inputs_path=validation_inputs.as_posix(),
            context_payloads_path=payloads.as_posix(),
            g1_raw_responses_path=g1_raw.as_posix(),
            g1_reports_path=g1_reports.as_posix(),
            g1_manifest_path=g1_manifest.as_posix(),
            g2_raw_responses_path=g2_raw.as_posix(),
            g2_queries_path=g2_queries.as_posix(),
            g2_external_evidence_path=g2_external.as_posix(),
            g2_video_reports_path=g2_reports.as_posix(),
            g2_manifest_path=g2_manifest.as_posix(),
        )

    def test_usage_context_gate_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixture(Path(temporary))
            artifacts = run_llm_token_context_audit(config)
            stance = {row["comment_id"]: row for row in _read_csv(artifacts.stance_audit_path)}
            self.assertEqual(stance["root-reported"]["usage_status"], "REPORTED")
            self.assertEqual(stance["reply-partial"]["usage_status"], "PARTIAL")
            self.assertEqual(stance["root-unknown"]["usage_status"], "UNKNOWN")
            self.assertEqual(stance["reply-partial"]["is_reply"], "True")
            self.assertEqual(stance["reply-partial"]["has_parent_context"], "True")

            rag = {(row["candidate_id"], row["stage"]): row for row in _read_csv(artifacts.rag_audit_path)}
            self.assertEqual(rag[("c1", "G1")]["comment_coverage"], "0.5")
            self.assertEqual(rag[("c1", "G1")]["video_coverage"], "0.5")
            self.assertEqual(rag[("c2", "G1")]["comment_coverage"], "")
            self.assertEqual(rag[("c1", "G2")]["serper_requests"], "2")
            self.assertEqual(rag[("c1", "G2")]["model_calls"], "2")
            self.assertEqual(rag[("c1", "G2")]["total_tokens"], "700")

            gate = _read_csv(artifacts.gate_audit_path)
            skipped = [
                row for row in gate
                if row["candidate_id"] == "c2"
                and row["gate_policy_candidate"] == "candidate_strict_confirmed_or_partial"
            ][0]
            self.assertEqual(skipped["counterfactual_decision"], "SKIP_G2")
            self.assertEqual(skipped["serper_requests_saved"], "1")
            self.assertEqual(skipped["model_calls_saved"], "2")
            self.assertEqual(skipped["tokens_saved"], "1100")
            self.assertEqual(skipped["external_support_filtered"], "True")

            summary = json.loads(artifacts.summary_path.read_text())
            self.assertEqual(summary["g2_usage"]["model_calls"], 4)
            self.assertEqual(
                summary["gate_counterfactual"]["policies"]["candidate_strict_confirmed_or_partial"]["externally_supported_candidates_filtered"],
                1,
            )
            manifest = json.loads(artifacts.manifest_path.read_text())
            self.assertEqual(manifest["real_model_calls"], 0)
            self.assertEqual(manifest["real_serper_requests"], 0)
            self.assertIn("summary_sha256", manifest["artifacts"])

    def test_rerun_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._fixture(Path(temporary))
            artifacts = run_llm_token_context_audit(config)
            before = {
                path.name: path.read_bytes()
                for path in (
                    artifacts.stance_audit_path,
                    artifacts.rag_audit_path,
                    artifacts.gate_audit_path,
                    artifacts.summary_path,
                    artifacts.manifest_path,
                )
            }
            run_llm_token_context_audit(config)
            after = {
                path.name: path.read_bytes()
                for path in (
                    artifacts.stance_audit_path,
                    artifacts.rag_audit_path,
                    artifacts.gate_audit_path,
                    artifacts.summary_path,
                    artifacts.manifest_path,
                )
            }
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
