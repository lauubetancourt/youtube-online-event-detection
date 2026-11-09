from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from youtube_pipeline.external_evidence_temporal_audit import (
    AUDIT_MANIFEST_FILE,
    CANDIDATE_SUMMARY_FILE,
    SOURCE_AUDIT_FILE,
    ExternalEvidenceTemporalAuditConfig,
    run_external_evidence_temporal_audit,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _candidate_dir(
    root: Path,
    candidate_id: str,
    *,
    evidence: list[dict],
    used_ids: list[str],
    candidate_type: str = "ACTIVITY_POLARIZATION",
) -> Path:
    target = root / candidate_id
    _write_json(
        target / "candidate_rag_validation_manifest.json",
        {
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "causality": {
                "candidate_time_utc": "2026-01-01T10:00:00+00:00",
                "candidate_created_at_cutoff_utc": "2026-01-01T10:30:00+00:00",
                "evidence_cutoff_utc": "2026-01-01T10:30:00+00:00",
            },
            "g2": {
                "execution_status": "EXECUTED",
                "gate": {"mode": "OFF"},
            },
        },
    )
    _write_jsonl(target / "candidate_g2_external_evidence.jsonl", evidence)
    _write_jsonl(
        target / "candidate_g2_video_results.jsonl",
        [
            {
                "candidate_id": candidate_id,
                "video_id": "video-a",
                "used_external_evidence_ids": used_ids,
                "cited_external_evidence_ids": [],
            }
        ],
    )
    _write_jsonl(
        target / "candidate_validation_results.jsonl",
        [{"candidate_id": candidate_id, "status": "VALIDATED"}],
    )
    return target


class ExternalEvidenceTemporalAuditTests(unittest.TestCase):
    def test_temporal_classification_aggregation_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = _candidate_dir(
                root,
                "candidate-a",
                evidence=[
                    {
                        "external_evidence_id": "negative",
                        "video_id": "video-a",
                        "link": "https://example/negative",
                        "external_source_published_at": "2026-01-01T09:00:00+00:00",
                    },
                    {
                        "external_evidence_id": "equal",
                        "video_id": "video-a",
                        "external_source_published_at": "2026-01-01T10:00:00+00:00",
                    },
                    {
                        "external_evidence_id": "positive",
                        "video_id": "video-b",
                        "external_source_published_at": "2026-01-01T11:00:00+00:00",
                    },
                    {
                        "external_evidence_id": "unknown",
                        "video_id": "video-b",
                        "external_source_published_at": "unknown",
                    },
                ],
                used_ids=["negative", "positive"],
            )
            empty = _candidate_dir(
                root,
                "candidate-empty",
                evidence=[],
                used_ids=[],
                candidate_type="POLARIZATION",
            )
            output = root / "audit"
            artifacts = run_external_evidence_temporal_audit(
                ExternalEvidenceTemporalAuditConfig(
                    output_dir=output.as_posix(),
                    candidate_rag_dirs=(candidate.as_posix(), empty.as_posix()),
                )
            )

            self.assertEqual(artifacts.source_count, 4)
            self.assertEqual(artifacts.candidate_count, 2)
            rows = {row["external_source_id"]: row for row in _read_csv(output / SOURCE_AUDIT_FILE)}
            self.assertEqual(rows["negative"]["external_lag_seconds"], "-3600.0")
            self.assertEqual(rows["negative"]["relative_to_candidate"], "PRE_CANDIDATE")
            self.assertEqual(rows["negative"]["relative_to_cutoff"], "CAUSALLY_AVAILABLE")
            self.assertEqual(rows["equal"]["external_lag_seconds"], "0.0")
            self.assertEqual(rows["equal"]["relative_to_candidate"], "POST_CANDIDATE")
            self.assertEqual(rows["equal"]["relative_to_cutoff"], "CAUSALLY_AVAILABLE")
            self.assertEqual(rows["positive"]["relative_to_candidate"], "POST_CANDIDATE")
            self.assertEqual(rows["positive"]["relative_to_cutoff"], "FUTURE_RELATIVE_TO_CUTOFF")
            self.assertEqual(rows["unknown"]["publication_time_status"], "PUBLICATION_TIME_UNKNOWN")
            self.assertEqual({row["video_id"] for row in rows.values()}, {"video-a", "video-b"})

            summaries = {row["candidate_id"]: row for row in _read_csv(output / CANDIDATE_SUMMARY_FILE)}
            self.assertEqual(summaries["candidate-a"]["total_sources"], "4")
            self.assertEqual(summaries["candidate-a"]["relevant_sources"], "2")
            self.assertEqual(
                summaries["candidate-a"]["counterfactual_causal_g2"],
                "REQUIRES_G2_REEVALUATION",
            )
            self.assertEqual(
                summaries["candidate-empty"]["candidate_temporal_category"],
                "NO_EXTERNAL_EVIDENCE",
            )
            manifest = json.loads((output / AUDIT_MANIFEST_FILE).read_text())
            self.assertEqual(manifest["counts"]["sources"], 4)
            self.assertEqual(manifest["counts"]["known_timestamp"], 3)
            self.assertEqual(manifest["external_lag_seconds"]["median"], 0.0)
            self.assertEqual(manifest["real_serper_requests"], 0)
            self.assertEqual(manifest["real_g2_model_calls"], 0)
            self.assertIn("source_audit_sha256", manifest["artifacts"])

    def test_date_only_keeps_same_day_unknown_and_prior_day_causal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = _candidate_dir(
                root,
                "candidate-date",
                evidence=[
                    {
                        "external_evidence_id": "prior-day",
                        "video_id": "video-a",
                        "published_at": "31/12/2025",
                    },
                    {
                        "external_evidence_id": "same-day",
                        "video_id": "video-a",
                        "published_at": "1/01/2026",
                    },
                ],
                used_ids=["prior-day"],
            )
            output = root / "audit"
            run_external_evidence_temporal_audit(
                ExternalEvidenceTemporalAuditConfig(
                    output_dir=output.as_posix(),
                    candidate_rag_dirs=(candidate.as_posix(),),
                )
            )
            rows = {row["external_source_id"]: row for row in _read_csv(output / SOURCE_AUDIT_FILE)}
            self.assertEqual(rows["prior-day"]["publication_time_status"], "DATE_ONLY")
            self.assertEqual(rows["prior-day"]["relative_to_candidate"], "PRE_CANDIDATE")
            self.assertEqual(rows["prior-day"]["relative_to_cutoff"], "CAUSALLY_AVAILABLE")
            self.assertEqual(rows["same-day"]["relative_to_candidate"], "UNKNOWN_SAME_DAY_DATE_ONLY")
            self.assertEqual(rows["same-day"]["relative_to_cutoff"], "UNKNOWN_SAME_DAY_DATE_ONLY")
            self.assertEqual(rows["same-day"]["external_lag_seconds"], "")

    def test_exact_cutoff_respects_strictly_before_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = _candidate_dir(
                root,
                "candidate-boundary",
                evidence=[
                    {
                        "external_evidence_id": "at-cutoff",
                        "video_id": "video-a",
                        "external_source_published_at": "2026-01-01T10:30:00+00:00",
                    }
                ],
                used_ids=["at-cutoff"],
            )
            output = root / "audit"
            run_external_evidence_temporal_audit(
                ExternalEvidenceTemporalAuditConfig(
                    output_dir=output.as_posix(),
                    candidate_rag_dirs=(candidate.as_posix(),),
                )
            )
            row = _read_csv(output / SOURCE_AUDIT_FILE)[0]
            self.assertEqual(row["relative_to_cutoff"], "FUTURE_RELATIVE_TO_CUTOFF")
            self.assertEqual(row["publication_time_precision"], "EXACT")

    def test_legacy_adapter_does_not_invent_evidence_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "legacy_external.jsonl"
            reports = root / "legacy_reports.jsonl"
            candidates = root / "event_candidates.csv"
            _write_jsonl(
                evidence,
                [
                    {
                        "external_evidence_id": "legacy-source",
                        "event_id": "event-1",
                        "video_id": "video-1",
                        "published_at": "2026-01-01T09:00:00+00:00",
                    }
                ],
            )
            _write_jsonl(
                reports,
                [{"event_id": "event-1", "used_external_evidence_ids": ["legacy-source"]}],
            )
            candidates.write_text(
                "event_id,trigger_time_utc\n"
                "event-1,2026-01-01T10:00:00+00:00\n",
                encoding="utf-8",
            )
            output = root / "audit"
            run_external_evidence_temporal_audit(
                ExternalEvidenceTemporalAuditConfig(
                    output_dir=output.as_posix(),
                    legacy_external_evidence_path=evidence.as_posix(),
                    legacy_event_candidates_path=candidates.as_posix(),
                    legacy_video_reports_path=reports.as_posix(),
                )
            )
            row = _read_csv(output / SOURCE_AUDIT_FILE)[0]
            self.assertEqual(row["relative_to_candidate"], "PRE_CANDIDATE")
            self.assertEqual(row["relative_to_cutoff"], "UNKNOWN_MISSING_EVIDENCE_CUTOFF")
            self.assertEqual(row["identity_mode"], "LEGACY_EVENT_COMPATIBILITY_ALIAS")
            summary = _read_csv(output / CANDIDATE_SUMMARY_FILE)[0]
            self.assertEqual(summary["candidate_temporal_category"], "TEMPORAL_AVAILABILITY_UNKNOWN")

    def test_requires_complete_legacy_input_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "Legacy auditing requires"):
            ExternalEvidenceTemporalAuditConfig(
                output_dir="out",
                legacy_external_evidence_path="external.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
