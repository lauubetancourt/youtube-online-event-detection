"""Deterministic temporal audit for persisted G-2 external evidence.

The audit never performs retrieval or model calls.  It accepts the current
candidate-RAG artifacts and, for historical comparison, an explicit legacy
event adapter.  Missing cutoffs and date-only publication metadata remain
unknown instead of being coerced into causal timestamps.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


AUDIT_POLICY_ID = "external_evidence_temporal_audit_v1"
SOURCE_AUDIT_FILE = "external_evidence_temporal_audit.csv"
CANDIDATE_SUMMARY_FILE = "external_evidence_candidate_summary.csv"
AUDIT_MANIFEST_FILE = "external_evidence_temporal_audit_manifest.json"

CAUSALLY_AVAILABLE = "CAUSALLY_AVAILABLE"
FUTURE_RELATIVE_TO_CUTOFF = "FUTURE_RELATIVE_TO_CUTOFF"
PUBLICATION_TIME_UNKNOWN = "PUBLICATION_TIME_UNKNOWN"
DATE_ONLY = "DATE_ONLY"
KNOWN_TIMESTAMP = "KNOWN_TIMESTAMP"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        output.append(value)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _publication_time(value: object) -> tuple[str, datetime | date | None]:
    if not isinstance(value, str) or not value.strip():
        return PUBLICATION_TIME_UNKNOWN, None
    raw = value.strip()
    has_clock = "T" in raw or ":" in raw
    if has_clock:
        parsed = _utc(raw)
        if parsed is not None:
            return KNOWN_TIMESTAMP, parsed
        return PUBLICATION_TIME_UNKNOWN, None
    for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return DATE_ONLY, datetime.strptime(raw, pattern).date()
        except ValueError:
            pass
    return PUBLICATION_TIME_UNKNOWN, None


def _source_id(record: Mapping[str, Any]) -> str:
    existing = record.get("external_evidence_id") or record.get("external_source_id")
    if existing:
        return str(existing)
    stable = "|".join(
        str(record.get(key) or "") for key in ("link", "title", "source", "rank")
    )
    return "extaudit_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _category(causal: int, future: int, unknown: int, total: int) -> str:
    if total == 0:
        return "NO_EXTERNAL_EVIDENCE"
    if unknown:
        return "TEMPORAL_AVAILABILITY_UNKNOWN"
    if causal and future:
        return "MIXED_CAUSAL_AND_FUTURE_EVIDENCE"
    if causal:
        return "ONLY_CAUSAL_EXTERNAL_EVIDENCE"
    if future:
        return "ONLY_FUTURE_EXTERNAL_EVIDENCE"
    return "TEMPORAL_AVAILABILITY_UNKNOWN"


@dataclass(frozen=True, slots=True)
class ExternalEvidenceTemporalAuditConfig:
    output_dir: str
    candidate_rag_dirs: tuple[str, ...] = ()
    legacy_external_evidence_path: str | None = None
    legacy_event_candidates_path: str | None = None
    legacy_video_reports_path: str | None = None
    legacy_g2_manifest_path: str | None = None

    def __post_init__(self) -> None:
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty.")
        if not self.candidate_rag_dirs and not self.legacy_external_evidence_path:
            raise ValueError("At least one persisted evidence source is required.")
        legacy_values = (
            self.legacy_external_evidence_path,
            self.legacy_event_candidates_path,
            self.legacy_video_reports_path,
        )
        if any(legacy_values) and not all(legacy_values):
            raise ValueError(
                "Legacy auditing requires evidence, event candidates, and video reports."
            )


@dataclass(frozen=True, slots=True)
class ExternalEvidenceTemporalAuditArtifacts:
    output_dir: Path
    source_audit_path: Path
    candidate_summary_path: Path
    manifest_path: Path
    source_count: int
    candidate_count: int


def _used_evidence_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    used: set[str] = set()
    for record in records:
        for field in ("used_external_evidence_ids", "cited_external_evidence_ids"):
            values = record.get(field) or []
            if isinstance(values, list):
                used.update(str(value) for value in values if value)
    return used


def _candidate_inputs(
    config: ExternalEvidenceTemporalAuditConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Path], list[str]]:
    sources: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    input_paths: list[Path] = []
    limitations: list[str] = []

    for raw_dir in config.candidate_rag_dirs:
        root = Path(raw_dir)
        manifest_path = root / "candidate_rag_validation_manifest.json"
        evidence_path = root / "candidate_g2_external_evidence.jsonl"
        video_path = root / "candidate_g2_video_results.jsonl"
        validation_path = root / "candidate_validation_results.jsonl"
        for path in (manifest_path, evidence_path, video_path, validation_path):
            if path.exists():
                input_paths.append(path)
        manifest = _read_json(manifest_path)
        candidate_id = str(manifest["candidate_id"])
        gate = (manifest.get("g2") or {}).get("gate") or {}
        if gate.get("mode") not in (None, "OFF"):
            limitations.append(
                f"{candidate_id}: G2 gate mode was {gate.get('mode')}, not OFF."
            )
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_type": manifest.get("candidate_type") or "UNKNOWN",
            "candidate_reference_time_utc": (manifest.get("causality") or {}).get(
                "candidate_time_utc"
            ),
            "candidate_created_at_cutoff_utc": (
                manifest.get("causality") or {}
            ).get("candidate_created_at_cutoff_utc"),
            "evidence_cutoff_utc": (manifest.get("causality") or {}).get(
                "evidence_cutoff_utc"
            ),
            "g2_status": (manifest.get("g2") or {}).get("execution_status"),
            "identity_mode": "CANDIDATE_ID",
        }
        used = _used_evidence_ids(_read_jsonl(video_path))
        for record in _read_jsonl(evidence_path):
            item = dict(record)
            item["candidate_id"] = candidate_id
            item["candidate_type"] = candidates[candidate_id]["candidate_type"]
            item["source_used_by_g2"] = _source_id(item) in used
            item["identity_mode"] = "CANDIDATE_ID"
            sources.append(item)

    if config.legacy_external_evidence_path:
        evidence_path = Path(config.legacy_external_evidence_path)
        candidate_path = Path(str(config.legacy_event_candidates_path))
        reports_path = Path(str(config.legacy_video_reports_path))
        input_paths.extend((evidence_path, candidate_path, reports_path))
        if config.legacy_g2_manifest_path:
            manifest_path = Path(config.legacy_g2_manifest_path)
            if manifest_path.exists():
                input_paths.append(manifest_path)
        with candidate_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                candidate_id = str(row["event_id"])
                candidates[candidate_id] = {
                    "candidate_id": candidate_id,
                    "candidate_type": "ACTIVITY",
                    "candidate_reference_time_utc": row.get("trigger_time_utc"),
                    "candidate_created_at_cutoff_utc": None,
                    "evidence_cutoff_utc": None,
                    "g2_status": "LEGACY_EXECUTED",
                    "identity_mode": "LEGACY_EVENT_COMPATIBILITY_ALIAS",
                }
        used = _used_evidence_ids(_read_jsonl(reports_path))
        for record in _read_jsonl(evidence_path):
            candidate_id = str(record.get("event_id") or "")
            if not candidate_id or candidate_id not in candidates:
                continue
            item = dict(record)
            item["candidate_id"] = candidate_id
            item["candidate_type"] = "ACTIVITY"
            item["source_used_by_g2"] = _source_id(item) in used
            item["identity_mode"] = "LEGACY_EVENT_COMPATIBILITY_ALIAS"
            sources.append(item)
        limitations.extend(
            [
                "Legacy event_id is exposed only as an explicit candidate_id compatibility alias.",
                "Legacy artifacts do not persist candidate_created_at_cutoff_utc or evidence_cutoff_utc.",
            ]
        )
    return sources, candidates, input_paths, limitations


def _audit_source(
    source: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_reference = _utc(candidate.get("candidate_reference_time_utc"))
    evidence_cutoff = _utc(candidate.get("evidence_cutoff_utc"))
    raw_publication = source.get("external_source_published_at")
    if raw_publication is None:
        raw_publication = source.get("published_at")
    publication_status, publication = _publication_time(raw_publication)
    lag: float | None = None
    relative_candidate = "UNKNOWN"
    relative_cutoff = "UNKNOWN"
    if isinstance(publication, datetime):
        if candidate_reference is not None:
            lag = (publication - candidate_reference).total_seconds()
            relative_candidate = (
                "PRE_CANDIDATE" if publication < candidate_reference else "POST_CANDIDATE"
            )
        if evidence_cutoff is not None:
            relative_cutoff = (
                CAUSALLY_AVAILABLE
                if publication <= evidence_cutoff
                else FUTURE_RELATIVE_TO_CUTOFF
            )
        else:
            relative_cutoff = "UNKNOWN_MISSING_EVIDENCE_CUTOFF"
    elif isinstance(publication, date):
        if candidate_reference is not None:
            if publication < candidate_reference.date():
                relative_candidate = "PRE_CANDIDATE"
            elif publication > candidate_reference.date():
                relative_candidate = "POST_CANDIDATE"
            else:
                relative_candidate = "UNKNOWN_SAME_DAY_DATE_ONLY"
        if evidence_cutoff is None:
            relative_cutoff = "UNKNOWN_MISSING_EVIDENCE_CUTOFF"
        elif publication < evidence_cutoff.date():
            relative_cutoff = CAUSALLY_AVAILABLE
        elif publication > evidence_cutoff.date():
            relative_cutoff = FUTURE_RELATIVE_TO_CUTOFF
        else:
            relative_cutoff = "UNKNOWN_SAME_DAY_DATE_ONLY"
    else:
        relative_candidate = PUBLICATION_TIME_UNKNOWN
        relative_cutoff = PUBLICATION_TIME_UNKNOWN
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate.get("candidate_type"),
        "video_id": source.get("video_id"),
        "candidate_reference_time_utc": candidate.get("candidate_reference_time_utc"),
        "candidate_created_at_cutoff_utc": candidate.get(
            "candidate_created_at_cutoff_utc"
        ),
        "evidence_cutoff_utc": candidate.get("evidence_cutoff_utc"),
        "external_source_id": _source_id(source),
        "external_source_url": source.get("link") or source.get("url"),
        "external_source_published_at": raw_publication,
        "publication_time_status": publication_status,
        "external_lag_seconds": lag,
        "relative_to_candidate": relative_candidate,
        "relative_to_cutoff": relative_cutoff,
        "source_retrieved": True,
        "source_used_by_g2": bool(source.get("source_used_by_g2")),
        "g2_status": candidate.get("g2_status"),
        "identity_mode": source.get("identity_mode") or candidate.get("identity_mode"),
    }


def _candidate_summaries(
    audited: list[dict[str, Any]], candidates: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {key: [] for key in candidates}
    for row in audited:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for candidate_id in sorted(by_candidate):
        rows = by_candidate[candidate_id]
        causal = sum(row["relative_to_cutoff"] == CAUSALLY_AVAILABLE for row in rows)
        future = sum(
            row["relative_to_cutoff"] == FUTURE_RELATIVE_TO_CUTOFF for row in rows
        )
        unknown = len(rows) - causal - future
        relevant = [row for row in rows if row["source_used_by_g2"]]
        relevant_causal = sum(
            row["relative_to_cutoff"] == CAUSALLY_AVAILABLE for row in relevant
        )
        relevant_future = sum(
            row["relative_to_cutoff"] == FUTURE_RELATIVE_TO_CUTOFF for row in relevant
        )
        relevant_unknown = len(relevant) - relevant_causal - relevant_future
        metadata = candidates[candidate_id]
        output.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": metadata.get("candidate_type"),
                "total_sources": len(rows),
                "sources_with_known_time": sum(
                    row["publication_time_status"] == KNOWN_TIMESTAMP for row in rows
                ),
                "sources_with_date_only": sum(
                    row["publication_time_status"] == DATE_ONLY for row in rows
                ),
                "sources_with_unknown_time": sum(
                    row["publication_time_status"] == PUBLICATION_TIME_UNKNOWN
                    for row in rows
                ),
                "pre_candidate_sources": sum(
                    row["relative_to_candidate"] == "PRE_CANDIDATE" for row in rows
                ),
                "post_candidate_sources": sum(
                    row["relative_to_candidate"] == "POST_CANDIDATE" for row in rows
                ),
                "candidate_relation_unknown_sources": sum(
                    row["relative_to_candidate"] not in {"PRE_CANDIDATE", "POST_CANDIDATE"}
                    for row in rows
                ),
                "causally_available_sources": causal,
                "future_relative_to_cutoff_sources": future,
                "causal_relation_unknown_sources": unknown,
                "relevant_sources": len(relevant),
                "relevant_causal_sources": relevant_causal,
                "relevant_future_sources": relevant_future,
                "relevant_unknown_sources": relevant_unknown,
                "candidate_temporal_category": _category(
                    causal, future, unknown, len(rows)
                ),
                "relevant_temporal_category": _category(
                    relevant_causal,
                    relevant_future,
                    relevant_unknown,
                    len(relevant),
                ),
                "counterfactual_causal_g2": (
                    "NO_RELEVANT_EXTERNAL_EVIDENCE"
                    if not relevant
                    else "REQUIRES_G2_REEVALUATION"
                    if relevant_future or relevant_unknown
                    else "ALL_USED_EVIDENCE_CAUSALLY_AVAILABLE"
                ),
                "identity_mode": metadata.get("identity_mode"),
            }
        )
    return output


def run_external_evidence_temporal_audit(
    config: ExternalEvidenceTemporalAuditConfig,
) -> ExternalEvidenceTemporalAuditArtifacts:
    sources, candidates, input_paths, limitations = _candidate_inputs(config)
    audited = [
        _audit_source(source, candidates[str(source["candidate_id"])])
        for source in sources
    ]
    audited.sort(
        key=lambda row: (
            str(row["candidate_id"]),
            str(row.get("video_id") or ""),
            str(row["external_source_id"]),
        )
    )
    summaries = _candidate_summaries(audited, candidates)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / SOURCE_AUDIT_FILE
    summary_path = output_dir / CANDIDATE_SUMMARY_FILE
    manifest_path = output_dir / AUDIT_MANIFEST_FILE
    source_fields = [
        "candidate_id",
        "candidate_type",
        "video_id",
        "candidate_reference_time_utc",
        "candidate_created_at_cutoff_utc",
        "evidence_cutoff_utc",
        "external_source_id",
        "external_source_url",
        "external_source_published_at",
        "publication_time_status",
        "external_lag_seconds",
        "relative_to_candidate",
        "relative_to_cutoff",
        "source_retrieved",
        "source_used_by_g2",
        "g2_status",
        "identity_mode",
    ]
    summary_fields = list(summaries[0]) if summaries else [
        "candidate_id",
        "candidate_type",
        "total_sources",
    ]
    _write_csv(source_path, audited, source_fields)
    _write_csv(summary_path, summaries, summary_fields)
    lags = [
        float(row["external_lag_seconds"])
        for row in audited
        if row["external_lag_seconds"] is not None
    ]
    input_records = [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in dict.fromkeys(input_paths)
    ]
    manifest: dict[str, Any] = {
        "audit_policy_id": AUDIT_POLICY_ID,
        "status": "executed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "g2_gate_mode_required_for_unbiased_new_samples": "OFF",
        "current_g2_temporal_policy": {
            "anchor": "candidate/event reference date",
            "window_rule": "calendar date +/- configured search_days_before/after",
            "candidate_route_boundary": "published_at strictly before evidence_cutoff_utc",
            "legacy_route_boundary": "no persisted causal cutoff",
        },
        "counts": {
            "candidates": len(summaries),
            "sources": len(audited),
            "relevant_sources": sum(row["source_used_by_g2"] for row in audited),
            "relevant_pre_candidate": sum(
                row["source_used_by_g2"]
                and row["relative_to_candidate"] == "PRE_CANDIDATE"
                for row in audited
            ),
            "relevant_post_candidate": sum(
                row["source_used_by_g2"]
                and row["relative_to_candidate"] == "POST_CANDIDATE"
                for row in audited
            ),
            "relevant_candidate_relation_unknown": sum(
                row["source_used_by_g2"]
                and row["relative_to_candidate"]
                not in {"PRE_CANDIDATE", "POST_CANDIDATE"}
                for row in audited
            ),
            "relevant_causally_available": sum(
                row["source_used_by_g2"]
                and row["relative_to_cutoff"] == CAUSALLY_AVAILABLE
                for row in audited
            ),
            "relevant_future_relative_to_cutoff": sum(
                row["source_used_by_g2"]
                and row["relative_to_cutoff"] == FUTURE_RELATIVE_TO_CUTOFF
                for row in audited
            ),
            "relevant_causal_relation_unknown": sum(
                row["source_used_by_g2"]
                and row["relative_to_cutoff"]
                not in {CAUSALLY_AVAILABLE, FUTURE_RELATIVE_TO_CUTOFF}
                for row in audited
            ),
            "known_timestamp": sum(
                row["publication_time_status"] == KNOWN_TIMESTAMP for row in audited
            ),
            "date_only": sum(
                row["publication_time_status"] == DATE_ONLY for row in audited
            ),
            "publication_time_unknown": sum(
                row["publication_time_status"] == PUBLICATION_TIME_UNKNOWN
                for row in audited
            ),
            "pre_candidate": sum(
                row["relative_to_candidate"] == "PRE_CANDIDATE" for row in audited
            ),
            "post_candidate": sum(
                row["relative_to_candidate"] == "POST_CANDIDATE" for row in audited
            ),
            "candidate_relation_unknown": sum(
                row["relative_to_candidate"] not in {"PRE_CANDIDATE", "POST_CANDIDATE"}
                for row in audited
            ),
            "causally_available": sum(
                row["relative_to_cutoff"] == CAUSALLY_AVAILABLE for row in audited
            ),
            "future_relative_to_cutoff": sum(
                row["relative_to_cutoff"] == FUTURE_RELATIVE_TO_CUTOFF for row in audited
            ),
            "causal_relation_unknown": sum(
                row["relative_to_cutoff"]
                not in {CAUSALLY_AVAILABLE, FUTURE_RELATIVE_TO_CUTOFF}
                for row in audited
            ),
        },
        "external_lag_seconds": {
            "count": len(lags),
            "min": min(lags) if lags else None,
            "p10": _quantile(lags, 0.10),
            "p25": _quantile(lags, 0.25),
            "median": _quantile(lags, 0.50),
            "p75": _quantile(lags, 0.75),
            "p90": _quantile(lags, 0.90),
            "max": max(lags) if lags else None,
        },
        "candidate_temporal_categories": {
            category: sum(
                row["candidate_temporal_category"] == category for row in summaries
            )
            for category in (
                "ONLY_CAUSAL_EXTERNAL_EVIDENCE",
                "MIXED_CAUSAL_AND_FUTURE_EVIDENCE",
                "ONLY_FUTURE_EXTERNAL_EVIDENCE",
                "NO_EXTERNAL_EVIDENCE",
                "TEMPORAL_AVAILABILITY_UNKNOWN",
            )
        },
        "candidate_sample": [row["candidate_id"] for row in summaries],
        "candidates_by_type": {
            candidate_type: sum(
                row["candidate_type"] == candidate_type for row in summaries
            )
            for candidate_type in sorted(
                {str(row["candidate_type"]) for row in summaries}
            )
        },
        "input_artifacts": input_records,
        "real_serper_requests": 0,
        "real_g2_model_calls": 0,
        "reused_external_results": len(audited),
        "limitations": sorted(set(limitations + [
            "Retrospective publication date does not prove historical search-index availability.",
            "Date-only publication metadata cannot support second-level lag or same-day ordering.",
            "Removing future evidence may change G-2 output and requires model reevaluation.",
        ])),
        "artifacts": {
            "source_audit": source_path.as_posix(),
            "candidate_summary": summary_path.as_posix(),
        },
    }
    _write_json(manifest_path, manifest)
    manifest["artifacts"]["source_audit_sha256"] = _sha256(source_path)
    manifest["artifacts"]["candidate_summary_sha256"] = _sha256(summary_path)
    _write_json(manifest_path, manifest)
    return ExternalEvidenceTemporalAuditArtifacts(
        output_dir=output_dir,
        source_audit_path=source_path,
        candidate_summary_path=summary_path,
        manifest_path=manifest_path,
        source_count=len(audited),
        candidate_count=len(summaries),
    )


__all__ = [
    "AUDIT_MANIFEST_FILE",
    "AUDIT_POLICY_ID",
    "CANDIDATE_SUMMARY_FILE",
    "CAUSALLY_AVAILABLE",
    "DATE_ONLY",
    "ExternalEvidenceTemporalAuditArtifacts",
    "ExternalEvidenceTemporalAuditConfig",
    "FUTURE_RELATIVE_TO_CUTOFF",
    "KNOWN_TIMESTAMP",
    "PUBLICATION_TIME_UNKNOWN",
    "SOURCE_AUDIT_FILE",
    "run_external_evidence_temporal_audit",
]
