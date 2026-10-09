"""Deterministic human and tabular views over persisted EventCandidate artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_CONFIRMED,
    normalize_validation_status,
)
from youtube_pipeline.validation_state import (
    COMPLETE,
    ValidationState,
    normalize_g1_state,
    normalize_g2_state,
)


NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_ORDER = (CONFIRMED, INSUFFICIENT_EVIDENCE, NOT_CONFIRMED)

CSV_FIELDS = (
    "candidate_index",
    "candidate_id",
    "run_id",
    "candidate_type",
    "reference_time_utc",
    "created_at_cutoff_utc",
    "associated_video_ids",
    "video_count",
    "association_id",
    "trigger_source",
    "activity_alert_count",
    "polarization_alert_count",
    "activity_signal",
    "activity_detector",
    "polarization_signals",
    "polarization_detectors",
    "association_policy_id",
    "association_tolerance",
    "association_finalization_reason",
    "inventory_comment_count",
    "selected_comment_count",
    "selected_fraction",
    "context_unit_count",
    "videos_with_internal_evidence",
    "internal_validation_status",
    "internal_execution_status",
    "internal_completeness_status",
    "internal_reason_code",
    "g1_confidence",
    "g1_event_interpretation",
    "g1_claim_summary",
    "g1_evidence_summary",
    "g1_limitations",
    "g1_cited_comment_count",
    "g1_cited_context_unit_count",
    "external_validation_status",
    "external_execution_status",
    "external_completeness_status",
    "external_reason_code",
    "g2_overall_interpretation",
    "g2_videos_evaluated_count",
    "g2_videos_evaluated",
    "g2_videos_with_external_evidence_count",
    "g2_videos_with_external_evidence",
    "g2_videos_without_external_evidence_count",
    "g2_videos_without_external_evidence",
    "g2_skipped_video_ids",
    "g2_failed_video_ids",
    "external_source_count",
    "agreement_between_videos",
    "human_interpretation",
    "internal_future_leak_count",
    "external_future_leak_count",
    "future_leak_count",
    "candidate_validation_id",
    "g1_validation_id",
    "g2_event_summary_id",
)


@dataclass(frozen=True)
class CandidateReviewArtifacts:
    output_dir: Path
    csv_path: Path
    markdown_path: Path
    candidate_count: int


def _read_jsonl(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required artifact: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _index_unique(
    rows: Iterable[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(field)
        if key is None:
            continue
        key_text = str(key)
        if key_text in output:
            raise ValueError(f"Duplicate {field}: {key_text}")
        output[key_text] = row
    return output


def _first_jsonl(path: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(path)
    return rows[0] if rows else None


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _stable_strings(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _pipe(values: Iterable[object]) -> str:
    return "|".join(_stable_strings(values))


def _json_list(values: object) -> str:
    return json.dumps(_list(values), ensure_ascii=False, separators=(",", ":"))


def _normalized_status(*values: object) -> str:
    for value in values:
        if value not in (None, ""):
            normalized = normalize_validation_status(value)
            if normalized is None:  # pragma: no cover - guarded by allow_none=False
                break
            return normalized
    return NOT_AVAILABLE


def _value(value: object) -> object:
    return NOT_AVAILABLE if value is None else value


def _count_list(value: object) -> int | str:
    return len(value) if isinstance(value, list) else NOT_AVAILABLE


def _trigger_source(activity_count: int, polarization_count: int) -> str:
    if activity_count and polarization_count:
        return "Activity + Polarization"
    if activity_count:
        return "Activity only"
    if polarization_count:
        return "Polarization only"
    return NOT_AVAILABLE


def describe_validation_pair(
    internal_status: str,
    external_status: str,
) -> str:
    """Describe G1 and G2 without deriving or implying a fused status."""

    internal_text = {
        CONFIRMED: "Se confirmó un patrón conversacional interno en YouTube",
        INSUFFICIENT_EVIDENCE: (
            "La evidencia interna disponible no permite establecer con suficiente "
            "sustento un patrón conversacional"
        ),
        NOT_CONFIRMED: (
            "La evidencia interna disponible no respalda la existencia del patrón "
            "conversacional propuesto"
        ),
        NOT_AVAILABLE: "La validación interna G1 no está disponible",
    }
    external_text = {
        CONFIRMED: "se obtuvo corroboración externa suficiente",
        INSUFFICIENT_EVIDENCE: (
            "no se obtuvo evidencia externa suficiente para corroborarlo"
        ),
        NOT_CONFIRMED: (
            "la evidencia externa disponible no respalda su corroboración"
        ),
        NOT_AVAILABLE: "la corroboración externa G2 no está disponible",
    }
    if internal_status not in internal_text:
        raise ValueError(f"Unknown internal status: {internal_status!r}")
    if external_status not in external_text:
        raise ValueError(f"Unknown external status: {external_status!r}")

    if internal_status == CONFIRMED:
        return f"{internal_text[internal_status]}, pero {external_text[external_status]}." if external_status != CONFIRMED else f"{internal_text[internal_status]} y {external_text[external_status]}."
    if external_status == NOT_AVAILABLE:
        return f"{internal_text[internal_status]}. La corroboración externa G2 no está disponible."
    return f"{internal_text[internal_status]}. Por separado, G2 indica que {external_text[external_status]}."


def _describe_validation_states(
    internal: ValidationState,
    external: ValidationState,
) -> str:
    if internal.completeness_status != COMPLETE:
        internal_text = "Internal validation could not be fully evaluated."
    else:
        internal_text = describe_validation_pair(
            internal.validation_status or NOT_AVAILABLE, NOT_AVAILABLE
        ).split(". La corroboración", 1)[0] + "."
    if external.completeness_status != COMPLETE:
        return f"{internal_text} External corroboration could not be fully evaluated."
    if internal.completeness_status == COMPLETE:
        return describe_validation_pair(
            internal.validation_status or NOT_AVAILABLE,
            external.validation_status or NOT_AVAILABLE,
        )
    external_text = {
        CONFIRMED: "External corroboration was confirmed.",
        INSUFFICIENT_EVIDENCE: "External evidence was insufficient for corroboration.",
        NOT_CONFIRMED: "External evidence did not corroborate the candidate.",
        None: "External corroboration was not evaluated.",
    }
    return f"{internal_text} {external_text[external.validation_status]}"


def _prefixed_state(
    validation: dict[str, Any],
    final_synthesis: dict[str, Any],
    prefix: str,
) -> ValidationState | None:
    values = {
        name: validation.get(f"{prefix}_{name}")
        if f"{prefix}_{name}" in validation
        else final_synthesis.get(f"{prefix}_{name}")
        for name in (
            "execution_status",
            "completeness_status",
            "validation_status",
            "reason_code",
        )
    }
    if values["execution_status"] is None or values["completeness_status"] is None:
        return None
    return ValidationState(
        str(values["execution_status"]),
        str(values["completeness_status"]),
        normalize_validation_status(values["validation_status"], allow_none=True),
        str(values["reason_code"]) if values["reason_code"] not in (None, "") else None,
    )


def _rag_records(
    run_dir: Path,
    candidate_id: str,
    validation: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_dir = run_dir / "candidate_rag" / candidate_id
    g1 = validation.get("g1_result")
    if not isinstance(g1, dict):
        g1 = _first_jsonl(candidate_dir / "candidate_g1_results.jsonl")
    if g1 is None:
        g1 = _first_jsonl(candidate_dir / "g1" / "rag_validation_reports.jsonl")
    g2 = validation.get("g2_result")
    if not isinstance(g2, dict):
        g2 = _first_jsonl(candidate_dir / "candidate_g2_results.jsonl")
    if g2 is None:
        g2 = _first_jsonl(
            candidate_dir / "g2_hierarchical" / "rag_event_validation_summary.jsonl"
        )
    video_rows = _read_jsonl(candidate_dir / "candidate_g2_video_results.jsonl")
    if not video_rows:
        video_rows = _read_jsonl(
            candidate_dir / "g2_hierarchical" / "rag_video_validation_reports.jsonl"
        )
    external_rows = _read_jsonl(candidate_dir / "candidate_g2_external_evidence.jsonl")
    if not external_rows:
        external_rows = _read_jsonl(
            candidate_dir / "g2_hierarchical" / "rag_video_external_evidence.jsonl"
        )
    query_rows = _read_jsonl(
        candidate_dir / "g2_hierarchical" / "rag_video_news_queries.jsonl"
    )
    raw_rows = _read_jsonl(
        candidate_dir / "g2_hierarchical" / "rag_raw_model_responses.jsonl"
    )
    return g1 or {}, g2 or {}, video_rows, external_rows, query_rows, raw_rows


def _external_source_count(
    external_rows: list[dict[str, Any]], video_rows: list[dict[str, Any]]
) -> int | str:
    if external_rows:
        keys = {
            str(row.get("external_evidence_id") or row.get("link") or index)
            for index, row in enumerate(external_rows)
        }
        return len(keys)
    if video_rows:
        return sum(int(row.get("external_evidence_count") or 0) for row in video_rows)
    return NOT_AVAILABLE


def _format_time(value: object) -> str:
    text = str(value or NOT_AVAILABLE)
    if text == NOT_AVAILABLE:
        return text
    if text.endswith("+00:00"):
        return text[:-6].replace("T", " ") + " UTC"
    if text.endswith("Z"):
        return text[:-1].replace("T", " ") + " UTC"
    return text


def _candidate_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _read_jsonl(run_dir / "event_candidates.jsonl", required=True)
    candidates.sort(
        key=lambda row: (
            str(row.get("reference_time_utc") or "9999-12-31T23:59:59Z"),
            str(row.get("candidate_id") or ""),
        )
    )
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("Every event candidate must contain candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("event_candidates.jsonl contains duplicate candidate_id values")

    validations = _index_unique(
        _read_jsonl(run_dir / "candidate_validation_results.jsonl"), "candidate_id"
    )
    evidence_packages = _index_unique(
        _read_jsonl(
            run_dir / "candidate_evidence" / "candidate_evidence_packages.jsonl"
        ),
        "candidate_id",
    )
    alerts = _index_unique(
        _read_jsonl(run_dir / "detection_alerts.jsonl"), "alert_id"
    )
    associations = _index_unique(
        _read_jsonl(run_dir / "detection_associations.jsonl"), "association_id"
    )
    context_videos: dict[str, set[str]] = defaultdict(set)
    for unit in _read_jsonl(
        run_dir / "candidate_evidence" / "candidate_rag_context_units.jsonl"
    ):
        if unit.get("candidate_id") and unit.get("video_id"):
            context_videos[str(unit["candidate_id"])].add(str(unit["video_id"]))

    rows: list[dict[str, Any]] = []
    video_details: dict[str, list[dict[str, Any]]] = {}
    g2_limitations: dict[str, list[str]] = {}
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["candidate_id"])
        validation = validations.get(candidate_id, {})
        package = evidence_packages.get(candidate_id, {})
        association = associations.get(str(candidate.get("association_id") or ""), {})
        g1, g2, g2_video_rows, external_rows, query_rows, raw_rows = _rag_records(
            run_dir, candidate_id, validation
        )
        final_synthesis = validation.get("final_synthesis")
        if not isinstance(final_synthesis, dict):
            final_synthesis = {}

        activity_ids = _list(candidate.get("activity_alert_ids"))
        polarization_ids = _list(candidate.get("polarization_alert_ids"))
        activity_alerts = [alerts[str(value)] for value in activity_ids if str(value) in alerts]
        polarization_alerts = [
            alerts[str(value)] for value in polarization_ids if str(value) in alerts
        ]
        evidence_state = validation.get("evidence_status") or package.get(
            "evidence_status"
        )
        evidence_available = evidence_state != "NO_EVIDENCE_FOUND"
        internal_state = _prefixed_state(
            validation, final_synthesis, "internal"
        ) or normalize_g1_state(g1 or None, evidence_available=evidence_available)
        external_state = _prefixed_state(
            validation, final_synthesis, "external"
        ) or normalize_g2_state(
            g2 or None,
            video_reports=g2_video_rows,
            query_records=query_rows,
            raw_records=raw_rows,
            evidence_available=evidence_available,
            attempted=bool(g2),
        )
        associated_video_ids = _stable_strings(
            _list(candidate.get("associated_video_ids"))
        )
        videos_evaluated = _list(g2.get("videos_evaluated"))
        videos_with_external = _list(g2.get("videos_with_external_evidence"))
        videos_without_external = _list(g2.get("videos_without_external_evidence"))
        candidate_validation_id = validation.get("candidate_validation_id") or validation.get(
            "validation_id"
        )

        row = {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "run_id": candidate.get("run_id") or run_dir.name,
            "candidate_type": _value(candidate.get("candidate_type")),
            "reference_time_utc": _value(candidate.get("reference_time_utc")),
            "created_at_cutoff_utc": _value(
                candidate.get("created_at_cutoff_utc")
            ),
            "associated_video_ids": _pipe(associated_video_ids),
            "video_count": len(associated_video_ids),
            "association_id": _value(candidate.get("association_id")),
            "trigger_source": _trigger_source(len(activity_ids), len(polarization_ids)),
            "activity_alert_count": len(activity_ids),
            "polarization_alert_count": len(polarization_ids),
            "activity_signal": _pipe(alert.get("signal_id") for alert in activity_alerts),
            "activity_detector": _pipe(
                alert.get("detector_id") for alert in activity_alerts
            ),
            "polarization_signals": _pipe(
                alert.get("signal_id") for alert in polarization_alerts
            ),
            "polarization_detectors": _pipe(
                alert.get("detector_id") for alert in polarization_alerts
            ),
            "association_policy_id": _value(
                candidate.get("association_policy_id") or association.get("policy_id")
            ),
            "association_tolerance": _value(
                candidate.get("association_tolerance") or association.get("tolerance")
            ),
            "association_finalization_reason": _value(
                candidate.get("association_finalization_reason")
                or association.get("finalization_reason")
            ),
            "inventory_comment_count": _value(
                validation.get("inventory_comment_count")
                if validation.get("inventory_comment_count") is not None
                else package.get("comment_count")
            ),
            "selected_comment_count": _value(
                validation.get("selected_comment_count")
            ),
            "selected_fraction": _value(validation.get("selected_fraction")),
            "context_unit_count": _value(package.get("context_unit_count")),
            "videos_with_internal_evidence": _pipe(context_videos[candidate_id]),
            "internal_validation_status": internal_state.validation_status,
            "internal_execution_status": internal_state.execution_status,
            "internal_completeness_status": internal_state.completeness_status,
            "internal_reason_code": internal_state.reason_code,
            "g1_confidence": _value(g1.get("confidence_label")),
            "g1_event_interpretation": _value(g1.get("event_interpretation")),
            "g1_claim_summary": _value(g1.get("claim_summary")),
            "g1_evidence_summary": _value(g1.get("evidence_summary")),
            "g1_limitations": _json_list(g1.get("limitations")),
            "g1_cited_comment_count": (
                len(_list(g1.get("cited_comment_ids"))) if g1 else NOT_AVAILABLE
            ),
            "g1_cited_context_unit_count": (
                len(_list(g1.get("cited_context_unit_ids")))
                if g1
                else NOT_AVAILABLE
            ),
            "external_validation_status": external_state.validation_status,
            "external_execution_status": external_state.execution_status,
            "external_completeness_status": external_state.completeness_status,
            "external_reason_code": external_state.reason_code,
            "g2_overall_interpretation": _value(g2.get("overall_interpretation")),
            "g2_videos_evaluated_count": _count_list(g2.get("videos_evaluated")),
            "g2_videos_evaluated": _pipe(videos_evaluated),
            "g2_videos_with_external_evidence_count": _count_list(
                g2.get("videos_with_external_evidence")
            ),
            "g2_videos_with_external_evidence": _pipe(videos_with_external),
            "g2_videos_without_external_evidence_count": _count_list(
                g2.get("videos_without_external_evidence")
            ),
            "g2_videos_without_external_evidence": _pipe(videos_without_external),
            "g2_skipped_video_ids": _pipe(_list(g2.get("skipped_video_ids"))),
            "g2_failed_video_ids": _pipe(_list(g2.get("failed_video_ids"))),
            "external_source_count": _external_source_count(
                external_rows, g2_video_rows
            ),
            "agreement_between_videos": _value(
                g2.get("agreement_between_videos")
            ),
            "human_interpretation": _describe_validation_states(
                internal_state, external_state
            ),
            "internal_future_leak_count": _value(
                validation.get("internal_future_leak_count")
            ),
            "external_future_leak_count": _value(
                validation.get("external_future_leak_count")
            ),
            "future_leak_count": _value(validation.get("future_leak_count")),
            "candidate_validation_id": _value(candidate_validation_id),
            "g1_validation_id": _value(g1.get("validation_id")),
            "g2_event_summary_id": _value(g2.get("event_summary_id")),
        }
        rows.append(row)
        video_details[candidate_id] = sorted(
            g2_video_rows,
            key=lambda item: (
                int(item.get("video_batch_order") or 10**9),
                str(item.get("video_id") or ""),
            ),
        )
        g2_limitations[candidate_id] = [
            str(item) for item in _list(g2.get("global_limitations"))
        ]

    return rows, {
        "video_details": video_details,
        "g2_limitations": g2_limitations,
        "replay_manifest": _read_json(run_dir / "causal_replay_manifest.json"),
    }


def _markdown_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "Not available"


def _markdown_report(
    run_dir: Path, rows: list[dict[str, Any]], details: dict[str, Any]
) -> str:
    run_id = str(rows[0]["run_id"] if rows else run_dir.name)
    type_counts = Counter(str(row["candidate_type"]) for row in rows)
    g1_counts = Counter(str(row["internal_validation_status"]) for row in rows)
    g2_counts = Counter(str(row["external_validation_status"]) for row in rows)
    replay_causality = details.get("replay_manifest", {}).get("causality", {})
    run_future_leak = (
        replay_causality.get("future_leak_count", NOT_AVAILABLE)
        if isinstance(replay_causality, dict)
        else NOT_AVAILABLE
    )
    lines = [
        f"# Candidate Review — {run_id}",
        "",
        f"Candidates: {len(rows)}",
        f"Run: {run_id}",
        "",
        "JSON and JSONL run artifacts remain authoritative. This report is a deterministic derived view.",
        "",
        "## Run summary",
        "",
        f"TOTAL CANDIDATES: {len(rows)}",
        "",
        "BY TYPE",
        f"- ACTIVITY: {type_counts.get('ACTIVITY', 0)}",
        f"- POLARIZATION: {type_counts.get('POLARIZATION', 0)}",
        f"- ACTIVITY_POLARIZATION: {type_counts.get('ACTIVITY_POLARIZATION', 0)}",
        "",
        "Internal validation — G1",
        f"- CONFIRMED: {g1_counts.get(CONFIRMED, 0)}",
        f"- INSUFFICIENT_EVIDENCE: {g1_counts.get(INSUFFICIENT_EVIDENCE, 0)}",
        f"- NOT_CONFIRMED: {g1_counts.get(NOT_CONFIRMED, 0)}",
        f"- NOT EVALUATED: {g1_counts.get('None', 0)}",
        "",
        "External corroboration — G2",
        f"- CONFIRMED: {g2_counts.get(CONFIRMED, 0)}",
        f"- INSUFFICIENT_EVIDENCE: {g2_counts.get(INSUFFICIENT_EVIDENCE, 0)}",
        f"- NOT_CONFIRMED: {g2_counts.get(NOT_CONFIRMED, 0)}",
        f"- NOT EVALUATED: {g2_counts.get('None', 0)}",
        "",
        "CAUSALITY",
        f"- future_leak_count: {run_future_leak}",
        "",
        "CSV list convention: identifier lists use `|`; textual limitations use compact JSON arrays.",
        "",
    ]

    video_details = details["video_details"]
    g2_limitations = details["g2_limitations"]
    for row in rows:
        candidate_id = str(row["candidate_id"])
        associated_videos = str(row["associated_video_ids"]) or NOT_AVAILABLE
        trigger = row["trigger_source"]
        signal_parts = [
            value
            for value in [row["activity_signal"], row["polarization_signals"]]
            if value
        ]
        detector_parts = [
            value
            for value in [row["activity_detector"], row["polarization_detectors"]]
            if value
        ]
        g1_limitations_values = json.loads(str(row["g1_limitations"]))
        combined_limitations = [
            *[str(value) for value in g1_limitations_values],
            *g2_limitations.get(candidate_id, []),
        ]
        lines.extend(
            [
                f"## Candidate {row['candidate_index']}",
                "",
                f"**ID:** {candidate_id}",
                f"**Time:** {_format_time(row['reference_time_utc'])}",
                f"**Type:** {row['candidate_type']}",
                f"**Trigger source:** {trigger}",
                f"**Signal:** {', '.join(signal_parts) if signal_parts else NOT_AVAILABLE}",
                f"**Detector:** {', '.join(detector_parts) if detector_parts else NOT_AVAILABLE}",
                f"**Videos:** {row['video_count']} ({associated_videos})",
                "",
                "**Internal evidence:**",
                f"- Inventory comments: {row['inventory_comment_count']}",
                f"- Selected comments: {row['selected_comment_count']}",
                f"- Context units: {row['context_unit_count']}",
                f"- Videos with internal evidence: {row['videos_with_internal_evidence'] or NOT_AVAILABLE}",
                "",
                "### G1 — Internal validation",
                "",
                f"**Result:** {row['internal_validation_status'] or 'NOT EVALUATED'}",
                f"**Execution:** {row['internal_execution_status']}",
                f"**Completeness:** {row['internal_completeness_status']}",
                *(
                    [f"**Reason:** {row['internal_reason_code']}"]
                    if row["internal_reason_code"]
                    else []
                ),
                f"**Confidence:** {row['g1_confidence']}",
                f"**Interpretation:** {row['g1_event_interpretation']}",
                "",
                "**Finding:**",
                str(row["g1_claim_summary"]),
                "",
                "**Evidence summary:**",
                str(row["g1_evidence_summary"]),
                "",
                "### G2 — External corroboration",
                "",
                f"**Result:** {row['external_validation_status'] or 'NOT EVALUATED'}",
                f"**Execution:** {row['external_execution_status']}",
                f"**Completeness:** {row['external_completeness_status']}",
                *(
                    [f"**Reason:** {row['external_reason_code']}"]
                    if row["external_reason_code"]
                    else []
                ),
                f"**Interpretation:** {row['g2_overall_interpretation']}",
                f"**Videos evaluated:** {row['g2_videos_evaluated_count']}",
                f"**Videos with external evidence:** {row['g2_videos_with_external_evidence_count']}",
                f"**Videos without external evidence:** {row['g2_videos_without_external_evidence_count']}",
                f"**External sources:** {row['external_source_count']}",
                "",
            ]
        )
        candidate_video_rows = video_details.get(candidate_id, [])
        if candidate_video_rows:
            lines.extend(["#### G2 per video", ""])
            for video in candidate_video_rows:
                external_count = int(video.get("external_evidence_count") or 0)
                video_status = _normalized_status(video.get("validation_status"))
                lines.extend(
                    [
                        f"**Video:** {video.get('video_id', NOT_AVAILABLE)}",
                        f"- External evidence: {'YES' if external_count > 0 else 'NO'}",
                        f"- Result: {'NOT EVALUATED' if video_status == NOT_AVAILABLE else video_status}",
                        f"- Execution: {video.get('execution_status', NOT_AVAILABLE)}",
                        f"- Completeness: {video.get('completeness_status', NOT_AVAILABLE)}",
                        f"- Reason: {video.get('reason_code') or NOT_AVAILABLE}",
                        f"- External assessment: {video.get('external_evidence_assessment', NOT_AVAILABLE)}",
                        "",
                    ]
                )
        else:
            lines.extend(["#### G2 per video", "", "Not available", ""])

        lines.extend(
            [
                "### Interpretation",
                "",
                str(row["human_interpretation"]),
                "",
                "### Limitations",
                "",
                _markdown_list(combined_limitations),
                "",
                "### Technical details",
                "",
                f"- Association policy: {row['association_policy_id']}",
                f"- Association tolerance: {row['association_tolerance']}",
                f"- Association finalization: {row['association_finalization_reason']}",
                f"- Internal future leak count: {row['internal_future_leak_count']}",
                f"- External future leak count: {row['external_future_leak_count']}",
                "",
                "### Technical references",
                "",
                f"- candidate_id: {candidate_id}",
                f"- association_id: {row['association_id']}",
                f"- candidate_validation_id: {row['candidate_validation_id']}",
                f"- g1_validation_id: {row['g1_validation_id']}",
                f"- g2_event_summary_id: {row['g2_event_summary_id']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_review(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> CandidateReviewArtifacts:
    """Build CSV and Markdown views without changing or re-running source stages."""

    source_root = Path(run_dir)
    destination = Path(output_dir) if output_dir else source_root / "candidate_review"
    rows, details = _candidate_rows(source_root)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "candidate_review.csv"
    markdown_path = destination / "candidate_review.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(
        _markdown_report(source_root, rows, details), encoding="utf-8"
    )
    return CandidateReviewArtifacts(
        output_dir=destination,
        csv_path=csv_path,
        markdown_path=markdown_path,
        candidate_count=len(rows),
    )


__all__ = [
    "CSV_FIELDS",
    "CandidateReviewArtifacts",
    "NOT_AVAILABLE",
    "build_candidate_review",
    "describe_validation_pair",
]
