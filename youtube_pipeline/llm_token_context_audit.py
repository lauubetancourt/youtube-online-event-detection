"""Deterministic audit of persisted LLM usage and RAG context coverage.

This module is analysis-only: it never invokes a model, Serper, context
selection, or a gate. Provider-reported usage and character-based estimates are
kept in separate fields.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from youtube_pipeline.rag_generation_g1 import _approx_tokens, _compact_event_payload
from youtube_pipeline.stance_classifier_prompt import SYSTEM_INSTRUCTIONS, build_messages


AUDIT_POLICY_ID = "llm_token_context_audit_v1"
STANCE_AUDIT_FILE = "stance_token_usage_audit.csv"
RAG_AUDIT_FILE = "rag_token_context_audit.csv"
GATE_AUDIT_FILE = "g1_g2_gate_counterfactual.csv"
SUMMARY_FILE = "llm_token_context_audit_summary.json"
MANIFEST_FILE = "llm_token_context_audit_manifest.json"

GATE_POLICIES: dict[str, frozenset[str]] = {
    "candidate_strict_confirmed_or_partial": frozenset(
        {"confirmed", "partially_confirmed"}
    ),
    "candidate_permissive_add_ambiguous": frozenset(
        {"confirmed", "partially_confirmed", "ambiguous"}
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{number}")
        output.append(value)
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _distribution(values: Iterable[object]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None and value != ""]
    return {
        "count": len(numeric),
        "min": min(numeric) if numeric else None,
        "p25": _quantile(numeric, 0.25),
        "median": _quantile(numeric, 0.50),
        "p75": _quantile(numeric, 0.75),
        "p90": _quantile(numeric, 0.90),
        "p95": _quantile(numeric, 0.95),
        "max": max(numeric) if numeric else None,
        "mean": sum(numeric) / len(numeric) if numeric else None,
        "sum": sum(numeric) if numeric else None,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _usage(record: Mapping[str, Any]) -> tuple[int | None, int | None, int | None, str]:
    explicit = record.get("token_usage")
    status = record.get("token_usage_status")
    if isinstance(explicit, Mapping):
        values = tuple(explicit.get(key) for key in ("input_tokens", "output_tokens", "total_tokens"))
        if all(isinstance(value, int) for value in values):
            return int(values[0]), int(values[1]), int(values[2]), str(status or "REPORTED")
    body = record.get("raw_response")
    raw = body.get("usage") if isinstance(body, Mapping) else None
    if isinstance(raw, Mapping):
        values = (
            raw.get("input_tokens", raw.get("prompt_tokens")),
            raw.get("output_tokens", raw.get("completion_tokens")),
            raw.get("total_tokens"),
        )
        if all(isinstance(value, int) for value in values):
            return int(values[0]), int(values[1]), int(values[2]), "REPORTED"
        if any(isinstance(value, int) for value in values):
            return (
                int(values[0]) if isinstance(values[0], int) else None,
                int(values[1]) if isinstance(values[1], int) else None,
                int(values[2]) if isinstance(values[2], int) else None,
                "PARTIAL",
            )
    attempts = record.get("attempts")
    return None, None, None, "NOT_APPLICABLE" if attempts == 0 else "UNKNOWN"


@dataclass(frozen=True, slots=True)
class LlmTokenContextAuditConfig:
    output_dir: str
    stance_artifact_dir: str
    stance_comments_path: str
    event_comment_inventory_path: str
    context_units_path: str
    context_unit_comment_map_path: str
    validation_inputs_path: str
    context_payloads_path: str
    g1_raw_responses_path: str
    g1_reports_path: str
    g1_manifest_path: str
    g2_raw_responses_path: str
    g2_queries_path: str
    g2_external_evidence_path: str
    g2_video_reports_path: str
    g2_manifest_path: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty.")


@dataclass(frozen=True, slots=True)
class LlmTokenContextAuditArtifacts:
    output_dir: Path
    stance_audit_path: Path
    rag_audit_path: Path
    gate_audit_path: Path
    summary_path: Path
    manifest_path: Path


def _stance_rows(config: LlmTokenContextAuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(config.stance_artifact_dir)
    manifest = _read_json(root / "classifier_manifest.json")
    stance_records = _read_jsonl(root / "comment_stances.jsonl")
    comments = {row.get("comment_id"): row for row in _read_csv(Path(config.stance_comments_path))}
    proposition = manifest.get("proposition") or {}
    prompt = manifest.get("prompt_artifact") or {}
    system_text = str(prompt.get("system_instructions") or SYSTEM_INSTRUCTIONS)
    proposition_text = str(proposition.get("proposition_text") or "")
    context_text = str(proposition.get("context") or "")
    proposition_id = str(proposition.get("proposition_id") or "")
    output: list[dict[str, Any]] = []
    for record in stance_records:
        source = comments.get(record.get("comment_id"), {})
        parent_id = record.get("parent_comment_id") or source.get("reply_to_comment_id") or None
        parent = comments.get(parent_id, {}) if parent_id else {}
        comment_text = str(source.get("text") or "")
        parent_text = str(parent.get("text") or "") if parent_id and parent else None
        input_tokens, output_tokens, total_tokens, usage_status = _usage(record)
        messages = build_messages(
            proposition_id=proposition_id,
            proposition_text=proposition_text,
            context=context_text,
            comment_text=comment_text,
            parent_text=parent_text,
        )
        estimated_total = round(sum(len(item["content"]) for item in messages) / 4)
        estimated_system = round(len(system_text) / 4)
        estimated_topic = round((len(proposition_id) + len(proposition_text) + len(context_text)) / 4)
        estimated_comment = round(len(comment_text) / 4)
        estimated_parent = round(len(parent_text or "") / 4)
        estimated_wrapper = max(
            0,
            estimated_total
            - estimated_system
            - estimated_topic
            - estimated_comment
            - estimated_parent,
        )
        output.append(
            {
                "classifier_run_id": manifest.get("classifier_run_id"),
                "comment_id": record.get("comment_id"),
                "video_id": source.get("video_id"),
                "is_reply": bool(parent_id),
                "has_parent_context": bool(parent_text),
                "classification_status": record.get("classification_status"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "usage_status": usage_status,
                "estimated_input_tokens_char4": estimated_total,
                "estimated_system_tokens_char4": estimated_system,
                "estimated_topic_profile_tokens_char4": estimated_topic,
                "estimated_comment_tokens_char4": estimated_comment,
                "estimated_parent_tokens_char4": estimated_parent,
                "estimated_wrapper_tokens_char4": estimated_wrapper,
                "estimation_method": "ROUND_UTF8_CHARACTERS_DIVIDED_BY_4",
                "model": manifest.get("model"),
                "prompt_id": manifest.get("prompt_id"),
                "prompt_version": manifest.get("prompt_version"),
                "topic_id": proposition_id,
            }
        )
    return output, manifest


def _project_inventory_comment(
    row: Mapping[str, Any], unit_id: str | None
) -> dict[str, Any]:
    is_reply = str(row.get("is_reply", "")).lower() == "true"
    return {
        "comment_id": row.get("comment_id"),
        "event_id": row.get("event_id"),
        "video_id": row.get("video_id"),
        "event_time_utc": row.get("event_time_utc"),
        "trigger_time_utc": row.get("trigger_time_utc"),
        "temporal_role": "alert_evidence",
        "available_at_trigger": True,
        "relative_to_trigger": "before",
        "is_post_trigger_context": False,
        "text": row.get("text"),
        "text_clean": row.get("text_clean"),
        "context_unit_id": unit_id,
        "root_comment_id": row.get("root_comment_id"),
        "parent_comment_id": row.get("parent_comment_id") or None,
        "is_reply": is_reply,
    }


def _project_inventory_unit(row: Mapping[str, Any]) -> dict[str, Any]:
    comment_ids = row.get("comment_ids") or []
    return {
        "context_unit_id": row.get("context_unit_id"),
        "event_id": row.get("event_id"),
        "video_id": row.get("video_id"),
        "context_type": row.get("context_type"),
        "trigger_time_utc": row.get("trigger_time_utc"),
        "window_start_utc": row.get("window_start_utc"),
        "window_end_utc": row.get("window_end_utc"),
        "time_start_utc": row.get("time_start_utc"),
        "time_end_utc": row.get("time_end_utc"),
        "comment_ids": comment_ids,
        "alert_evidence_comment_ids": comment_ids,
        "post_trigger_comment_ids": [],
        "comment_count": row.get("comment_count"),
        "post_trigger_context_used": False,
        "selection_reason": row.get("selection_reason"),
        "context_text": row.get("context_text"),
    }


def _rag_inputs(config: LlmTokenContextAuditConfig) -> dict[str, Any]:
    inventory = _read_csv(Path(config.event_comment_inventory_path))
    units = _read_jsonl(Path(config.context_units_path))
    mapping = _read_csv(Path(config.context_unit_comment_map_path))
    validation_inputs = _read_jsonl(Path(config.validation_inputs_path))
    payloads = _read_jsonl(Path(config.context_payloads_path))
    g1_raw = _read_jsonl(Path(config.g1_raw_responses_path))
    g1_reports = _read_jsonl(Path(config.g1_reports_path))
    g1_manifest = _read_json(Path(config.g1_manifest_path))
    g2_raw = _read_jsonl(Path(config.g2_raw_responses_path))
    g2_queries = _read_jsonl(Path(config.g2_queries_path))
    g2_external = _read_jsonl(Path(config.g2_external_evidence_path))
    g2_reports = _read_jsonl(Path(config.g2_video_reports_path))
    g2_manifest = _read_json(Path(config.g2_manifest_path))
    return locals()


def _group(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            output[str(value)].append(row)
    return dict(output)


def _sum_usage(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    usages = [_usage(record) for record in records]
    reported = [usage for usage in usages if usage[3] == "REPORTED"]
    partial = [usage for usage in usages if usage[3] == "PARTIAL"]
    known = reported + partial
    return {
        "calls": len(records),
        "reported_calls": len(reported),
        "partial_calls": len(partial),
        "unknown_calls": len(records) - len(known),
        "input_tokens": sum(value[0] or 0 for value in known) if known else None,
        "output_tokens": sum(value[1] or 0 for value in known) if known else None,
        "total_tokens": sum(value[2] or 0 for value in known) if known else None,
        "usage_status": (
            "REPORTED"
            if len(reported) == len(records) and records
            else "UNKNOWN"
            if not known
            else "PARTIAL"
        ),
    }


def _size_band(value: int, p25: float, p50: float, p75: float) -> str:
    if value <= p25:
        return "SMALL_LE_P25"
    if value <= p50:
        return "MEDIUM_P25_P50"
    if value <= p75:
        return "LARGE_P50_P75"
    return "VERY_LARGE_GT_P75"


def _rag_rows(config: LlmTokenContextAuditConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = _rag_inputs(config)
    grouped = {
        name: _group(data[name], "event_id")
        for name in (
            "inventory",
            "units",
            "mapping",
            "g1_raw",
            "g1_reports",
            "g2_raw",
            "g2_queries",
            "g2_external",
            "g2_reports",
        )
    }
    inputs = {str(row["event_id"]): row for row in data["validation_inputs"]}
    payloads = {str(row["event_id"]): row for row in data["payloads"]}
    candidate_ids = sorted(set(inputs) | set(payloads) | set(grouped["inventory"]))
    selected_estimates = [
        _approx_tokens(_compact_event_payload(inputs[key], payloads[key]))
        for key in candidate_ids
        if key in inputs and key in payloads
    ]
    p25 = _quantile([float(value) for value in selected_estimates], 0.25) or 0
    p50 = _quantile([float(value) for value in selected_estimates], 0.50) or 0
    p75 = _quantile([float(value) for value in selected_estimates], 0.75) or 0
    unit_for_comment = {
        str(row.get("comment_id")): str(row.get("context_unit_id"))
        for row in data["mapping"]
        if row.get("comment_id") and row.get("context_unit_id")
    }
    rows: list[dict[str, Any]] = []
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        inventory = list(grouped["inventory"].get(candidate_id, []))
        units = list(grouped["units"].get(candidate_id, []))
        selected_payload = payloads.get(candidate_id, {})
        selected_comments = list(selected_payload.get("used_context_comments") or [])
        selected_units = list(selected_payload.get("selected_context_units") or [])
        validation_input = inputs.get(candidate_id, {"event_id": candidate_id})
        full_payload = {
            "post_trigger_context_used": False,
            "post_trigger_comment_ids": [],
            "selected_context_units": [_project_inventory_unit(row) for row in units],
            "used_context_comments": [
                _project_inventory_comment(row, unit_for_comment.get(str(row.get("comment_id"))))
                for row in inventory
            ],
        }
        inventory_estimate = _approx_tokens(
            _compact_event_payload(validation_input, full_payload)
        )
        selected_estimate = (
            _approx_tokens(_compact_event_payload(validation_input, selected_payload))
            if selected_payload
            else 0
        )
        inventory_videos = {str(row.get("video_id")) for row in inventory if row.get("video_id")}
        selected_videos = {str(row.get("video_id")) for row in selected_comments if row.get("video_id")}
        inventory_threads = {str(row.get("root_comment_id")) for row in inventory if row.get("root_comment_id")}
        selected_threads = {str(row.get("root_comment_id")) for row in selected_comments if row.get("root_comment_id")}
        selected_video_counts: dict[str, int] = defaultdict(int)
        for row in selected_comments:
            if row.get("video_id"):
                selected_video_counts[str(row["video_id"])] += 1
        inventory_thread_comments: dict[str, set[str]] = defaultdict(set)
        selected_thread_comments: dict[str, set[str]] = defaultdict(set)
        for row in inventory:
            if row.get("root_comment_id") and row.get("comment_id"):
                inventory_thread_comments[str(row["root_comment_id"])].add(str(row["comment_id"]))
        for row in selected_comments:
            if row.get("root_comment_id") and row.get("comment_id"):
                selected_thread_comments[str(row["root_comment_id"])].add(str(row["comment_id"]))
        complete_selected_threads = sum(
            comments == inventory_thread_comments.get(root_id, set())
            for root_id, comments in selected_thread_comments.items()
        )
        inventory_times = sorted(
            str(row["event_time_utc"]) for row in inventory if row.get("event_time_utc")
        )
        selected_times = sorted(
            str(row["event_time_utc"])
            for row in selected_comments
            if row.get("event_time_utc")
        )
        g1_records = list(grouped["g1_raw"].get(candidate_id, []))
        g1_usage = _sum_usage(g1_records)
        g1_report = next(iter(grouped["g1_reports"].get(candidate_id, [])), {})
        base = {
            "candidate_id": candidate_id,
            "candidate_type": "ACTIVITY",
            "inventory_comment_count": len(inventory),
            "selected_comment_count": len(selected_comments),
            "comment_coverage": _ratio(len(selected_comments), len(inventory)),
            "inventory_context_unit_count": len(units),
            "selected_context_unit_count": len(selected_units),
            "context_unit_coverage": _ratio(len(selected_units), len(units)),
            "inventory_video_count": len(inventory_videos),
            "selected_video_count": len(selected_videos),
            "video_coverage": _ratio(len(selected_videos), len(inventory_videos)),
            "inventory_thread_count": len(inventory_threads),
            "selected_thread_count": len(selected_threads),
            "thread_coverage": _ratio(len(selected_threads), len(inventory_threads)),
            "selected_thread_completeness": _ratio(
                complete_selected_threads, len(selected_threads)
            ),
            "selected_max_video_comment_fraction": (
                max(selected_video_counts.values()) / len(selected_comments)
                if selected_comments and selected_video_counts
                else None
            ),
            "inventory_time_start_utc": inventory_times[0] if inventory_times else None,
            "inventory_time_end_utc": inventory_times[-1] if inventory_times else None,
            "selected_time_start_utc": selected_times[0] if selected_times else None,
            "selected_time_end_utc": selected_times[-1] if selected_times else None,
            "temporal_boundary_coverage": (
                bool(inventory_times)
                and bool(selected_times)
                and inventory_times[0] == selected_times[0]
                and inventory_times[-1] == selected_times[-1]
            ),
            "estimated_inventory_tokens": inventory_estimate,
            "estimated_selected_context_tokens": selected_estimate,
            "estimated_token_coverage": _ratio(selected_estimate, inventory_estimate),
            "candidate_size_band": _size_band(selected_estimate, p25, p50, p75),
        }
        g1_row = {
            **base,
            "stage": "G1",
            "model_calls": g1_usage["calls"],
            "query_generation_calls": 0,
            "video_validation_calls": 0,
            "candidate_synthesis_calls": 0,
            "serper_requests": 0,
            "sources_returned": 0,
            "sources_retained": 0,
            "sources_used_or_cited": 0,
            "input_tokens": g1_usage["input_tokens"],
            "output_tokens": g1_usage["output_tokens"],
            "total_tokens": g1_usage["total_tokens"],
            "usage_status": g1_usage["usage_status"],
            "validation_status": g1_report.get("validation_status"),
            "external_support": None,
        }
        g2_records = list(grouped["g2_raw"].get(candidate_id, []))
        query_records = [
            row for row in g2_records if row.get("record_type") == "video_query_generation"
        ]
        validation_records = [
            row for row in g2_records if row.get("record_type") == "video_validation_generation"
        ]
        g2_usage = _sum_usage(g2_records)
        queries = list(grouped["g2_queries"].get(candidate_id, []))
        external = list(grouped["g2_external"].get(candidate_id, []))
        reports = list(grouped["g2_reports"].get(candidate_id, []))
        used_external: set[str] = set()
        for report in reports:
            for field in ("used_external_evidence_ids", "cited_external_evidence_ids"):
                used_external.update(str(value) for value in (report.get(field) or []))
        external_support = any(
            report.get("external_evidence_assessment") == "supports" for report in reports
        )
        g2_row = {
            **base,
            "stage": "G2",
            "model_calls": g2_usage["calls"],
            "query_generation_calls": len(query_records),
            "video_validation_calls": len(validation_records),
            "candidate_synthesis_calls": 0,
            "serper_requests": len(queries),
            "sources_returned": len(external),
            "sources_retained": len(external),
            "sources_used_or_cited": len(used_external),
            "input_tokens": g2_usage["input_tokens"],
            "output_tokens": g2_usage["output_tokens"],
            "total_tokens": g2_usage["total_tokens"],
            "usage_status": g2_usage["usage_status"],
            "validation_status": None,
            "external_support": external_support,
        }
        known_totals = [
            row["total_tokens"]
            for row in (g1_row, g2_row)
            if isinstance(row["total_tokens"], int)
        ]
        total_row = {
            **base,
            "stage": "TOTAL_RAG",
            "model_calls": g1_row["model_calls"] + g2_row["model_calls"],
            "query_generation_calls": g2_row["query_generation_calls"],
            "video_validation_calls": g2_row["video_validation_calls"],
            "candidate_synthesis_calls": 0,
            "serper_requests": g2_row["serper_requests"],
            "sources_returned": g2_row["sources_returned"],
            "sources_retained": g2_row["sources_retained"],
            "sources_used_or_cited": g2_row["sources_used_or_cited"],
            "input_tokens": (
                (g1_row["input_tokens"] or 0) + (g2_row["input_tokens"] or 0)
                if g1_row["input_tokens"] is not None
                else None
            ),
            "output_tokens": (
                (g1_row["output_tokens"] or 0) + (g2_row["output_tokens"] or 0)
                if g1_row["output_tokens"] is not None
                else None
            ),
            "total_tokens": sum(known_totals) if len(known_totals) == 2 else None,
            "usage_status": (
                "REPORTED"
                if g1_row["usage_status"] == g2_row["usage_status"] == "REPORTED"
                else "PARTIAL"
            ),
            "validation_status": g1_row["validation_status"],
            "external_support": external_support,
        }
        rows.extend((g1_row, g2_row, total_row))
        candidate_metrics[candidate_id] = {
            "g1_status": g1_row["validation_status"],
            "g2": g2_row,
        }
    gate_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        metric = candidate_metrics[candidate_id]
        g1_status = metric["g1_status"]
        g2 = metric["g2"]
        for policy_id, allowed in GATE_POLICIES.items():
            if g1_status is None:
                decision = "NOT_EVALUABLE_MISSING_G1"
                saved_requests = saved_calls = saved_tokens = None
                external_filtered = None
            else:
                decision = "RUN_G2" if g1_status in allowed else "SKIP_G2"
                skip = decision == "SKIP_G2"
                saved_requests = g2["serper_requests"] if skip else 0
                saved_calls = g2["model_calls"] if skip else 0
                saved_tokens = g2["total_tokens"] if skip else 0
                external_filtered = bool(skip and g2["external_support"])
            gate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "g1_status": g1_status,
                    "observed_g2_status": (
                        "EXTERNAL_SUPPORT_FOUND"
                        if g2["external_support"]
                        else "NO_SUPPORT_FOUND"
                    ),
                    "gate_policy_candidate": policy_id,
                    "eligible_g1_statuses": ",".join(sorted(allowed)),
                    "parameter_status": "NOT_CALIBRATED",
                    "counterfactual_decision": decision,
                    "serper_requests_saved": saved_requests,
                    "model_calls_saved": saved_calls,
                    "tokens_saved": saved_tokens,
                    "external_support_filtered": external_filtered,
                }
            )
    return rows, gate_rows, data


def run_llm_token_context_audit(
    config: LlmTokenContextAuditConfig,
) -> LlmTokenContextAuditArtifacts:
    stance_rows, stance_manifest = _stance_rows(config)
    rag_rows, gate_rows, rag_data = _rag_rows(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stance_path = output_dir / STANCE_AUDIT_FILE
    rag_path = output_dir / RAG_AUDIT_FILE
    gate_path = output_dir / GATE_AUDIT_FILE
    summary_path = output_dir / SUMMARY_FILE
    manifest_path = output_dir / MANIFEST_FILE
    stance_fields = list(stance_rows[0]) if stance_rows else ["comment_id"]
    rag_fields = list(rag_rows[0]) if rag_rows else ["candidate_id", "stage"]
    gate_fields = list(gate_rows[0]) if gate_rows else ["candidate_id"]
    _write_csv(stance_path, stance_rows, stance_fields)
    _write_csv(rag_path, rag_rows, rag_fields)
    _write_csv(gate_path, gate_rows, gate_fields)

    stance_token_usage = stance_manifest.get("token_usage") or {}
    stance_reported_attempts = int(stance_token_usage.get("reported_attempts", 0) or 0)
    stance_provider_attempts = int(stance_token_usage.get("provider_attempts", 0) or 0)
    g1_rows = [row for row in rag_rows if row["stage"] == "G1"]
    g2_rows = [row for row in rag_rows if row["stage"] == "G2"]
    total_rows = [row for row in rag_rows if row["stage"] == "TOTAL_RAG"]
    g2_call_usage = [_usage(row) for row in rag_data["g2_raw"]]
    g2_query_usage = [
        _usage(row)
        for row in rag_data["g2_raw"]
        if row.get("record_type") == "video_query_generation"
    ]
    g2_validation_usage = [
        _usage(row)
        for row in rag_data["g2_raw"]
        if row.get("record_type") == "video_validation_generation"
    ]
    g1_call_usage = [_usage(row) for row in rag_data["g1_raw"]]
    selected_estimates = [row["estimated_selected_context_tokens"] for row in g1_rows]
    coverage_fields = (
        "comment_coverage",
        "context_unit_coverage",
        "video_coverage",
        "thread_coverage",
    )
    gate_summary: dict[str, Any] = {}
    for policy_id in GATE_POLICIES:
        rows = [row for row in gate_rows if row["gate_policy_candidate"] == policy_id]
        evaluable = [row for row in rows if row["counterfactual_decision"] != "NOT_EVALUABLE_MISSING_G1"]
        gate_summary[policy_id] = {
            "candidate_count": len(rows),
            "evaluable_candidates": len(evaluable),
            "not_evaluable_candidates": len(rows) - len(evaluable),
            "g2_executions_avoided": sum(
                row["counterfactual_decision"] == "SKIP_G2" for row in evaluable
            ),
            "serper_requests_avoided": sum(
                int(row["serper_requests_saved"] or 0) for row in evaluable
            ),
            "g2_model_calls_avoided": sum(
                int(row["model_calls_saved"] or 0) for row in evaluable
            ),
            "g2_tokens_avoided": sum(
                int(row["tokens_saved"] or 0) for row in evaluable
            ),
            "externally_supported_candidates_filtered": sum(
                row["external_support_filtered"] is True for row in evaluable
            ),
        }
    multi_video_cost: dict[str, Any] = {}
    for video_count in sorted({int(row["serper_requests"]) for row in g2_rows}):
        matches = [row for row in g2_rows if int(row["serper_requests"]) == video_count]
        multi_video_cost[str(video_count)] = {
            "candidate_count": len(matches),
            "mean_model_calls": sum(int(row["model_calls"]) for row in matches) / len(matches),
            "mean_total_tokens": sum(int(row["total_tokens"]) for row in matches) / len(matches),
            "min_total_tokens": min(int(row["total_tokens"]) for row in matches),
            "max_total_tokens": max(int(row["total_tokens"]) for row in matches),
        }
    g1_g2_relationship: dict[str, int] = defaultdict(int)
    g2_by_candidate = {row["candidate_id"]: row for row in g2_rows}
    for row in g1_rows:
        if row["validation_status"] is None:
            continue
        external = g2_by_candidate[row["candidate_id"]]["external_support"]
        key = f"{row['validation_status']}|{'EXTERNAL_SUPPORT_FOUND' if external else 'NO_SUPPORT_FOUND'}"
        g1_g2_relationship[key] += 1
    summary = {
        "audit_policy_id": AUDIT_POLICY_ID,
        "stance": {
            "calls": stance_provider_attempts,
            "aggregate_reported_calls": stance_reported_attempts,
            "row_attributable_reported_calls": sum(
                row["usage_status"] == "REPORTED" for row in stance_rows
            ),
            "aggregate_usage": stance_token_usage,
            "input_tokens_per_comment": _distribution(
                row["input_tokens"] for row in stance_rows
            ),
            "output_tokens_per_comment": _distribution(
                row["output_tokens"] for row in stance_rows
            ),
            "total_tokens_per_comment": _distribution(
                row["total_tokens"] for row in stance_rows
            ),
            "aggregate_mean_per_provider_attempt": {
                key: (
                    stance_token_usage.get(key) / stance_provider_attempts
                    if stance_provider_attempts and isinstance(stance_token_usage.get(key), int)
                    else None
                )
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
            "estimated_input_tokens_char4": _distribution(
                row["estimated_input_tokens_char4"] for row in stance_rows
            ),
            "root_estimated_input_tokens_char4": _distribution(
                row["estimated_input_tokens_char4"] for row in stance_rows if not row["is_reply"]
            ),
            "reply_estimated_input_tokens_char4": _distribution(
                row["estimated_input_tokens_char4"] for row in stance_rows if row["is_reply"]
            ),
            "topic_profile_estimated_tokens_char4": (
                stance_rows[0]["estimated_topic_profile_tokens_char4"] if stance_rows else None
            ),
            "system_estimated_tokens_char4": (
                stance_rows[0]["estimated_system_tokens_char4"] if stance_rows else None
            ),
            "estimated_by_classification_status": {
                status: _distribution(
                    row["estimated_input_tokens_char4"]
                    for row in stance_rows
                    if row["classification_status"] == status
                )
                for status in sorted(
                    {str(row["classification_status"]) for row in stance_rows}
                )
            },
            "estimated_fixed_plus_topic_fraction_of_mean_input": (
                (
                    stance_rows[0]["estimated_system_tokens_char4"]
                    + stance_rows[0]["estimated_topic_profile_tokens_char4"]
                )
                / _distribution(
                    row["estimated_input_tokens_char4"] for row in stance_rows
                )["mean"]
                if stance_rows
                else None
            ),
        },
        "g1_context": {
            "candidates": len(g1_rows),
            "inventory_comments": sum(row["inventory_comment_count"] for row in g1_rows),
            "selected_comments": sum(row["selected_comment_count"] for row in g1_rows),
            "inventory_context_units": sum(row["inventory_context_unit_count"] for row in g1_rows),
            "selected_context_units": sum(row["selected_context_unit_count"] for row in g1_rows),
            "coverage": {
                field: _distribution(row[field] for row in g1_rows)
                for field in coverage_fields
            },
            "estimated_selected_context_tokens": _distribution(selected_estimates),
            "inventory_comment_count": _distribution(
                row["inventory_comment_count"] for row in g1_rows
            ),
            "inventory_context_unit_count": _distribution(
                row["inventory_context_unit_count"] for row in g1_rows
            ),
            "inventory_video_count": _distribution(
                row["inventory_video_count"] for row in g1_rows
            ),
            "inventory_thread_count": _distribution(
                row["inventory_thread_count"] for row in g1_rows
            ),
            "selected_max_video_comment_fraction": _distribution(
                row["selected_max_video_comment_fraction"] for row in g1_rows
            ),
            "selected_thread_completeness": _distribution(
                row["selected_thread_completeness"] for row in g1_rows
            ),
            "temporal_boundary_coverage_candidates": sum(
                row["temporal_boundary_coverage"] is True for row in g1_rows
            ),
            "truncated_candidates": sum(
                row["selected_comment_count"] < row["inventory_comment_count"]
                or row["selected_context_unit_count"] < row["inventory_context_unit_count"]
                for row in g1_rows
            ),
        },
        "g1_usage": {
            "candidate_calls": sum(row["model_calls"] for row in g1_rows),
            "candidates_with_reported_usage": sum(
                row["usage_status"] == "REPORTED" for row in g1_rows
            ),
            "input_tokens_per_call": _distribution(value[0] for value in g1_call_usage),
            "output_tokens_per_call": _distribution(value[1] for value in g1_call_usage),
            "total_tokens_per_call": _distribution(value[2] for value in g1_call_usage),
            "reported_input_minus_estimated_selected_context": _distribution(
                row["input_tokens"] - row["estimated_selected_context_tokens"]
                for row in g1_rows
                if isinstance(row["input_tokens"], int)
            ),
        },
        "g2_usage": {
            "serper_requests": sum(row["serper_requests"] for row in g2_rows),
            "serper_request_count_status": "INFERRED_FROM_EXECUTED_QUERY_RECORDS",
            "model_calls": sum(row["model_calls"] for row in g2_rows),
            "query_generation_calls": sum(row["query_generation_calls"] for row in g2_rows),
            "video_validation_calls": sum(row["video_validation_calls"] for row in g2_rows),
            "candidate_synthesis_calls": 0,
            "sources_returned": sum(row["sources_returned"] for row in g2_rows),
            "sources_used_or_cited": sum(row["sources_used_or_cited"] for row in g2_rows),
            "input_tokens_per_call": _distribution(value[0] for value in g2_call_usage),
            "output_tokens_per_call": _distribution(value[1] for value in g2_call_usage),
            "total_tokens_per_call": _distribution(value[2] for value in g2_call_usage),
            "query_generation_usage": {
                "input_tokens": _distribution(value[0] for value in g2_query_usage),
                "output_tokens": _distribution(value[1] for value in g2_query_usage),
                "total_tokens": _distribution(value[2] for value in g2_query_usage),
            },
            "video_validation_usage": {
                "input_tokens": _distribution(value[0] for value in g2_validation_usage),
                "output_tokens": _distribution(value[1] for value in g2_validation_usage),
                "total_tokens": _distribution(value[2] for value in g2_validation_usage),
            },
            "input_tokens_per_candidate": _distribution(row["input_tokens"] for row in g2_rows),
            "output_tokens_per_candidate": _distribution(row["output_tokens"] for row in g2_rows),
            "total_tokens_per_candidate": _distribution(row["total_tokens"] for row in g2_rows),
            "multi_video_cost_by_executed_video_count": multi_video_cost,
        },
        "end_to_end_rag": {
            "fully_reported_candidates": sum(
                row["usage_status"] == "REPORTED" for row in total_rows
            ),
            "partially_reported_candidates": sum(
                row["usage_status"] != "REPORTED" for row in total_rows
            ),
            "total_tokens_per_fully_reported_candidate": _distribution(
                row["total_tokens"]
                for row in total_rows
                if row["usage_status"] == "REPORTED"
            ),
        },
        "gate_counterfactual": {
            "baseline": "OFF",
            "policies": gate_summary,
        },
        "g1_g2_relationship": dict(sorted(g1_g2_relationship.items())),
        "budget_candidates": {
            "derivation": "observed estimated selected-context percentiles; not recommendations",
            "p50": round(_quantile([float(value) for value in selected_estimates], 0.50) or 0),
            "p75": round(_quantile([float(value) for value in selected_estimates], 0.75) or 0),
            "p90": round(_quantile([float(value) for value in selected_estimates], 0.90) or 0),
            "near_full_observed_max": max(selected_estimates) if selected_estimates else None,
            "parameter_status": "AUDIT_DERIVED_NOT_CALIBRATED",
        },
        "usage_reporting_coverage": {
            "stance_aggregate": _ratio(stance_reported_attempts, stance_provider_attempts),
            "stance_row_attribution": _ratio(
                sum(row["usage_status"] == "REPORTED" for row in stance_rows),
                len(stance_rows),
            ),
            "g1_executed_calls": _ratio(
                sum(value[3] == "REPORTED" for value in g1_call_usage), len(g1_call_usage)
            ),
            "g1_candidate_set": _ratio(
                sum(row["usage_status"] == "REPORTED" for row in g1_rows), len(g1_rows)
            ),
            "g2_calls": _ratio(
                sum(value[3] == "REPORTED" for value in g2_call_usage), len(g2_call_usage)
            ),
        },
    }
    _write_json(summary_path, summary)

    source_paths = [
        Path(config.stance_artifact_dir) / "comment_stances.jsonl",
        Path(config.stance_artifact_dir) / "classifier_manifest.json",
        Path(config.stance_comments_path),
        *[
            Path(getattr(config, field_name))
            for field_name in config.__dataclass_fields__
            if field_name.endswith("_path") and field_name != "stance_comments_path"
        ],
    ]
    source_paths = list(dict.fromkeys(source_paths))
    manifest = {
        "audit_policy_id": AUDIT_POLICY_ID,
        "status": "executed",
        "candidate_set": sorted(
            {row["candidate_id"] for row in rag_rows if row["candidate_id"]}
        ),
        "models": {
            "stance": stance_manifest.get("model"),
            "g1": rag_data["g1_manifest"].get("model_name"),
            "g2_query": rag_data["g2_manifest"].get("query_model_name"),
            "g2_validation": rag_data["g2_manifest"].get("validation_model_name"),
        },
        "prompt_versions": {
            "stance": stance_manifest.get("prompt_version"),
            "g1": rag_data["g1_manifest"].get("prompt_version"),
            "g2_query": rag_data["g2_manifest"].get("query_prompt_version"),
            "g2_validation": rag_data["g2_manifest"].get("validation_prompt_version"),
        },
        "topic_profile": stance_manifest.get("proposition"),
        "gate_baseline": "OFF",
        "usage_reporting_coverage": summary["usage_reporting_coverage"],
        "model_context_limits": {
            "stance_model_maximum": "UNKNOWN_NOT_PERSISTED",
            "stance_application_input_budget": (
                stance_manifest.get("inference_parameters") or {}
            ).get("max_input_characters"),
            "stance_application_output_budget": (
                stance_manifest.get("inference_parameters") or {}
            ).get("max_completion_tokens"),
            "g1_model_maximum": "UNKNOWN_NOT_PERSISTED",
            "g1_application_approx_budget": (
                rag_data["g1_manifest"].get("context_policy") or {}
            ).get("max_approx_tokens"),
            "g2_model_maximum": "UNKNOWN_NOT_PERSISTED",
            "g2_event_batch_estimated_budget": (
                rag_data["g2_manifest"].get("batch_limits") or {}
            ).get("max_estimated_tokens_per_event_batch"),
        },
        "real_model_calls": 0,
        "real_serper_requests": 0,
        "reused_results": {
            "stance_calls": stance_provider_attempts,
            "g1_calls": len(rag_data["g1_raw"]),
            "g2_calls": len(rag_data["g2_raw"]),
            "historical_serper_requests_inferred": len(rag_data["g2_queries"]),
        },
        "source_artifacts": [
            {"path": path.as_posix(), "sha256": _sha256(path)} for path in source_paths
        ],
        "limitations": [
            "Stance usage is aggregate in the retained smoke and cannot be attributed to comments.",
            "Stance root/reply and prompt-component values are character/4 estimates, not provider tokens.",
            "Only two of ten candidates have persisted G-1 executions.",
            "Historical Serper request count is inferred from executed query records.",
            "Legacy event_id is used as the candidate compatibility identity.",
            "Model maximum context limits are not persisted and are not inferred.",
            "Gate counterfactuals do not constitute ground truth or gate calibration.",
        ],
        "artifacts": {
            "stance_audit": stance_path.as_posix(),
            "rag_audit": rag_path.as_posix(),
            "gate_counterfactual": gate_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
    }
    for key, path in (
        ("stance_audit_sha256", stance_path),
        ("rag_audit_sha256", rag_path),
        ("gate_counterfactual_sha256", gate_path),
        ("summary_sha256", summary_path),
    ):
        manifest["artifacts"][key] = _sha256(path)
    _write_json(manifest_path, manifest)
    return LlmTokenContextAuditArtifacts(
        output_dir=output_dir,
        stance_audit_path=stance_path,
        rag_audit_path=rag_path,
        gate_audit_path=gate_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )


__all__ = [
    "AUDIT_POLICY_ID",
    "GATE_AUDIT_FILE",
    "LlmTokenContextAuditArtifacts",
    "LlmTokenContextAuditConfig",
    "MANIFEST_FILE",
    "RAG_AUDIT_FILE",
    "STANCE_AUDIT_FILE",
    "SUMMARY_FILE",
    "run_llm_token_context_audit",
]
