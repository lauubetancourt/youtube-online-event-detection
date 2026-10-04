from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from youtube_pipeline.external_publication_time import (
    CAUSALLY_AVAILABLE,
    FUTURE_RELATIVE_TO_CUTOFF,
    SAME_DAY_TIME_UNKNOWN,
    external_publication_time_from_record,
    parse_external_publication_time,
    temporal_relation_to_cutoff,
)
from youtube_pipeline.rag_generation_g1 import (
    OPENAI_RESPONSES_URL,
    _check_model_available,
    _extract_event,
    _load_api_key,
    _normalize_path,
    _parse_model_json,
    _read_json,
    _read_jsonl_records,
    _sha256_file,
    _short_hash,
    _should_send_temperature,
    _temperature_sent_value,
    _utc_now_iso,
    _write_json,
    _write_jsonl,
)
from youtube_pipeline.rag_generation_g2 import (
    CONFIDENCE_LABEL_VALUES,
    DEFAULT_SERPER_URL,
    EVENT_INTERPRETATION_VALUES,
    EXTERNAL_EVIDENCE_ASSESSMENT_VALUES,
    VALIDATION_STATUS_VALUES,
    _call_serper_news,
    _create_json_response,
    _load_serper_api_key,
    _safe_text,
    _schema_errors,
    _temporal_relation,
    _tokens,
)
from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_CONFIRMED,
    normalize_validation_status,
)


RAG_G2H_ARTIFACT_VERSION = "rag_generation_g2_hierarchical_v3"
QUERY_PROMPT_VERSION = "rag_video_news_query_prompt_v0.3"
VALIDATION_PROMPT_VERSION = "rag_video_validation_g2_prompt_v0.3"

RAG_VALIDATION_INPUTS_FILE = "rag_validation_inputs.jsonl"
RAG_CONTEXT_PAYLOADS_FILE = "rag_context_payloads.jsonl"
RAG_CONSUMER_MANIFEST_FILE = "rag_consumer_manifest.json"

RAG_VIDEO_NEWS_QUERIES_FILE = "rag_video_news_queries.jsonl"
RAG_VIDEO_EXTERNAL_EVIDENCE_FILE = "rag_video_external_evidence.jsonl"
RAG_VIDEO_VALIDATION_REPORTS_FILE = "rag_video_validation_reports.jsonl"
RAG_EVENT_VALIDATION_SUMMARY_FILE = "rag_event_validation_summary.jsonl"
RAG_GENERATION_MANIFEST_FILE = "rag_generation_manifest.json"
RAG_RAW_MODEL_RESPONSES_FILE = "rag_raw_model_responses.jsonl"

AGREEMENT_VALUES = {"consistent", "mixed", "contradictory", "insufficient"}
DEFAULT_QUERY_MODEL = "gpt-5-mini"
DEFAULT_VALIDATION_MODEL = "gpt-5-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SERPER_GL = "co"
DEFAULT_SERPER_HL = "es"
DEFAULT_SERPER_TYPE = "news"
DEFAULT_SERPER_NUM_RESULTS = 5
DEFAULT_SEARCH_DAYS_BEFORE = 1
DEFAULT_SEARCH_DAYS_AFTER = 1
DEFAULT_MAX_VIDEOS_PER_EVENT_BATCH = 5
DEFAULT_MAX_ESTIMATED_TOKENS_PER_EVENT_BATCH = 80_000
DEFAULT_MAX_LLM_CALLS_PER_BATCH = 100
DEFAULT_MAX_SERPER_CALLS_PER_BATCH = 50
DEFAULT_MAX_ESTIMATED_COST_USD_PER_BATCH = 1.0
DEFAULT_BATCH_INDEX = 1
DETERMINISTIC_METADATA = "DETERMINISTIC_METADATA"
LLM_FALLBACK = "LLM_FALLBACK"
QUERY_GENERATION_MODE_VALUES = {DETERMINISTIC_METADATA, LLM_FALLBACK}
FALLBACK_QUERY_REASONING = "low"
FALLBACK_QUERY_MAX_OUTPUT_TOKENS = 512
_GENERIC_VIDEO_TITLES = {
    "video",
    "en vivo",
    "live",
    "stream",
    "untitled",
}
VIDEO_BATCH_STATUS_VALUES = {
    "processed",
    "pending_batch",
    "pending_budget_limit",
    "skipped_no_context",
    "failed_retrieval",
    "failed_validation",
}
EVENT_BATCH_STATUS_VALUES = {
    "complete",
    "partial_pending_batch",
    "partial_errors",
    "not_started",
    "failed",
}


@dataclass(frozen=True)
class RagG2HierarchicalConfig:
    consumer_dir: str
    output_dir: str
    event_id: str = "__all__"
    query_model: str = DEFAULT_QUERY_MODEL
    validation_model: str = DEFAULT_VALIDATION_MODEL
    provider: str = "openai"
    temperature: float = DEFAULT_TEMPERATURE
    serper_url: str = DEFAULT_SERPER_URL
    serper_gl: str = DEFAULT_SERPER_GL
    serper_hl: str = DEFAULT_SERPER_HL
    serper_type: str = DEFAULT_SERPER_TYPE
    serper_num_results: int = DEFAULT_SERPER_NUM_RESULTS
    search_days_before: int = DEFAULT_SEARCH_DAYS_BEFORE
    search_days_after: int = DEFAULT_SEARCH_DAYS_AFTER
    max_videos_per_event_batch: int = DEFAULT_MAX_VIDEOS_PER_EVENT_BATCH
    max_estimated_tokens_per_event_batch: int = (
        DEFAULT_MAX_ESTIMATED_TOKENS_PER_EVENT_BATCH
    )
    max_llm_calls_per_batch: int = DEFAULT_MAX_LLM_CALLS_PER_BATCH
    max_serper_calls_per_batch: int = DEFAULT_MAX_SERPER_CALLS_PER_BATCH
    max_estimated_cost_usd_per_batch: float = (
        DEFAULT_MAX_ESTIMATED_COST_USD_PER_BATCH
    )
    batch_index: int = DEFAULT_BATCH_INDEX
    batch_id: str | None = None
    max_retries: int = 1
    request_timeout_seconds: int = 120
    serper_timeout_seconds: int = 60
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    external_evidence_cutoff_utc: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RagG2HierarchicalConfig":
        missing = [
            key
            for key in ["consumer_dir", "output_dir"]
            if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                "RAG G-2 hierarchical config missing required fields: "
                + ", ".join(missing)
            )
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        return cls(
            consumer_dir=str(payload["consumer_dir"]),
            output_dir=str(payload["output_dir"]),
            event_id=str(payload.get("event_id", "__all__")),
            query_model=str(payload.get("query_model", DEFAULT_QUERY_MODEL)),
            validation_model=str(
                payload.get("validation_model", DEFAULT_VALIDATION_MODEL)
            ),
            provider=str(payload.get("provider", "openai")),
            temperature=float(payload.get("temperature", DEFAULT_TEMPERATURE)),
            serper_url=str(payload.get("serper_url", DEFAULT_SERPER_URL)),
            serper_gl=str(payload.get("serper_gl", DEFAULT_SERPER_GL)),
            serper_hl=str(payload.get("serper_hl", DEFAULT_SERPER_HL)),
            serper_type=str(payload.get("serper_type", DEFAULT_SERPER_TYPE)),
            serper_num_results=int(
                payload.get("serper_num_results", DEFAULT_SERPER_NUM_RESULTS)
            ),
            search_days_before=int(
                payload.get("search_days_before", DEFAULT_SEARCH_DAYS_BEFORE)
            ),
            search_days_after=int(
                payload.get("search_days_after", DEFAULT_SEARCH_DAYS_AFTER)
            ),
            max_videos_per_event_batch=int(
                payload.get(
                    "max_videos_per_event_batch",
                    payload.get(
                        "max_videos_per_event",
                        DEFAULT_MAX_VIDEOS_PER_EVENT_BATCH,
                    ),
                )
            ),
            max_estimated_tokens_per_event_batch=int(
                payload.get(
                    "max_estimated_tokens_per_event_batch",
                    DEFAULT_MAX_ESTIMATED_TOKENS_PER_EVENT_BATCH,
                )
            ),
            max_llm_calls_per_batch=int(
                payload.get("max_llm_calls_per_batch", DEFAULT_MAX_LLM_CALLS_PER_BATCH)
            ),
            max_serper_calls_per_batch=int(
                payload.get(
                    "max_serper_calls_per_batch", DEFAULT_MAX_SERPER_CALLS_PER_BATCH
                )
            ),
            max_estimated_cost_usd_per_batch=float(
                payload.get(
                    "max_estimated_cost_usd_per_batch",
                    DEFAULT_MAX_ESTIMATED_COST_USD_PER_BATCH,
                )
            ),
            batch_index=int(payload.get("batch_index", DEFAULT_BATCH_INDEX)),
            batch_id=payload.get("batch_id"),
            max_retries=int(payload.get("max_retries", 1)),
            request_timeout_seconds=int(payload.get("request_timeout_seconds", 120)),
            serper_timeout_seconds=int(payload.get("serper_timeout_seconds", 60)),
            notes=payload.get("notes"),
            params=params,
            external_evidence_cutoff_utc=payload.get(
                "external_evidence_cutoff_utc"
            ),
        )


def _extract_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("rag_generation_g2_hierarchical")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError(
            "rag_generation_g2_hierarchical config section must be an object."
        )
    return nested


def _merge_config_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for key, value in payload.items():
            if value is None:
                continue
            if key == "params":
                current = merged.get(key, {})
                if not isinstance(current, dict) or not isinstance(value, dict):
                    raise ValueError("params must be an object.")
                merged[key] = {**current, **value}
            else:
                merged[key] = value
    return merged


def load_rag_g2_hierarchical_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> RagG2HierarchicalConfig:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG G-2 hierarchical config must be an object: {p}")
    base = _extract_config_payload(payload)
    merged = _merge_config_payloads(base, overrides or {})
    return RagG2HierarchicalConfig.from_mapping(merged)


def _validate_config(
    config: RagG2HierarchicalConfig,
    *,
    require_single_event: bool,
) -> None:
    if config.provider != "openai":
        raise ValueError("G-2 hierarchical currently supports only provider='openai'.")
    if config.external_evidence_cutoff_utc is not None:
        cutoff = pd.Timestamp(config.external_evidence_cutoff_utc)
        if cutoff.tzinfo is None:
            raise ValueError("external_evidence_cutoff_utc must be timezone-aware.")
    if require_single_event and _is_all_events_request(config.event_id):
        raise ValueError("event_id is required for real G-2 execution.")
    if config.batch_index < 1:
        raise ValueError("batch_index must be >= 1.")
    if config.max_videos_per_event_batch < 1:
        raise ValueError("max_videos_per_event_batch must be >= 1.")
    if config.max_llm_calls_per_batch < 1:
        raise ValueError("max_llm_calls_per_batch must be >= 1.")
    if config.max_serper_calls_per_batch < 1:
        raise ValueError("max_serper_calls_per_batch must be >= 1.")
    if config.max_estimated_tokens_per_event_batch < 1:
        raise ValueError("max_estimated_tokens_per_event_batch must be >= 1.")
    if config.search_days_before < 0 or config.search_days_after < 0:
        raise ValueError("search_days_before/search_days_after must be >= 0.")
    if config.serper_num_results < 1:
        raise ValueError("serper_num_results must be >= 1.")


def _cost_guard_status(config: RagG2HierarchicalConfig) -> str:
    cost_config = config.params.get("cost_estimation") or {}
    if not isinstance(cost_config, dict):
        return "not_enforced_missing_rates"
    has_rate = any(
        float(cost_config.get(key, 0) or 0) > 0
        for key in [
            "usd_per_1k_estimated_tokens",
            "usd_per_llm_call",
            "usd_per_serper_call",
        ]
    )
    return "enforced" if has_rate else "not_enforced_missing_rates"


def _is_all_events_request(event_id: str | None) -> bool:
    return event_id in {None, "", "__all__", "all", "*"}


def _consumer_paths(consumer_dir: str | Path) -> dict[str, Path]:
    root = Path(consumer_dir)
    return {
        "rag_validation_inputs": root / RAG_VALIDATION_INPUTS_FILE,
        "rag_context_payloads": root / RAG_CONTEXT_PAYLOADS_FILE,
        "rag_consumer_manifest": root / RAG_CONSUMER_MANIFEST_FILE,
    }


def _output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "rag_video_news_queries": root / RAG_VIDEO_NEWS_QUERIES_FILE,
        "rag_video_external_evidence": root / RAG_VIDEO_EXTERNAL_EVIDENCE_FILE,
        "rag_video_validation_reports": root / RAG_VIDEO_VALIDATION_REPORTS_FILE,
        "rag_event_validation_summary": root / RAG_EVENT_VALIDATION_SUMMARY_FILE,
        "rag_generation_manifest": root / RAG_GENERATION_MANIFEST_FILE,
        "rag_raw_model_responses": root / RAG_RAW_MODEL_RESPONSES_FILE,
    }


def _query_candidate_id(event_id: str, video_id: str, prompt_version: str, model: str) -> str:
    return "qryv_" + _short_hash(event_id, video_id, prompt_version, model)


def _video_validation_id(event_id: str, video_id: str, prompt_version: str, model: str) -> str:
    return "valv_" + _short_hash(event_id, video_id, prompt_version, model)


def _event_summary_id(event_id: str, video_validation_ids: list[str]) -> str:
    return "evsum_" + _short_hash(event_id, ",".join(sorted(video_validation_ids)))


def _external_evidence_id(
    event_id: str,
    video_id: str,
    query_candidate_id: str,
    rank: int,
    result: dict[str, Any],
) -> str:
    stable = result.get("link") or result.get("title") or json.dumps(result, sort_keys=True)
    return "extv_" + _short_hash(event_id, video_id, query_candidate_id, rank, stable)


def _search_window(trigger_time_utc: str, days_before: int, days_after: int) -> tuple[str, str]:
    trigger = pd.Timestamp(trigger_time_utc)
    start = (trigger.date() - timedelta(days=days_before)).isoformat()
    end = (trigger.date() + timedelta(days=days_after)).isoformat()
    return start, end


def _bounded_search_end(search_end: str, cutoff_utc: str | None) -> str:
    if cutoff_utc is None:
        return search_end
    return min(search_end, pd.Timestamp(cutoff_utc).date().isoformat())


def _comment_excerpt(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "comment_id": comment.get("comment_id"),
        "context_unit_id": comment.get("context_unit_id"),
        "video_id": comment.get("video_id"),
        "event_time_utc": comment.get("event_time_utc"),
        "temporal_role": comment.get("temporal_role"),
        "available_at_trigger": comment.get("available_at_trigger"),
        "relative_to_trigger": comment.get("relative_to_trigger"),
        "is_post_trigger_context": comment.get("is_post_trigger_context"),
        "text": _safe_text(comment.get("text"), 650),
    }


def _build_video_bundles(
    *,
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    comments = context_payload.get("used_context_comments", [])
    units = context_payload.get("selected_context_units", [])
    bundles: list[dict[str, Any]] = []
    for video in validation_input.get("associated_videos", []):
        video_id = video.get("video_id")
        if not video_id:
            continue
        video_units = [unit for unit in units if unit.get("video_id") == video_id]
        video_comments = [
            comment for comment in comments if comment.get("video_id") == video_id
        ]
        unit_ids = [
            unit.get("context_unit_id")
            for unit in video_units
            if unit.get("context_unit_id")
        ]
        comment_ids = [
            comment.get("comment_id")
            for comment in video_comments
            if comment.get("comment_id")
        ]
        bundles.append(
            {
                "event_id": validation_input.get("event_id"),
                "video_id": video_id,
                "video": video,
                "context_units": video_units,
                "comments": video_comments,
                "associated_context_unit_ids": unit_ids,
                "source_comment_ids": comment_ids,
                "post_trigger_context_used": any(
                    bool(comment.get("is_post_trigger_context"))
                    for comment in video_comments
                ),
            }
        )
    return bundles


def _verify_video_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for bundle in bundles:
        video_id = bundle["video_id"]
        for unit in bundle["context_units"]:
            if unit.get("video_id") != video_id:
                errors.append(
                    {
                        "type": "context_unit_video_mismatch",
                        "video_id": video_id,
                        "context_unit_id": unit.get("context_unit_id"),
                        "unit_video_id": unit.get("video_id"),
                    }
                )
        for comment in bundle["comments"]:
            if comment.get("video_id") != video_id:
                errors.append(
                    {
                        "type": "comment_video_mismatch",
                        "video_id": video_id,
                        "comment_id": comment.get("comment_id"),
                        "comment_video_id": comment.get("video_id"),
                    }
                )
    return {"valid": not errors, "errors": errors}


def _first_comment_time_utc(bundle: dict[str, Any]) -> str:
    times = [
        str(comment.get("event_time_utc"))
        for comment in bundle.get("comments", [])
        if comment.get("event_time_utc")
    ]
    return min(times) if times else "9999-12-31T23:59:59+00:00"


def _video_batch_sort_key(bundle: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        -len(bundle.get("source_comment_ids", [])),
        -len(bundle.get("associated_context_unit_ids", [])),
        _first_comment_time_utc(bundle),
        str(bundle.get("video_id") or ""),
    )


def _estimate_bundle_tokens(
    *,
    validation_input: dict[str, Any],
    bundle: dict[str, Any],
) -> int:
    query_payload = _build_video_query_payload(
        validation_input=validation_input,
        bundle=bundle,
    )
    validation_payload = _build_video_validation_payload(
        validation_input=validation_input,
        bundle=bundle,
        query_record=None,
        external_evidence=[],
    )
    text = json.dumps(
        {
            "query_payload": query_payload,
            "validation_payload_without_external_evidence": validation_payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, round(len(text) / 4))


def _estimate_incremental_cost_usd(
    *,
    config: RagG2HierarchicalConfig,
    estimated_tokens: int,
    llm_calls: int,
    serper_calls: int,
) -> float | None:
    cost_config = config.params.get("cost_estimation") or {}
    if not isinstance(cost_config, dict):
        return None
    token_cost_per_1k = float(cost_config.get("usd_per_1k_estimated_tokens", 0) or 0)
    llm_call_cost = float(cost_config.get("usd_per_llm_call", 0) or 0)
    serper_call_cost = float(cost_config.get("usd_per_serper_call", 0) or 0)
    if token_cost_per_1k <= 0 and llm_call_cost <= 0 and serper_call_cost <= 0:
        return None
    return (
        (estimated_tokens / 1000) * token_cost_per_1k
        + llm_calls * llm_call_cost
        + serper_calls * serper_call_cost
    )


def _video_batch_record(
    *,
    bundle: dict[str, Any],
    batch_id: str,
    order_index: int,
    status: str,
    status_reason: str,
    estimated_tokens: int,
    estimated_llm_calls: int,
    estimated_serper_calls: int,
    estimated_cost_usd: float | None,
) -> dict[str, Any]:
    return {
        "event_id": bundle.get("event_id"),
        "video_id": bundle.get("video_id"),
        "batch_id": batch_id,
        "video_batch_order": order_index,
        "video_batch_status": status,
        "status_reason": status_reason,
        "comment_count": len(bundle.get("source_comment_ids", [])),
        "context_unit_count": len(bundle.get("associated_context_unit_ids", [])),
        "first_comment_time_utc": _first_comment_time_utc(bundle),
        "estimated_tokens": estimated_tokens,
        "estimated_llm_calls": estimated_llm_calls,
        "estimated_serper_calls": estimated_serper_calls,
        "estimated_cost_usd": estimated_cost_usd,
    }


def _select_video_batch(
    *,
    validation_input: dict[str, Any],
    bundles: list[dict[str, Any]],
    config: RagG2HierarchicalConfig,
    batch_id: str,
) -> dict[str, Any]:
    sorted_bundles = sorted(bundles, key=_video_batch_sort_key)
    start_index = max(0, (config.batch_index - 1) * config.max_videos_per_event_batch)

    selected: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    limit_reasons: list[str] = []
    used_tokens = 0
    used_llm_calls = 0
    used_serper_calls = 0
    used_cost = 0.0
    cost_enforced = False
    batch_closed_reason: str | None = None

    sorted_ids = [bundle["video_id"] for bundle in sorted_bundles]
    previous_batch_ids = set(sorted_ids[:start_index])

    for absolute_index, bundle in enumerate(sorted_bundles, start=1):
        estimated_tokens = _estimate_bundle_tokens(
            validation_input=validation_input,
            bundle=bundle,
        )
        estimated_llm_calls = 1 + int(
            not has_sufficient_search_metadata(bundle)
        )
        estimated_serper_calls = 1
        estimated_cost = _estimate_incremental_cost_usd(
            config=config,
            estimated_tokens=estimated_tokens,
            llm_calls=estimated_llm_calls,
            serper_calls=estimated_serper_calls,
        )
        if estimated_cost is not None:
            cost_enforced = True

        if bundle["video_id"] in previous_batch_ids:
            records.append(
                _video_batch_record(
                    bundle=bundle,
                    batch_id=batch_id,
                    order_index=absolute_index,
                    status="pending_batch",
                    status_reason="belongs_to_previous_batch_window",
                    estimated_tokens=estimated_tokens,
                    estimated_llm_calls=estimated_llm_calls,
                    estimated_serper_calls=estimated_serper_calls,
                    estimated_cost_usd=estimated_cost,
                )
            )
            continue

        if batch_closed_reason:
            status = (
                "pending_batch"
                if batch_closed_reason == "max_videos_per_event_batch"
                else "pending_budget_limit"
            )
            records.append(
                _video_batch_record(
                    bundle=bundle,
                    batch_id=batch_id,
                    order_index=absolute_index,
                    status=status,
                    status_reason=batch_closed_reason,
                    estimated_tokens=estimated_tokens,
                    estimated_llm_calls=estimated_llm_calls,
                    estimated_serper_calls=estimated_serper_calls,
                    estimated_cost_usd=estimated_cost,
                )
            )
            continue

        if not bundle.get("comments") or not bundle.get("context_units"):
            records.append(
                _video_batch_record(
                    bundle=bundle,
                    batch_id=batch_id,
                    order_index=absolute_index,
                    status="skipped_no_context",
                    status_reason="missing_comments_or_context_units_for_video",
                    estimated_tokens=estimated_tokens,
                    estimated_llm_calls=0,
                    estimated_serper_calls=0,
                    estimated_cost_usd=0.0 if cost_enforced else None,
                )
            )
            continue

        next_cost = used_cost + (estimated_cost or 0.0)
        budget_limit = None
        if len(selected) >= config.max_videos_per_event_batch:
            budget_limit = "max_videos_per_event_batch"
        elif (
            used_tokens + estimated_tokens
            > config.max_estimated_tokens_per_event_batch
        ):
            budget_limit = "max_estimated_tokens_per_event_batch"
        elif used_llm_calls + estimated_llm_calls > config.max_llm_calls_per_batch:
            budget_limit = "max_llm_calls_per_batch"
        elif (
            used_serper_calls + estimated_serper_calls
            > config.max_serper_calls_per_batch
        ):
            budget_limit = "max_serper_calls_per_batch"
        elif (
            cost_enforced
            and next_cost > config.max_estimated_cost_usd_per_batch
        ):
            budget_limit = "max_estimated_cost_usd_per_batch"

        if budget_limit:
            batch_closed_reason = budget_limit
            status = (
                "pending_batch"
                if budget_limit == "max_videos_per_event_batch"
                else "pending_budget_limit"
            )
            if budget_limit not in limit_reasons:
                limit_reasons.append(budget_limit)
            records.append(
                _video_batch_record(
                    bundle=bundle,
                    batch_id=batch_id,
                    order_index=absolute_index,
                    status=status,
                    status_reason=budget_limit,
                    estimated_tokens=estimated_tokens,
                    estimated_llm_calls=estimated_llm_calls,
                    estimated_serper_calls=estimated_serper_calls,
                    estimated_cost_usd=estimated_cost,
                )
            )
            continue

        selected.append(bundle)
        used_tokens += estimated_tokens
        used_llm_calls += estimated_llm_calls
        used_serper_calls += estimated_serper_calls
        used_cost = next_cost
        records.append(
            _video_batch_record(
                bundle=bundle,
                batch_id=batch_id,
                order_index=absolute_index,
                status="processed",
                status_reason="selected_for_current_batch",
                estimated_tokens=estimated_tokens,
                estimated_llm_calls=estimated_llm_calls,
                estimated_serper_calls=estimated_serper_calls,
                estimated_cost_usd=estimated_cost,
            )
        )

    return {
        "batch_id": batch_id,
        "batch_index": config.batch_index,
        "ordered_video_ids": sorted_ids,
        "selected_bundles": selected,
        "video_batch_records": records,
        "pending_video_ids": [
            record["video_id"]
            for record in records
            if str(record.get("video_batch_status", "")).startswith("pending_")
        ],
        "skipped_video_ids": [
            record["video_id"]
            for record in records
            if record.get("video_batch_status") == "skipped_no_context"
        ],
        "limits_reached": limit_reasons,
        "batch_estimates": {
            "estimated_tokens": used_tokens,
            "estimated_llm_calls": used_llm_calls,
            "estimated_serper_calls": used_serper_calls,
            "estimated_cost_usd": used_cost if cost_enforced else None,
            "cost_enforced": cost_enforced,
            "cost_estimation_note": (
                "USD cost limit enforced from config.params.cost_estimation."
                if cost_enforced
                else "USD cost limit recorded but not enforced because no pricing rates were provided."
            ),
        },
    }


def _query_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }


def _build_video_query_payload(
    *,
    validation_input: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "video_id": bundle["video_id"],
        "video": {
            "video_id": bundle["video"].get("video_id"),
            "title": bundle["video"].get("title"),
            "channel_title": bundle["video"].get("channel_title"),
        },
        "context_units": [
            {
                "context_unit_id": unit.get("context_unit_id"),
                "context_type": unit.get("context_type"),
                "comment_ids": unit.get("comment_ids", []),
            }
            for unit in bundle["context_units"]
        ],
        "comments": [_comment_excerpt(comment) for comment in bundle["comments"]],
        "query_constraints": {
            "target_words": "4-10",
            "language": "es",
            "must_use_only_this_video_evidence": True,
            "do_not_use_other_videos_in_same_event": True,
            "do_not_invent_entities_places_or_facts": True,
            "allowed_extraction": [
                "observable_person_or_entity",
                "observable_topic",
                "observable_slogan_or_phrase",
            ],
        },
    }


def _clean_metadata_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _video_metadata(bundle_or_video: dict[str, Any]) -> dict[str, Any]:
    candidate = bundle_or_video.get("video")
    return candidate if isinstance(candidate, dict) else bundle_or_video


def has_sufficient_search_metadata(bundle_or_video: dict[str, Any]) -> bool:
    """A descriptive title is sufficient; channel/date only enrich it."""

    video = _video_metadata(bundle_or_video)
    title = _clean_metadata_text(video.get("title") or video.get("video_title"))
    if not title:
        return False
    normalized = title.casefold().strip(" .,:;!?-_()[]{}")
    return normalized not in _GENERIC_VIDEO_TITLES


def _deterministic_metadata_query(
    bundle_or_video: dict[str, Any],
) -> tuple[str, list[str]]:
    if not has_sufficient_search_metadata(bundle_or_video):
        raise ValueError("Video metadata is insufficient for deterministic query generation.")
    video = _video_metadata(bundle_or_video)
    title = _clean_metadata_text(video.get("title") or video.get("video_title"))
    channel = _clean_metadata_text(
        video.get("channel_title") or video.get("channel_name")
    )
    raw_date = _clean_metadata_text(
        video.get("published_at")
        or video.get("published_at_utc")
        or video.get("video_published_at")
        or video.get("publication_date")
    )
    date = raw_date[:10] if len(raw_date) >= 10 else raw_date
    parts = [title]
    fields = ["video_title"]
    if channel and channel.casefold() not in title.casefold():
        parts.append(channel)
        fields.append("channel_name")
    if date and date not in title:
        parts.append(date)
        fields.append("video_date")
    return " ".join(parts), fields


def _video_evidence_text(query_payload: dict[str, Any]) -> str:
    parts = [
        str(query_payload.get("video", {}).get("title") or ""),
        str(query_payload.get("video", {}).get("channel_title") or ""),
    ]
    for unit in query_payload.get("context_units", []):
        parts.append(str(unit.get("context_text") or ""))
    for comment in query_payload.get("comments", []):
        parts.append(str(comment.get("text") or ""))
    return "\n".join(parts)


def _classify_video_query(
    query: str,
    query_payload: dict[str, Any],
    *,
    query_kind: str = "query",
) -> tuple[str, list[str], dict[str, Any]]:
    clean_query = " ".join(str(query or "").split())
    if not clean_query:
        return "invalid_invented_content", ["empty_query"], {}
    query_tokens = [token for token in _tokens(clean_query) if len(token) > 2]
    evidence_tokens = set(_tokens(_video_evidence_text(query_payload)))
    unseen_tokens = [token for token in query_tokens if token not in evidence_tokens]
    limitations: list[str] = []
    title_text = " ".join(
        [
            str(query_payload.get("video", {}).get("title") or ""),
            str(query_payload.get("video", {}).get("channel_title") or ""),
        ]
    )
    title_tokens = set(_tokens(title_text))
    unsupported_event_phrases = [
        "aparicion publica",
        "aparición pública",
        "declaracion",
        "declaración",
        "entrevista",
        "acto de campaña",
        "video oficial",
        "campaign event",
        "public appearance",
        "official video",
    ]
    evidence_text = _video_evidence_text(query_payload).casefold()
    unsupported_phrases = [
        phrase
        for phrase in unsupported_event_phrases
        if phrase in clean_query.casefold() and phrase not in evidence_text
    ]
    primary_claim_only_tokens: list[str] = []
    if unsupported_phrases:
        return (
            "invalid_invented_content",
            ["query_contains_unsupported_event_inference"],
            {
                "query_tokens": query_tokens,
                "unseen_tokens": unseen_tokens,
                "unseen_ratio": (
                    len(unseen_tokens) / len(query_tokens) if query_tokens else None
                ),
                "unsupported_phrases": unsupported_phrases,
                "primary_claim_only_tokens": [],
            },
        )
    if query_tokens and len(unseen_tokens) / len(query_tokens) > 0.6:
        return (
            "invalid_invented_content",
            ["query_contains_many_terms_not_present_in_this_video_evidence"],
            {
                "query_tokens": query_tokens,
                "unseen_tokens": unseen_tokens,
                "unseen_ratio": len(unseen_tokens) / len(query_tokens),
                "primary_claim_only_tokens": primary_claim_only_tokens,
            },
        )
    if len(clean_query.split()) > 10:
        limitations.append("query_exceeds_recommended_word_count")
    if len(query_tokens) < 3:
        limitations.append("query_has_few_informative_terms")
    return (
        "broad_but_valid" if limitations else "valid",
        limitations,
        {
            "query_tokens": query_tokens,
            "unseen_tokens": unseen_tokens,
            "unseen_ratio": (
                len(unseen_tokens) / len(query_tokens) if query_tokens else None
            ),
            "primary_claim_only_tokens": primary_claim_only_tokens,
        },
    )


def _query_prompt_text(query_payload: dict[str, Any]) -> tuple[str, str]:
    developer_prompt = (
        "Eres un asistente de recuperacion de noticias para una validacion RAG "
        "academica. La metadata del video fue insuficiente; genera una sola query "
        "de busqueda para ese video.\n\n"
        "Usa unicamente los comentarios y relaciones incluidos en el payload. "
        "No uses entidades, comentarios ni "
        "hechos de otros videos del mismo evento. No inventes nombres, lugares, "
        "organizaciones, hechos ni contexto externo.\n\n"
        "Extrae solamente la persona o entidad principal observable, el tema "
        "observable y, si ayuda, un eslogan o frase que aparezca en la evidencia. "
        "No infieras entrevista, aparicion publica, declaracion, acto de campaña, "
        "video oficial, evento, manifestacion ni coordinacion salvo que esas palabras "
        "aparezcan explicitamente en la evidencia.\n\n"
        "La query debe ser neutral, orientada a busqueda, idealmente de 4 a 10 "
        "palabras. Devuelve solo JSON con la propiedad query."
    )
    user_prompt = (
        "Payload para generar query externa por video:\n\n"
        + json.dumps(query_payload, ensure_ascii=False, indent=2)
    )
    return developer_prompt, user_prompt


def _query_record_from_model(
    *,
    query_candidate_id: str,
    query_result: dict[str, Any],
    query_status: str,
    query_limitations: list[str],
    query_scope_verification: dict[str, Any],
    validation_input: dict[str, Any],
    bundle: dict[str, Any],
    search_start: str,
    search_end: str,
    config: RagG2HierarchicalConfig,
    generated_at_utc: str,
) -> dict[str, Any]:
    query = " ".join(str(query_result.get("query") or "").split())
    return {
        "query_candidate_id": query_candidate_id,
        "event_id": validation_input.get("event_id"),
        "video_id": bundle["video_id"],
        "query_text": query,
        "query": query,
        "query_status": query_status,
        "query_scope_verification": query_scope_verification,
        "query_generation_mode": LLM_FALLBACK,
        "metadata_fields_used": [],
        "query_prompt_version": QUERY_PROMPT_VERSION,
        "model_name": config.query_model,
        "provider": config.provider,
        "temperature_requested": config.temperature,
        "temperature_sent": _temperature_sent_value(
            config.query_model, config.temperature
        ),
        "trigger_time_utc": validation_input.get("trigger_time_utc"),
        "query_time_window_start_utc": search_start,
        "query_time_window_end_utc": search_end,
        "generated_at_utc": generated_at_utc,
        "limitations": query_limitations,
    }


def _query_record_from_metadata(
    *,
    query_candidate_id: str,
    validation_input: dict[str, Any],
    bundle: dict[str, Any],
    search_start: str,
    search_end: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    query, fields = _deterministic_metadata_query(bundle)
    return {
        "query_candidate_id": query_candidate_id,
        "event_id": validation_input.get("event_id"),
        "video_id": bundle["video_id"],
        "query_text": query,
        "query": query,
        "query_status": "valid",
        "query_scope_verification": {
            "deterministic": True,
            "source": "video_metadata",
        },
        "query_generation_mode": DETERMINISTIC_METADATA,
        "metadata_fields_used": fields,
        "trigger_time_utc": validation_input.get("trigger_time_utc"),
        "query_time_window_start_utc": search_start,
        "query_time_window_end_utc": search_end,
        "generated_at_utc": generated_at_utc,
        "limitations": [],
    }


def _normalize_video_external_evidence(
    *,
    event_id: str,
    video_id: str,
    query_candidate_id: str,
    query: str,
    serper_response: dict[str, Any],
    trigger_time_utc: str,
    retrieved_at_utc: str,
) -> list[dict[str, Any]]:
    body = serper_response.get("body", {})
    results = body.get("news", []) if isinstance(body, dict) else []
    records: list[dict[str, Any]] = []
    for rank, result in enumerate(results or [], start=1):
        if not isinstance(result, dict):
            continue
        published_at = result.get("date")
        publication_time = parse_external_publication_time(published_at)
        records.append(
            {
                "external_evidence_id": _external_evidence_id(
                    event_id, video_id, query_candidate_id, rank, result
                ),
                "event_id": event_id,
                "video_id": video_id,
                "query_candidate_id": query_candidate_id,
                "query": query,
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "source": result.get("source"),
                "link": result.get("link"),
                "published_at": published_at,
                **publication_time.to_mapping(),
                "retrieved_at_utc": retrieved_at_utc,
                "provider": "serper_news",
                "rank": rank,
                "temporal_relation_to_trigger": _temporal_relation(
                    published_at, trigger_time_utc
                ),
                "raw_result_ref": {
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "source": result.get("source"),
                    "link": result.get("link"),
                    "date": result.get("date"),
                    "position": result.get("position"),
                },
            }
        )
    return records


def _filter_external_evidence_by_cutoff(
    records: list[dict[str, Any]],
    cutoff_utc: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the optional causal boundary without changing legacy G-2 behavior."""

    if cutoff_utc is None:
        return records, {
            "kept": len(records),
            "excluded_at_or_after_cutoff": 0,
            "excluded_unknown_published_at": 0,
            "same_day_time_unknown": 0,
        }
    kept: list[dict[str, Any]] = []
    excluded_future = 0
    excluded_unknown = 0
    same_day_unknown = 0
    for record in records:
        publication_time = external_publication_time_from_record(record)
        relation = temporal_relation_to_cutoff(publication_time, cutoff_utc)
        enriched = {
            **record,
            **publication_time.to_mapping(),
            "temporal_relation_to_cutoff": relation,
        }
        if relation == FUTURE_RELATIVE_TO_CUTOFF:
            excluded_future += 1
            continue
        if relation != CAUSALLY_AVAILABLE:
            excluded_unknown += 1
            if relation == SAME_DAY_TIME_UNKNOWN:
                same_day_unknown += 1
            continue
        kept.append(enriched)
    return kept, {
        "kept": len(kept),
        "excluded_at_or_after_cutoff": excluded_future,
        "excluded_unknown_published_at": excluded_unknown,
        "same_day_time_unknown": same_day_unknown,
    }


def _video_validation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "validation_status": {
                "type": "string",
                "enum": sorted(VALIDATION_STATUS_VALUES),
            },
            "event_interpretation": {
                "type": "string",
                "enum": sorted(EVENT_INTERPRETATION_VALUES),
            },
            "confidence_label": {
                "type": "string",
                "enum": sorted(CONFIDENCE_LABEL_VALUES),
            },
            "internal_evidence_summary": {"type": "string"},
            "external_evidence_summary": {"type": "string"},
            "external_evidence_assessment": {
                "type": "string",
                "enum": sorted(EXTERNAL_EVIDENCE_ASSESSMENT_VALUES),
            },
            "reasoning_summary": {"type": "string"},
            "cited_comment_ids": {"type": "array", "items": {"type": "string"}},
            "cited_context_unit_ids": {"type": "array", "items": {"type": "string"}},
            "cited_external_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "validation_status",
            "event_interpretation",
            "confidence_label",
            "internal_evidence_summary",
            "external_evidence_summary",
            "external_evidence_assessment",
            "reasoning_summary",
            "cited_comment_ids",
            "cited_context_unit_ids",
            "cited_external_evidence_ids",
            "limitations",
        ],
    }


def _build_video_validation_payload(
    *,
    validation_input: dict[str, Any],
    bundle: dict[str, Any],
    query_record: dict[str, Any] | None,
    external_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "event_metadata": {
            "event_id": validation_input.get("event_id"),
            "trigger_time_utc": validation_input.get("trigger_time_utc"),
            "window_start_utc": validation_input.get("window_start_utc"),
            "window_end_utc": validation_input.get("window_end_utc"),
            "trigger_volume": validation_input.get("trigger_volume"),
            "trigger_strength": validation_input.get("trigger_strength"),
        },
        "video": {
            key: value
            for key, value in bundle["video"].items()
            if key in {"video_id", "title", "channel_title", "published_at"}
        },
        "hierarchy_policy": {
            "validation_unit": "event_id + video_id",
            "do_not_use_evidence_from_other_videos": True,
            "do_not_attribute_external_evidence_to_other_videos": True,
        },
        "temporal_policy": {
            "alert_evidence_comments": "window_start_utc <= event_time_utc <= trigger_time_utc",
            "validation_context_comments": "window_start_utc <= event_time_utc <= window_end_utc",
            "post_trigger_context_used": bundle["post_trigger_context_used"],
            "do_not_treat_post_trigger_comments_as_alert_cause": True,
        },
        "internal_evidence": {
            "context_units": [
                {key: value for key, value in unit.items() if key != "context_text"}
                for unit in bundle["context_units"]
            ],
            "comments": [_comment_excerpt(comment) for comment in bundle["comments"]],
        },
        "external_retrieval": {
            "query": (
                {
                    key: query_record.get(key)
                    for key in [
                        "query_candidate_id",
                        "query_text",
                        "query_generation_mode",
                        "metadata_fields_used",
                    ]
                }
                if query_record
                else None
            ),
            "external_evidence": [
                {
                    key: record.get(key)
                    for key in [
                        "external_evidence_id",
                        "title",
                        "snippet",
                        "source",
                        "link",
                        "raw_publication_date",
                        "parsed_publication_time_utc",
                        "publication_time_precision",
                        "temporal_relation_to_cutoff",
                    ]
                    if key in record
                }
                for record in external_evidence
            ],
        },
        "allowed_validation_status": sorted(VALIDATION_STATUS_VALUES),
        "allowed_event_interpretation": sorted(EVENT_INTERPRETATION_VALUES),
        "allowed_external_evidence_assessment": sorted(
            EXTERNAL_EVIDENCE_ASSESSMENT_VALUES
        ),
    }


def _video_validation_prompt_text(payload: dict[str, Any]) -> tuple[str, str]:
    developer_prompt = (
        "Eres un evaluador academico para una validacion RAG G-2 por video. "
        "Tu unidad de analisis es exactamente event_id + video_id.\n\n"
        "Usa solo la evidencia interna y externa con ese mismo event_id y video_id. "
        "No uses evidencia de otros videos del mismo evento. No atribuyas evidencia "
        "externa recuperada para este video a otros videos. No uses conocimiento "
        "externo no incluido en el payload.\n\n"
        "Cita comment_id cuando uses comentarios, context_unit_id cuando uses "
        "unidades internas y external_evidence_id cuando uses evidencia externa. "
        "G-2 responde exclusivamente si la evidencia externa causalmente disponible "
        "corrobora el fenomeno. Estados: CONFIRMED requiere evidencia externa valida "
        "que lo respalde; INSUFFICIENT_EVIDENCE significa que la evidencia externa "
        "falta, es parcial, debil o inconclusa; NOT_CONFIRMED requiere evidencia "
        "externa suficiente para evaluar que no lo respalda o lo contradice. La "
        "evidencia interna nunca basta para CONFIRMED ni NOT_CONFIRMED en G-2.\n\n"
        "external_event requiere al menos una cita externa valida. Si este video no "
        "recupero evidencia externa, usa INSUFFICIENT_EVIDENCE.\n\n"
        "No presentes comentarios posteriores al trigger como causa de la alerta. "
        "No modifiques la decision original del detector; evalua solo la validacion "
        "posterior para este video.\n\n"
        "Devuelve solo JSON valido que cumpla el esquema."
    )
    user_prompt = (
        "Payload de validacion G-2 por video:\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return developer_prompt, user_prompt


def _verify_video_report(
    *,
    report: dict[str, Any],
    bundle: dict[str, Any],
    external_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_comment_ids = set(bundle["source_comment_ids"])
    allowed_context_unit_ids = set(bundle["associated_context_unit_ids"])
    allowed_external_ids = {
        record["external_evidence_id"] for record in external_evidence
    }
    cited_comment_ids = set(report.get("cited_comment_ids") or [])
    cited_context_unit_ids = set(report.get("cited_context_unit_ids") or [])
    cited_external_ids = set(report.get("cited_external_evidence_ids") or [])
    errors: list[str] = []

    invalid_cited_comment_ids = sorted(cited_comment_ids - allowed_comment_ids)
    invalid_cited_context_unit_ids = sorted(
        cited_context_unit_ids - allowed_context_unit_ids
    )
    invalid_cited_external_ids = sorted(cited_external_ids - allowed_external_ids)
    if invalid_cited_comment_ids:
        errors.append("invalid_cited_comment_ids_for_video")
    if invalid_cited_context_unit_ids:
        errors.append("invalid_cited_context_unit_ids_for_video")
    if invalid_cited_external_ids:
        errors.append("invalid_cited_external_evidence_ids_for_video")
    valid_internal_citation = bool(
        cited_comment_ids & allowed_comment_ids
        or cited_context_unit_ids & allowed_context_unit_ids
    )
    valid_external_citation = bool(cited_external_ids & allowed_external_ids)
    status = report.get("validation_status")
    assessment = report.get("external_evidence_assessment")
    if status in {CONFIRMED, NOT_CONFIRMED} and not valid_external_citation:
        errors.append("decisive_g2_status_requires_valid_external_citation")
    if status == CONFIRMED and assessment != "supports":
        errors.append("confirmed_requires_supporting_external_assessment")
    if status == NOT_CONFIRMED and assessment != "contradicts":
        errors.append("not_confirmed_requires_contradicting_external_assessment")
    if not allowed_external_ids and status != INSUFFICIENT_EVIDENCE:
        errors.append("no_external_evidence_requires_insufficient_evidence")
    if report.get("event_interpretation") == "external_event" and not valid_external_citation:
        errors.append("external_event_requires_valid_external_citation")
    if not allowed_external_ids and report.get("event_interpretation") == "external_event":
        errors.append("external_event_without_external_evidence_for_video")
    return {
        "valid": not errors,
        "errors": errors,
        "invalid_cited_comment_ids": invalid_cited_comment_ids,
        "invalid_cited_context_unit_ids": invalid_cited_context_unit_ids,
        "invalid_cited_external_evidence_ids": invalid_cited_external_ids,
        "invalid_used_context_comment_ids": [],
        "invalid_used_external_evidence_ids": [],
        "valid_internal_citation": valid_internal_citation,
        "valid_external_citation": valid_external_citation,
        "allowed_comment_count": len(allowed_comment_ids),
        "allowed_context_unit_count": len(allowed_context_unit_ids),
        "allowed_external_evidence_count": len(allowed_external_ids),
        "allowed_comment_ids": sorted(allowed_comment_ids),
        "allowed_external_evidence_ids": sorted(allowed_external_ids),
        "post_trigger_context_used": bool(bundle["post_trigger_context_used"]),
    }


def _finalize_video_report(
    *,
    model_report: dict[str, Any],
    video_validation_id: str,
    event_id: str,
    video_id: str,
    model: str,
    validated_at_utc: str,
    verification: dict[str, Any],
    attempt_count: int,
    raw_response_id: str | None,
    query_candidate_id: str | None,
    external_evidence_count: int,
) -> dict[str, Any]:
    report = dict(model_report)
    report["video_validation_id"] = video_validation_id
    report["event_id"] = event_id
    report["video_id"] = video_id
    report["model_name"] = model
    report["prompt_version"] = VALIDATION_PROMPT_VERSION
    report["validated_at_utc"] = validated_at_utc
    report["used_context_comment_ids"] = list(
        verification.get("allowed_comment_ids", [])
    )
    report["used_external_evidence_ids"] = list(
        verification.get("allowed_external_evidence_ids", [])
    )
    report["post_trigger_context_used"] = bool(
        verification.get("post_trigger_context_used")
    )
    report["citation_verification"] = verification
    report["attempt_count"] = attempt_count
    report["raw_response_id"] = raw_response_id
    report["query_candidate_id"] = query_candidate_id
    report["external_evidence_count"] = external_evidence_count
    return report


def _error_video_report(
    *,
    video_validation_id: str,
    event_id: str,
    video_id: str,
    model: str,
    validated_at_utc: str,
    bundle: dict[str, Any],
    errors: list[str],
    attempt_count: int,
    query_candidate_id: str | None,
    external_evidence_count: int,
) -> dict[str, Any]:
    return {
        "video_validation_id": video_validation_id,
        "event_id": event_id,
        "video_id": video_id,
        "validation_status": INSUFFICIENT_EVIDENCE,
        "event_interpretation": "unclear",
        "confidence_label": "low",
        "internal_evidence_summary": "",
        "external_evidence_summary": "",
        "external_evidence_assessment": (
            "no_external_evidence" if external_evidence_count == 0 else "inconclusive"
        ),
        "reasoning_summary": "",
        "cited_comment_ids": [],
        "cited_context_unit_ids": [],
        "cited_external_evidence_ids": [],
        "used_context_comment_ids": bundle["source_comment_ids"],
        "used_external_evidence_ids": [],
        "post_trigger_context_used": bundle["post_trigger_context_used"],
        "limitations": [
            "La validacion G-2 por video fallo o no cumplio el contrato requerido.",
            *errors,
        ],
        "model_name": model,
        "prompt_version": VALIDATION_PROMPT_VERSION,
        "validated_at_utc": validated_at_utc,
        "citation_verification": {"valid": False, "errors": errors},
        "attempt_count": attempt_count,
        "query_candidate_id": query_candidate_id,
        "external_evidence_count": external_evidence_count,
        "generation_error": True,
    }


def _build_event_summary(
    *,
    event_id: str,
    video_reports: list[dict[str, Any]],
    external_evidence: list[dict[str, Any]],
    video_batch_records: list[dict[str, Any]],
    batch_id: str,
    created_at_utc: str,
) -> dict[str, Any]:
    video_validation_ids = [
        report.get("video_validation_id")
        for report in video_reports
        if report.get("video_validation_id")
    ]
    videos_evaluated = [
        report.get("video_id") for report in video_reports if report.get("video_id")
    ]
    videos_with_external_evidence = sorted(
        {
            record["video_id"]
            for record in external_evidence
            if record.get("event_id") == event_id and record.get("video_id")
        }
    )
    videos_without_external_evidence = sorted(
        set(videos_evaluated) - set(videos_with_external_evidence)
    )
    videos_total = len(video_batch_records)
    videos_processed = len(
        [
            record
            for record in video_batch_records
            if record.get("video_batch_status") == "processed"
        ]
    )
    pending_video_ids = [
        record.get("video_id")
        for record in video_batch_records
        if str(record.get("video_batch_status", "")).startswith("pending_")
    ]
    skipped_video_ids = [
        record.get("video_id")
        for record in video_batch_records
        if record.get("video_batch_status") == "skipped_no_context"
    ]
    failed_video_ids = [
        record.get("video_id")
        for record in video_batch_records
        if record.get("video_batch_status") in {"failed_retrieval", "failed_validation"}
    ]
    videos_pending = len(pending_video_ids)
    next_batch_required = bool(pending_video_ids)
    summary_completeness = (
        "complete" if videos_processed + len(skipped_video_ids) == videos_total else "partial"
    )
    assessments = {
        report.get("video_id"): report.get("external_evidence_assessment")
        for report in video_reports
    }
    interpretations = {
        report.get("video_id"): report.get("event_interpretation")
        for report in video_reports
    }
    statuses = {
        report.get("video_id"): normalize_validation_status(
            report.get("validation_status")
        )
        for report in video_reports
    }

    if not videos_with_external_evidence:
        agreement = "insufficient"
        overall_status = INSUFFICIENT_EVIDENCE
        if set(interpretations.values()) == {"internal_community_reaction"}:
            overall_interpretation = "internal_community_reaction"
        else:
            overall_interpretation = "unclear"
    elif "contradicts" in set(assessments.values()) and "supports" in set(
        assessments.values()
    ):
        agreement = "contradictory"
        overall_status = INSUFFICIENT_EVIDENCE
        overall_interpretation = "unclear"
    elif len(set(interpretations.values())) == 1 and len(set(assessments.values())) == 1:
        agreement = "consistent"
        overall_status = next(iter(statuses.values()), INSUFFICIENT_EVIDENCE)
        overall_interpretation = next(iter(interpretations.values()), "unclear")
    else:
        agreement = "mixed"
        overall_status = INSUFFICIENT_EVIDENCE
        overall_interpretation = "unclear"

    if pending_video_ids:
        summary_completeness = "partial"
        overall_status = INSUFFICIENT_EVIDENCE
        if not video_reports:
            overall_interpretation = "unclear"

    # A decisive candidate-level status is emitted only when every evaluated
    # video agrees. Mixed, incomplete or inconclusive evidence stays unknown.
    normalized_statuses = set(statuses.values())
    if videos_with_external_evidence and not pending_video_ids:
        if normalized_statuses == {CONFIRMED}:
            overall_status = CONFIRMED
        elif normalized_statuses == {NOT_CONFIRMED}:
            overall_status = NOT_CONFIRMED
        elif normalized_statuses:
            overall_status = INSUFFICIENT_EVIDENCE

    global_limitations: list[str] = []
    if videos_without_external_evidence:
        global_limitations.append(
            "No todos los videos recuperaron evidencia externa: "
            + ", ".join(videos_without_external_evidence)
        )
    if agreement in {"mixed", "contradictory"}:
        global_limitations.append(
            "La sintesis global no transfiere evidencia externa entre videos."
        )
    if agreement == "insufficient":
        global_limitations.append(
            "No se recupero evidencia externa suficiente para contrastar el evento global."
        )
    if pending_video_ids:
        global_limitations.append(
            "Sintesis parcial: quedan videos pendientes de procesamiento por lote: "
            + ", ".join(str(video_id) for video_id in pending_video_ids)
        )

    if not video_reports and failed_video_ids:
        event_batch_status = "failed"
    elif not video_reports and pending_video_ids:
        event_batch_status = "not_started"
    elif failed_video_ids:
        event_batch_status = "partial_errors"
    elif pending_video_ids:
        event_batch_status = "partial_pending_batch"
    else:
        event_batch_status = "complete"

    return {
        "event_summary_id": _event_summary_id(event_id, video_validation_ids),
        "event_id": event_id,
        "batch_id": batch_id,
        "video_validation_ids": video_validation_ids,
        "videos_evaluated": videos_evaluated,
        "videos_total": videos_total,
        "videos_processed": videos_processed,
        "videos_pending": videos_pending,
        "pending_video_ids": pending_video_ids,
        "skipped_video_ids": skipped_video_ids,
        "failed_video_ids": failed_video_ids,
        "next_batch_required": next_batch_required,
        "summary_completeness": summary_completeness,
        "event_batch_status": event_batch_status,
        "videos_with_external_evidence": videos_with_external_evidence,
        "videos_without_external_evidence": videos_without_external_evidence,
        "overall_validation_status": overall_status,
        "overall_interpretation": overall_interpretation,
        "agreement_between_videos": agreement,
        "summary_reasoning": (
            "Sintesis determinista calculada desde los reportes por video. "
            "La evidencia externa de un video no se atribuye a otros videos."
        ),
        "global_limitations": global_limitations,
        "created_at_utc": created_at_utc,
    }


def _deterministic_query_preview(bundle: dict[str, Any]) -> str | None:
    if not has_sufficient_search_metadata(bundle):
        return None
    return _deterministic_metadata_query(bundle)[0]


def _dry_run_event_plan(
    *,
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
    config: RagG2HierarchicalConfig,
    created_at_utc: str,
) -> dict[str, Any]:
    event_id = str(validation_input.get("event_id") or "")
    bundles = _build_video_bundles(
        validation_input=validation_input,
        context_payload=context_payload,
    )
    bundle_verification = _verify_video_bundles(bundles)
    batch_id = config.batch_id or (
        "dryrun_batch_"
        + _short_hash(
            event_id,
            str(config.batch_index),
            str(config.max_videos_per_event_batch),
            created_at_utc,
        )
    )
    batch_plan = _select_video_batch(
        validation_input=validation_input,
        bundles=bundles,
        config=config,
        batch_id=batch_id,
    )
    search_start, search_end = _search_window(
        str(validation_input.get("trigger_time_utc")),
        config.search_days_before,
        config.search_days_after,
    )
    search_end = _bounded_search_end(
        search_end,
        config.external_evidence_cutoff_utc,
    )
    bundle_by_video = {bundle["video_id"]: bundle for bundle in bundles}
    video_plans: list[dict[str, Any]] = []
    for record in batch_plan["video_batch_records"]:
        bundle = bundle_by_video.get(record.get("video_id"), {})
        processed = record.get("video_batch_status") == "processed"
        preview = _deterministic_query_preview(bundle) if processed else None
        mode = (
            DETERMINISTIC_METADATA
            if processed and has_sufficient_search_metadata(bundle)
            else LLM_FALLBACK if processed else None
        )
        video_plans.append(
            {
                **record,
                "query_generation": {
                    "query_generation_mode": mode,
                    "llm_fallback_would_be_used": mode == LLM_FALLBACK,
                    "query_preview_type": (
                        "deterministic_metadata_query"
                        if mode == DETERMINISTIC_METADATA
                        else "not_available_until_llm_fallback"
                        if mode == LLM_FALLBACK
                        else "not_applicable"
                    ),
                    "query_text_preview": preview,
                },
                "search_window": {
                    "window_start_date": search_start,
                    "window_end_date": search_end,
                    "provider": "serper_news",
                    "serper_call_would_be_made": processed,
                },
            }
        )

    videos_total = len(bundles)
    videos_selected = len(batch_plan["selected_bundles"])
    videos_pending = len(batch_plan["pending_video_ids"])
    videos_skipped = len(batch_plan["skipped_video_ids"])
    if videos_selected == 0 and videos_pending == 0 and videos_skipped == 0:
        estimated_event_status = "not_started"
    elif any(
        record.get("video_batch_status") == "pending_budget_limit"
        for record in batch_plan["video_batch_records"]
    ):
        estimated_event_status = "pending_budget_limit"
    elif videos_pending:
        estimated_event_status = "partial_pending_batch"
    elif videos_selected + videos_skipped == videos_total:
        estimated_event_status = "complete_in_one_batch"
    else:
        estimated_event_status = "not_started"

    return {
        "event_id": event_id,
        "batch_id": batch_id,
        "batch_index": config.batch_index,
        "estimated_event_status": estimated_event_status,
        "status_reason": (
            "dry_run_batch_plan_only_no_llm_no_serper_no_writes"
        ),
        "videos_total": videos_total,
        "videos_selected_for_current_batch": videos_selected,
        "videos_pending": videos_pending,
        "pending_video_ids": batch_plan["pending_video_ids"],
        "videos_skipped": videos_skipped,
        "comments_total": sum(len(bundle.get("source_comment_ids", [])) for bundle in bundles),
        "context_units_total": sum(
            len(bundle.get("associated_context_unit_ids", [])) for bundle in bundles
        ),
        "ordered_video_ids": batch_plan["ordered_video_ids"],
        "video_batch_plan": video_plans,
        "batch_estimates": batch_plan["batch_estimates"],
        "limits_reached": batch_plan["limits_reached"],
        "bundle_verification": bundle_verification,
        "search_policy": {
            "provider": "serper_news",
            "endpoint": config.serper_url,
            "gl": config.serper_gl,
            "hl": config.serper_hl,
            "type": config.serper_type,
            "num": config.serper_num_results,
            "window_start_date": search_start,
            "window_end_date": search_end,
            "window_rule": "trigger date +/- configured calendar days",
            "external_evidence_cutoff_utc": config.external_evidence_cutoff_utc,
            "unknown_publication_time_policy": (
                "exclude_when_causal_cutoff_is_configured"
                if config.external_evidence_cutoff_utc is not None
                else "legacy_preserve"
            ),
        },
        "query_policy": {
            "strategy": "metadata_first_deterministic",
            "fallback": "restricted_llm_when_metadata_insufficient",
            "fallback_reasoning": FALLBACK_QUERY_REASONING,
            "claim_verification_query_generated": False,
        },
    }


def plan_rag_g2_hierarchical_dry_run(
    config: RagG2HierarchicalConfig,
) -> dict[str, Any]:
    """Build a deterministic G-2 hierarchical batch plan without network or writes."""
    _validate_config(config, require_single_event=False)
    consumer_paths = _consumer_paths(config.consumer_dir)
    validation_inputs = _read_jsonl_records(consumer_paths["rag_validation_inputs"])
    context_payloads = _read_jsonl_records(consumer_paths["rag_context_payloads"])
    consumer_manifest = _read_json(consumer_paths["rag_consumer_manifest"])
    contexts_by_event = {
        record.get("event_id"): record
        for record in context_payloads
        if record.get("event_id")
    }
    if _is_all_events_request(config.event_id):
        selected_inputs = validation_inputs
    else:
        selected_inputs = [
            record
            for record in validation_inputs
            if record.get("event_id") == config.event_id
        ]
        if not selected_inputs:
            raise ValueError(f"event_id not found in consumer artifacts: {config.event_id}")

    created_at_utc = _utc_now_iso()
    event_plans: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for validation_input in selected_inputs:
        event_id = validation_input.get("event_id")
        context_payload = contexts_by_event.get(event_id)
        if context_payload is None:
            errors.append(
                {
                    "event_id": event_id,
                    "type": "missing_context_payload",
                }
            )
            continue
        event_plans.append(
            _dry_run_event_plan(
                validation_input=validation_input,
                context_payload=context_payload,
                config=config,
                created_at_utc=created_at_utc,
            )
        )

    videos_total = sum(plan["videos_total"] for plan in event_plans)
    videos_processable = sum(
        plan["videos_selected_for_current_batch"] for plan in event_plans
    )
    videos_pending = sum(plan["videos_pending"] for plan in event_plans)
    tokens_estimated = sum(
        int(plan["batch_estimates"].get("estimated_tokens") or 0)
        for plan in event_plans
    )
    llm_calls_estimated = sum(
        int(plan["batch_estimates"].get("estimated_llm_calls") or 0)
        for plan in event_plans
    )
    serper_calls_estimated = sum(
        int(plan["batch_estimates"].get("estimated_serper_calls") or 0)
        for plan in event_plans
    )
    cost_guard_status = _cost_guard_status(config)

    return {
        "run_id": "ragg2h_dryrun_" + _short_hash(created_at_utc, config.event_id),
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_generation_g2_hierarchical_dry_run",
        "mode": "dry_run_no_network_no_writes",
        "dry_run": True,
        "artifact_version": RAG_G2H_ARTIFACT_VERSION,
        "network_calls_made": False,
        "openai_calls_made": False,
        "serper_calls_made": False,
        "embeddings_used": False,
        "vectorstore_used": False,
        "claim_verification_query_generated": False,
        "event_ids_requested": (
            "all_events" if _is_all_events_request(config.event_id) else [config.event_id]
        ),
        "events_planned": len(event_plans),
        "events_with_errors": len(errors),
        "summary": {
            "events_total": len(event_plans),
            "videos_total": videos_total,
            "videos_processable_in_current_batch": videos_processable,
            "videos_pending": videos_pending,
            "estimated_tokens": tokens_estimated,
            "estimated_llm_calls": llm_calls_estimated,
            "estimated_serper_calls": serper_calls_estimated,
            "cost_guard_status": cost_guard_status,
        },
        "configuration": {
            "query_model": config.query_model,
            "validation_model": config.validation_model,
            "provider": config.provider,
            "temperature_requested": config.temperature,
            "query_temperature_sent": _temperature_sent_value(
                config.query_model, config.temperature
            ),
            "validation_temperature_sent": _temperature_sent_value(
                config.validation_model, config.temperature
            ),
            "serper_url": config.serper_url,
            "serper_gl": config.serper_gl,
            "serper_hl": config.serper_hl,
            "serper_type": config.serper_type,
            "serper_num_results": config.serper_num_results,
            "search_days_before": config.search_days_before,
            "search_days_after": config.search_days_after,
            "max_videos_per_event_batch": config.max_videos_per_event_batch,
            "max_estimated_tokens_per_event_batch": (
                config.max_estimated_tokens_per_event_batch
            ),
            "max_llm_calls_per_batch": config.max_llm_calls_per_batch,
            "max_serper_calls_per_batch": config.max_serper_calls_per_batch,
            "max_estimated_cost_usd_per_batch": (
                config.max_estimated_cost_usd_per_batch
            ),
            "batch_index": config.batch_index,
            "cost_guard_status": cost_guard_status,
        },
        "consumer_run_id": consumer_manifest.get("run_id"),
        "input_paths": {
            name: _normalize_path(path) for name, path in consumer_paths.items()
        },
        "scope_policy": {
            "validation_unit": "event_id + video_id",
            "event_summary_unit": "event_id",
            "number_of_videos_is_not_an_exclusion_criterion": True,
            "pending_videos_are_marked_not_excluded": True,
            "query_strategy": "metadata_first_deterministic",
            "claim_verification_query_generated": False,
            "does_not_modify_sidecars": True,
            "does_not_modify_consumer": True,
            "does_not_modify_pipeline": True,
        },
        "video_batch_policy": {
            "batch_order": [
                "comment_count_desc",
                "context_unit_count_desc",
                "first_comment_time_utc_asc",
                "video_id_asc",
            ],
            "allowed_video_batch_status": sorted(VIDEO_BATCH_STATUS_VALUES),
            "allowed_event_batch_status": sorted(EVENT_BATCH_STATUS_VALUES),
        },
        "events": event_plans,
        "errors": errors,
    }


def run_rag_g2_hierarchical_from_config(
    config: RagG2HierarchicalConfig,
) -> dict[str, Any]:
    _validate_config(config, require_single_event=True)

    consumer_paths = _consumer_paths(config.consumer_dir)
    output_paths = _output_paths(config.output_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_inputs = _read_jsonl_records(consumer_paths["rag_validation_inputs"])
    context_payloads = _read_jsonl_records(consumer_paths["rag_context_payloads"])
    consumer_manifest = _read_json(consumer_paths["rag_consumer_manifest"])
    validation_input, context_payload = _extract_event(
        inputs=validation_inputs,
        contexts=context_payloads,
        event_id=config.event_id,
    )
    bundles = _build_video_bundles(
        validation_input=validation_input,
        context_payload=context_payload,
    )
    bundle_verification = _verify_video_bundles(bundles)
    if not bundle_verification["valid"]:
        raise ValueError("Invalid video bundle scope: " + json.dumps(bundle_verification))

    created_at_utc = _utc_now_iso()
    batch_id = config.batch_id or (
        "batch_"
        + _short_hash(
            config.event_id,
            str(config.batch_index),
            str(config.max_videos_per_event_batch),
            created_at_utc,
        )
    )
    batch_plan = _select_video_batch(
        validation_input=validation_input,
        bundles=bundles,
        config=config,
        batch_id=batch_id,
    )
    selected_bundles = batch_plan["selected_bundles"]
    video_batch_records = batch_plan["video_batch_records"]
    search_start, search_end = _search_window(
        str(validation_input.get("trigger_time_utc")),
        config.search_days_before,
        config.search_days_after,
    )
    search_end = _bounded_search_end(
        search_end,
        config.external_evidence_cutoff_utc,
    )
    api_key = _load_api_key()
    serper_api_key = _load_serper_api_key()
    manifest_errors: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    all_external_evidence: list[dict[str, Any]] = []
    video_reports: list[dict[str, Any]] = []
    per_video_runs: list[dict[str, Any]] = []
    external_causality_audit: list[dict[str, Any]] = []
    serper_request_count = 0

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found; no provider fallback attempted.")
    if not serper_api_key:
        raise RuntimeError("SERPER_API_KEY was not found; no provider fallback attempted.")

    query_fallback_required = any(
        not has_sufficient_search_metadata(bundle) for bundle in selected_bundles
    )
    query_model_availability = (
        _check_model_available(
            api_key, config.query_model, config.request_timeout_seconds
        )
        if query_fallback_required
        else {"available": True, "check_skipped": "no_llm_query_fallback_required"}
    )
    validation_model_availability = (
        query_model_availability
        if query_fallback_required and config.validation_model == config.query_model
        else _check_model_available(
            api_key, config.validation_model, config.request_timeout_seconds
        )
    )
    if not query_model_availability.get("available"):
        raise RuntimeError(f"Approved query model {config.query_model} is not available.")
    if not validation_model_availability.get("available"):
        raise RuntimeError(
            f"Approved validation model {config.validation_model} is not available."
        )

    query_prompt_hashes: dict[str, str] = {}
    validation_prompt_hashes: dict[str, str] = {}

    batch_record_by_video = {
        record["video_id"]: record for record in video_batch_records
    }

    for bundle in selected_bundles:
        video_id = bundle["video_id"]
        batch_record = batch_record_by_video[video_id]
        query_candidate_id = _query_candidate_id(
            config.event_id, video_id, QUERY_PROMPT_VERSION, config.query_model
        )
        video_validation_id = _video_validation_id(
            config.event_id, video_id, VALIDATION_PROMPT_VERSION, config.validation_model
        )
        query_record = None
        external_evidence: list[dict[str, Any]] = []
        retrieval_status = "not_started"
        query_payload = _build_video_query_payload(
            validation_input=validation_input,
            bundle=bundle,
        )
        if has_sufficient_search_metadata(bundle):
            query_record = _query_record_from_metadata(
                query_candidate_id=query_candidate_id,
                validation_input=validation_input,
                bundle=bundle,
                search_start=search_start,
                search_end=search_end,
                generated_at_utc=_utc_now_iso(),
            )
        else:
            query_developer_prompt, query_user_prompt = _query_prompt_text(query_payload)
            query_prompt_hash = hashlib.sha256(
                (query_developer_prompt + "\n\n" + query_user_prompt).encode("utf-8")
            ).hexdigest()
            query_prompt_hashes[video_id] = query_prompt_hash
            query_response = _create_json_response(
                api_key=api_key,
                model=config.query_model,
                temperature=config.temperature,
                developer_prompt=query_developer_prompt,
                user_prompt=query_user_prompt,
                schema_name="rag_g2_video_news_query_fallback",
                schema=_query_response_schema(),
                timeout=config.request_timeout_seconds,
                reasoning_effort=FALLBACK_QUERY_REASONING,
                max_output_tokens=FALLBACK_QUERY_MAX_OUTPUT_TOKENS,
            )
            parsed_query, query_response_text, query_parse_error = _parse_model_json(
                query_response
            )
            query_schema_errors = _schema_errors(parsed_query, _query_response_schema())
            raw_records.append(
                {
                    "record_type": "video_query_generation_fallback",
                    "query_candidate_id": query_candidate_id,
                    "event_id": config.event_id,
                    "video_id": video_id,
                    "attempt": 1,
                    "model": config.query_model,
                    "reasoning_effort": FALLBACK_QUERY_REASONING,
                    "max_output_tokens": FALLBACK_QUERY_MAX_OUTPUT_TOKENS,
                    "prompt_version": QUERY_PROMPT_VERSION,
                    "prompt_sha256": query_prompt_hash,
                    "response_status_code": query_response.get("status_code"),
                    "response_text": query_response_text,
                    "parsed_json": parsed_query,
                    "parse_error": query_parse_error,
                    "schema_errors": query_schema_errors,
                    "raw_response": query_response.get("body"),
                    "created_at_utc": _utc_now_iso(),
                }
            )
            if query_parse_error or query_schema_errors or parsed_query is None:
                manifest_errors.append(
                    {
                        "type": "video_query_generation_failed",
                        "video_id": video_id,
                        "parse_error": query_parse_error,
                        "schema_errors": query_schema_errors,
                    }
                )
            else:
                query_status, query_limitations, query_scope_verification = (
                    _classify_video_query(parsed_query.get("query", ""), query_payload)
                )
                query_record = _query_record_from_model(
                    query_candidate_id=query_candidate_id,
                    query_result=parsed_query,
                    query_status=query_status,
                    query_limitations=query_limitations,
                    query_scope_verification=query_scope_verification,
                    validation_input=validation_input,
                    bundle=bundle,
                    search_start=search_start,
                    search_end=search_end,
                    config=config,
                    generated_at_utc=_utc_now_iso(),
                )

        if query_record is not None:
            query_records.append(query_record)
            if query_record["query_status"] == "invalid_invented_content":
                retrieval_status = "skipped_invalid_query"
                batch_record["video_batch_status"] = "failed_retrieval"
                batch_record["status_reason"] = "invalid_video_query"
                manifest_errors.append(
                    {
                        "type": "invalid_video_query",
                        "video_id": video_id,
                        "query_candidate_id": query_candidate_id,
                    }
                )
            else:
                retrieved_at_utc = _utc_now_iso()
                serper_request_count += 1
                serper_response = _call_serper_news(
                    api_key=serper_api_key,
                    query=str(query_record.get("query_text", "")),
                    window_start_date=search_start,
                    window_end_date=search_end,
                    config=config,
                )
                if serper_response.get("status_code") in {401, 403}:
                    raise RuntimeError(
                        "Serper authentication failed; no provider fallback attempted."
                    )
                if not serper_response.get("ok"):
                    retrieval_status = "error"
                    batch_record["video_batch_status"] = "failed_retrieval"
                    batch_record["status_reason"] = "serper_request_failed"
                    manifest_errors.append(
                        {
                            "type": "serper_request_failed",
                            "video_id": video_id,
                            "status_code": serper_response.get("status_code"),
                            "body": serper_response.get("body"),
                        }
                    )
                else:
                    normalized_external_evidence = _normalize_video_external_evidence(
                        event_id=config.event_id,
                        video_id=video_id,
                        query_candidate_id=query_candidate_id,
                        query=str(query_record.get("query_text", "")),
                        serper_response=serper_response,
                        trigger_time_utc=str(validation_input.get("trigger_time_utc")),
                        retrieved_at_utc=retrieved_at_utc,
                    )
                    external_evidence, causal_audit = (
                        _filter_external_evidence_by_cutoff(
                            normalized_external_evidence,
                            config.external_evidence_cutoff_utc,
                        )
                    )
                    external_causality_audit.append(
                        {"video_id": video_id, **causal_audit}
                    )
                    retrieval_status = "success" if external_evidence else "no_results"
                    all_external_evidence.extend(external_evidence)

        validation_payload = _build_video_validation_payload(
            validation_input=validation_input,
            bundle=bundle,
            query_record=query_record,
            external_evidence=external_evidence,
        )
        validation_developer_prompt, validation_user_prompt = (
            _video_validation_prompt_text(validation_payload)
        )
        validation_prompt_hash = hashlib.sha256(
            (validation_developer_prompt + "\n\n" + validation_user_prompt).encode(
                "utf-8"
            )
        ).hexdigest()
        validation_prompt_hashes[video_id] = validation_prompt_hash

        last_errors: list[str] = []
        video_report = None
        for attempt in range(1, config.max_retries + 2):
            validation_response = _create_json_response(
                api_key=api_key,
                model=config.validation_model,
                temperature=config.temperature,
                developer_prompt=validation_developer_prompt,
                user_prompt=validation_user_prompt,
                schema_name="rag_g2_video_validation_report",
                schema=_video_validation_schema(),
                timeout=config.request_timeout_seconds,
            )
            response_body = validation_response.get("body", {})
            raw_id = response_body.get("id") if isinstance(response_body, dict) else None
            parsed_report, response_text, parse_error = _parse_model_json(
                validation_response
            )
            schema_errors = _schema_errors(parsed_report, _video_validation_schema())
            verification = (
                _verify_video_report(
                    report=parsed_report,
                    bundle=bundle,
                    external_evidence=external_evidence,
                )
                if parsed_report is not None and not schema_errors
                else {"valid": False, "errors": ["schema_or_parse_error"]}
            )
            raw_records.append(
                {
                    "record_type": "video_validation_generation",
                    "video_validation_id": video_validation_id,
                    "event_id": config.event_id,
                    "video_id": video_id,
                    "attempt": attempt,
                    "model": config.validation_model,
                    "temperature_requested": config.temperature,
                    "temperature_sent": _temperature_sent_value(
                        config.validation_model, config.temperature
                    ),
                    "temperature_parameter_sent": validation_response.get(
                        "temperature_parameter_sent"
                    ),
                    "prompt_version": VALIDATION_PROMPT_VERSION,
                    "prompt_sha256": validation_prompt_hash,
                    "response_status_code": validation_response.get("status_code"),
                    "response_id": raw_id,
                    "response_text": response_text,
                    "parsed_json": parsed_report,
                    "parse_error": parse_error,
                    "schema_errors": schema_errors,
                    "citation_verification": verification,
                    "raw_response": response_body,
                    "created_at_utc": _utc_now_iso(),
                }
            )
            last_errors = []
            if parse_error:
                last_errors.append(parse_error)
            last_errors.extend(schema_errors)
            if parsed_report is not None and not schema_errors and verification["valid"]:
                video_report = _finalize_video_report(
                    model_report=parsed_report,
                    video_validation_id=video_validation_id,
                    event_id=config.event_id,
                    video_id=video_id,
                    model=config.validation_model,
                    validated_at_utc=created_at_utc,
                    verification=verification,
                    attempt_count=attempt,
                    raw_response_id=raw_id,
                    query_candidate_id=query_candidate_id if query_record else None,
                    external_evidence_count=len(external_evidence),
                )
                break
            if parsed_report is not None and not schema_errors:
                last_errors.extend(verification.get("errors", []))
            if attempt <= config.max_retries:
                manifest_errors.append(
                    {
                        "type": "retry",
                        "stage": "video_validation_generation",
                        "video_id": video_id,
                        "attempt": attempt,
                        "errors": last_errors,
                    }
                )
        if video_report is None:
            batch_record["video_batch_status"] = "failed_validation"
            batch_record["status_reason"] = "video_validation_generation_failed"
            video_report = _error_video_report(
                video_validation_id=video_validation_id,
                event_id=config.event_id,
                video_id=video_id,
                model=config.validation_model,
                validated_at_utc=created_at_utc,
                bundle=bundle,
                errors=last_errors or ["generation_failed"],
                attempt_count=sum(
                    1
                    for item in raw_records
                    if item.get("record_type") == "video_validation_generation"
                    and item.get("video_id") == video_id
                ),
                query_candidate_id=query_candidate_id if query_record else None,
                external_evidence_count=len(external_evidence),
            )
        video_report["video_batch_status"] = batch_record["video_batch_status"]
        video_report["batch_id"] = batch_id
        video_report["video_batch_order"] = batch_record["video_batch_order"]
        if batch_record["video_batch_status"] == "processed" and retrieval_status == "error":
            batch_record["video_batch_status"] = "failed_retrieval"
            batch_record["status_reason"] = "serper_request_failed"
            video_report["video_batch_status"] = "failed_retrieval"
        video_reports.append(video_report)
        per_video_runs.append(
            {
                "event_id": config.event_id,
                "video_id": video_id,
                "batch_id": batch_id,
                "video_batch_order": batch_record["video_batch_order"],
                "video_batch_status": batch_record["video_batch_status"],
                "status_reason": batch_record["status_reason"],
                "query_candidate_id": query_candidate_id,
                "query_text": query_record.get("query_text") if query_record else None,
                "query_generation_mode": (
                    query_record.get("query_generation_mode") if query_record else None
                ),
                "metadata_fields_used": (
                    query_record.get("metadata_fields_used") if query_record else []
                ),
                "query_status": query_record.get("query_status") if query_record else None,
                "retrieval_status": retrieval_status,
                "external_evidence_count": len(external_evidence),
                "video_validation_id": video_validation_id,
                "validation_status": video_report.get("validation_status"),
                "event_interpretation": video_report.get("event_interpretation"),
                "external_evidence_assessment": video_report.get(
                    "external_evidence_assessment"
                ),
                "citation_verification": video_report.get("citation_verification"),
            }
        )

    event_summary = _build_event_summary(
        event_id=config.event_id,
        video_reports=video_reports,
        external_evidence=all_external_evidence,
        video_batch_records=video_batch_records,
        batch_id=batch_id,
        created_at_utc=created_at_utc,
    )

    input_hashes = {
        name: _sha256_file(path)
        for name, path in consumer_paths.items()
        if path.exists()
    }
    manifest = {
        "run_id": "ragg2h_" + _short_hash(config.event_id, created_at_utc),
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_generation_g2_hierarchical_external_evidence",
        "mode": "event_video_external_validation_with_deterministic_event_summary",
        "artifact_version": RAG_G2H_ARTIFACT_VERSION,
        "event_ids": [config.event_id],
        "batch_id": batch_id,
        "batch_index": config.batch_index,
        "videos_total": len(bundles),
        "videos_evaluated": [bundle["video_id"] for bundle in selected_bundles],
        "videos_pending": batch_plan["pending_video_ids"],
        "videos_skipped": batch_plan["skipped_video_ids"],
        "max_videos_per_event": config.max_videos_per_event_batch,
        "max_videos_per_event_batch": config.max_videos_per_event_batch,
        "max_estimated_tokens_per_event_batch": (
            config.max_estimated_tokens_per_event_batch
        ),
        "max_llm_calls_per_batch": config.max_llm_calls_per_batch,
        "max_serper_calls_per_batch": config.max_serper_calls_per_batch,
        "max_estimated_cost_usd_per_batch": (
            config.max_estimated_cost_usd_per_batch
        ),
        "video_batch_policy": {
            "number_of_videos_is_not_an_exclusion_criterion": True,
            "batch_order": [
                "comment_count_desc",
                "context_unit_count_desc",
                "first_comment_time_utc_asc",
                "video_id_asc",
            ],
            "allowed_video_batch_status": sorted(VIDEO_BATCH_STATUS_VALUES),
            "allowed_event_batch_status": sorted(EVENT_BATCH_STATUS_VALUES),
        },
        "video_batch_plan": video_batch_records,
        "batch_estimates": batch_plan["batch_estimates"],
        "cost_guard_status": _cost_guard_status(config),
        "limits_reached": batch_plan["limits_reached"],
        "next_batch_required": bool(batch_plan["pending_video_ids"]),
        "provider": config.provider,
        "query_model_name": config.query_model,
        "validation_model_name": config.validation_model,
        "temperature_requested": config.temperature,
        "query_temperature_sent": _temperature_sent_value(
            config.query_model, config.temperature
        ),
        "validation_temperature_sent": _temperature_sent_value(
            config.validation_model, config.temperature
        ),
        "temperature_parameter_sent": {
            "query": _should_send_temperature(config.query_model),
            "validation": _should_send_temperature(config.validation_model),
        },
        "temperature_effective_note": (
            "temperature parameter not sent because gpt-5-mini does not support it"
            if not _should_send_temperature(config.query_model)
            else "temperature parameter sent to API"
        ),
        "query_prompt_version": QUERY_PROMPT_VERSION,
        "validation_prompt_version": VALIDATION_PROMPT_VERSION,
        "query_prompt_sha256_by_video": query_prompt_hashes,
        "validation_prompt_sha256_by_video": validation_prompt_hashes,
        "consumer_run_id": consumer_manifest.get("run_id"),
        "input_paths": {
            name: _normalize_path(path) for name, path in consumer_paths.items()
        },
        "input_hashes": input_hashes,
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "scope_policy": {
            "validation_unit": "event_id + video_id",
            "event_summary_unit": "event_id",
            "internal_youtube_evidence_used": True,
            "external_news_evidence_used": True,
            "serper_used": True,
            "embeddings_used": False,
            "vectorstore_used": False,
            "chroma_used": False,
            "query_expansion_used": False,
            "multi_query_used": False,
            "query_strategy": "metadata_first_deterministic",
            "llm_query_fallback_only_when_metadata_insufficient": True,
            "claim_verification_query_generated": False,
            "does_not_modify_sidecars": True,
            "does_not_modify_consumer": True,
            "does_not_modify_g1": True,
            "does_not_modify_g2_global": True,
            "does_not_modify_pipeline": True,
            "does_not_change_event_id": True,
            "does_not_change_roles_temporales": True,
        },
        "search_policy": {
            "provider": "serper_news",
            "endpoint": config.serper_url,
            "gl": config.serper_gl,
            "hl": config.serper_hl,
            "type": config.serper_type,
            "num": config.serper_num_results,
            "window_start_date": search_start,
            "window_end_date": search_end,
            "window_rule": "trigger date +/- approved calendar days",
            "external_evidence_cutoff_utc": config.external_evidence_cutoff_utc,
            "published_at_boundary": "strictly_before_cutoff",
            "unknown_publication_time_policy": (
                "exclude_when_causal_cutoff_is_configured"
                if config.external_evidence_cutoff_utc is not None
                else "legacy_preserve"
            ),
        },
        "external_evidence_causality": {
            "cutoff_utc": config.external_evidence_cutoff_utc,
            "audits_by_video": external_causality_audit,
            "kept": sum(item["kept"] for item in external_causality_audit),
            "excluded_at_or_after_cutoff": sum(
                item["excluded_at_or_after_cutoff"]
                for item in external_causality_audit
            ),
            "excluded_unknown_published_at": sum(
                item["excluded_unknown_published_at"]
                for item in external_causality_audit
            ),
            "same_day_time_unknown": sum(
                item["same_day_time_unknown"]
                for item in external_causality_audit
            ),
            "future_leak_count": 0,
        },
        "per_video_runs": per_video_runs,
        "pending_video_runs": [
            record
            for record in video_batch_records
            if str(record.get("video_batch_status", "")).startswith("pending_")
        ],
        "skipped_video_runs": [
            record
            for record in video_batch_records
            if record.get("video_batch_status") == "skipped_no_context"
        ],
        "external_evidence_count": len(all_external_evidence),
        "event_summary": event_summary,
        "raw_response_count": len(raw_records),
        "query_call_count": sum(
            1
            for item in raw_records
            if item.get("record_type") == "video_query_generation_fallback"
        ),
        "deterministic_query_count": sum(
            1
            for item in query_records
            if item.get("query_generation_mode") == DETERMINISTIC_METADATA
        ),
        "llm_fallback_query_count": sum(
            1
            for item in query_records
            if item.get("query_generation_mode") == LLM_FALLBACK
        ),
        "validation_call_count": sum(
            1
            for item in raw_records
            if item.get("record_type") == "video_validation_generation"
        ),
        "serper_request_count": serper_request_count,
        "max_retries": config.max_retries,
        "retry_count": sum(1 for item in manifest_errors if item.get("type") == "retry"),
        "errors": manifest_errors,
        "bundle_verification": bundle_verification,
        "notes": config.notes,
        "params": config.params,
    }

    _write_jsonl(output_paths["rag_video_news_queries"], query_records)
    _write_jsonl(output_paths["rag_video_external_evidence"], all_external_evidence)
    _write_jsonl(output_paths["rag_video_validation_reports"], video_reports)
    _write_jsonl(output_paths["rag_event_validation_summary"], [event_summary])
    _write_jsonl(output_paths["rag_raw_model_responses"], raw_records)
    _write_json(output_paths["rag_generation_manifest"], manifest)

    return {
        "run_id": manifest["run_id"],
        "event_id": config.event_id,
        "output_dir": _normalize_path(output_dir),
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "videos_evaluated": manifest["videos_evaluated"],
        "videos_total": manifest["videos_total"],
        "videos_pending": manifest["videos_pending"],
        "batch_id": batch_id,
        "next_batch_required": manifest["next_batch_required"],
        "query_call_count": manifest["query_call_count"],
        "validation_call_count": manifest["validation_call_count"],
        "serper_request_count": manifest["serper_request_count"],
        "external_evidence_count": len(all_external_evidence),
        "per_video_runs": per_video_runs,
        "pending_video_runs": manifest["pending_video_runs"],
        "event_summary": event_summary,
        "errors": manifest_errors,
    }


def run_rag_g2_hierarchical(
    *,
    consumer_dir: str | Path,
    output_dir: str | Path,
    event_id: str,
    query_model: str = DEFAULT_QUERY_MODEL,
    validation_model: str = DEFAULT_VALIDATION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_videos_per_event_batch: int = DEFAULT_MAX_VIDEOS_PER_EVENT_BATCH,
    max_retries: int = 1,
    notes: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = RagG2HierarchicalConfig(
        consumer_dir=str(consumer_dir),
        output_dir=str(output_dir),
        event_id=event_id,
        query_model=query_model,
        validation_model=validation_model,
        temperature=temperature,
        max_videos_per_event_batch=max_videos_per_event_batch,
        max_retries=max_retries,
        notes=notes,
        params=params or {},
    )
    return run_rag_g2_hierarchical_from_config(config)


__all__ = [
    "DETERMINISTIC_METADATA",
    "LLM_FALLBACK",
    "QUERY_PROMPT_VERSION",
    "RAG_G2H_ARTIFACT_VERSION",
    "VALIDATION_PROMPT_VERSION",
    "RagG2HierarchicalConfig",
    "has_sufficient_search_metadata",
    "load_rag_g2_hierarchical_config",
    "plan_rag_g2_hierarchical_dry_run",
    "run_rag_g2_hierarchical",
    "run_rag_g2_hierarchical_from_config",
]
