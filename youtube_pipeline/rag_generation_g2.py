from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from youtube_pipeline.validation_status import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_CONFIRMED,
    VALIDATION_STATUS_VALUES,
)

from youtube_pipeline.external_publication_time import (
    AFTER_REFERENCE,
    AT_REFERENCE,
    BEFORE_REFERENCE,
    SAME_DAY_TIME_UNKNOWN,
    parse_external_publication_time,
    temporal_relation_to_reference,
)
from youtube_pipeline.rag_generation_g1 import (
    OPENAI_RESPONSES_URL,
    _approx_tokens,
    _check_model_available,
    _extract_event,
    _json_safe,
    _load_api_key,
    _normalize_path,
    _parse_model_json,
    _read_existing_json,
    _read_existing_jsonl_records,
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


RAG_G2_ARTIFACT_VERSION = "rag_generation_g2_v2"
QUERY_PROMPT_VERSION = "rag_news_query_prompt_v0.1"
VALIDATION_PROMPT_VERSION = "rag_validation_g2_prompt_v0.2"

RAG_VALIDATION_INPUTS_FILE = "rag_validation_inputs.jsonl"
RAG_CONTEXT_PAYLOADS_FILE = "rag_context_payloads.jsonl"
RAG_CONSUMER_MANIFEST_FILE = "rag_consumer_manifest.json"
G1_VALIDATION_REPORTS_FILE = "rag_validation_reports.jsonl"

RAG_NEWS_QUERIES_FILE = "rag_news_queries.jsonl"
RAG_EXTERNAL_EVIDENCE_FILE = "rag_external_evidence.jsonl"
RAG_VALIDATION_REPORTS_FILE = "rag_validation_reports.jsonl"
RAG_GENERATION_MANIFEST_FILE = "rag_generation_manifest.json"
RAG_RAW_MODEL_RESPONSES_FILE = "rag_raw_model_responses.jsonl"

EVENT_INTERPRETATION_VALUES = {
    "external_event",
    "internal_community_reaction",
    "possible_noise",
    "possible_disinformation",
    "unclear",
}
CONFIDENCE_LABEL_VALUES = {"low", "medium", "high"}
EXTERNAL_EVIDENCE_ASSESSMENT_VALUES = {
    "supports",
    "contradicts",
    "inconclusive",
    "no_external_evidence",
}

DEFAULT_SERPER_URL = "https://google.serper.dev/news"


@dataclass(frozen=True)
class RagG2Config:
    consumer_dir: str
    g1_dir: str
    output_dir: str
    event_id: str
    query_model: str = "gpt-5-mini"
    validation_model: str = "gpt-5-mini"
    provider: str = "openai"
    temperature: float = 0.0
    serper_url: str = DEFAULT_SERPER_URL
    serper_gl: str = "co"
    serper_hl: str = "es"
    serper_type: str = "news"
    serper_num_results: int = 5
    search_days_before: int = 1
    search_days_after: int = 1
    max_retries: int = 1
    request_timeout_seconds: int = 120
    serper_timeout_seconds: int = 60
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RagG2Config":
        missing = [
            key
            for key in ["consumer_dir", "g1_dir", "output_dir", "event_id"]
            if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                "RAG G-2 config missing required fields: " + ", ".join(missing)
            )
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        return cls(
            consumer_dir=str(payload["consumer_dir"]),
            g1_dir=str(payload["g1_dir"]),
            output_dir=str(payload["output_dir"]),
            event_id=str(payload["event_id"]),
            query_model=str(payload.get("query_model", "gpt-5-mini")),
            validation_model=str(payload.get("validation_model", "gpt-5-mini")),
            provider=str(payload.get("provider", "openai")),
            temperature=float(payload.get("temperature", 0.0)),
            serper_url=str(payload.get("serper_url", DEFAULT_SERPER_URL)),
            serper_gl=str(payload.get("serper_gl", "co")),
            serper_hl=str(payload.get("serper_hl", "es")),
            serper_type=str(payload.get("serper_type", "news")),
            serper_num_results=int(payload.get("serper_num_results", 5)),
            search_days_before=int(payload.get("search_days_before", 1)),
            search_days_after=int(payload.get("search_days_after", 1)),
            max_retries=int(payload.get("max_retries", 1)),
            request_timeout_seconds=int(payload.get("request_timeout_seconds", 120)),
            serper_timeout_seconds=int(payload.get("serper_timeout_seconds", 60)),
            notes=payload.get("notes"),
            params=params,
        )


def _extract_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("rag_generation_g2")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("rag_generation_g2 config section must be an object.")
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


def load_rag_g2_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> RagG2Config:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG G-2 config must be an object: {p}")
    base = _extract_config_payload(payload)
    merged = _merge_config_payloads(base, overrides or {})
    return RagG2Config.from_mapping(merged)


def _consumer_paths(consumer_dir: str | Path) -> dict[str, Path]:
    root = Path(consumer_dir)
    return {
        "rag_validation_inputs": root / RAG_VALIDATION_INPUTS_FILE,
        "rag_context_payloads": root / RAG_CONTEXT_PAYLOADS_FILE,
        "rag_consumer_manifest": root / RAG_CONSUMER_MANIFEST_FILE,
    }


def _g1_paths(g1_dir: str | Path) -> dict[str, Path]:
    root = Path(g1_dir)
    return {"g1_validation_reports": root / G1_VALIDATION_REPORTS_FILE}


def _output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "rag_news_queries": root / RAG_NEWS_QUERIES_FILE,
        "rag_external_evidence": root / RAG_EXTERNAL_EVIDENCE_FILE,
        "rag_validation_reports": root / RAG_VALIDATION_REPORTS_FILE,
        "rag_generation_manifest": root / RAG_GENERATION_MANIFEST_FILE,
        "rag_raw_model_responses": root / RAG_RAW_MODEL_RESPONSES_FILE,
    }


def _load_serper_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("SERPER_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "SERPER_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


def _validation_id(event_id: str, prompt_version: str, model: str) -> str:
    return "valg2_" + _short_hash(event_id, prompt_version, model)


def _query_id(event_id: str, prompt_version: str, model: str) -> str:
    return "qry_" + _short_hash(event_id, prompt_version, model)


def _external_evidence_id(event_id: str, query_id: str, rank: int, result: dict[str, Any]) -> str:
    stable = result.get("link") or result.get("title") or json.dumps(result, sort_keys=True)
    return "ext_" + _short_hash(event_id, query_id, rank, stable)


def _load_g1_report(g1_dir: str | Path, event_id: str) -> dict[str, Any] | None:
    path = _g1_paths(g1_dir)["g1_validation_reports"]
    if not path.exists():
        return None
    for record in _read_jsonl_records(path):
        if record.get("event_id") == event_id:
            return record
    return None


def _safe_text(value: Any, max_chars: int = 450) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


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
        "text": _safe_text(comment.get("text"), 450),
    }


def _build_query_payload(
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
    g1_report: dict[str, Any] | None,
) -> dict[str, Any]:
    comments = context_payload.get("used_context_comments", [])
    cited_ids = set((g1_report or {}).get("cited_comment_ids") or [])
    if cited_ids:
        representative = [c for c in comments if c.get("comment_id") in cited_ids]
    else:
        representative = comments[:8]
    return {
        "event_id": validation_input.get("event_id"),
        "trigger_time_utc": validation_input.get("trigger_time_utc"),
        "window_start_utc": validation_input.get("window_start_utc"),
        "window_end_utc": validation_input.get("window_end_utc"),
        "associated_videos": [
            {
                "video_id": video.get("video_id"),
                "title": video.get("title"),
                "channel_title": video.get("channel_title"),
                "inventory_comment_count": video.get("inventory_comment_count"),
            }
            for video in validation_input.get("associated_videos", [])
        ],
        "g1_internal_validation": {
            "validation_status": (g1_report or {}).get("validation_status"),
            "event_interpretation": (g1_report or {}).get("event_interpretation"),
            "claim_summary": (g1_report or {}).get("claim_summary"),
            "evidence_summary": (g1_report or {}).get("evidence_summary"),
            "reasoning_summary": (g1_report or {}).get("reasoning_summary"),
            "cited_comment_ids": (g1_report or {}).get("cited_comment_ids", []),
            "cited_context_unit_ids": (g1_report or {}).get(
                "cited_context_unit_ids", []
            ),
        },
        "representative_comments": [_comment_excerpt(c) for c in representative[:10]],
        "query_constraints": {
            "max_words": 8,
            "language": "es",
            "must_use_only_terms_present_in_evidence": True,
            "do_not_invent_entities_places_or_facts": True,
            "if_insufficient_specific_evidence": (
                "Use a conservative query composed only of terms present in the "
                "evidence and state the limitation."
            ),
        },
    }


def _query_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
            "input_context_summary": {"type": "string"},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query", "input_context_summary", "limitations"],
    }


def _query_prompt_text(query_payload: dict[str, Any]) -> tuple[str, str]:
    developer_prompt = (
        "Eres un asistente de recuperacion de noticias para una validacion RAG "
        "academica sobre eventos detectados en comentarios de YouTube.\n\n"
        "Genera una unica query breve para buscar evidencia externa en noticias. "
        "Usa unicamente la evidencia suministrada: event_id, titulos de video, "
        "ventana temporal, resumen interno y comentarios citables.\n\n"
        "Prohibido inventar nombres, lugares, organizaciones, hechos o contexto "
        "externo. Prohibido usar conocimiento fuera de los insumos. Si no hay "
        "suficiente evidencia para formular una query especifica, devuelve una "
        "query conservadora basada solo en terminos presentes en la evidencia y "
        "registra la limitacion.\n\n"
        "La query debe estar en espanol, ser especifica y tener idealmente entre "
        "5 y 8 palabras. Devuelve solo JSON valido que cumpla el esquema."
    )
    user_prompt = (
        "Payload para generar query externa G-2:\n\n"
        + json.dumps(query_payload, ensure_ascii=False, indent=2)
    )
    return developer_prompt, user_prompt


def _create_json_response(
    *,
    api_key: str,
    model: str,
    temperature: float,
    developer_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    timeout: int,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    temperature_parameter_sent = _should_send_temperature(model)
    payload = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": developer_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    if temperature_parameter_sent:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    try:
        body = response.json()
    except Exception:
        body = {"text": response.text}
    return {
        "status_code": response.status_code,
        "ok": response.ok,
        "body": body,
        "temperature_parameter_sent": temperature_parameter_sent,
    }


def _schema_errors(payload: dict[str, Any] | None, schema: dict[str, Any]) -> list[str]:
    if payload is None:
        return ["missing_payload"]
    errors: list[str] = []
    missing = sorted(set(schema.get("required", [])).difference(payload))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    for field, definition in schema.get("properties", {}).items():
        if field not in payload:
            continue
        expected_type = definition.get("type")
        value = payload[field]
        expected_types = (
            expected_type if isinstance(expected_type, list) else [expected_type]
        )
        type_matches = False
        for item_type in expected_types:
            if item_type == "string" and isinstance(value, str):
                type_matches = True
            elif item_type == "boolean" and isinstance(value, bool):
                type_matches = True
            elif item_type == "array" and isinstance(value, list):
                type_matches = True
            elif item_type == "null" and value is None:
                type_matches = True
        if expected_type and not type_matches:
            errors.append(
                f"{field}_is_not_{'_or_'.join(str(t) for t in expected_types)}"
            )
        enum_values = definition.get("enum")
        if enum_values and value not in enum_values:
            errors.append(f"{field}_invalid_enum")
    return errors


_STOPWORDS = {
    "a",
    "al",
    "ante",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "su",
    "sus",
    "un",
    "una",
    "y",
    "o",
    "noticia",
    "noticias",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-záéíóúñü0-9#]+", text.lower())
        if token and token not in _STOPWORDS
    ]


def _evidence_text_for_query(query_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for video in query_payload.get("associated_videos", []):
        parts.append(str(video.get("title") or ""))
        parts.append(str(video.get("channel_title") or ""))
    g1 = query_payload.get("g1_internal_validation", {})
    parts.extend(
        [
            str(g1.get("claim_summary") or ""),
            str(g1.get("evidence_summary") or ""),
            str(g1.get("reasoning_summary") or ""),
        ]
    )
    for comment in query_payload.get("representative_comments", []):
        parts.append(str(comment.get("text") or ""))
    return "\n".join(parts)


def _classify_query(query: str, query_payload: dict[str, Any]) -> tuple[str, list[str]]:
    query = " ".join(str(query or "").split())
    limitations: list[str] = []
    if not query:
        return "invalid_invented_content", ["empty_query"]
    query_tokens = _tokens(query)
    evidence_tokens = set(_tokens(_evidence_text_for_query(query_payload)))
    informative_tokens = [token for token in query_tokens if len(token) > 2]
    unseen_tokens = [
        token for token in informative_tokens if token not in evidence_tokens
    ]
    if informative_tokens and len(unseen_tokens) / len(informative_tokens) > 0.6:
        return (
            "invalid_invented_content",
            ["query_contains_many_terms_not_present_in_internal_evidence"],
        )
    if len(query.split()) > 8:
        limitations.append("query_exceeds_recommended_word_count")
    if len(informative_tokens) < 3:
        limitations.append("query_has_few_informative_terms")
    status = "broad_but_valid" if limitations else "valid"
    return status, limitations


def _search_window(trigger_time_utc: str, days_before: int, days_after: int) -> tuple[str, str]:
    trigger = pd.Timestamp(trigger_time_utc)
    start = (trigger.date() - timedelta(days=days_before)).isoformat()
    end = (trigger.date() + timedelta(days=days_after)).isoformat()
    return start, end


def _serper_date(date_iso: str) -> str:
    dt = datetime.fromisoformat(date_iso)
    return f"{dt.month}/{dt.day}/{dt.year}"


def _call_serper_news(
    *,
    api_key: str,
    query: str,
    window_start_date: str,
    window_end_date: str,
    config: RagG2Config,
) -> dict[str, Any]:
    payload = {
        "q": query,
        "gl": config.serper_gl,
        "hl": config.serper_hl,
        "tbs": (
            "cdr:1,"
            f"cd_min:{_serper_date(window_start_date)},"
            f"cd_max:{_serper_date(window_end_date)}"
        ),
        "type": config.serper_type,
        "num": config.serper_num_results,
    }
    response = requests.post(
        config.serper_url,
        json=payload,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=config.serper_timeout_seconds,
    )
    try:
        body = response.json()
    except Exception:
        body = {"text": response.text[:1000]}
    return {
        "status_code": response.status_code,
        "ok": response.ok,
        "body": body,
        "request_payload": payload,
    }


def _temporal_relation(published_at: Any, trigger_time_utc: str) -> str:
    relation = temporal_relation_to_reference(
        parse_external_publication_time(published_at),
        trigger_time_utc,
    )
    return {
        BEFORE_REFERENCE: "before_trigger",
        AT_REFERENCE: "at_trigger",
        AFTER_REFERENCE: "after_trigger",
        SAME_DAY_TIME_UNKNOWN: "unknown_same_day_date_only",
    }.get(relation, "unknown")


def _normalize_external_evidence(
    *,
    event_id: str,
    query_id: str,
    query: str,
    serper_response: dict[str, Any],
    trigger_time_utc: str,
    provider: str,
    retrieved_at_utc: str,
) -> list[dict[str, Any]]:
    body = serper_response.get("body", {})
    results = body.get("news", []) if isinstance(body, dict) else []
    records: list[dict[str, Any]] = []
    for index, result in enumerate(results or [], start=1):
        if not isinstance(result, dict):
            continue
        published_at = result.get("date")
        publication_time = parse_external_publication_time(published_at)
        records.append(
            {
                "external_evidence_id": _external_evidence_id(
                    event_id, query_id, index, result
                ),
                "event_id": event_id,
                "query_id": query_id,
                "query": query,
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "source": result.get("source"),
                "link": result.get("link"),
                "published_at": published_at,
                **publication_time.to_mapping(),
                "retrieved_at_utc": retrieved_at_utc,
                "provider": provider,
                "rank": index,
                "raw_result_ref": {
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "source": result.get("source"),
                    "link": result.get("link"),
                    "date": result.get("date"),
                    "imageUrl": result.get("imageUrl"),
                    "position": result.get("position"),
                },
                "temporal_relation_to_trigger": _temporal_relation(
                    published_at, trigger_time_utc
                ),
            }
        )
    return records


def _compact_validation_payload(
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
    g1_report: dict[str, Any] | None,
    query_record: dict[str, Any],
    external_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "event_metadata": {
            "event_id": validation_input.get("event_id"),
            "run_id": validation_input.get("run_id"),
            "detector_name": validation_input.get("detector_name"),
            "trigger_time_utc": validation_input.get("trigger_time_utc"),
            "window_start_utc": validation_input.get("window_start_utc"),
            "window_end_utc": validation_input.get("window_end_utc"),
            "trigger_volume": validation_input.get("trigger_volume"),
            "trigger_strength": validation_input.get("trigger_strength"),
            "associated_videos": validation_input.get("associated_videos", []),
        },
        "temporal_policy": {
            "alert_evidence_comments": "window_start_utc <= event_time_utc <= trigger_time_utc",
            "validation_context_comments": "window_start_utc <= event_time_utc <= window_end_utc",
            "post_trigger_context_used": context_payload.get("post_trigger_context_used"),
            "post_trigger_comment_ids": context_payload.get("post_trigger_comment_ids", []),
            "do_not_treat_post_trigger_comments_as_alert_cause": True,
        },
        "g1_internal_validation_reference": {
            "validation_status": (g1_report or {}).get("validation_status"),
            "event_interpretation": (g1_report or {}).get("event_interpretation"),
            "confidence_label": (g1_report or {}).get("confidence_label"),
            "claim_summary": (g1_report or {}).get("claim_summary"),
            "evidence_summary": (g1_report or {}).get("evidence_summary"),
            "reasoning_summary": (g1_report or {}).get("reasoning_summary"),
            "cited_comment_ids": (g1_report or {}).get("cited_comment_ids", []),
            "cited_context_unit_ids": (g1_report or {}).get(
                "cited_context_unit_ids", []
            ),
        },
        "internal_evidence": {
            "selected_context_units": context_payload.get("selected_context_units", []),
            "used_context_comments": context_payload.get("used_context_comments", []),
            "used_context_comment_ids": [
                comment.get("comment_id")
                for comment in context_payload.get("used_context_comments", [])
                if comment.get("comment_id")
            ],
        },
        "external_retrieval": {
            "query": query_record,
            "external_evidence": external_evidence,
        },
        "allowed_validation_status": sorted(VALIDATION_STATUS_VALUES),
        "allowed_event_interpretation": sorted(EVENT_INTERPRETATION_VALUES),
        "allowed_external_evidence_assessment": sorted(
            EXTERNAL_EVIDENCE_ASSESSMENT_VALUES
        ),
    }


def _validation_response_schema() -> dict[str, Any]:
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
            "claim_summary": {"type": "string"},
            "internal_evidence_summary": {"type": "string"},
            "external_evidence_summary": {"type": "string"},
            "reasoning_summary": {"type": "string"},
            "external_evidence_assessment": {
                "type": "string",
                "enum": sorted(EXTERNAL_EVIDENCE_ASSESSMENT_VALUES),
            },
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
            "claim_summary",
            "internal_evidence_summary",
            "external_evidence_summary",
            "reasoning_summary",
            "external_evidence_assessment",
            "cited_comment_ids",
            "cited_context_unit_ids",
            "cited_external_evidence_ids",
            "limitations",
        ],
    }


def _validation_prompt_text(compact_payload: dict[str, Any]) -> tuple[str, str]:
    developer_prompt = (
        "Eres un evaluador academico para una validacion RAG G-2 de eventos "
        "detectados en comentarios de YouTube.\n\n"
        "Tu tarea es contrastar un patron conversacional interno con evidencia "
        "externa recuperada. Usa solo los insumos proporcionados: evidencia interna "
        "de YouTube, unidades de contexto, comentarios citables, query generada y "
        "evidencia externa normalizada.\n\n"
        "Reglas metodologicas:\n"
        "- No uses conocimiento externo que no este en el payload.\n"
        "- No inventes noticias, enlaces, hechos ni fuentes.\n"
        "- Separa evidencia interna, evidencia externa e inferencia.\n"
        "- Cita comment_id cuando uses comentarios.\n"
        "- Cita context_unit_id cuando uses unidades internas.\n"
        "- Cita external_evidence_id cuando uses noticias o evidencia externa.\n"
        "- No presentes comentarios posteriores al trigger como causa de la alerta.\n"
        "- No modifiques la decision original del detector; evalua solo la "
        "validacion posterior.\n"
        "- external_event requiere al menos una cita externa valida.\n"
        "- CONFIRMED requiere evidencia externa valida que corrobore el fenomeno.\n"
        "- INSUFFICIENT_EVIDENCE significa que la evidencia externa falta, es "
        "parcial, debil o inconclusa.\n"
        "- NOT_CONFIRMED requiere evidencia externa suficiente para evaluar que no "
        "respalda o contradice sustantivamente la hipotesis.\n"
        "- La evidencia interna por si sola nunca permite CONFIRMED ni NOT_CONFIRMED.\n"
        "- possible_disinformation solo debe usarse si la evidencia disponible "
        "contradice de forma importante afirmaciones presentes en los comentarios "
        "o muestra falta clara de respaldo.\n"
        "- INSUFFICIENT_EVIDENCE debe usarse cuando no hay evidencia suficiente, no "
        "por dudas vagas.\n\n"
        "Devuelve solo JSON valido que cumpla el esquema."
    )
    user_prompt = (
        "Payload de validacion G-2 con evidencia interna y externa:\n\n"
        + json.dumps(compact_payload, ensure_ascii=False, indent=2)
    )
    return developer_prompt, user_prompt


def _citation_verification(
    *,
    report: dict[str, Any],
    context_payload: dict[str, Any],
    external_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_comment_ids = {
        comment.get("comment_id")
        for comment in context_payload.get("used_context_comments", [])
        if comment.get("comment_id")
    }
    allowed_context_unit_ids = {
        unit.get("context_unit_id")
        for unit in context_payload.get("selected_context_units", [])
        if unit.get("context_unit_id")
    }
    allowed_external_ids = {
        item.get("external_evidence_id")
        for item in external_evidence
        if item.get("external_evidence_id")
    }
    cited_comment_ids = set(report.get("cited_comment_ids") or [])
    cited_context_unit_ids = set(report.get("cited_context_unit_ids") or [])
    cited_external_ids = set(report.get("cited_external_evidence_ids") or [])

    invalid_comment_ids = sorted(cited_comment_ids.difference(allowed_comment_ids))
    invalid_context_unit_ids = sorted(
        cited_context_unit_ids.difference(allowed_context_unit_ids)
    )
    invalid_external_ids = sorted(cited_external_ids.difference(allowed_external_ids))

    valid_internal_citation = bool(
        cited_comment_ids.intersection(allowed_comment_ids)
        or cited_context_unit_ids.intersection(allowed_context_unit_ids)
    )
    valid_external_citation = bool(cited_external_ids.intersection(allowed_external_ids))
    has_valid_citation = valid_internal_citation or valid_external_citation
    status_requires_citation = report.get("validation_status") in {
        CONFIRMED,
        NOT_CONFIRMED,
    }
    event_interpretation_requires_external = (
        report.get("event_interpretation") == "external_event"
    )
    confirmed_external_event = report.get("validation_status") == CONFIRMED
    payload_post_trigger = bool(context_payload.get("post_trigger_context_used"))

    errors: list[str] = []
    if invalid_comment_ids:
        errors.append("invalid_cited_comment_ids")
    if invalid_context_unit_ids:
        errors.append("invalid_cited_context_unit_ids")
    if invalid_external_ids:
        errors.append("invalid_cited_external_evidence_ids")
    if status_requires_citation and not valid_external_citation:
        errors.append("decisive_g2_status_requires_valid_external_citation")
    if event_interpretation_requires_external and not valid_external_citation:
        errors.append("external_event_requires_valid_external_citation")
    if confirmed_external_event and not valid_external_citation:
        errors.append("confirmed_external_event_requires_external_citation")
    if not allowed_external_ids and report.get("event_interpretation") == "external_event":
        errors.append("external_event_without_external_evidence")
    if not allowed_external_ids and report.get("validation_status") != INSUFFICIENT_EVIDENCE:
        errors.append("no_external_evidence_requires_insufficient_evidence")
    if report.get("validation_status") == CONFIRMED and report.get(
        "external_evidence_assessment"
    ) != "supports":
        errors.append("confirmed_requires_supporting_external_assessment")
    if report.get("validation_status") == NOT_CONFIRMED and report.get(
        "external_evidence_assessment"
    ) != "contradicts":
        errors.append("not_confirmed_requires_contradicting_external_assessment")

    return {
        "valid": not errors,
        "errors": errors,
        "invalid_cited_comment_ids": invalid_comment_ids,
        "invalid_cited_context_unit_ids": invalid_context_unit_ids,
        "invalid_cited_external_evidence_ids": invalid_external_ids,
        "invalid_used_context_comment_ids": [],
        "invalid_used_external_evidence_ids": [],
        "status_requires_citation": status_requires_citation,
        "has_valid_citation": has_valid_citation,
        "valid_internal_citation": valid_internal_citation,
        "valid_external_citation": valid_external_citation,
        "event_interpretation_requires_external": event_interpretation_requires_external,
        "payload_post_trigger_context_used": payload_post_trigger,
        "report_post_trigger_context_used": payload_post_trigger,
        "allowed_comment_count": len(allowed_comment_ids),
        "allowed_context_unit_count": len(allowed_context_unit_ids),
        "allowed_external_evidence_count": len(allowed_external_ids),
        "allowed_comment_ids": sorted(allowed_comment_ids),
        "allowed_external_evidence_ids": sorted(allowed_external_ids),
        "post_trigger_context_used": payload_post_trigger,
    }


def _finalize_report(
    *,
    model_report: dict[str, Any],
    validation_id: str,
    event_id: str,
    model: str,
    validated_at_utc: str,
    citation_verification: dict[str, Any],
    attempt_count: int,
    raw_response_id: str | None,
    query_id: str,
    external_evidence_count: int,
) -> dict[str, Any]:
    report = dict(model_report)
    report["validation_id"] = validation_id
    report["event_id"] = event_id
    report["model_name"] = model
    report["prompt_version"] = VALIDATION_PROMPT_VERSION
    report["validated_at_utc"] = validated_at_utc
    report["used_context_comment_ids"] = citation_verification.get(
        "allowed_comment_ids", []
    )
    report["used_external_evidence_ids"] = citation_verification.get(
        "allowed_external_evidence_ids", []
    )
    report["post_trigger_context_used"] = citation_verification.get(
        "post_trigger_context_used", False
    )
    report["citation_verification"] = citation_verification
    report["attempt_count"] = attempt_count
    report["raw_response_id"] = raw_response_id
    report["generation_scope"] = "internal_youtube_plus_external_news_evidence"
    report["external_evidence_used"] = external_evidence_count > 0
    report["query_id"] = query_id
    report["external_evidence_count"] = external_evidence_count
    return report


def _error_report(
    *,
    validation_id: str,
    event_id: str,
    model: str,
    validated_at_utc: str,
    used_context_comment_ids: list[str],
    post_trigger_context_used: bool,
    errors: list[str],
    attempt_count: int,
    external_evidence_count: int,
    query_id: str | None,
) -> dict[str, Any]:
    return {
        "validation_id": validation_id,
        "event_id": event_id,
        "validation_status": INSUFFICIENT_EVIDENCE,
        "event_interpretation": "unclear",
        "confidence_label": "low",
        "claim_summary": "No se obtuvo una validacion G-2 estructurada valida.",
        "internal_evidence_summary": "",
        "external_evidence_summary": "",
        "reasoning_summary": "",
        "external_evidence_assessment": (
            "no_external_evidence" if external_evidence_count == 0 else "inconclusive"
        ),
        "cited_comment_ids": [],
        "cited_context_unit_ids": [],
        "cited_external_evidence_ids": [],
        "used_context_comment_ids": used_context_comment_ids,
        "used_external_evidence_ids": [],
        "post_trigger_context_used": post_trigger_context_used,
        "limitations": [
            "La generacion G-2 con LLM fallo o no cumplio el esquema/citas requeridas.",
            *errors,
        ],
        "model_name": model,
        "prompt_version": VALIDATION_PROMPT_VERSION,
        "validated_at_utc": validated_at_utc,
        "citation_verification": {"valid": False, "errors": errors},
        "attempt_count": attempt_count,
        "generation_scope": "internal_youtube_plus_external_news_evidence",
        "external_evidence_used": external_evidence_count > 0,
        "query_id": query_id,
        "external_evidence_count": external_evidence_count,
        "generation_error": True,
    }


def _query_record_from_model(
    *,
    query_id: str,
    event_id: str,
    query_result: dict[str, Any],
    query_status: str,
    query_status_limitations: list[str],
    validation_input: dict[str, Any],
    search_start: str,
    search_end: str,
    config: RagG2Config,
    generated_at_utc: str,
) -> dict[str, Any]:
    limitations = list(query_result.get("limitations") or [])
    for limitation in query_status_limitations:
        if limitation not in limitations:
            limitations.append(limitation)
    return {
        "query_id": query_id,
        "event_id": event_id,
        "query": query_result.get("query", ""),
        "query_status": query_status,
        "trigger_time_utc": validation_input.get("trigger_time_utc"),
        "query_time_window_start_utc": search_start,
        "query_time_window_end_utc": search_end,
        "query_prompt_version": QUERY_PROMPT_VERSION,
        "model_name": config.query_model,
        "provider": config.provider,
        "temperature_requested": config.temperature,
        "temperature_sent": _temperature_sent_value(
            config.query_model, config.temperature
        ),
        "generated_at_utc": generated_at_utc,
        "input_context_summary": query_result.get("input_context_summary", ""),
        "limitations": limitations,
    }


def run_rag_g2_validation_from_config(config: RagG2Config) -> dict[str, Any]:
    if config.provider != "openai":
        raise ValueError("G-2 currently supports only provider='openai'.")
    if config.query_model != "gpt-5-mini":
        raise ValueError("Approved G-2 query model is gpt-5-mini.")
    if config.validation_model != "gpt-5-mini":
        raise ValueError("Approved G-2 validation model is gpt-5-mini.")
    if config.event_id != "evt_34d7999bde8c":
        raise ValueError("Approved first G-2 event is evt_34d7999bde8c only.")
    if config.max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

    consumer_paths = _consumer_paths(config.consumer_dir)
    g1_paths = _g1_paths(config.g1_dir)
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
    g1_report = _load_g1_report(config.g1_dir, config.event_id)

    created_at_utc = _utc_now_iso()
    retrieved_at_utc = None
    validation_id = _validation_id(
        config.event_id, VALIDATION_PROMPT_VERSION, config.validation_model
    )
    query_id = _query_id(config.event_id, QUERY_PROMPT_VERSION, config.query_model)
    search_start, search_end = _search_window(
        str(validation_input.get("trigger_time_utc")),
        config.search_days_before,
        config.search_days_after,
    )

    query_payload = _build_query_payload(validation_input, context_payload, g1_report)
    query_developer_prompt, query_user_prompt = _query_prompt_text(query_payload)
    query_prompt_hash = hashlib.sha256(
        (query_developer_prompt + "\n\n" + query_user_prompt).encode("utf-8")
    ).hexdigest()

    api_key = _load_api_key()
    serper_api_key = _load_serper_api_key()
    manifest_errors: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    external_evidence: list[dict[str, Any]] = []
    validation_reports: list[dict[str, Any]] = []
    serper_response: dict[str, Any] | None = None
    model_availability: dict[str, Any] | None = None
    query_model_availability: dict[str, Any] | None = None
    validation_model_availability: dict[str, Any] | None = None
    retrieval_status = "not_started"
    query_status = "not_generated"
    query_record: dict[str, Any] | None = None

    used_context_comment_ids = [
        comment.get("comment_id")
        for comment in context_payload.get("used_context_comments", [])
        if comment.get("comment_id")
    ]

    if not api_key:
        manifest_errors.append(
            {
                "type": "missing_openai_api_key",
                "message": "OPENAI_API_KEY was not found in environment or .env.",
            }
        )
    if not serper_api_key:
        manifest_errors.append(
            {
                "type": "missing_serper_api_key",
                "message": "SERPER_API_KEY was not found in environment or .env.",
            }
        )

    if api_key and serper_api_key:
        query_model_availability = _check_model_available(
            api_key, config.query_model, config.request_timeout_seconds
        )
        validation_model_availability = (
            query_model_availability
            if config.validation_model == config.query_model
            else _check_model_available(
                api_key, config.validation_model, config.request_timeout_seconds
            )
        )
        model_availability = {
            "query_model": query_model_availability,
            "validation_model": validation_model_availability,
        }
        if not query_model_availability.get("available"):
            manifest_errors.append(
                {
                    "type": "query_model_unavailable",
                    "message": f"Approved model {config.query_model} is not available.",
                    "details": query_model_availability,
                }
            )
        elif not validation_model_availability.get("available"):
            manifest_errors.append(
                {
                    "type": "validation_model_unavailable",
                    "message": (
                        f"Approved model {config.validation_model} is not available."
                    ),
                    "details": validation_model_availability,
                }
            )
        else:
            query_response = _create_json_response(
                api_key=api_key,
                model=config.query_model,
                temperature=config.temperature,
                developer_prompt=query_developer_prompt,
                user_prompt=query_user_prompt,
                schema_name="rag_g2_news_query",
                schema=_query_response_schema(),
                timeout=config.request_timeout_seconds,
            )
            parsed_query, query_response_text, query_parse_error = _parse_model_json(
                query_response
            )
            query_schema_errors = _schema_errors(
                parsed_query, _query_response_schema()
            )
            raw_records.append(
                {
                    "record_type": "query_generation",
                    "query_id": query_id,
                    "event_id": config.event_id,
                    "attempt": 1,
                    "model": config.query_model,
                    "temperature_requested": config.temperature,
                    "temperature_sent": _temperature_sent_value(
                        config.query_model, config.temperature
                    ),
                    "temperature_parameter_sent": query_response.get(
                        "temperature_parameter_sent"
                    ),
                    "prompt_version": QUERY_PROMPT_VERSION,
                    "response_status_code": query_response.get("status_code"),
                    "response_id": (
                        query_response.get("body", {}).get("id")
                        if isinstance(query_response.get("body"), dict)
                        else None
                    ),
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
                        "type": "query_generation_failed",
                        "parse_error": query_parse_error,
                        "schema_errors": query_schema_errors,
                    }
                )
            else:
                query_status, query_status_limitations = _classify_query(
                    parsed_query.get("query", ""), query_payload
                )
                query_record = _query_record_from_model(
                    query_id=query_id,
                    event_id=config.event_id,
                    query_result=parsed_query,
                    query_status=query_status,
                    query_status_limitations=query_status_limitations,
                    validation_input=validation_input,
                    search_start=search_start,
                    search_end=search_end,
                    config=config,
                    generated_at_utc=_utc_now_iso(),
                )
                query_records.append(query_record)
                if query_status == "invalid_invented_content":
                    manifest_errors.append(
                        {
                            "type": "invalid_query_invented_content",
                            "message": "Generated query was not sent to Serper.",
                            "limitations": query_record.get("limitations", []),
                        }
                    )
                else:
                    retrieved_at_utc = _utc_now_iso()
                    serper_response = _call_serper_news(
                        api_key=serper_api_key,
                        query=str(query_record.get("query", "")),
                        window_start_date=search_start,
                        window_end_date=search_end,
                        config=config,
                    )
                    if serper_response.get("status_code") in {401, 403}:
                        retrieval_status = "auth_failed"
                        manifest_errors.append(
                            {
                                "type": "serper_auth_failed",
                                "message": (
                                    "Serper authentication failed; no provider "
                                    "fallback was attempted."
                                ),
                                "status_code": serper_response.get("status_code"),
                            }
                        )
                    elif not serper_response.get("ok"):
                        retrieval_status = "error"
                        manifest_errors.append(
                            {
                                "type": "serper_request_failed",
                                "status_code": serper_response.get("status_code"),
                                "body": serper_response.get("body"),
                            }
                        )
                    else:
                        external_evidence = _normalize_external_evidence(
                            event_id=config.event_id,
                            query_id=query_id,
                            query=str(query_record.get("query", "")),
                            serper_response=serper_response,
                            trigger_time_utc=str(validation_input.get("trigger_time_utc")),
                            provider="serper_news",
                            retrieved_at_utc=retrieved_at_utc,
                        )
                        retrieval_status = (
                            "success" if external_evidence else "no_results"
                        )

                    if retrieval_status in {"success", "no_results"}:
                        compact_validation_payload = _compact_validation_payload(
                            validation_input,
                            context_payload,
                            g1_report,
                            query_record,
                            external_evidence,
                        )
                        validation_developer_prompt, validation_user_prompt = (
                            _validation_prompt_text(compact_validation_payload)
                        )
                        validation_prompt_hash = hashlib.sha256(
                            (
                                validation_developer_prompt
                                + "\n\n"
                                + validation_user_prompt
                            ).encode("utf-8")
                        ).hexdigest()
                        last_errors: list[str] = []
                        for attempt in range(1, config.max_retries + 2):
                            validation_response = _create_json_response(
                                api_key=api_key,
                                model=config.validation_model,
                                temperature=config.temperature,
                                developer_prompt=validation_developer_prompt,
                                user_prompt=validation_user_prompt,
                                schema_name="rag_g2_external_validation_report",
                                schema=_validation_response_schema(),
                                timeout=config.request_timeout_seconds,
                            )
                            response_body = validation_response.get("body", {})
                            raw_id = (
                                response_body.get("id")
                                if isinstance(response_body, dict)
                                else None
                            )
                            parsed_report, response_text, parse_error = (
                                _parse_model_json(validation_response)
                            )
                            schema_errors = _schema_errors(
                                parsed_report, _validation_response_schema()
                            )
                            citation_verification = (
                                _citation_verification(
                                    report=parsed_report,
                                    context_payload=context_payload,
                                    external_evidence=external_evidence,
                                )
                                if parsed_report is not None and not schema_errors
                                else {
                                    "valid": False,
                                    "errors": ["schema_or_parse_error"],
                                }
                            )
                            raw_records.append(
                                {
                                    "record_type": "validation_generation",
                                    "validation_id": validation_id,
                                    "event_id": config.event_id,
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
                                    "response_status_code": validation_response.get(
                                        "status_code"
                                    ),
                                    "response_id": raw_id,
                                    "response_text": response_text,
                                    "parsed_json": parsed_report,
                                    "parse_error": parse_error,
                                    "schema_errors": schema_errors,
                                    "citation_verification": citation_verification,
                                    "raw_response": response_body,
                                    "created_at_utc": _utc_now_iso(),
                                }
                            )
                            last_errors = []
                            if parse_error:
                                last_errors.append(parse_error)
                            last_errors.extend(schema_errors)
                            if (
                                parsed_report is not None
                                and not schema_errors
                                and citation_verification["valid"]
                            ):
                                validation_reports.append(
                                    _finalize_report(
                                        model_report=parsed_report,
                                        validation_id=validation_id,
                                        event_id=config.event_id,
                                        model=config.validation_model,
                                        validated_at_utc=created_at_utc,
                                        citation_verification=citation_verification,
                                        attempt_count=attempt,
                                        raw_response_id=raw_id,
                                        query_id=query_id,
                                        external_evidence_count=len(external_evidence),
                                    )
                                )
                                break
                            if parsed_report is not None and not schema_errors:
                                last_errors.extend(citation_verification.get("errors", []))
                            if attempt <= config.max_retries:
                                manifest_errors.append(
                                    {
                                        "type": "retry",
                                        "stage": "validation_generation",
                                        "attempt": attempt,
                                        "errors": last_errors,
                                    }
                                )
                        if not validation_reports:
                            manifest_errors.append(
                                {
                                    "type": "validation_generation_failed",
                                    "message": "All allowed attempts failed validation.",
                                    "errors": last_errors,
                                }
                            )
                            validation_reports.append(
                                _error_report(
                                    validation_id=validation_id,
                                    event_id=config.event_id,
                                    model=config.validation_model,
                                    validated_at_utc=created_at_utc,
                                    used_context_comment_ids=used_context_comment_ids,
                                    post_trigger_context_used=bool(
                                        context_payload.get("post_trigger_context_used")
                                    ),
                                    errors=last_errors or ["generation_failed"],
                                    attempt_count=sum(
                                        1
                                        for item in raw_records
                                        if item.get("record_type")
                                        == "validation_generation"
                                    ),
                                    external_evidence_count=len(external_evidence),
                                    query_id=query_id,
                                )
                            )
                    else:
                        manifest_errors.append(
                            {
                                "type": "validation_skipped_due_to_retrieval_status",
                                "retrieval_status": retrieval_status,
                            }
                        )

    if not validation_reports and manifest_errors:
        validation_reports.append(
            _error_report(
                validation_id=validation_id,
                event_id=config.event_id,
                model=config.validation_model,
                validated_at_utc=created_at_utc,
                used_context_comment_ids=used_context_comment_ids,
                post_trigger_context_used=bool(
                    context_payload.get("post_trigger_context_used")
                ),
                errors=[
                    str(error.get("type", "unknown_error")) for error in manifest_errors
                ],
                attempt_count=sum(
                    1
                    for item in raw_records
                    if item.get("record_type") == "validation_generation"
                ),
                external_evidence_count=len(external_evidence),
                query_id=query_id if query_record else None,
            )
        )

    report = validation_reports[0] if validation_reports else None
    validation_prompt_hash = None
    validation_prompt_record = None
    if query_record is not None and retrieval_status in {"success", "no_results"}:
        compact_validation_payload = _compact_validation_payload(
            validation_input,
            context_payload,
            g1_report,
            query_record,
            external_evidence,
        )
        validation_developer_prompt, validation_user_prompt = _validation_prompt_text(
            compact_validation_payload
        )
        validation_prompt_hash = hashlib.sha256(
            (validation_developer_prompt + "\n\n" + validation_user_prompt).encode(
                "utf-8"
            )
        ).hexdigest()
        validation_prompt_record = {
            "prompt_version": VALIDATION_PROMPT_VERSION,
            "developer_prompt": validation_developer_prompt,
            "user_prompt": validation_user_prompt,
            "prompt_sha256": validation_prompt_hash,
            "response_schema": _validation_response_schema(),
        }

    input_hashes = {
        name: _sha256_file(path)
        for name, path in {**consumer_paths, **g1_paths}.items()
        if path.exists()
    }
    query_prompt_record = {
        "prompt_version": QUERY_PROMPT_VERSION,
        "developer_prompt": query_developer_prompt,
        "user_prompt": query_user_prompt,
        "prompt_sha256": query_prompt_hash,
        "response_schema": _query_response_schema(),
    }
    serper_request_payload = (
        serper_response.get("request_payload")
        if isinstance(serper_response, dict)
        else None
    )
    manifest = {
        "run_id": "ragg2_" + _short_hash(config.event_id, created_at_utc),
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_generation_g2_external_evidence",
        "mode": "single_event_internal_plus_serper_news_llm_validation",
        "artifact_version": RAG_G2_ARTIFACT_VERSION,
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
        "query_prompt_sha256": query_prompt_hash,
        "validation_prompt_sha256": validation_prompt_hash,
        "prompts": {
            "query": query_prompt_record,
            "validation": validation_prompt_record,
        },
        "event_ids": [config.event_id],
        "approved_event_count": 1,
        "executed_event_count": 1,
        "consumer_run_id": consumer_manifest.get("run_id"),
        "input_paths": {
            name: _normalize_path(path) for name, path in {**consumer_paths, **g1_paths}.items()
        },
        "input_hashes": input_hashes,
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "scope_policy": {
            "internal_youtube_evidence_used": True,
            "external_news_evidence_used": True,
            "serper_used": bool(serper_response),
            "embeddings_used": False,
            "vectorstore_used": False,
            "chroma_used": False,
            "query_expansion_used": False,
            "multi_query_used": False,
            "does_not_modify_sidecars": True,
            "does_not_modify_consumer": True,
            "does_not_modify_g1": True,
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
            "request_payload_without_key": serper_request_payload,
        },
        "query": query_record,
        "query_status": query_status,
        "retrieval_status": retrieval_status,
        "external_evidence_count": len(external_evidence),
        "external_evidence_ids": [
            item.get("external_evidence_id") for item in external_evidence
        ],
        "model_availability": model_availability,
        "raw_response_count": len(raw_records),
        "validation_attempts": sum(
            1 for item in raw_records if item.get("record_type") == "validation_generation"
        ),
        "max_retries": config.max_retries,
        "retry_count": sum(1 for item in manifest_errors if item.get("type") == "retry"),
        "errors": manifest_errors,
        "citation_verification": (
            report.get("citation_verification") if report else None
        ),
        "validation_status": report.get("validation_status") if report else None,
        "event_interpretation": report.get("event_interpretation") if report else None,
        "external_evidence_assessment": (
            report.get("external_evidence_assessment") if report else None
        ),
        "notes": config.notes,
        "params": config.params,
    }

    _write_jsonl(output_paths["rag_news_queries"], query_records)
    _write_jsonl(output_paths["rag_external_evidence"], external_evidence)
    _write_jsonl(output_paths["rag_validation_reports"], validation_reports)
    _write_jsonl(output_paths["rag_raw_model_responses"], raw_records)
    _write_json(output_paths["rag_generation_manifest"], manifest)

    return {
        "run_id": manifest["run_id"],
        "event_id": config.event_id,
        "output_dir": _normalize_path(output_dir),
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "query": query_record.get("query") if query_record else None,
        "query_status": query_status,
        "retrieval_status": retrieval_status,
        "external_evidence_count": len(external_evidence),
        "query_model_name": config.query_model,
        "validation_model_name": config.validation_model,
        "temperature_requested": config.temperature,
        "query_temperature_sent": _temperature_sent_value(
            config.query_model, config.temperature
        ),
        "validation_temperature_sent": _temperature_sent_value(
            config.validation_model, config.temperature
        ),
        "validation_status": report.get("validation_status") if report else None,
        "event_interpretation": report.get("event_interpretation") if report else None,
        "confidence_label": report.get("confidence_label") if report else None,
        "external_evidence_assessment": (
            report.get("external_evidence_assessment") if report else None
        ),
        "citation_verification": (
            report.get("citation_verification") if report else None
        ),
        "errors": manifest_errors,
    }


def run_rag_g2_validation(
    *,
    consumer_dir: str | Path,
    g1_dir: str | Path,
    output_dir: str | Path,
    event_id: str,
    query_model: str = "gpt-5-mini",
    validation_model: str = "gpt-5-mini",
    temperature: float = 0.0,
    max_retries: int = 1,
    notes: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = RagG2Config(
        consumer_dir=str(consumer_dir),
        g1_dir=str(g1_dir),
        output_dir=str(output_dir),
        event_id=event_id,
        query_model=query_model,
        validation_model=validation_model,
        temperature=temperature,
        max_retries=max_retries,
        notes=notes,
        params=params or {},
    )
    return run_rag_g2_validation_from_config(config)


__all__ = [
    "QUERY_PROMPT_VERSION",
    "RAG_G2_ARTIFACT_VERSION",
    "VALIDATION_PROMPT_VERSION",
    "RagG2Config",
    "load_rag_g2_config",
    "run_rag_g2_validation",
    "run_rag_g2_validation_from_config",
]
