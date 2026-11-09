from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from youtube_pipeline.validation_state import (
    COMPLETE,
    CONTEXT_LIMIT_EXCEEDED,
    EXECUTED,
    FAILED,
    INVALID_CITATION,
    MISSING_PAYLOAD,
    PARTIAL,
    PROVIDER_ERROR,
    TIMEOUT,
)


RAG_G1_ARTIFACT_VERSION = "rag_generation_g1_v3"
PROMPT_VERSION = "rag_validation_prompt_v0.2"

RAG_VALIDATION_INPUTS_FILE = "rag_validation_inputs.jsonl"
RAG_CONTEXT_PAYLOADS_FILE = "rag_context_payloads.jsonl"
RAG_CONSUMER_MANIFEST_FILE = "rag_consumer_manifest.json"

RAG_VALIDATION_REPORTS_FILE = "rag_validation_reports.jsonl"
RAG_GENERATION_MANIFEST_FILE = "rag_generation_manifest.json"
RAG_RAW_MODEL_RESPONSES_FILE = "rag_raw_model_responses.jsonl"

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL_URL = "https://api.openai.com/v1/models/{model}"

EVENT_INTERPRETATION_VALUES = {
    "internal_community_reaction",
    "possible_external_event",
    "possible_noise",
    "possible_disinformation",
    "unclear",
}
CONFIDENCE_LABEL_VALUES = {"low", "medium", "high"}


@dataclass(frozen=True)
class RagG1Config:
    consumer_dir: str
    output_dir: str
    event_id: str
    model: str = "gpt-5-mini"
    provider: str = "openai"
    temperature: float = 0.0
    max_approx_tokens: int = 16_000
    max_retries: int = 1
    request_timeout_seconds: int = 120
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RagG1Config":
        missing = [
            key
            for key in ["consumer_dir", "output_dir", "event_id"]
            if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                "RAG G-1 config missing required fields: " + ", ".join(missing)
            )
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        return cls(
            consumer_dir=str(payload["consumer_dir"]),
            output_dir=str(payload["output_dir"]),
            event_id=str(payload["event_id"]),
            model=str(payload.get("model", "gpt-5-mini")),
            provider=str(payload.get("provider", "openai")),
            temperature=float(payload.get("temperature", 0.0)),
            max_approx_tokens=int(payload.get("max_approx_tokens", 16_000)),
            max_retries=int(payload.get("max_retries", 1)),
            request_timeout_seconds=int(payload.get("request_timeout_seconds", 120)),
            notes=payload.get("notes"),
            params=params,
        )


def _extract_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("rag_generation_g1")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("rag_generation_g1 config section must be an object.")
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


def load_rag_g1_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> RagG1Config:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG G-1 config must be an object: {p}")
    base = _extract_config_payload(payload)
    merged = _merge_config_payloads(base, overrides or {})
    return RagG1Config.from_mapping(merged)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()


def _short_hash(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_safe(record), ensure_ascii=False) for record in records]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_existing_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {p}")
    return payload


def _read_existing_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return _read_json(p)


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSONL artifact: {p}")
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
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
        if name.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


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
        "rag_validation_reports": root / RAG_VALIDATION_REPORTS_FILE,
        "rag_generation_manifest": root / RAG_GENERATION_MANIFEST_FILE,
        "rag_raw_model_responses": root / RAG_RAW_MODEL_RESPONSES_FILE,
    }


def _extract_event(
    *,
    inputs: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    event_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_matches = [record for record in inputs if record.get("event_id") == event_id]
    context_matches = [record for record in contexts if record.get("event_id") == event_id]
    if len(input_matches) != 1:
        raise ValueError(
            f"Expected exactly one validation input for {event_id}, got {len(input_matches)}."
        )
    if len(context_matches) != 1:
        raise ValueError(
            f"Expected exactly one context payload for {event_id}, got {len(context_matches)}."
        )
    return input_matches[0], context_matches[0]


def _approx_tokens(payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return round(len(text) / 4)


def _validation_id(event_id: str, prompt_version: str, model: str) -> str:
    return "valg1_" + _short_hash(event_id, prompt_version, model)


def _compact_event_payload(
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
) -> dict[str, Any]:
    event_metadata = {
        key: validation_input.get(key)
        for key in [
            "validation_id",
            "event_id",
            "run_id",
            "detector_name",
            "trigger_time_utc",
            "window_start_utc",
            "window_end_utc",
            "trigger_volume",
            "trigger_strength",
        ]
        if key in validation_input
    }
    compact_units = []
    for unit in context_payload.get("selected_context_units", []):
        # Comment text is carried exactly once in used_context_comments.  Units
        # retain their IDs, membership and temporal/thread relationships.
        compact_units.append(
            {key: value for key, value in unit.items() if key != "context_text"}
        )
    return {
        "event_metadata": event_metadata,
        "temporal_policy": {
            "alert_evidence_comments": "window_start_utc <= event_time_utc <= trigger_time_utc",
            "validation_context_comments": "window_start_utc <= event_time_utc <= window_end_utc",
            "post_trigger_context_used": context_payload.get("post_trigger_context_used"),
            "confirmed_scope": (
                "In G-1, confirmed means internally confirmed as a conversational "
                "pattern in YouTube evidence, not confirmed as an external public fact."
            ),
        },
        "selected_context_units": compact_units,
        "used_context_comments": context_payload.get("used_context_comments", []),
        "allowed_validation_status": sorted(VALIDATION_STATUS_VALUES),
        "allowed_event_interpretation": sorted(EVENT_INTERPRETATION_VALUES),
    }


def _prompt_text(compact_payload: dict[str, Any]) -> tuple[str, str]:
    developer_prompt = (
        "Eres un evaluador academico de evidencia interna para un prototipo de "
        "deteccion en linea de eventos en YouTube.\n\n"
        "Tu tarea es evaluar si la evidencia interna proporcionada permite sostener "
        "que hubo un patron o evento conversacional asociado a la alerta detectada.\n\n"
        "No estas verificando todavia si ocurrio un hecho publico externo. Esa "
        "validacion requiere una etapa posterior con noticias u otras fuentes externas.\n\n"
        "Usa unicamente la evidencia incluida en este payload. No inventes informacion "
        "externa. No menciones noticias ni fuentes externas. No modifiques la decision "
        "original del detector. No presentes comentarios posteriores al trigger como "
        "causa de la alerta.\n\n"
        "Estados permitidos para G-1:\n"
        "- CONFIRMED: hay evidencia interna suficiente para sostener un patron "
        "conversacional coherente.\n"
        "- INSUFFICIENT_EVIDENCE: la evidencia interna no permite decidir con "
        "suficiente sustento, incluida evidencia parcial o ambigua.\n"
        "- NOT_CONFIRMED: hay evidencia interna suficiente para evaluar, pero no "
        "respalda el patron propuesto.\n"
        "- possible_external_event es solo una hipotesis interpretativa, no una "
        "confirmacion factual externa.\n\n"
        "Debes citar comment_id cuando uses comentarios y context_unit_id cuando uses "
        "unidades de contexto. Separa evidencia observada de inferencia, declara "
        "incertidumbre, reporta limitaciones y respeta los roles temporales.\n\n"
        "Regla de consistencia: CONFIRMED y NOT_CONFIRMED requieren al menos una "
        "cita interna valida. Si no hay evidencia suficiente para decidir, usa "
        "INSUFFICIENT_EVIDENCE.\n\n"
        "Devuelve solo JSON valido que cumpla el esquema."
    )
    user_prompt = (
        "Payload de validacion G-1 con evidencia interna de YouTube:\n\n"
        + json.dumps(compact_payload, ensure_ascii=False, indent=2)
    )
    return developer_prompt, user_prompt


def _response_schema() -> dict[str, Any]:
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
            "evidence_summary": {"type": "string"},
            "reasoning_summary": {"type": "string"},
            "cited_comment_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cited_context_unit_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "validation_status",
            "event_interpretation",
            "confidence_label",
            "claim_summary",
            "evidence_summary",
            "reasoning_summary",
            "cited_comment_ids",
            "cited_context_unit_ids",
            "limitations",
        ],
    }


def _check_model_available(api_key: str, model: str, timeout: int) -> dict[str, Any]:
    response = requests.get(
        OPENAI_MODEL_URL.format(model=model),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    out = {"status_code": response.status_code, "available": response.status_code == 200}
    if response.status_code != 200:
        try:
            payload = response.json()
        except Exception:
            payload = {"text": response.text[:500]}
        out["error"] = payload.get("error", payload)
    return out


def _should_send_temperature(model: str) -> bool:
    return model not in {"gpt-5-mini"}


def _temperature_sent_value(model: str, temperature: float) -> float | None:
    return temperature if _should_send_temperature(model) else None


def _create_response(
    *,
    api_key: str,
    model: str,
    temperature: float,
    developer_prompt: str,
    user_prompt: str,
    timeout: int,
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
                "name": "rag_g1_internal_validation_report",
                "strict": True,
                "schema": _response_schema(),
            }
        },
    }
    if temperature_parameter_sent:
        payload["temperature"] = temperature
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


def _extract_response_text(response_body: dict[str, Any]) -> str | None:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text
    texts: list[str] = []
    for item in response_body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(texts) if texts else None


def _parse_model_json(response: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not response.get("ok"):
        return None, None, f"OpenAI API error status {response.get('status_code')}"
    text = _extract_response_text(response.get("body", {}))
    if not text:
        return None, None, "Response did not contain output text."
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, text, f"Invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, text, "Parsed JSON is not an object."
    return parsed, text, None


def _schema_errors(report: dict[str, Any]) -> list[str]:
    required = set(_response_schema()["required"])
    errors: list[str] = []
    missing = sorted(required.difference(report))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    if report.get("validation_status") not in VALIDATION_STATUS_VALUES:
        errors.append("invalid_validation_status")
    if report.get("event_interpretation") not in EVENT_INTERPRETATION_VALUES:
        errors.append("invalid_event_interpretation")
    if report.get("confidence_label") not in CONFIDENCE_LABEL_VALUES:
        errors.append("invalid_confidence_label")
    for field in [
        "cited_comment_ids",
        "cited_context_unit_ids",
        "limitations",
    ]:
        if field in report and not isinstance(report[field], list):
            errors.append(f"{field}_is_not_array")
    return errors


def _citation_verification(
    *,
    report: dict[str, Any],
    context_payload: dict[str, Any],
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
    cited_comment_ids = set(report.get("cited_comment_ids") or [])
    cited_context_unit_ids = set(report.get("cited_context_unit_ids") or [])

    invalid_comment_ids = sorted(cited_comment_ids.difference(allowed_comment_ids))
    invalid_context_unit_ids = sorted(
        cited_context_unit_ids.difference(allowed_context_unit_ids)
    )
    status_requires_citation = report.get("validation_status") in {
        CONFIRMED,
        NOT_CONFIRMED,
    }
    has_valid_citation = bool(
        cited_comment_ids.intersection(allowed_comment_ids)
        or cited_context_unit_ids.intersection(allowed_context_unit_ids)
    )
    payload_post_trigger = bool(context_payload.get("post_trigger_context_used"))
    errors: list[str] = []
    if invalid_comment_ids:
        errors.append("invalid_cited_comment_ids")
    if invalid_context_unit_ids:
        errors.append("invalid_cited_context_unit_ids")
    if status_requires_citation and not has_valid_citation:
        errors.append("status_requires_at_least_one_valid_citation")
    return {
        "valid": not errors,
        "errors": errors,
        "invalid_cited_comment_ids": invalid_comment_ids,
        "invalid_cited_context_unit_ids": invalid_context_unit_ids,
        "invalid_used_context_comment_ids": [],
        "status_requires_citation": status_requires_citation,
        "has_valid_citation": has_valid_citation,
        "payload_post_trigger_context_used": payload_post_trigger,
        "report_post_trigger_context_used": payload_post_trigger,
        "allowed_comment_count": len(allowed_comment_ids),
        "allowed_context_unit_count": len(allowed_context_unit_ids),
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
    used_context_comment_ids: list[str],
    post_trigger_context_used: bool,
) -> dict[str, Any]:
    report = dict(model_report)
    if citation_verification.get("valid"):
        report.update(
            execution_status=EXECUTED,
            completeness_status=COMPLETE,
            reason_code=None,
        )
    else:
        report["model_validation_status"] = report.get("validation_status")
        report.update(
            validation_status=None,
            execution_status=FAILED,
            completeness_status=PARTIAL,
            reason_code=INVALID_CITATION,
        )
    report["validation_id"] = validation_id
    report["event_id"] = event_id
    report["model_name"] = model
    report["prompt_version"] = PROMPT_VERSION
    report["validated_at_utc"] = validated_at_utc
    report["used_context_comment_ids"] = used_context_comment_ids
    report["post_trigger_context_used"] = post_trigger_context_used
    report["citation_verification"] = citation_verification
    report["attempt_count"] = attempt_count
    report["raw_response_id"] = raw_response_id
    report["generation_scope"] = "internal_youtube_evidence_only"
    report["external_evidence_used"] = False
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
) -> dict[str, Any]:
    error_text = " ".join(errors).lower()
    if "context_limit_exceeded" in error_text:
        reason_code = CONTEXT_LIMIT_EXCEEDED
    elif "timeout" in error_text:
        reason_code = TIMEOUT
    elif "citation" in error_text:
        reason_code = INVALID_CITATION
    elif "parse" in error_text or "schema" in error_text or "payload" in error_text:
        reason_code = MISSING_PAYLOAD
    else:
        reason_code = PROVIDER_ERROR
    return {
        "validation_id": validation_id,
        "event_id": event_id,
        "validation_status": None,
        "execution_status": FAILED,
        "completeness_status": PARTIAL,
        "reason_code": reason_code,
        "event_interpretation": "unclear",
        "confidence_label": "low",
        "claim_summary": "No se obtuvo una validacion generativa estructurada valida.",
        "evidence_summary": "",
        "reasoning_summary": "",
        "cited_comment_ids": [],
        "cited_context_unit_ids": [],
        "used_context_comment_ids": used_context_comment_ids,
        "post_trigger_context_used": post_trigger_context_used,
        "limitations": [
            "La generacion con LLM fallo o no cumplio el esquema/citas requeridas.",
            *errors,
        ],
        "model_name": model,
        "prompt_version": PROMPT_VERSION,
        "validated_at_utc": validated_at_utc,
        "citation_verification": {
            "valid": False,
            "errors": errors,
        },
        "attempt_count": attempt_count,
        "generation_scope": "internal_youtube_evidence_only",
        "external_evidence_used": False,
        "generation_error": True,
    }


def run_rag_g1_validation_from_config(config: RagG1Config) -> dict[str, Any]:
    if config.provider != "openai":
        raise ValueError("G-1 currently supports only provider='openai'.")
    if config.model != "gpt-5-mini":
        raise ValueError("Approved G-1 model is gpt-5-mini; refusing to change model.")
    if config.max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

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
    compact_payload = _compact_event_payload(validation_input, context_payload)
    approx_tokens = _approx_tokens(compact_payload)
    created_at_utc = _utc_now_iso()
    validation_id = _validation_id(config.event_id, PROMPT_VERSION, config.model)

    prompt_developer, prompt_user = _prompt_text(compact_payload)
    prompt_hash = hashlib.sha256(
        (prompt_developer + "\n\n" + prompt_user).encode("utf-8")
    ).hexdigest()
    used_context_comment_ids = [
        comment.get("comment_id")
        for comment in context_payload.get("used_context_comments", [])
        if comment.get("comment_id")
    ]
    raw_records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    manifest_errors: list[dict[str, Any]] = []

    context_limit_exceeded = approx_tokens > config.max_approx_tokens
    api_key = _load_api_key()
    model_availability = None

    if context_limit_exceeded:
        manifest_errors.append(
            {
                "type": "context_limit_exceeded",
                "message": (
                    f"Approx token estimate {approx_tokens} exceeds "
                    f"limit {config.max_approx_tokens}; no model call made."
                ),
            }
        )
        reports.append(
            _error_report(
                validation_id=validation_id,
                event_id=config.event_id,
                model=config.model,
                validated_at_utc=created_at_utc,
                used_context_comment_ids=used_context_comment_ids,
                post_trigger_context_used=bool(
                    context_payload.get("post_trigger_context_used")
                ),
                errors=["context_limit_exceeded"],
                attempt_count=0,
            )
        )
    elif not api_key:
        manifest_errors.append(
            {
                "type": "missing_openai_api_key",
                "message": "OPENAI_API_KEY was not found in environment or .env.",
            }
        )
        reports.append(
            _error_report(
                validation_id=validation_id,
                event_id=config.event_id,
                model=config.model,
                validated_at_utc=created_at_utc,
                used_context_comment_ids=used_context_comment_ids,
                post_trigger_context_used=bool(
                    context_payload.get("post_trigger_context_used")
                ),
                errors=["missing_openai_api_key"],
                attempt_count=0,
            )
        )
    else:
        model_availability = _check_model_available(
            api_key, config.model, config.request_timeout_seconds
        )
        if not model_availability.get("available"):
            manifest_errors.append(
                {
                    "type": "model_unavailable",
                    "message": f"Approved model {config.model} is not available.",
                    "details": model_availability,
                }
            )
            reports.append(
                _error_report(
                    validation_id=validation_id,
                    event_id=config.event_id,
                    model=config.model,
                    validated_at_utc=created_at_utc,
                    used_context_comment_ids=used_context_comment_ids,
                    post_trigger_context_used=bool(
                        context_payload.get("post_trigger_context_used")
                    ),
                    errors=["model_unavailable"],
                    attempt_count=0,
                )
            )
        else:
            last_errors: list[str] = []
            for attempt in range(1, config.max_retries + 2):
                response = _create_response(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    developer_prompt=prompt_developer,
                    user_prompt=prompt_user,
                    timeout=config.request_timeout_seconds,
                )
                response_body = response.get("body", {})
                raw_id = response_body.get("id") if isinstance(response_body, dict) else None
                parsed, response_text, parse_error = _parse_model_json(response)
                schema_errors = _schema_errors(parsed) if parsed is not None else []
                citation_verification = (
                    _citation_verification(
                        report=parsed,
                        context_payload=context_payload,
                    )
                    if parsed is not None and not schema_errors
                    else {
                        "valid": False,
                        "errors": ["schema_or_parse_error"],
                    }
                )
                raw_records.append(
                    {
                        "validation_id": validation_id,
                        "event_id": config.event_id,
                        "attempt": attempt,
                        "model": config.model,
                        "temperature_requested": config.temperature,
                        "temperature_sent": _temperature_sent_value(
                            config.model, config.temperature
                        ),
                        "temperature_parameter_sent": response.get(
                            "temperature_parameter_sent"
                        ),
                        "prompt_version": PROMPT_VERSION,
                        "request_scope": "internal_youtube_evidence_only",
                        "response_status_code": response.get("status_code"),
                        "response_id": raw_id,
                        "response_text": response_text,
                        "parsed_json": parsed,
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
                if parsed is not None and not schema_errors and citation_verification["valid"]:
                    reports.append(
                        _finalize_report(
                            model_report=parsed,
                            validation_id=validation_id,
                            event_id=config.event_id,
                            model=config.model,
                            validated_at_utc=created_at_utc,
                            citation_verification=citation_verification,
                            attempt_count=attempt,
                            raw_response_id=raw_id,
                            used_context_comment_ids=used_context_comment_ids,
                            post_trigger_context_used=bool(
                                context_payload.get("post_trigger_context_used")
                            ),
                        )
                    )
                    break
                if parsed is not None and not schema_errors:
                    manifest_errors.append(
                        {
                            "type": "citation_verification_failed",
                            "attempt": attempt,
                            "errors": citation_verification.get("errors", []),
                        }
                    )
                    reports.append(
                        _finalize_report(
                            model_report=parsed,
                            validation_id=validation_id,
                            event_id=config.event_id,
                            model=config.model,
                            validated_at_utc=created_at_utc,
                            citation_verification=citation_verification,
                            attempt_count=attempt,
                            raw_response_id=raw_id,
                            used_context_comment_ids=used_context_comment_ids,
                            post_trigger_context_used=bool(
                                context_payload.get("post_trigger_context_used")
                            ),
                        )
                    )
                    break
                if attempt <= config.max_retries:
                    manifest_errors.append(
                        {
                            "type": "retry",
                            "attempt": attempt,
                            "errors": last_errors,
                        }
                    )
            if not reports:
                manifest_errors.append(
                    {
                        "type": "generation_failed",
                        "message": "All allowed attempts failed validation.",
                        "errors": last_errors,
                    }
                )
                reports.append(
                    _error_report(
                        validation_id=validation_id,
                        event_id=config.event_id,
                        model=config.model,
                        validated_at_utc=created_at_utc,
                        used_context_comment_ids=used_context_comment_ids,
                        post_trigger_context_used=bool(
                            context_payload.get("post_trigger_context_used")
                        ),
                        errors=last_errors or ["generation_failed"],
                        attempt_count=len(raw_records),
                    )
                )

    report = reports[0]
    prompt_record = {
        "prompt_version": PROMPT_VERSION,
        "developer_prompt": prompt_developer,
        "user_prompt": prompt_user,
        "prompt_sha256": prompt_hash,
        "response_schema": _response_schema(),
    }
    input_hashes = {
        name: _sha256_file(path)
        for name, path in consumer_paths.items()
        if path.exists()
    }
    manifest = {
        "run_id": "ragg1_" + _short_hash(config.event_id, PROMPT_VERSION, config.model),
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_generation_g1_internal_only",
        "mode": "single_event_internal_evidence_llm_validation",
        "artifact_version": RAG_G1_ARTIFACT_VERSION,
        "provider": config.provider,
        "model_name": config.model,
        "temperature_requested": config.temperature,
        "temperature_sent": _temperature_sent_value(config.model, config.temperature),
        "temperature_parameter_sent": _should_send_temperature(config.model),
        "temperature_effective_note": (
            "temperature parameter not sent because gpt-5-mini does not support it"
            if not _should_send_temperature(config.model)
            else "temperature parameter sent to API"
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "prompt": prompt_record,
        "event_ids": [config.event_id],
        "approved_event_count": 1,
        "executed_event_count": 1,
        "consumer_run_id": consumer_manifest.get("run_id"),
        "input_paths": {
            name: _normalize_path(path) for name, path in consumer_paths.items()
        },
        "input_hashes": input_hashes,
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "context_policy": {
            "max_approx_tokens": config.max_approx_tokens,
            "approx_tokens": approx_tokens,
            "context_limit_exceeded": context_limit_exceeded,
            "automatic_truncation": False,
            "used_all_context_units_for_event": True,
            "used_context_unit_count": len(context_payload.get("selected_context_units", [])),
            "used_context_comment_count": len(
                context_payload.get("used_context_comments", [])
            ),
        },
        "scope_policy": {
            "internal_youtube_evidence_only": True,
            "external_evidence_used": False,
            "news_used": False,
            "serper_used": False,
            "embeddings_used": False,
            "vectorstore_used": False,
            "does_not_modify_sidecars": True,
            "does_not_modify_consumer": True,
            "does_not_modify_pipeline": True,
            "does_not_change_event_id": True,
            "does_not_change_trigger_time_video_id": True,
        },
        "model_availability": model_availability,
        "attempts": len(raw_records),
        "max_retries": config.max_retries,
        "retry_count": max(0, len(raw_records) - 1),
        "errors": manifest_errors,
        "citation_verification": report.get("citation_verification"),
        "validation_status": report.get("validation_status"),
        "execution_status": report.get("execution_status"),
        "completeness_status": report.get("completeness_status"),
        "reason_code": report.get("reason_code"),
        "event_interpretation": report.get("event_interpretation"),
        "notes": config.notes,
        "params": config.params,
    }

    current_event_run = {
        "event_id": config.event_id,
        "run_id": manifest["run_id"],
        "created_at_utc": created_at_utc,
        "validation_id": validation_id,
        "validation_status": report.get("validation_status"),
        "execution_status": report.get("execution_status"),
        "completeness_status": report.get("completeness_status"),
        "reason_code": report.get("reason_code"),
        "event_interpretation": report.get("event_interpretation"),
        "confidence_label": report.get("confidence_label"),
        "approx_tokens": approx_tokens,
        "attempts": len(raw_records),
        "retry_count": manifest["retry_count"],
        "citation_verification": report.get("citation_verification"),
        "context_policy": manifest["context_policy"],
        "errors": manifest_errors,
    }
    existing_manifest = _read_existing_json(output_paths["rag_generation_manifest"])
    event_runs: list[dict[str, Any]] = []
    if existing_manifest:
        existing_event_runs = existing_manifest.get("event_runs")
        if isinstance(existing_event_runs, list):
            event_runs = [
                item
                for item in existing_event_runs
                if isinstance(item, dict) and item.get("event_id") != config.event_id
            ]
        elif existing_manifest.get("event_ids"):
            for event in existing_manifest.get("event_ids", []):
                if event == config.event_id:
                    continue
                event_runs.append(
                    {
                        "event_id": event,
                        "run_id": existing_manifest.get("run_id"),
                        "created_at_utc": existing_manifest.get("created_at_utc"),
                        "validation_status": existing_manifest.get(
                            "validation_status"
                        ),
                        "event_interpretation": existing_manifest.get(
                            "event_interpretation"
                        ),
                        "attempts": existing_manifest.get("attempts"),
                        "retry_count": existing_manifest.get("retry_count"),
                        "citation_verification": existing_manifest.get(
                            "citation_verification"
                        ),
                        "context_policy": existing_manifest.get("context_policy"),
                        "errors": existing_manifest.get("errors", []),
                    }
                )
    event_runs.append(current_event_run)
    event_ids: list[str] = []
    for event_run in event_runs:
        event_id = event_run.get("event_id")
        if event_id and event_id not in event_ids:
            event_ids.append(str(event_id))
    manifest["event_ids"] = event_ids
    manifest["approved_event_count"] = len(event_ids)
    manifest["executed_event_count"] = len(event_ids)
    manifest["event_runs"] = event_runs
    manifest["latest_event_id"] = config.event_id
    manifest["aggregate_manifest"] = True

    existing_reports = _read_existing_jsonl_records(
        output_paths["rag_validation_reports"]
    )
    combined_reports = [
        item for item in existing_reports if item.get("event_id") != config.event_id
    ] + reports
    existing_raw_records = _read_existing_jsonl_records(
        output_paths["rag_raw_model_responses"]
    )
    combined_raw_records = [
        item
        for item in existing_raw_records
        if item.get("validation_id") != validation_id
    ] + raw_records

    _write_jsonl(output_paths["rag_validation_reports"], combined_reports)
    _write_jsonl(output_paths["rag_raw_model_responses"], combined_raw_records)
    _write_json(output_paths["rag_generation_manifest"], manifest)

    return {
        "run_id": manifest["run_id"],
        "event_id": config.event_id,
        "output_dir": _normalize_path(output_dir),
        "output_paths": {
            name: _normalize_path(path) for name, path in output_paths.items()
        },
        "model_name": config.model,
        "temperature": config.temperature,
        "temperature_sent": _temperature_sent_value(config.model, config.temperature),
        "temperature_parameter_sent": _should_send_temperature(config.model),
        "prompt_version": PROMPT_VERSION,
        "approx_tokens": approx_tokens,
        "attempts": len(raw_records),
        "retry_count": manifest["retry_count"],
        "validation_status": report.get("validation_status"),
        "execution_status": report.get("execution_status"),
        "completeness_status": report.get("completeness_status"),
        "reason_code": report.get("reason_code"),
        "event_interpretation": report.get("event_interpretation"),
        "confidence_label": report.get("confidence_label"),
        "citation_verification": report.get("citation_verification"),
        "errors": manifest_errors,
    }


def run_rag_g1_validation(
    *,
    consumer_dir: str | Path,
    output_dir: str | Path,
    event_id: str,
    model: str = "gpt-5-mini",
    temperature: float = 0.0,
    max_approx_tokens: int = 16_000,
    max_retries: int = 1,
    notes: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = RagG1Config(
        consumer_dir=str(consumer_dir),
        output_dir=str(output_dir),
        event_id=event_id,
        model=model,
        temperature=temperature,
        max_approx_tokens=max_approx_tokens,
        max_retries=max_retries,
        notes=notes,
        params=params or {},
    )
    return run_rag_g1_validation_from_config(config)


__all__ = [
    "PROMPT_VERSION",
    "RAG_G1_ARTIFACT_VERSION",
    "RagG1Config",
    "load_rag_g1_config",
    "run_rag_g1_validation",
    "run_rag_g1_validation_from_config",
]
