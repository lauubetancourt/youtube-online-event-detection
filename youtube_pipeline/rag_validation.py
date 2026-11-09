from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .rag_evidence import read_table, write_json, write_jsonl


RAG_VALIDATION_ARTIFACT_VERSION = "rag_validation_harness_v1"

VALIDATION_MANIFEST_FILE = "rag_validation_manifest.json"
VALIDATION_TASKS_FILE = "rag_validation_tasks.csv"
RETRIEVAL_QUESTIONS_FILE = "rag_retrieval_questions.csv"
RAG_QUERIES_FILE = "rag_queries.csv"
EXTERNAL_EVIDENCE_FILE = "external_evidence.csv"
VALIDATION_RESULTS_FILE = "validation_results.csv"
VALIDATION_SUMMARY_FILE = "rag_validation_summary.json"

VALIDATION_LABELS = [
    "confirmed",
    "partially_confirmed",
    "not_confirmed",
    "ambiguous",
]

RETRIEVAL_QUESTION_TEMPLATES = [
    (
        "event_identity",
        "What public event, if any, occurred near {trigger_time_utc}?",
    ),
    (
        "entities",
        "Which people, institutions, places, or topics are central to this event candidate?",
    ),
    (
        "external_coverage",
        "Do reliable external sources mention the same event within the validation time window?",
    ),
    (
        "reaction_type",
        "Is the YouTube activity reacting to a documented public event, to video publication dynamics, or to platform-internal discussion?",
    ),
    (
        "claim_support",
        "Are the strongest claims in the comments supported, contradicted, or absent in external sources?",
    ),
    (
        "topic_coherence",
        "Does the candidate involve one coherent event, multiple sub-events, or unrelated topics that only coincide temporally?",
    ),
]


@dataclass(frozen=True)
class RagValidationPrepareConfig:
    evidence_packages_path: str
    output_dir: str
    validation_run_id: str | None = None
    validator: str = "manual_pending"
    query_language: str = "es"
    max_videos_per_event: int = 5
    notes: str | None = None
    validation_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RagValidationPrepareConfig":
        missing = [
            key
            for key in ["evidence_packages_path", "output_dir"]
            if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                "RAG validation config missing required fields: "
                + ", ".join(missing)
            )
        validation_params = payload.get("validation_params") or {}
        if not isinstance(validation_params, dict):
            raise ValueError("validation_params must be an object.")
        return cls(
            evidence_packages_path=str(payload["evidence_packages_path"]),
            output_dir=str(payload["output_dir"]),
            validation_run_id=payload.get("validation_run_id"),
            validator=str(payload.get("validator", "manual_pending")),
            query_language=str(payload.get("query_language", "es")),
            max_videos_per_event=int(payload.get("max_videos_per_event", 5)),
            notes=payload.get("notes"),
            validation_params=validation_params,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).as_posix()


def _short_hash(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _validation_run_id(evidence_packages_path: str | Path, output_dir: str | Path) -> str:
    return f"ragval_{_short_hash(_normalize_path(evidence_packages_path), _normalize_path(output_dir))}"


def derive_rag_validation_run_id(config: RagValidationPrepareConfig) -> str:
    """Return the existing path-derived validation stage identity."""

    return _validation_run_id(config.evidence_packages_path, config.output_dir)


def _extract_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("rag_validation")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise ValueError("rag_validation config section must be an object.")
    return nested


def _merge_config_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for key, value in payload.items():
            if value is None:
                continue
            if key == "validation_params":
                current = merged.get(key, {})
                if not isinstance(current, dict) or not isinstance(value, dict):
                    raise ValueError("validation_params must be an object.")
                merged[key] = {**current, **value}
            else:
                merged[key] = value
    return merged


def load_rag_validation_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> RagValidationPrepareConfig:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG validation config must be an object: {p}")
    base = _extract_config_payload(payload)
    merged = _merge_config_payloads(base, overrides or {})
    return RagValidationPrepareConfig.from_mapping(merged)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    records: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object: {p}")
            records.append(payload)
    return records


def _package_path(packages_path: str | Path) -> str:
    return _normalize_path(packages_path) or ""


def _load_package_table(path: str | Path, field: str) -> pd.DataFrame:
    packages = read_jsonl(path)
    if not packages:
        return pd.DataFrame()
    artifact_path = packages[0].get(field)
    if not artifact_path:
        return pd.DataFrame()
    return read_table(artifact_path)


def _event_video_candidates(
    event_comment_map: pd.DataFrame,
    *,
    max_videos_per_event: int,
) -> dict[str, list[dict[str, Any]]]:
    if event_comment_map.empty:
        return {}
    required = {"event_id", "video_id"}
    missing = required.difference(event_comment_map.columns)
    if missing:
        raise KeyError(
            "event_comment_map missing required columns: " + ", ".join(sorted(missing))
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    title_col = "title" if "title" in event_comment_map.columns else None
    grouped = (
        event_comment_map.groupby(["event_id", "video_id"], dropna=False)
        .size()
        .reset_index(name="comment_count")
        .sort_values(["event_id", "comment_count", "video_id"], ascending=[True, False, True])
    )
    if title_col:
        titles = event_comment_map[["event_id", "video_id", title_col]].drop_duplicates(
            subset=["event_id", "video_id"]
        )
        grouped = grouped.merge(titles, on=["event_id", "video_id"], how="left")
    else:
        grouped["title"] = pd.NA
    for event_id, group in grouped.groupby("event_id", sort=False):
        rows[event_id] = group.head(max_videos_per_event).to_dict("records")
    return rows


def build_validation_manifest(
    *,
    validation_run_id: str,
    evidence_packages_path: str | Path,
    output_dir: str | Path,
    validator: str,
    query_language: str,
    validation_params: dict[str, Any],
    notes: str | None,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "validation_run_id": validation_run_id,
        "created_at_utc": created_at_utc,
        "pipeline_stage": "rag_validation_preparation",
        "mode": "contract_only_no_retrieval_no_generation",
        "evidence_packages_path": _normalize_path(evidence_packages_path),
        "output_dir": _normalize_path(output_dir),
        "validator": validator,
        "query_language": query_language,
        "validation_params": validation_params,
        "validation_labels": VALIDATION_LABELS,
        "artifact_version": RAG_VALIDATION_ARTIFACT_VERSION,
        "notes": notes,
    }


def build_validation_tasks(
    packages: list[dict[str, Any]],
    *,
    validation_run_id: str,
    evidence_packages_path: str | Path,
    validator: str,
    created_at_utc: str,
) -> pd.DataFrame:
    rows = []
    for package in packages:
        event_id = package["event_id"]
        rows.append(
            {
                "validation_task_id": f"valtask_{_short_hash(validation_run_id, event_id)}",
                "validation_run_id": validation_run_id,
                "event_id": event_id,
                "run_id": package.get("run_id"),
                "trigger_time_utc": package.get("trigger_time_utc"),
                "window_start_utc": package.get("window_start_utc"),
                "window_end_utc": package.get("window_end_utc"),
                "event_evidence_package_path": _package_path(evidence_packages_path),
                "event_candidate_path": package.get("event_candidate_path"),
                "event_comment_map_path": package.get("event_comment_map_path"),
                "event_signal_snapshot_map_path": package.get(
                    "event_signal_snapshot_map_path"
                ),
                "comment_count": package.get("comment_count", 0),
                "signal_snapshot_count": package.get("signal_snapshot_count", 0),
                "rag_readiness_status": package.get("rag_readiness_status"),
                "validation_status": "needs_external_evidence",
                "validator": validator,
                "created_at_utc": created_at_utc,
                "artifact_version": RAG_VALIDATION_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def build_retrieval_questions(
    validation_tasks: pd.DataFrame,
    *,
    validation_run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, task in validation_tasks.iterrows():
        event_id = task["event_id"]
        for order, (question_type, template) in enumerate(
            RETRIEVAL_QUESTION_TEMPLATES, start=1
        ):
            question_text = template.format(
                trigger_time_utc=task.get("trigger_time_utc")
            )
            rows.append(
                {
                    "question_id": f"rq_{_short_hash(validation_run_id, event_id, question_type)}",
                    "validation_run_id": validation_run_id,
                    "event_id": event_id,
                    "question_order": order,
                    "question_type": question_type,
                    "question_text": question_text,
                    "question_status": "pending_retrieval",
                    "artifact_version": RAG_VALIDATION_ARTIFACT_VERSION,
                }
            )
    return pd.DataFrame(rows)


def build_rag_query_placeholders(
    validation_tasks: pd.DataFrame,
    event_comment_map: pd.DataFrame,
    *,
    validation_run_id: str,
    query_language: str,
    max_videos_per_event: int,
) -> pd.DataFrame:
    video_candidates = _event_video_candidates(
        event_comment_map,
        max_videos_per_event=max_videos_per_event,
    )
    rows: list[dict[str, Any]] = []
    for _, task in validation_tasks.iterrows():
        event_id = task["event_id"]
        videos = video_candidates.get(event_id, []) or [{}]
        for order, video in enumerate(videos, start=1):
            video_id = video.get("video_id")
            rows.append(
                {
                    "query_id": f"query_{_short_hash(validation_run_id, event_id, video_id, order)}",
                    "validation_run_id": validation_run_id,
                    "event_id": event_id,
                    "trigger_time_utc": task.get("trigger_time_utc"),
                    "video_id": video_id,
                    "title": video.get("title"),
                    "news_api_query": "",
                    "query_language": query_language,
                    "query_time_window_start_utc": task.get("window_start_utc"),
                    "query_time_window_end_utc": task.get("window_end_utc"),
                    "query_source": "pending_design",
                    "query_status": "pending_query_design",
                    "artifact_version": RAG_VALIDATION_ARTIFACT_VERSION,
                }
            )
    return pd.DataFrame(rows)


def build_empty_external_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "evidence_id",
            "query_id",
            "event_id",
            "title",
            "link",
            "snippet",
            "date",
            "source",
            "retrieved_at_utc",
            "retrieval_provider",
            "artifact_version",
        ]
    )


def build_pending_validation_results(
    validation_tasks: pd.DataFrame,
    *,
    validation_run_id: str,
    validator: str,
    created_at_utc: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, task in validation_tasks.iterrows():
        event_id = task["event_id"]
        rows.append(
            {
                "validation_id": f"val_{_short_hash(validation_run_id, event_id)}",
                "validation_run_id": validation_run_id,
                "event_id": event_id,
                "validation_label": "",
                "validation_status": "needs_external_evidence",
                "n_external_sources": 0,
                "rationale": "Pending external retrieval and validator decision.",
                "supporting_evidence_ids": "",
                "contradictory_evidence_ids": "",
                "limitations": "No external evidence retrieved by this preparation stage.",
                "validated_at_utc": "",
                "prepared_at_utc": created_at_utc,
                "validator": validator,
                "artifact_version": RAG_VALIDATION_ARTIFACT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def prepare_rag_validation_artifacts(
    *,
    evidence_packages_path: str | Path,
    output_dir: str | Path,
    validation_run_id: str | None = None,
    validator: str = "manual_pending",
    query_language: str = "es",
    max_videos_per_event: int = 5,
    notes: str | None = None,
    validation_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_validation_run_id = validation_run_id or _validation_run_id(
        evidence_packages_path, output_dir
    )
    created_at = _utc_now_iso()
    packages = read_jsonl(evidence_packages_path)
    event_comment_map = _load_package_table(
        evidence_packages_path, "event_comment_map_path"
    )

    manifest = build_validation_manifest(
        validation_run_id=resolved_validation_run_id,
        evidence_packages_path=evidence_packages_path,
        output_dir=output_root,
        validator=validator,
        query_language=query_language,
        validation_params=validation_params or {},
        notes=notes,
        created_at_utc=created_at,
    )
    validation_tasks = build_validation_tasks(
        packages,
        validation_run_id=resolved_validation_run_id,
        evidence_packages_path=evidence_packages_path,
        validator=validator,
        created_at_utc=created_at,
    )
    retrieval_questions = build_retrieval_questions(
        validation_tasks,
        validation_run_id=resolved_validation_run_id,
    )
    rag_queries = build_rag_query_placeholders(
        validation_tasks,
        event_comment_map,
        validation_run_id=resolved_validation_run_id,
        query_language=query_language,
        max_videos_per_event=max_videos_per_event,
    )
    external_evidence = build_empty_external_evidence()
    validation_results = build_pending_validation_results(
        validation_tasks,
        validation_run_id=resolved_validation_run_id,
        validator=validator,
        created_at_utc=created_at,
    )

    output_paths = {
        "manifest": (output_root / VALIDATION_MANIFEST_FILE).as_posix(),
        "validation_tasks": (output_root / VALIDATION_TASKS_FILE).as_posix(),
        "retrieval_questions": (output_root / RETRIEVAL_QUESTIONS_FILE).as_posix(),
        "rag_queries": (output_root / RAG_QUERIES_FILE).as_posix(),
        "external_evidence": (output_root / EXTERNAL_EVIDENCE_FILE).as_posix(),
        "validation_results": (output_root / VALIDATION_RESULTS_FILE).as_posix(),
        "summary": (output_root / VALIDATION_SUMMARY_FILE).as_posix(),
    }

    write_json(output_paths["manifest"], manifest)
    validation_tasks.to_csv(output_paths["validation_tasks"], index=False)
    retrieval_questions.to_csv(output_paths["retrieval_questions"], index=False)
    rag_queries.to_csv(output_paths["rag_queries"], index=False)
    external_evidence.to_csv(output_paths["external_evidence"], index=False)
    validation_results.to_csv(output_paths["validation_results"], index=False)

    summary = {
        "validation_run_id": resolved_validation_run_id,
        "artifact_version": RAG_VALIDATION_ARTIFACT_VERSION,
        "output_paths": output_paths,
        "validation_task_count": int(len(validation_tasks)),
        "retrieval_question_count": int(len(retrieval_questions)),
        "rag_query_placeholder_count": int(len(rag_queries)),
        "external_evidence_count": int(len(external_evidence)),
        "pending_validation_count": int(
            (validation_results["validation_status"] == "needs_external_evidence").sum()
        ),
        "mode": "contract_only_no_retrieval_no_generation",
    }
    write_json(output_paths["summary"], summary)
    return summary


def prepare_rag_validation_artifacts_from_config(
    config: RagValidationPrepareConfig,
) -> dict[str, Any]:
    return prepare_rag_validation_artifacts(
        evidence_packages_path=config.evidence_packages_path,
        output_dir=config.output_dir,
        validation_run_id=config.validation_run_id,
        validator=config.validator,
        query_language=config.query_language,
        max_videos_per_event=config.max_videos_per_event,
        notes=config.notes,
        validation_params=config.validation_params,
    )


__all__ = [
    "RAG_VALIDATION_ARTIFACT_VERSION",
    "RagValidationPrepareConfig",
    "build_empty_external_evidence",
    "build_pending_validation_results",
    "build_rag_query_placeholders",
    "build_retrieval_questions",
    "build_validation_manifest",
    "build_validation_tasks",
    "derive_rag_validation_run_id",
    "load_rag_validation_config",
    "prepare_rag_validation_artifacts",
    "prepare_rag_validation_artifacts_from_config",
    "read_jsonl",
]
