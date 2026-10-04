from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from youtube_pipeline.association_candidates import (
    AssociatedEventCandidate,
    load_associated_event_candidates,
)
from youtube_pipeline.candidate_evidence import (
    CANDIDATE_COMMENT_INVENTORY_FILE,
    CANDIDATE_EVIDENCE_MANIFEST_FILE,
    CANDIDATE_EVIDENCE_PACKAGES_FILE,
    EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
    EVIDENCE_PENDING_HORIZON,
    EVIDENCE_READY,
    NO_EVIDENCE_FOUND,
    RAG_COMPATIBILITY_DIR,
)
from youtube_pipeline.external_publication_time import (
    AFTER_REFERENCE,
    AT_REFERENCE,
    BEFORE_REFERENCE,
    CAUSALLY_AVAILABLE,
    FUTURE_RELATIVE_TO_CUTOFF,
    SAME_DAY_TIME_UNKNOWN,
    external_publication_time_from_record,
    temporal_relation_to_cutoff,
    temporal_relation_to_reference,
)
from youtube_pipeline.g2_gate import (
    G2_EXECUTED,
    G2_FAILED,
    G2_NOT_ELIGIBLE,
    G2_SKIPPED_BY_G1_GATE,
    RUN_G2,
    G2GateDecision,
    G2GatePolicyConfig,
    decide_g2,
)
from youtube_pipeline.rag_consumer import (
    RAG_CONTEXT_PAYLOADS_FILE,
    RAG_CONSUMER_MANIFEST_FILE,
    RAG_VALIDATION_INPUTS_FILE,
    RagConsumerConfig,
    write_rag_consumer_artifacts_from_config,
)
from youtube_pipeline.rag_generation_g1 import (
    PROMPT_VERSION as G1_PROMPT_VERSION,
    RAG_G1_ARTIFACT_VERSION,
    RAG_GENERATION_MANIFEST_FILE as G1_MANIFEST_FILE,
    RAG_RAW_MODEL_RESPONSES_FILE as G1_RAW_RESPONSES_FILE,
    RAG_VALIDATION_REPORTS_FILE as G1_REPORTS_FILE,
    RagG1Config,
    run_rag_g1_validation_from_config,
)
from youtube_pipeline.rag_generation_g2_hierarchical import (
    QUERY_PROMPT_VERSION as G2_QUERY_PROMPT_VERSION,
    RAG_EVENT_VALIDATION_SUMMARY_FILE,
    RAG_GENERATION_MANIFEST_FILE as G2_MANIFEST_FILE,
    RAG_G2H_ARTIFACT_VERSION,
    RAG_RAW_MODEL_RESPONSES_FILE as G2_RAW_RESPONSES_FILE,
    RAG_VIDEO_EXTERNAL_EVIDENCE_FILE,
    RAG_VIDEO_NEWS_QUERIES_FILE,
    RAG_VIDEO_VALIDATION_REPORTS_FILE,
    VALIDATION_PROMPT_VERSION as G2_VALIDATION_PROMPT_VERSION,
    RagG2HierarchicalConfig,
    run_rag_g2_hierarchical_from_config,
)
from youtube_pipeline.validation_status import normalize_validation_status


CANDIDATE_RAG_INTEGRATION_V1 = "candidate_rag_integration_v1"
CANDIDATE_RAG_ARTIFACT_VERSION = "candidate_rag_validation_v2"
EXTERNAL_CAUSAL_POLICY_V1 = "published_strictly_before_evidence_cutoff_v1"

VALIDATED = "VALIDATED"
VALIDATION_ERROR = "VALIDATION_ERROR"

CANDIDATE_G1_RESULTS_FILE = "candidate_g1_results.jsonl"
CANDIDATE_G2_VIDEO_RESULTS_FILE = "candidate_g2_video_results.jsonl"
CANDIDATE_G2_RESULTS_FILE = "candidate_g2_results.jsonl"
CANDIDATE_G2_EXTERNAL_EVIDENCE_FILE = "candidate_g2_external_evidence.jsonl"
CANDIDATE_G2_GATE_DECISION_FILE = "candidate_g2_gate_decision.json"
CANDIDATE_VALIDATION_RESULTS_FILE = "candidate_validation_results.jsonl"
CANDIDATE_RAG_MANIFEST_FILE = "candidate_rag_validation_manifest.json"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _utc(value: object, *, name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid ISO-8601 datetime.") from exc
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime or ISO-8601 string.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _short_hash(*parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON objects in {target}.")
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class CandidateRagValidationConfig:
    candidates_path: str
    evidence_dir: str
    output_dir: str
    candidate_id: str
    integration_policy_id: str = CANDIDATE_RAG_INTEGRATION_V1
    g1_model: str = "gpt-5-mini"
    g2_query_model: str = "gpt-5-mini"
    g2_validation_model: str = "gpt-5-mini"
    provider: str = "openai"
    temperature: float = 0.0
    g1_max_approx_tokens: int = 16_000
    max_retries: int = 1
    request_timeout_seconds: int = 120
    serper_timeout_seconds: int = 60
    serper_num_results: int = 5
    search_days_before: int = 1
    search_days_after: int = 1
    max_videos_per_event_batch: int = 5
    external_causal_policy_id: str = EXTERNAL_CAUSAL_POLICY_V1
    notes: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidates_path",
            "evidence_dir",
            "output_dir",
            "candidate_id",
            "integration_policy_id",
            "g1_model",
            "g2_query_model",
            "g2_validation_model",
            "provider",
            "external_causal_policy_id",
        ):
            _require_text(name, getattr(self, name))
        if self.integration_policy_id != CANDIDATE_RAG_INTEGRATION_V1:
            raise ValueError("Unsupported candidate RAG integration policy.")
        if self.external_causal_policy_id != EXTERNAL_CAUSAL_POLICY_V1:
            raise ValueError("Unsupported external evidence causal policy.")
        if self.provider != "openai":
            raise ValueError("The reused G-1/G-2 implementation requires openai.")
        if self.g1_model != "gpt-5-mini":
            raise ValueError("The existing G-1 contract fixes model gpt-5-mini.")
        if self.g2_query_model != "gpt-5-mini" or self.g2_validation_model != "gpt-5-mini":
            raise ValueError("The existing hierarchical G-2 contract fixes gpt-5-mini.")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be numeric.")
        for name in (
            "g1_max_approx_tokens",
            "request_timeout_seconds",
            "serper_timeout_seconds",
            "serper_num_results",
            "max_videos_per_event_batch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("max_retries must be an integer.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        for name in ("search_days_before", "search_days_after"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        if not isinstance(self.params, dict):
            raise TypeError("params must be an object.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CandidateRagValidationConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("candidate RAG config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown candidate RAG fields: {sorted(unknown)}")
        missing = {
            "candidates_path",
            "evidence_dir",
            "output_dir",
            "candidate_id",
        } - set(payload)
        if missing:
            raise ValueError(f"Missing candidate RAG fields: {sorted(missing)}")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class CandidateValidationResult:
    validation_id: str
    candidate_id: str
    candidate_type: str
    candidate_run_id: str
    candidate_config_hash: str
    evidence_status: str
    status: str
    evidence_policy_id: str
    evidence_cutoff_utc: str
    g1_result: dict[str, Any] | None
    g2_result: dict[str, Any] | None
    g2_execution_status: str
    g2_gate_decision: dict[str, Any] | None
    final_synthesis: dict[str, Any] | None
    inventory_comment_count: int
    selected_comment_count: int
    selected_fraction: float
    internal_future_leak_count: int
    external_future_leak_count: int
    limitations: tuple[str, ...]
    created_at_utc: str
    artifact_version: str = CANDIDATE_RAG_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "validation_id",
            "candidate_id",
            "candidate_type",
            "candidate_run_id",
            "candidate_config_hash",
            "evidence_status",
            "status",
            "evidence_policy_id",
            "evidence_cutoff_utc",
            "created_at_utc",
            "g2_execution_status",
        ):
            _require_text(name, getattr(self, name))
        _utc(self.evidence_cutoff_utc, name="evidence_cutoff_utc")
        _utc(self.created_at_utc, name="created_at_utc")
        if self.status not in {
            VALIDATED,
            VALIDATION_ERROR,
            NO_EVIDENCE_FOUND,
            EVIDENCE_PENDING_HORIZON,
            EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE,
        }:
            raise ValueError("Unsupported candidate validation status.")
        if self.g2_execution_status not in {
            G2_EXECUTED,
            G2_SKIPPED_BY_G1_GATE,
            G2_NOT_ELIGIBLE,
            G2_FAILED,
        }:
            raise ValueError("Unsupported G-2 execution status.")
        if (
            self.g2_execution_status == G2_SKIPPED_BY_G1_GATE
            and self.g2_gate_decision is None
        ):
            raise ValueError("A gate skip requires its decision lineage.")
        if self.inventory_comment_count < 0 or self.selected_comment_count < 0:
            raise ValueError("Comment counts must be non-negative.")
        if not 0.0 <= self.selected_fraction <= 1.0:
            raise ValueError("selected_fraction must be in [0, 1].")
        if self.internal_future_leak_count or self.external_future_leak_count:
            raise ValueError("Candidate validation results cannot contain future leaks.")
        if any(not isinstance(item, str) or not item.strip() for item in self.limitations):
            raise ValueError("limitations must contain non-empty strings.")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateRagArtifacts:
    output_dir: Path
    validation_results_path: Path
    manifest_path: Path
    candidate_id: str
    status: str


def _candidate_and_package(
    config: CandidateRagValidationConfig,
) -> tuple[AssociatedEventCandidate, dict[str, Any], dict[str, Any]]:
    candidates = load_associated_event_candidates(config.candidates_path)
    matches = [item for item in candidates if item.candidate_id == config.candidate_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one candidate {config.candidate_id!r}, got {len(matches)}."
        )
    evidence_root = Path(config.evidence_dir)
    packages = _read_jsonl(evidence_root / CANDIDATE_EVIDENCE_PACKAGES_FILE)
    package_matches = [
        item for item in packages if item.get("candidate_id") == config.candidate_id
    ]
    if len(package_matches) != 1:
        raise ValueError(
            f"Expected exactly one evidence package for {config.candidate_id!r}, "
            f"got {len(package_matches)}."
        )
    manifest = _read_json(evidence_root / CANDIDATE_EVIDENCE_MANIFEST_FILE)
    candidate = matches[0]
    package = package_matches[0]
    if package.get("candidate_type") != candidate.candidate_type:
        raise ValueError("Candidate type differs between candidate and evidence package.")
    if package.get("run_id") != candidate.run_id:
        raise ValueError("Candidate run lineage differs from evidence package.")
    source = package.get("source") or {}
    if source.get("input_policy_id") != candidate.input_policy_id:
        raise ValueError("Candidate input policy differs from evidence source policy.")
    return candidate, package, manifest


def _selected_context_metrics(
    *, evidence_dir: Path, consumer_dir: Path, candidate_id: str
) -> tuple[int, int, float, int, dict[str, int]]:
    inventory = pd.read_csv(evidence_dir / CANDIDATE_COMMENT_INVENTORY_FILE)
    candidate_inventory = inventory.loc[
        inventory["candidate_id"].astype(str) == candidate_id
    ].copy()
    inventory_ids = set(candidate_inventory["comment_id"].astype(str))
    payloads = _read_jsonl(consumer_dir / RAG_CONTEXT_PAYLOADS_FILE)
    matches = [item for item in payloads if item.get("event_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one consumer context payload for candidate.")
    context_payload = matches[0]
    used_comments = context_payload.get("used_context_comments", [])
    selected_ids = {
        str(item.get("comment_id"))
        for item in used_comments
        if item.get("comment_id") is not None
    }
    if not selected_ids.issubset(inventory_ids):
        raise ValueError("RAG context contains comments outside candidate inventory.")
    package_records = _read_jsonl(evidence_dir / CANDIDATE_EVIDENCE_PACKAGES_FILE)
    package = next(item for item in package_records if item["candidate_id"] == candidate_id)
    cutoff = _utc(
        package["evidence_scope"]["evidence_cutoff_utc"],
        name="evidence_cutoff_utc",
    )
    future = 0
    by_id = {
        str(row.comment_id): _utc(row.event_time_utc, name="event_time_utc")
        for row in candidate_inventory.itertuples()
    }
    for comment_id in selected_ids:
        future += by_id[comment_id] >= cutoff
    inventory_count = len(inventory_ids)
    selected_count = len(selected_ids)
    fraction = selected_count / inventory_count if inventory_count else 0.0
    selected_units = context_payload.get("selected_context_units", [])
    selected_videos = {
        str(item.get("video_id"))
        for item in used_comments
        if item.get("video_id") is not None
    }
    selected_threads = {
        str(
            item.get("thread_id")
            or item.get("root_comment_id")
            or item.get("context_unit_id")
        )
        for item in selected_units
        if (
            item.get("thread_id")
            or item.get("root_comment_id")
            or item.get("context_unit_id")
        ) is not None
    }
    coverage = {
        "inventory_comments": inventory_count,
        "selected_comments": selected_count,
        "inventory_context_units": int(package.get("context_unit_count", 0)),
        "selected_context_units": len(selected_units),
        "inventory_videos": int(package.get("video_count", 0)),
        "selected_videos": len(selected_videos),
        "inventory_threads": int(package.get("thread_count", 0)),
        "selected_threads": len(selected_threads),
    }
    return inventory_count, selected_count, fraction, future, coverage


def _candidate_alias(records: list[dict[str, Any]], candidate_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("event_id") != candidate_id:
            continue
        converted = dict(record)
        converted["compatibility_event_id"] = converted.pop("event_id")
        converted["candidate_id"] = candidate_id
        output.append(converted)
    return output


def _token_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = output_tokens = total_tokens = 0
    complete = True
    response_count = 0
    for record in records:
        body = record.get("raw_response")
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            complete = False
            continue
        response_count += 1
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        total_tokens += int(
            usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
            or 0
        )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_complete": complete,
        "responses_with_usage": response_count,
    }


def _external_future_leaks(records: list[dict[str, Any]], cutoff: datetime) -> int:
    return sum(
        temporal_relation_to_cutoff(
            external_publication_time_from_record(record), cutoff
        )
        != CAUSALLY_AVAILABLE
        for record in records
    )


def _external_timing_records(
    records: list[dict[str, Any]],
    *,
    candidate: AssociatedEventCandidate,
    evidence_cutoff: datetime,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reference = candidate.reference_time_utc
    for record in records:
        enriched = dict(record)
        raw = record.get("published_at")
        publication_time = external_publication_time_from_record(record)
        reference_relation = temporal_relation_to_reference(
            publication_time, reference
        )
        cutoff_relation = temporal_relation_to_cutoff(
            publication_time, evidence_cutoff
        )
        enriched.update({
            **publication_time.to_mapping(),
            "candidate_reference_time_utc": reference.isoformat(),
            "candidate_created_at_cutoff_utc": (
                candidate.created_at_cutoff_utc.isoformat()
            ),
            "evidence_cutoff_utc": evidence_cutoff.isoformat(),
            "external_source_published_at": raw,
            "external_lag_seconds": None,
            "external_temporal_relation": "UNKNOWN",
            "temporal_relation_to_cutoff": cutoff_relation,
        })
        if publication_time.timestamp_utc is not None:
            published_at = publication_time.timestamp_utc
            enriched["external_lag_seconds"] = (
                published_at - reference
            ).total_seconds()
        if reference_relation == BEFORE_REFERENCE:
            enriched["external_temporal_relation"] = "PRE_CANDIDATE"
        elif cutoff_relation == FUTURE_RELATIVE_TO_CUTOFF:
            enriched["external_temporal_relation"] = "FUTURE_RELATIVE_TO_CUTOFF"
        elif reference_relation in {AT_REFERENCE, AFTER_REFERENCE}:
            enriched["external_temporal_relation"] = (
                "POST_CANDIDATE_CAUSALLY_AVAILABLE"
            )
        elif reference_relation == SAME_DAY_TIME_UNKNOWN:
            enriched["external_temporal_relation"] = "UNKNOWN_SAME_DAY_DATE_ONLY"
        output.append(enriched)
    return output


def _g1_projection(g1_dir: Path, candidate_id: str) -> list[dict[str, Any]]:
    return _candidate_alias(_read_jsonl(g1_dir / G1_REPORTS_FILE), candidate_id)


def _g2_projections(
    g2_dir: Path, candidate_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    video = _candidate_alias(
        _read_jsonl(g2_dir / RAG_VIDEO_VALIDATION_REPORTS_FILE), candidate_id
    )
    summary = _candidate_alias(
        _read_jsonl(g2_dir / RAG_EVENT_VALIDATION_SUMMARY_FILE), candidate_id
    )
    evidence = _candidate_alias(
        _read_jsonl(g2_dir / RAG_VIDEO_EXTERNAL_EVIDENCE_FILE), candidate_id
    )
    return video, summary, evidence


def _final_synthesis(
    g1: dict[str, Any] | None,
    g2: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if g1 is None and g2 is None:
        return None
    internal_status = normalize_validation_status(
        g1.get("validation_status") if g1 is not None else None,
        allow_none=True,
    )
    external_status = normalize_validation_status(
        g2.get("overall_validation_status") if g2 is not None else None,
        allow_none=True,
    )
    return {
        "policy": "separate_internal_validation_and_external_corroboration_v2",
        "internal_validation_status": (
            internal_status
        ),
        "external_validation_status": (
            external_status
        ),
        "event_interpretation": (
            g2.get("overall_interpretation")
            if g2 is not None
            else g1.get("event_interpretation") if g1 is not None else None
        ),
        "g1_and_g2_remain_distinct": True,
        "no_score_fusion": True,
        "external_evidence_absence_is_not_candidate_rejection": True,
    }


def run_candidate_rag_validation(
    config: CandidateRagValidationConfig,
    *,
    execute: bool = False,
    g2_gate: G2GatePolicyConfig | None = None,
    g1_runner: Callable[[RagG1Config], dict[str, Any]] = run_rag_g1_validation_from_config,
    g2_runner: Callable[[RagG2HierarchicalConfig], dict[str, Any]] = run_rag_g2_hierarchical_from_config,
) -> CandidateRagArtifacts:
    """Run the legacy RAG stages through an explicit candidate compatibility view.

    Network-capable execution is guarded by ``execute``. Tests inject deterministic
    runners while exercising the same persisted-artifact boundary.
    """

    if not isinstance(config, CandidateRagValidationConfig):
        raise TypeError("config must be CandidateRagValidationConfig.")
    gate_config = g2_gate if g2_gate is not None else G2GatePolicyConfig()
    if not isinstance(gate_config, G2GatePolicyConfig):
        raise TypeError("g2_gate must be G2GatePolicyConfig or None.")
    if not execute:
        raise RuntimeError("Candidate RAG execution requires explicit execute=True.")
    candidate, package, evidence_manifest = _candidate_and_package(config)
    root = Path(config.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Candidate RAG output directory must be empty or absent.")
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / CANDIDATE_VALIDATION_RESULTS_FILE
    manifest_path = root / CANDIDATE_RAG_MANIFEST_FILE
    created_at = datetime.now(timezone.utc).isoformat()
    evidence_status = str(package["evidence_status"])
    cutoff = _utc(
        package["evidence_scope"]["evidence_cutoff_utc"],
        name="evidence_cutoff_utc",
    )
    validation_id = "candval_" + _short_hash(
        candidate.candidate_id,
        evidence_manifest.get("artifact_hashes", {}).get(
            "candidate_evidence_packages_sha256"
        ),
        config.integration_policy_id,
        G1_PROMPT_VERSION,
        G2_QUERY_PROMPT_VERSION,
        G2_VALIDATION_PROMPT_VERSION,
    )

    output_paths = {
        "candidate_g1_results": root / CANDIDATE_G1_RESULTS_FILE,
        "candidate_g2_video_results": root / CANDIDATE_G2_VIDEO_RESULTS_FILE,
        "candidate_g2_results": root / CANDIDATE_G2_RESULTS_FILE,
        "candidate_g2_external_evidence": (
            root / CANDIDATE_G2_EXTERNAL_EVIDENCE_FILE
        ),
        "candidate_g2_gate_decision": root / CANDIDATE_G2_GATE_DECISION_FILE,
        "candidate_validation_results": result_path,
        "candidate_rag_validation_manifest": manifest_path,
    }
    consumer_dir = root / "rag_consumer"
    g1_dir = root / "g1"
    g2_dir = root / "g2_hierarchical"
    g1_records: list[dict[str, Any]] = []
    g2_video_records: list[dict[str, Any]] = []
    g2_records: list[dict[str, Any]] = []
    external_records: list[dict[str, Any]] = []
    g1_summary: dict[str, Any] | None = None
    g2_summary: dict[str, Any] | None = None
    inventory_count = int(package.get("comment_count", 0))
    selected_count = 0
    selected_fraction = 0.0
    internal_future_leak_count = 0
    external_future_leak_count = 0
    context_coverage = {
        "inventory_comments": inventory_count,
        "selected_comments": 0,
        "inventory_context_units": int(package.get("context_unit_count", 0)),
        "selected_context_units": 0,
        "inventory_videos": int(package.get("video_count", 0)),
        "selected_videos": 0,
        "inventory_threads": int(package.get("thread_count", 0)),
        "selected_threads": 0,
    }
    gate_decision: G2GateDecision | None = None
    g2_execution_status = G2_NOT_ELIGIBLE
    stage_summaries: dict[str, Any] = {}
    execution_errors: list[str] = []
    limitations = [
        "Historical web retrieval cannot prove index availability at the original cutoff."
    ]
    video_metadata_audit = {
        "videos": 0,
        "missing_title": 0,
        "missing_channel_title": 0,
    }

    if evidence_status == EVIDENCE_READY:
        compatibility_dir = Path(config.evidence_dir) / RAG_COMPATIBILITY_DIR
        consumer_summary = write_rag_consumer_artifacts_from_config(
            RagConsumerConfig(
                sidecars_dir=str(compatibility_dir),
                output_dir=str(consumer_dir),
                run_id="candidate_consumer_" + _short_hash(candidate.candidate_id),
                notes="Explicit candidate_id -> event_id compatibility adapter.",
                params={"candidate_id": candidate.candidate_id},
            )
        )
        stage_summaries["consumer"] = consumer_summary
        validation_inputs = _read_jsonl(consumer_dir / RAG_VALIDATION_INPUTS_FILE)
        candidate_inputs = [
            item for item in validation_inputs if item.get("event_id") == candidate.candidate_id
        ]
        if len(candidate_inputs) != 1:
            raise ValueError("Expected exactly one RAG validation input for candidate.")
        associated_videos = candidate_inputs[0].get("associated_videos", [])
        video_metadata_audit = {
            "videos": len(associated_videos),
            "missing_title": sum(not item.get("title") for item in associated_videos),
            "missing_channel_title": sum(
                not item.get("channel_title") for item in associated_videos
            ),
        }
        if video_metadata_audit["missing_title"] or video_metadata_audit["missing_channel_title"]:
            limitations.append(
                "Some associated videos lack title/channel metadata; G-2 remains scoped "
                "to their IDs and internal comments."
            )
        (
            inventory_count,
            selected_count,
            selected_fraction,
            internal_future_leak_count,
            context_coverage,
        ) = _selected_context_metrics(
            evidence_dir=Path(config.evidence_dir),
            consumer_dir=consumer_dir,
            candidate_id=candidate.candidate_id,
        )
        if internal_future_leak_count:
            raise AssertionError("Future internal evidence reached RAG context.")

        g1_config = RagG1Config(
            consumer_dir=str(consumer_dir),
            output_dir=str(g1_dir),
            event_id=candidate.candidate_id,
            model=config.g1_model,
            provider=config.provider,
            temperature=config.temperature,
            max_approx_tokens=config.g1_max_approx_tokens,
            max_retries=config.max_retries,
            request_timeout_seconds=config.request_timeout_seconds,
            notes=config.notes,
            params={**config.params, "candidate_id": candidate.candidate_id},
        )
        stage_summaries["g1"] = g1_runner(g1_config)
        g1_records = _g1_projection(g1_dir, candidate.candidate_id)
        if len(g1_records) != 1:
            execution_errors.append("expected_exactly_one_g1_result")
        g1_summary = g1_records[0] if g1_records else None
        gate_decision = decide_g2(
            candidate_id=candidate.candidate_id,
            config=gate_config,
            g1_result=g1_summary,
        )
        if gate_decision.decision == RUN_G2:
            g2_config = RagG2HierarchicalConfig(
                consumer_dir=str(consumer_dir),
                output_dir=str(g2_dir),
                event_id=candidate.candidate_id,
                query_model=config.g2_query_model,
                validation_model=config.g2_validation_model,
                provider=config.provider,
                temperature=config.temperature,
                serper_num_results=config.serper_num_results,
                search_days_before=config.search_days_before,
                search_days_after=config.search_days_after,
                max_videos_per_event_batch=config.max_videos_per_event_batch,
                max_retries=config.max_retries,
                request_timeout_seconds=config.request_timeout_seconds,
                serper_timeout_seconds=config.serper_timeout_seconds,
                notes=config.notes,
                params={**config.params, "candidate_id": candidate.candidate_id},
                external_evidence_cutoff_utc=cutoff.isoformat(),
            )
            try:
                stage_summaries["g2_hierarchical"] = g2_runner(g2_config)
                g2_video_records, g2_records, external_records = _g2_projections(
                    g2_dir, candidate.candidate_id
                )
                external_future_leak_count = _external_future_leaks(
                    external_records, cutoff
                )
                if external_future_leak_count:
                    raise AssertionError(
                        "Future or temporally unknown external evidence reached G-2."
                    )
                if len(g2_records) != 1:
                    execution_errors.append(
                        "expected_exactly_one_g2_candidate_summary"
                    )
                g2_summary = g2_records[0] if g2_records else None
                g2_execution_status = G2_EXECUTED
            except AssertionError:
                raise
            except Exception as exc:
                g2_execution_status = G2_FAILED
                execution_errors.append(
                    f"g2_failed:{type(exc).__name__}:{exc}"
                )
        else:
            g2_execution_status = G2_SKIPPED_BY_G1_GATE
        status = VALIDATION_ERROR if execution_errors else VALIDATED
    else:
        status = evidence_status

    final_synthesis = _final_synthesis(g1_summary, g2_summary)
    result = CandidateValidationResult(
        validation_id=validation_id,
        candidate_id=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
        candidate_run_id=candidate.run_id,
        candidate_config_hash=candidate.config_hash,
        evidence_status=evidence_status,
        status=status,
        evidence_policy_id=str(package["evidence_policy"]["policy_id"]),
        evidence_cutoff_utc=cutoff.isoformat(),
        g1_result=g1_summary,
        g2_result=g2_summary,
        g2_execution_status=g2_execution_status,
        g2_gate_decision=(
            gate_decision.to_mapping() if gate_decision is not None else None
        ),
        final_synthesis=final_synthesis,
        inventory_comment_count=inventory_count,
        selected_comment_count=selected_count,
        selected_fraction=selected_fraction,
        internal_future_leak_count=internal_future_leak_count,
        external_future_leak_count=external_future_leak_count,
        limitations=tuple(limitations),
        created_at_utc=created_at,
    )
    _write_jsonl(output_paths["candidate_g1_results"], g1_records)
    _write_jsonl(output_paths["candidate_g2_video_results"], g2_video_records)
    _write_jsonl(output_paths["candidate_g2_results"], g2_records)
    external_timing_records = _external_timing_records(
        external_records,
        candidate=candidate,
        evidence_cutoff=cutoff,
    )
    _write_jsonl(
        output_paths["candidate_g2_external_evidence"],
        external_timing_records,
    )
    _write_json(
        output_paths["candidate_g2_gate_decision"],
        (
            gate_decision.to_mapping()
            if gate_decision is not None
            else {
                "candidate_id": candidate.candidate_id,
                "policy_id": gate_config.policy_id,
                "mode": gate_config.mode,
                "decision": None,
                "reason_code": "G2_NOT_ELIGIBLE",
            }
        ),
    )
    _write_jsonl(result_path, [result.to_mapping()])

    g1_raw = _read_jsonl(g1_dir / G1_RAW_RESPONSES_FILE) if g1_dir.exists() else []
    g2_raw = _read_jsonl(g2_dir / G2_RAW_RESPONSES_FILE) if g2_dir.exists() else []
    token_usage = _token_usage(g1_raw + g2_raw)
    g1_token_usage = _token_usage(g1_raw)
    g2_token_usage = _token_usage(g2_raw)
    g1_manifest = _read_json(g1_dir / G1_MANIFEST_FILE) if (g1_dir / G1_MANIFEST_FILE).exists() else None
    g2_manifest = _read_json(g2_dir / G2_MANIFEST_FILE) if (g2_dir / G2_MANIFEST_FILE).exists() else None
    candidate_manifest = {
        "run_id": candidate.run_id,
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "validation_id": validation_id,
        "status": status,
        "created_at_utc": created_at,
        "pipeline_stage": "candidate_evidence_rag_integration",
        "artifact_version": CANDIDATE_RAG_ARTIFACT_VERSION,
        "integration_policy_id": config.integration_policy_id,
        "candidate_lineage": {
            "candidate_config_hash": candidate.config_hash,
            "association_id": candidate.association_id,
            "association_policy_id": candidate.association_policy_id,
            "association_tolerance": candidate.association_tolerance,
            "input_policy_id": candidate.input_policy_id,
        },
        "evidence": {
            "status": evidence_status,
            "policy": package.get("evidence_policy"),
            "cutoff_utc": cutoff.isoformat(),
            "source": package.get("source"),
            "source_manifest_sha256": _sha256(
                Path(config.evidence_dir) / CANDIDATE_EVIDENCE_MANIFEST_FILE
            ),
            "inventory_comment_count": inventory_count,
            "selected_comment_count": selected_count,
            "selected_fraction": selected_fraction,
            "context_coverage": context_coverage,
        },
        "compatibility": {
            "canonical_identity": "candidate_id",
            "transport_alias": "event_id := candidate_id",
            "legacy_rag_modified": False,
        },
        "g1": {
            "executed": evidence_status == EVIDENCE_READY,
            "model": config.g1_model,
            "prompt_version": G1_PROMPT_VERSION,
            "artifact_version": RAG_G1_ARTIFACT_VERSION,
            "manifest": g1_manifest,
            "token_usage": g1_token_usage,
            "estimated_context_tokens": (
                (g1_manifest or {}).get("context_policy", {}).get("approx_tokens")
            ),
            "context_coverage": context_coverage,
        },
        "g2": {
            "execution_status": g2_execution_status,
            "executed": g2_execution_status == G2_EXECUTED,
            "gate": (
                gate_decision.to_mapping()
                if gate_decision is not None
                else None
            ),
            "provider": "serper_news",
            "query_model": config.g2_query_model,
            "validation_model": config.g2_validation_model,
            "query_prompt_version": G2_QUERY_PROMPT_VERSION,
            "validation_prompt_version": G2_VALIDATION_PROMPT_VERSION,
            "artifact_version": RAG_G2H_ARTIFACT_VERSION,
            "external_causal_policy_id": config.external_causal_policy_id,
            "external_evidence_cutoff_utc": cutoff.isoformat(),
            "manifest": g2_manifest,
            "video_metadata_audit": video_metadata_audit,
            "token_usage": g2_token_usage,
        },
        "causality": {
            "internal_future_leak_count": internal_future_leak_count,
            "external_future_leak_count": external_future_leak_count,
            "future_leak_count": internal_future_leak_count + external_future_leak_count,
            "candidate_time_utc": candidate.reference_time_utc.isoformat(),
            "candidate_created_at_cutoff_utc": (
                candidate.created_at_cutoff_utc.isoformat()
            ),
            "evidence_cutoff_utc": cutoff.isoformat(),
            "evidence_ready_time_utc": package["evidence_scope"]["evidence_ready_at_utc"],
            "rag_execution_time_utc": created_at,
        },
        "stage_summaries": stage_summaries,
        "counts": {
            "candidate_count": 1,
            "g1_results": len(g1_records),
            "g1_executions": int(bool(g1_records)),
            "g2_video_results": len(g2_video_records),
            "g2_candidate_results": len(g2_records),
            "external_evidence_records": len(external_records),
            "g2_eligible": int(gate_decision is not None),
            "g2_executed": int(g2_execution_status == G2_EXECUTED),
            "g2_skipped_by_gate": int(
                g2_execution_status == G2_SKIPPED_BY_G1_GATE
            ),
            "g2_failed": int(g2_execution_status == G2_FAILED),
        },
        "token_usage": {
            **token_usage,
            "g1": g1_token_usage,
            "g2": g2_token_usage,
        },
        "external_calls": {
            "provider": "serper_news",
            "records_returned_after_causal_filter": len(external_records),
            "serper_request_count": (g2_manifest or {}).get(
                "serper_request_count", 0
            ),
            "query_call_count": (g2_manifest or {}).get("query_call_count", 0),
            "validation_call_count": (g2_manifest or {}).get("validation_call_count", 0),
            "g2_model_call_count": int(
                (g2_manifest or {}).get("query_call_count", 0) or 0
            ) + int(
                (g2_manifest or {}).get("validation_call_count", 0) or 0
            ),
        },
        "output_paths": {
            key: path.as_posix() for key, path in output_paths.items()
        },
        "errors": execution_errors,
        "limitations": limitations,
        "scientific_validation_performed": False,
        "calibration_performed": False,
    }
    _write_json(manifest_path, candidate_manifest)
    return CandidateRagArtifacts(
        output_dir=root,
        validation_results_path=result_path,
        manifest_path=manifest_path,
        candidate_id=candidate.candidate_id,
        status=status,
    )


__all__ = [
    "CANDIDATE_G1_RESULTS_FILE",
    "CANDIDATE_G2_RESULTS_FILE",
    "CANDIDATE_G2_EXTERNAL_EVIDENCE_FILE",
    "CANDIDATE_G2_GATE_DECISION_FILE",
    "CANDIDATE_G2_VIDEO_RESULTS_FILE",
    "CANDIDATE_RAG_ARTIFACT_VERSION",
    "CANDIDATE_RAG_INTEGRATION_V1",
    "CANDIDATE_RAG_MANIFEST_FILE",
    "CANDIDATE_VALIDATION_RESULTS_FILE",
    "CandidateRagArtifacts",
    "CandidateRagValidationConfig",
    "CandidateValidationResult",
    "EXTERNAL_CAUSAL_POLICY_V1",
    "VALIDATED",
    "VALIDATION_ERROR",
    "run_candidate_rag_validation",
]
