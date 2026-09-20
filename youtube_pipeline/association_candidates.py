"""Neutral projection from finalized detection associations to event candidates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from youtube_pipeline.detection_alerts import ACTIVITY, POLARIZATION
from youtube_pipeline.detection_association import (
    ASSOCIATION_ARTIFACT,
    ASSOCIATION_TYPES,
    END_OF_STREAM,
    FINALIZED,
    TOLERANCE_ELAPSED,
    DetectionAssociation,
)


ASSOCIATION_TO_CANDIDATE_V1 = "association_to_candidate_v1"
READY_FOR_EVIDENCE = "READY_FOR_EVIDENCE"
EVENT_CANDIDATE_ARTIFACT = "event_candidates.jsonl"
EVENT_CANDIDATE_MANIFEST = "event_candidate_manifest.json"
EVENT_CANDIDATE_ARTIFACT_VERSION = "neutral_event_candidate_v1"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def _require_utc(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC.")
    return value


def _normalized_ids(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of strings, not a string.")
    result = tuple(values)
    for value in result:
        _require_text(f"{name} item", value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(sorted(result))


def _candidate_id(association: DetectionAssociation) -> str:
    identity = json.dumps(
        {
            "run_id": association.run_id,
            "association_id": association.association_id,
            "candidate_type": association.association_type,
            "projection_policy_id": ASSOCIATION_TO_CANDIDATE_V1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "candidate_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class AssociatedEventCandidate:
    """A traceable event hypothesis projected from one finalized association.

    This contract intentionally carries no comments, evidence window, validation
    label, combined detector score, confidence, severity, or ranking.
    """

    candidate_id: str
    run_id: str
    config_hash: str
    association_id: str
    association_ref: str
    projection_policy_id: str
    association_policy_id: str
    association_tolerance: str
    association_parameter_status: str
    candidate_type: str
    activity_alert_ids: tuple[str, ...]
    polarization_alert_ids: tuple[str, ...]
    start_time_utc: datetime
    end_time_utc: datetime
    reference_time_utc: datetime
    association_finalized_at_cutoff_utc: datetime
    created_at_cutoff_utc: datetime
    association_finalization_reason: str
    status: str
    input_policy_id: str | None = None
    associated_video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "run_id",
            "config_hash",
            "association_id",
            "association_ref",
            "projection_policy_id",
            "association_policy_id",
            "association_tolerance",
            "association_parameter_status",
        ):
            _require_text(name, getattr(self, name))
        if self.projection_policy_id != ASSOCIATION_TO_CANDIDATE_V1:
            raise ValueError(
                f"projection_policy_id must be {ASSOCIATION_TO_CANDIDATE_V1!r}."
            )
        if self.association_parameter_status not in {
            "TEST_ONLY",
            "DEVELOPMENT_NOT_CALIBRATED",
        }:
            raise ValueError("Unsupported association_parameter_status.")
        if self.candidate_type not in ASSOCIATION_TYPES:
            raise ValueError(
                f"candidate_type must be one of {sorted(ASSOCIATION_TYPES)!r}."
            )
        activity_ids = _normalized_ids(
            "activity_alert_ids", self.activity_alert_ids
        )
        polarization_ids = _normalized_ids(
            "polarization_alert_ids", self.polarization_alert_ids
        )
        if set(activity_ids).intersection(polarization_ids):
            raise ValueError("An alert cannot belong to both signal families.")
        expected_type = (
            "ACTIVITY_POLARIZATION"
            if activity_ids and polarization_ids
            else (
                ACTIVITY
                if activity_ids
                else POLARIZATION if polarization_ids else None
            )
        )
        if self.candidate_type != expected_type:
            raise ValueError("candidate_type does not match its alert families.")
        object.__setattr__(self, "activity_alert_ids", activity_ids)
        object.__setattr__(self, "polarization_alert_ids", polarization_ids)
        object.__setattr__(
            self,
            "associated_video_ids",
            _normalized_ids("associated_video_ids", self.associated_video_ids),
        )
        for name in (
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "association_finalized_at_cutoff_utc",
            "created_at_cutoff_utc",
        ):
            _require_utc(name, getattr(self, name))
        if self.start_time_utc > self.end_time_utc:
            raise ValueError("start_time_utc must not follow end_time_utc.")
        if self.reference_time_utc != self.end_time_utc:
            raise ValueError("reference_time_utc must equal end_time_utc.")
        if self.association_finalized_at_cutoff_utc <= self.end_time_utc:
            raise ValueError("Association finalization must follow its alert interval.")
        if self.created_at_cutoff_utc < self.association_finalized_at_cutoff_utc:
            raise ValueError(
                "created_at_cutoff_utc must not precede association finalization."
            )
        if self.association_finalization_reason not in {
            TOLERANCE_ELAPSED,
            END_OF_STREAM,
        }:
            raise ValueError("Unsupported association_finalization_reason.")
        if self.status != READY_FOR_EVIDENCE:
            raise ValueError(f"status must be {READY_FOR_EVIDENCE!r}.")
        if self.input_policy_id is not None:
            _require_text("input_policy_id", self.input_policy_id)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "association_finalized_at_cutoff_utc",
            "created_at_cutoff_utc",
        ):
            payload[name] = getattr(self, name).isoformat()
        for name in (
            "activity_alert_ids",
            "polarization_alert_ids",
            "associated_video_ids",
        ):
            payload[name] = list(getattr(self, name))
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AssociatedEventCandidate:
        if not isinstance(payload, Mapping):
            raise TypeError("AssociatedEventCandidate payload must be a mapping.")
        expected = {
            "candidate_id",
            "run_id",
            "config_hash",
            "association_id",
            "association_ref",
            "projection_policy_id",
            "association_policy_id",
            "association_tolerance",
            "association_parameter_status",
            "candidate_type",
            "activity_alert_ids",
            "polarization_alert_ids",
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "association_finalized_at_cutoff_utc",
            "created_at_cutoff_utc",
            "association_finalization_reason",
            "status",
            "input_policy_id",
            "associated_video_ids",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise ValueError(
                f"Unknown AssociatedEventCandidate fields: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(
                f"Missing AssociatedEventCandidate fields: {sorted(missing)}"
            )
        parsed = dict(payload)
        for name in (
            "start_time_utc",
            "end_time_utc",
            "reference_time_utc",
            "association_finalized_at_cutoff_utc",
            "created_at_cutoff_utc",
        ):
            value = parsed[name]
            if not isinstance(value, str):
                raise TypeError(f"{name} must be an ISO-8601 string.")
            try:
                parsed[name] = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a valid ISO-8601 datetime.") from exc
        for name in (
            "activity_alert_ids",
            "polarization_alert_ids",
            "associated_video_ids",
        ):
            value = parsed[name]
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list.")
            parsed[name] = tuple(value)
        return cls(**parsed)


def project_detection_association(
    association: DetectionAssociation,
    *,
    config_hash: str,
    created_at_cutoff_utc: datetime,
    input_policy_id: str | None = None,
    association_ref: str | None = None,
    associated_video_ids: Iterable[str] = (),
) -> AssociatedEventCandidate | None:
    """Project one finalized association without re-running association logic."""

    if not isinstance(association, DetectionAssociation):
        raise TypeError("association must be a DetectionAssociation.")
    if association.status != FINALIZED:
        return None
    _require_text("config_hash", config_hash)
    cutoff = _require_utc("created_at_cutoff_utc", created_at_cutoff_utc)
    if cutoff < association.finalized_at_cutoff_utc:
        raise ValueError("Candidate creation cannot precede association finalization.")
    if input_policy_id is not None:
        _require_text("input_policy_id", input_policy_id)
    reference = (
        f"{ASSOCIATION_ARTIFACT}#{association.association_id}"
        if association_ref is None
        else _require_text("association_ref", association_ref)
    )
    return AssociatedEventCandidate(
        candidate_id=_candidate_id(association),
        run_id=association.run_id,
        config_hash=config_hash,
        association_id=association.association_id,
        association_ref=reference,
        projection_policy_id=ASSOCIATION_TO_CANDIDATE_V1,
        association_policy_id=association.policy_id,
        association_tolerance=association.tolerance,
        association_parameter_status=association.parameter_status,
        candidate_type=association.association_type,
        activity_alert_ids=association.activity_alert_ids,
        polarization_alert_ids=association.polarization_alert_ids,
        start_time_utc=association.start_time_utc,
        end_time_utc=association.end_time_utc,
        reference_time_utc=association.reference_time_utc,
        association_finalized_at_cutoff_utc=association.finalized_at_cutoff_utc,
        created_at_cutoff_utc=cutoff,
        association_finalization_reason=association.finalization_reason,
        status=READY_FOR_EVIDENCE,
        input_policy_id=input_policy_id,
        associated_video_ids=tuple(associated_video_ids),
    )


class EventCandidateProjector:
    """Incremental, in-run one-shot projection of finalized associations."""

    def __init__(
        self,
        *,
        run_id: str,
        config_hash: str,
        input_policy_id: str | None = None,
        association_artifact_ref: str = ASSOCIATION_ARTIFACT,
    ) -> None:
        self.run_id = _require_text("run_id", run_id)
        self.config_hash = _require_text("config_hash", config_hash)
        if input_policy_id is not None:
            _require_text("input_policy_id", input_policy_id)
        self.input_policy_id = input_policy_id
        self.association_artifact_ref = _require_text(
            "association_artifact_ref", association_artifact_ref
        )
        self._promoted_association_ids: set[str] = set()
        self._association_signature: tuple[str, str, str] | None = None
        self._last_cutoff_utc: datetime | None = None

    @property
    def promoted_association_ids(self) -> frozenset[str]:
        return frozenset(self._promoted_association_ids)

    def promote(
        self,
        association: DetectionAssociation,
        *,
        created_at_cutoff_utc: datetime,
        associated_video_ids: Iterable[str] = (),
    ) -> AssociatedEventCandidate | None:
        if not isinstance(association, DetectionAssociation):
            raise TypeError("association must be a DetectionAssociation.")
        cutoff = _require_utc("created_at_cutoff_utc", created_at_cutoff_utc)
        if self._last_cutoff_utc is not None and cutoff < self._last_cutoff_utc:
            raise ValueError("Candidate creation cutoffs must not regress.")
        self._last_cutoff_utc = cutoff
        if association.status != FINALIZED:
            return None
        if association.run_id != self.run_id:
            raise ValueError(
                "DetectionAssociation run_id does not match projection run."
            )
        signature = (
            association.policy_id,
            association.tolerance,
            association.parameter_status,
        )
        if self._association_signature is None:
            self._association_signature = signature
        elif signature != self._association_signature:
            raise ValueError(
                "Association policy parameters must remain fixed within a run."
            )
        if association.association_id in self._promoted_association_ids:
            return None
        candidate = project_detection_association(
            association,
            config_hash=self.config_hash,
            created_at_cutoff_utc=cutoff,
            input_policy_id=self.input_policy_id,
            association_ref=(
                f"{self.association_artifact_ref}#{association.association_id}"
            ),
            associated_video_ids=associated_video_ids,
        )
        if candidate is not None:
            self._promoted_association_ids.add(association.association_id)
        return candidate


@dataclass(frozen=True, slots=True)
class EventCandidateArtifacts:
    candidate_path: Path
    manifest_path: Path
    associations_input: int
    candidates_produced: int
    activity: int
    polarization: int
    joint: int


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            encoded = json.dumps(
                row, ensure_ascii=False, sort_keys=True, allow_nan=False
            ) + "\n"
            target.write(encoded)
            digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def run_event_candidate_fusion(
    *,
    associations: Iterable[DetectionAssociation],
    run_id: str,
    config_hash: str,
    created_at_cutoff_utc: datetime,
    output_dir: str | Path,
    input_policy_id: str | None = None,
    association_artifact_ref: str = ASSOCIATION_ARTIFACT,
    associated_video_ids_by_association: Mapping[str, Iterable[str]] | None = None,
) -> EventCandidateArtifacts:
    """Persist one neutral candidate per finalized association."""

    cutoff = _require_utc("created_at_cutoff_utc", created_at_cutoff_utc)
    association_rows = list(associations)
    for association in association_rows:
        if not isinstance(association, DetectionAssociation):
            raise TypeError("associations must contain DetectionAssociation values.")
    association_ids = [item.association_id for item in association_rows]
    if len(set(association_ids)) != len(association_ids):
        raise ValueError("associations must not contain duplicate association IDs.")
    video_map = dict(associated_video_ids_by_association or {})
    known_ids = set(association_ids)
    unknown_video_keys = set(video_map) - known_ids
    if unknown_video_keys:
        raise ValueError(
            "Video lineage references unknown association IDs: "
            f"{sorted(unknown_video_keys)}"
        )
    projector = EventCandidateProjector(
        run_id=run_id,
        config_hash=config_hash,
        input_policy_id=input_policy_id,
        association_artifact_ref=association_artifact_ref,
    )
    candidates: list[AssociatedEventCandidate] = []
    for association in sorted(
        association_rows,
        key=lambda item: (
            item.finalized_at_cutoff_utc,
            item.start_time_utc,
            item.association_id,
        ),
    ):
        candidate = projector.promote(
            association,
            created_at_cutoff_utc=cutoff,
            associated_video_ids=video_map.get(association.association_id, ()),
        )
        if candidate is not None:
            candidates.append(candidate)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / EVENT_CANDIDATE_ARTIFACT
    manifest_path = root / EVENT_CANDIDATE_MANIFEST
    if candidate_path.exists() or manifest_path.exists():
        raise FileExistsError("Candidate outputs already exist; choose a fresh output.")
    artifact_hash = _write_jsonl_exclusive(
        candidate_path, (candidate.to_mapping() for candidate in candidates)
    )
    counts = {
        ACTIVITY: sum(item.candidate_type == ACTIVITY for item in candidates),
        POLARIZATION: sum(
            item.candidate_type == POLARIZATION for item in candidates
        ),
        "ACTIVITY_POLARIZATION": sum(
            item.candidate_type == "ACTIVITY_POLARIZATION" for item in candidates
        ),
    }
    association_policies = sorted(
        {
            (
                item.policy_id,
                item.tolerance,
                item.parameter_status,
            )
            for item in association_rows
        }
    )
    manifest = {
        "artifact_version": EVENT_CANDIDATE_ARTIFACT_VERSION,
        "status": "executed",
        "run_id": run_id,
        "config_hash": config_hash,
        "candidate_projection": {
            "policy_id": ASSOCIATION_TO_CANDIDATE_V1,
            "one_candidate_per_finalized_association": True,
            "score_fusion": False,
            "ranking": False,
        },
        "association_policies": [
            {
                "policy_id": policy_id,
                "tolerance": tolerance,
                "parameter_status": parameter_status,
            }
            for policy_id, tolerance, parameter_status in association_policies
        ],
        "input": {
            "input_policy_id": input_policy_id,
            "association_artifact": association_artifact_ref,
        },
        "counts": {
            "associations_input": len(association_rows),
            "candidates": len(candidates),
            "activity": counts[ACTIVITY],
            "polarization": counts[POLARIZATION],
            "activity_polarization": counts["ACTIVITY_POLARIZATION"],
        },
        "causality": {
            "created_at_cutoff_utc": cutoff.isoformat(),
            "requires_finalized_association": True,
            "future_lookahead": False,
            "durable_cross_restart_state": False,
            "A9_status": "DEFERRED",
        },
        "artifacts": {
            "event_candidates": EVENT_CANDIDATE_ARTIFACT,
            "event_candidates_sha256": artifact_hash,
        },
        "downstream": {
            "evidence_window_policy_applied": False,
            "evidence_retrieved": False,
            "rag_executed": False,
            "candidate_validated": False,
        },
    }
    with manifest_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return EventCandidateArtifacts(
        candidate_path=candidate_path,
        manifest_path=manifest_path,
        associations_input=len(association_rows),
        candidates_produced=len(candidates),
        activity=counts[ACTIVITY],
        polarization=counts[POLARIZATION],
        joint=counts["ACTIVITY_POLARIZATION"],
    )


def load_associated_event_candidates(
    path: str | Path,
) -> tuple[AssociatedEventCandidate, ...]:
    """Load neutral event candidates from their JSONL artifact."""

    source = Path(path)
    candidates: list[AssociatedEventCandidate] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid candidate JSON on line {line_number}.") from exc
        candidates.append(AssociatedEventCandidate.from_mapping(payload))
    return tuple(candidates)


__all__ = [
    "ASSOCIATION_TO_CANDIDATE_V1",
    "AssociatedEventCandidate",
    "EVENT_CANDIDATE_ARTIFACT",
    "EVENT_CANDIDATE_ARTIFACT_VERSION",
    "EVENT_CANDIDATE_MANIFEST",
    "EventCandidateArtifacts",
    "EventCandidateProjector",
    "READY_FOR_EVIDENCE",
    "load_associated_event_candidates",
    "project_detection_association",
    "run_event_candidate_fusion",
]
