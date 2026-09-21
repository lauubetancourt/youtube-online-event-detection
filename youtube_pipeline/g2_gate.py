"""Configurable, non-calibrated G-1 to G-2 resource gate."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from youtube_pipeline.rag_generation_g1 import VALIDATION_STATUS_VALUES


G1_GATE_V1 = "g1_gate_v1"
OFF = "OFF"
PERMISSIVE = "PERMISSIVE"
STRICT = "STRICT"
G2_GATE_MODES = frozenset({OFF, PERMISSIVE, STRICT})

RUN_G2 = "RUN_G2"
SKIP_G2 = "SKIP_G2"

G2_EXECUTED = "EXECUTED"
G2_SKIPPED_BY_G1_GATE = "SKIPPED_BY_G1_GATE"
G2_NOT_ELIGIBLE = "NOT_ELIGIBLE"
G2_FAILED = "FAILED"


def _text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


@dataclass(frozen=True, slots=True)
class G2GatePolicyConfig:
    """Gate definition; non-OFF status sets remain explicit experiment inputs."""

    policy_id: str = G1_GATE_V1
    mode: str = OFF
    run_g2_for_g1_statuses: tuple[str, ...] = ()
    parameter_status: str = "DEVELOPMENT_NOT_CALIBRATED"

    def __post_init__(self) -> None:
        _text("policy_id", self.policy_id)
        _text("mode", self.mode)
        _text("parameter_status", self.parameter_status)
        if self.policy_id != G1_GATE_V1:
            raise ValueError("Unsupported G-2 gate policy.")
        if self.mode not in G2_GATE_MODES:
            raise ValueError(f"Unsupported G-2 gate mode {self.mode!r}.")
        if not isinstance(self.run_g2_for_g1_statuses, tuple):
            raise TypeError("run_g2_for_g1_statuses must be a tuple.")
        invalid = sorted(
            set(self.run_g2_for_g1_statuses) - VALIDATION_STATUS_VALUES
        )
        if invalid:
            raise ValueError(f"Unknown G-1 validation statuses: {invalid}")
        if len(set(self.run_g2_for_g1_statuses)) != len(
            self.run_g2_for_g1_statuses
        ):
            raise ValueError("run_g2_for_g1_statuses must not contain duplicates.")
        if self.mode == OFF and self.run_g2_for_g1_statuses:
            raise ValueError("OFF mode must not define eligible G-1 statuses.")
        if self.mode != OFF and not self.run_g2_for_g1_statuses:
            raise ValueError(
                "Non-OFF modes require an explicit, non-calibrated G-1 status set."
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> G2GatePolicyConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("g2_gate must be a JSON object.")
        values = dict(payload)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown G-2 gate fields: {sorted(unknown)}")
        statuses = values.get("run_g2_for_g1_statuses")
        if statuses is not None:
            if not isinstance(statuses, list):
                raise TypeError("run_g2_for_g1_statuses must be a JSON array.")
            values["run_g2_for_g1_statuses"] = tuple(statuses)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class G2GateDecision:
    candidate_id: str
    policy_id: str
    mode: str
    g1_result_ref: str | None
    g1_validation_status: str | None
    decision: str
    reason_code: str

    def __post_init__(self) -> None:
        for name in ("candidate_id", "policy_id", "mode", "decision", "reason_code"):
            _text(name, getattr(self, name))
        if self.g1_result_ref is not None:
            _text("g1_result_ref", self.g1_result_ref)
        if self.g1_validation_status is not None:
            _text("g1_validation_status", self.g1_validation_status)
        if self.decision not in {RUN_G2, SKIP_G2}:
            raise ValueError("Unsupported G-2 gate decision.")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def decide_g2(
    *,
    candidate_id: str,
    config: G2GatePolicyConfig,
    g1_result: Mapping[str, Any] | None,
) -> G2GateDecision:
    if not isinstance(config, G2GatePolicyConfig):
        raise TypeError("config must be G2GatePolicyConfig.")
    status = (
        str(g1_result.get("validation_status"))
        if isinstance(g1_result, Mapping)
        and g1_result.get("validation_status") is not None
        else None
    )
    result_ref = (
        str(g1_result.get("validation_id"))
        if isinstance(g1_result, Mapping) and g1_result.get("validation_id")
        else None
    )
    if config.mode == OFF:
        decision, reason = RUN_G2, "GATE_OFF_BASELINE"
    elif status is None:
        decision, reason = SKIP_G2, "G1_RESULT_UNAVAILABLE"
    elif status in config.run_g2_for_g1_statuses:
        decision, reason = RUN_G2, "G1_STATUS_CONFIGURED_TO_RUN"
    else:
        decision, reason = SKIP_G2, "G1_STATUS_CONFIGURED_TO_SKIP"
    return G2GateDecision(
        candidate_id=candidate_id,
        policy_id=config.policy_id,
        mode=config.mode,
        g1_result_ref=result_ref,
        g1_validation_status=status,
        decision=decision,
        reason_code=reason,
    )


__all__ = [
    "G1_GATE_V1",
    "G2_EXECUTED",
    "G2_FAILED",
    "G2_GATE_MODES",
    "G2_NOT_ELIGIBLE",
    "G2_SKIPPED_BY_G1_GATE",
    "G2GateDecision",
    "G2GatePolicyConfig",
    "OFF",
    "PERMISSIVE",
    "RUN_G2",
    "SKIP_G2",
    "STRICT",
    "decide_g2",
]
