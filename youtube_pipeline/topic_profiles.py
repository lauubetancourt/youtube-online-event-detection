"""Immutable monitored-topic configuration for stance-enabled runs."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from youtube_pipeline.stance_classifier import StanceProposition
from youtube_pipeline.stance_classifier_prompt import PROMPT_ID, PROMPT_VERSION


def _text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    return value if allow_empty else normalized


@dataclass(frozen=True, slots=True)
class TopicProfile:
    """One topic, proposition, and context frozen for an entire run."""

    topic_id: str
    topic_description: str
    proposition: StanceProposition
    stance_prompt_id: str = PROMPT_ID
    stance_prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        _text("topic_id", self.topic_id)
        _text("topic_description", self.topic_description)
        if not isinstance(self.proposition, StanceProposition):
            raise TypeError("proposition must be StanceProposition.")
        _text("context", self.proposition.context)
        _text("stance_prompt_id", self.stance_prompt_id)
        _text("stance_prompt_version", self.stance_prompt_version)
        if (self.stance_prompt_id, self.stance_prompt_version) != (
            PROMPT_ID,
            PROMPT_VERSION,
        ):
            raise ValueError("Unsupported stance prompt identity/version.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TopicProfile:
        if not isinstance(payload, Mapping):
            raise TypeError("topic_profile must be a JSON object.")
        values = dict(payload)
        allowed_resolved = {
            "topic_id",
            "topic_description",
            "proposition",
            "stance_prompt_id",
            "stance_prompt_version",
        }
        allowed_file = {"topic_id", "topic_description", "stance"}
        if "stance" in values:
            unknown = set(values) - allowed_file
            if unknown:
                raise ValueError(f"Unknown TopicProfile fields: {sorted(unknown)}")
            stance = values.pop("stance")
            if not isinstance(stance, Mapping):
                raise TypeError("topic_profile.stance must be a JSON object.")
            stance_values = dict(stance)
            allowed_stance = {
                "proposition_id",
                "proposition",
                "context",
                "context_version",
                "prompt_id",
                "prompt_version",
            }
            unknown_stance = set(stance_values) - allowed_stance
            if unknown_stance:
                raise ValueError(
                    "Unknown TopicProfile stance fields: "
                    f"{sorted(unknown_stance)}"
                )
            missing = {"proposition_id", "proposition", "context"} - set(
                stance_values
            )
            if missing:
                raise ValueError(
                    f"Missing TopicProfile stance fields: {sorted(missing)}"
                )
            values["proposition"] = {
                "proposition_id": stance_values["proposition_id"],
                "proposition_text": stance_values["proposition"],
                "context": stance_values["context"],
                "context_version": stance_values.get(
                    "context_version", "provided_v1"
                ),
            }
            values["stance_prompt_id"] = stance_values.get("prompt_id", PROMPT_ID)
            values["stance_prompt_version"] = stance_values.get(
                "prompt_version", PROMPT_VERSION
            )
        else:
            unknown = set(values) - allowed_resolved
            if unknown:
                raise ValueError(f"Unknown TopicProfile fields: {sorted(unknown)}")
        missing = {"topic_id", "topic_description", "proposition"} - set(values)
        if missing:
            raise ValueError(f"Missing TopicProfile fields: {sorted(missing)}")
        proposition = values["proposition"]
        if isinstance(proposition, Mapping):
            proposition = StanceProposition.from_mapping(proposition)
        values["proposition"] = proposition
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_topic_profile(path: str | Path) -> TopicProfile:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Topic profile not found: {target}")
    if not target.is_file():
        raise ValueError(f"Topic profile path must be a file: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in topic profile {target} at line {exc.lineno}, "
            f"column {exc.colno}."
        ) from exc
    return TopicProfile.from_mapping(payload)


__all__ = ["TopicProfile", "load_topic_profile"]
