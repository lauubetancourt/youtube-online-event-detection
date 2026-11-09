"""Numeric signal values and explicit absences; activity's legacy contract is unchanged."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta
from numbers import Real
from types import MappingProxyType
from typing import Any

VALUE_AVAILABLE = "VALUE_AVAILABLE"
NO_INPUT = "NO_INPUT"
NO_CLASSIFIABLE_OPINIONS = "NO_CLASSIFIABLE_OPINIONS"
NO_REPRESENTABLE_AUTHORS = "NO_REPRESENTABLE_AUTHORS"
SIGNAL_STATUSES = frozenset({VALUE_AVAILABLE, NO_INPUT, NO_CLASSIFIABLE_OPINIONS,
                             NO_REPRESENTABLE_AUTHORS})


def require_text(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def require_utc(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC.")


def freeze_json(value: Any) -> Any:
    """Defensive, recursive snapshot of JSON-compatible metadata."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            require_text("metadata key", key)
            result[key] = freeze_json(item)
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("metadata must contain finite JSON-compatible values.")


def to_json_mapping(value: Any) -> Any:
    """Serialize immutable signal artifacts without deepcopying mapping proxies."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_mapping(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: to_json_mapping(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_mapping(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SignalObservation:
    signal_id: str
    observation_time_utc: datetime
    value: float | None
    status: str
    support_count: int
    quality: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_text("signal_id", self.signal_id)
        require_utc("observation_time_utc", self.observation_time_utc)
        require_text("quality", self.quality)
        if self.status not in SIGNAL_STATUSES:
            raise ValueError(f"Unsupported signal status: {self.status!r}.")
        if isinstance(self.support_count, bool) or not isinstance(self.support_count, int):
            raise TypeError("support_count must be an integer.")
        if self.support_count < 0:
            raise ValueError("support_count must be >= 0.")
        if self.status == VALUE_AVAILABLE:
            if isinstance(self.value, bool) or not isinstance(self.value, Real):
                raise TypeError("VALUE_AVAILABLE requires a real value.")
            if not math.isfinite(float(self.value)):
                raise ValueError("value must be finite.")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            raise ValueError("An absent signal value must be None.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    def to_mapping(self) -> dict[str, Any]:
        return to_json_mapping(self)
