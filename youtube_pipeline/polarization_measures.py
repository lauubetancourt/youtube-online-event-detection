from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Protocol

import numpy as np
from measures.metrics.literature import EMDPol, EstebanRay
from measures.metrics.proposed import MEC


LIKERT5_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
AUTHOR_MASS = "author"
DISTRIBUTION_READY = "ready"
NO_CLASSIFIABLE_OPINIONS = "no_classifiable_opinions"
MEASUREMENT_COMPUTED = "computed"
MEASUREMENT_NOT_COMPUTED = "not_computed_no_classifiable_opinions"

ESTEBAN_RAY_MEASURE = "esteban_ray"
EMD_POL_MEASURE = "emd_pol"
MEC_MEASURE = "mec"

_COMPONENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_WEIGHT_SUM_TOLERANCE = 1e-9


def _require_nonempty_string(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _require_component_id(field_name: str, value: object) -> None:
    _require_nonempty_string(field_name, value)
    assert isinstance(value, str)
    if _COMPONENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must contain only lowercase letters, digits, "
            "and underscores, and must start with a letter or digit."
        )


def _require_integer(field_name: str, value: object, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")


def _finite_float(field_name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite.")
    return numeric_value


def _freeze_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("parameters keys must be non-empty strings.")
        if value is None:
            frozen[key] = None
        else:
            frozen[key] = _finite_float(f"parameters.{key}", value)
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class OpinionDistribution:
    """One validated, measure-neutral Likert-5 opinion distribution.

    This contract receives a distribution already built under the approved
    author-mass policy. It does not classify comments or aggregate authors.
    """

    proposition_id: str
    distribution_policy_id: str
    classification_run_id: str
    positions: tuple[float, ...]
    bin_counts: tuple[int, ...]
    weights: tuple[float, ...] | None
    support_count: int
    unique_author_count: int
    input_count: int
    classified_comment_count: int
    excluded_counts: Mapping[str, int]
    quality: str
    status: str = DISTRIBUTION_READY
    mass_unit: str = AUTHOR_MASS
    quality_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "proposition_id",
            "distribution_policy_id",
            "classification_run_id",
            "quality",
        ):
            _require_nonempty_string(field_name, getattr(self, field_name))
        if self.mass_unit != AUTHOR_MASS:
            raise ValueError("mass_unit must be 'author' for the approved contract.")
        if self.status not in {
            DISTRIBUTION_READY,
            NO_CLASSIFIABLE_OPINIONS,
        }:
            raise ValueError(
                "status must be 'ready' or 'no_classifiable_opinions'."
            )

        positions = tuple(
            _finite_float("positions[]", value) for value in self.positions
        )
        if positions != LIKERT5_POSITIONS:
            raise ValueError(
                "positions must be the complete Likert-5 support "
                "(0.0, 0.25, 0.5, 0.75, 1.0)."
            )
        object.__setattr__(self, "positions", positions)

        _require_integer("support_count", self.support_count)
        _require_integer("unique_author_count", self.unique_author_count)
        _require_integer("input_count", self.input_count)
        _require_integer("classified_comment_count", self.classified_comment_count)
        if self.unique_author_count != self.support_count:
            raise ValueError(
                "unique_author_count must equal support_count for author mass."
            )
        if self.classified_comment_count < self.support_count:
            raise ValueError(
                "classified_comment_count must be >= support_count."
            )
        if self.input_count < self.classified_comment_count:
            raise ValueError("input_count must be >= classified_comment_count.")

        if not isinstance(self.bin_counts, (list, tuple)):
            raise TypeError("bin_counts must be a list or tuple of integers.")
        bin_counts = tuple(self.bin_counts)
        if len(bin_counts) != len(LIKERT5_POSITIONS):
            raise ValueError("bin_counts must contain exactly five Likert bins.")
        for count in bin_counts:
            _require_integer("bin_counts[]", count)
        if sum(bin_counts) != self.support_count:
            raise ValueError("bin_counts must sum to support_count.")
        object.__setattr__(self, "bin_counts", bin_counts)

        if not isinstance(self.excluded_counts, Mapping):
            raise TypeError("excluded_counts must be a mapping.")
        excluded_counts: dict[str, int] = {}
        for reason, count in self.excluded_counts.items():
            _require_component_id("excluded_counts key", reason)
            _require_integer(f"excluded_counts.{reason}", count)
            excluded_counts[reason] = count
        object.__setattr__(self, "excluded_counts", MappingProxyType(excluded_counts))

        if not isinstance(self.quality_reasons, (list, tuple)):
            raise TypeError("quality_reasons must be a list or tuple of strings.")
        reasons = tuple(self.quality_reasons)
        for reason in reasons:
            _require_component_id("quality_reasons[]", reason)
        object.__setattr__(self, "quality_reasons", reasons)

        if self.status == NO_CLASSIFIABLE_OPINIONS:
            if self.weights is not None:
                raise ValueError(
                    "weights must be None when there are no classifiable opinions."
                )
            if self.support_count != 0:
                raise ValueError(
                    "support_count must be 0 when there are no classifiable opinions."
                )
            return

        if self.support_count < 1:
            raise ValueError("A ready distribution must have support_count >= 1.")
        if self.weights is None:
            raise ValueError("A ready distribution must provide weights.")
        if not isinstance(self.weights, (list, tuple)):
            raise TypeError("weights must be a list or tuple of real numbers.")
        weights = tuple(_finite_float("weights[]", value) for value in self.weights)
        if len(weights) != len(LIKERT5_POSITIONS):
            raise ValueError("weights must contain exactly five Likert-bin masses.")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative.")
        if not math.isclose(
            math.fsum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=_WEIGHT_SUM_TOLERANCE,
        ):
            raise ValueError("weights must sum to 1 within absolute tolerance 1e-9.")
        expected_weights = tuple(
            count / self.support_count for count in self.bin_counts
        )
        if any(
            not math.isclose(
                weight,
                expected,
                rel_tol=0.0,
                abs_tol=_WEIGHT_SUM_TOLERANCE,
            )
            for weight, expected in zip(weights, expected_weights, strict=True)
        ):
            raise ValueError("weights must equal bin_counts / support_count.")
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, slots=True)
class PolarizationMeasurement:
    """Measure-neutral result for one isolated opinion distribution."""

    measure_id: str
    value: float | None
    parameters: Mapping[str, Any]
    quality: str
    quality_reasons: tuple[str, ...]
    support_count: int
    unique_author_count: int
    proposition_id: str
    distribution_policy_id: str
    classification_run_id: str
    status: str = MEASUREMENT_COMPUTED

    def __post_init__(self) -> None:
        _require_component_id("measure_id", self.measure_id)
        for field_name in (
            "quality",
            "proposition_id",
            "distribution_policy_id",
            "classification_run_id",
        ):
            _require_nonempty_string(field_name, getattr(self, field_name))
        if self.status not in {
            MEASUREMENT_COMPUTED,
            MEASUREMENT_NOT_COMPUTED,
        }:
            raise ValueError("Unsupported polarization measurement status.")
        _require_integer("support_count", self.support_count)
        _require_integer("unique_author_count", self.unique_author_count)
        if self.value is None:
            if self.status != MEASUREMENT_NOT_COMPUTED:
                raise ValueError("value may be None only when measurement was skipped.")
        else:
            if self.status != MEASUREMENT_COMPUTED:
                raise ValueError("A skipped measurement must not contain a value.")
            object.__setattr__(self, "value", _finite_float("value", self.value))
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))
        if not isinstance(self.quality_reasons, (list, tuple)):
            raise TypeError("quality_reasons must be a list or tuple of strings.")
        reasons = tuple(self.quality_reasons)
        for reason in reasons:
            _require_component_id("quality_reasons[]", reason)
        object.__setattr__(self, "quality_reasons", reasons)


@dataclass(frozen=True, slots=True)
class EstebanRayConfig:
    """Library defaults; not methodologically validated project parameters."""

    alpha: float = 0.8
    K: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EstebanRayConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("Esteban-Ray config must be an object.")
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown Esteban-Ray config fields: {unknown}")
        return cls(**dict(payload))

    def __post_init__(self) -> None:
        alpha = _finite_float("alpha", self.alpha)
        if not 0 < alpha <= 1.6:
            raise ValueError("alpha must be in the interval (0, 1.6].")
        object.__setattr__(self, "alpha", alpha)
        if self.K is not None:
            normalization_constant = _finite_float("K", self.K)
            if normalization_constant <= 0:
                raise ValueError("K must be > 0 when provided.")
            object.__setattr__(self, "K", normalization_constant)


@dataclass(frozen=True, slots=True)
class EMDPolConfig:
    """EMDPol has no public constructor parameters in the pinned version."""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EMDPolConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("EMDPol config must be an object.")
        if payload:
            raise ValueError(f"Unknown EMDPol config fields: {sorted(payload)}")
        return cls()


@dataclass(frozen=True, slots=True)
class MECConfig:
    """Library defaults; not methodologically validated project parameters."""

    alpha: float = 2.0
    beta: float = 1.15

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MECConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("MEC config must be an object.")
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown MEC config fields: {unknown}")
        return cls(**dict(payload))

    def __post_init__(self) -> None:
        for field_name in ("alpha", "beta"):
            value = _finite_float(field_name, getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0.")
            object.__setattr__(self, field_name, value)


class PolarizationMeasure(Protocol):
    """Minimal interface shared by isolated polarization-measure adapters."""

    measure_id: str

    def measure(
        self,
        distribution: OpinionDistribution,
    ) -> PolarizationMeasurement:
        ...


class PolarizationCalculationError(RuntimeError):
    """The library failed to return a finite scalar for valid input."""


class _PolMeasuresAdapter:
    measure_id: str

    def __init__(self, *, library_measure: Callable[..., Any]) -> None:
        self._library_measure = library_measure

    @property
    def parameters(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def measure(
        self,
        distribution: OpinionDistribution,
    ) -> PolarizationMeasurement:
        if not isinstance(distribution, OpinionDistribution):
            raise TypeError("distribution must be an OpinionDistribution.")
        if distribution.status == NO_CLASSIFIABLE_OPINIONS:
            return self._result(
                distribution=distribution,
                value=None,
                status=MEASUREMENT_NOT_COMPUTED,
            )
        assert distribution.weights is not None
        positions = np.asarray(distribution.positions, dtype=np.float64)
        weights = np.asarray(distribution.weights, dtype=np.float64)
        try:
            raw_value = self._library_measure(
                positions,
                weights,
                normalize_weights=False,
                normalize_positions=False,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            raise PolarizationCalculationError(
                f"{self.measure_id} calculation failed in pol_measures."
            ) from exc
        try:
            value = _finite_float("pol_measures result", raw_value)
        except (TypeError, ValueError) as exc:
            raise PolarizationCalculationError(
                f"{self.measure_id} did not return a finite scalar."
            ) from exc
        return self._result(
            distribution=distribution,
            value=value,
            status=MEASUREMENT_COMPUTED,
        )

    def _result(
        self,
        *,
        distribution: OpinionDistribution,
        value: float | None,
        status: str,
    ) -> PolarizationMeasurement:
        return PolarizationMeasurement(
            measure_id=self.measure_id,
            value=value,
            parameters=self.parameters,
            quality=distribution.quality,
            quality_reasons=distribution.quality_reasons,
            support_count=distribution.support_count,
            unique_author_count=distribution.unique_author_count,
            proposition_id=distribution.proposition_id,
            distribution_policy_id=distribution.distribution_policy_id,
            classification_run_id=distribution.classification_run_id,
            status=status,
        )


class EstebanRayAdapter(_PolMeasuresAdapter):
    measure_id = ESTEBAN_RAY_MEASURE

    def __init__(self, *, config: EstebanRayConfig | None = None) -> None:
        if config is not None and not isinstance(config, EstebanRayConfig):
            raise TypeError("config must be EstebanRayConfig or None.")
        self.config = config or EstebanRayConfig()
        super().__init__(
            library_measure=EstebanRay(alpha=self.config.alpha, K=self.config.K)
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {"alpha": self.config.alpha, "K": self.config.K}


class EMDPolAdapter(_PolMeasuresAdapter):
    measure_id = EMD_POL_MEASURE

    def __init__(self, *, config: EMDPolConfig | None = None) -> None:
        if config is not None and not isinstance(config, EMDPolConfig):
            raise TypeError("config must be EMDPolConfig or None.")
        self.config = config or EMDPolConfig()
        super().__init__(library_measure=EMDPol())

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {}


class MECAdapter(_PolMeasuresAdapter):
    measure_id = MEC_MEASURE

    def __init__(self, *, config: MECConfig | None = None) -> None:
        if config is not None and not isinstance(config, MECConfig):
            raise TypeError("config must be MECConfig or None.")
        self.config = config or MECConfig()
        super().__init__(
            library_measure=MEC(alpha=self.config.alpha, beta=self.config.beta)
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {"alpha": self.config.alpha, "beta": self.config.beta}


POLARIZATION_MEASURE_REGISTRY: dict[
    str,
    Callable[..., PolarizationMeasure],
] = {
    ESTEBAN_RAY_MEASURE: EstebanRayAdapter,
    EMD_POL_MEASURE: EMDPolAdapter,
    MEC_MEASURE: MECAdapter,
}


def get_polarization_measure_names() -> tuple[str, ...]:
    return tuple(sorted(POLARIZATION_MEASURE_REGISTRY))


def create_polarization_measure(
    measure_id: str,
    *,
    config: EstebanRayConfig | EMDPolConfig | MECConfig | None = None,
) -> PolarizationMeasure:
    _require_component_id("measure_id", measure_id)
    try:
        factory = POLARIZATION_MEASURE_REGISTRY[measure_id]
    except KeyError as exc:
        available = ", ".join(get_polarization_measure_names())
        raise ValueError(
            f"Unknown polarization measure {measure_id!r}; available measures: "
            f"{available}."
        ) from exc
    return factory(config=config)


__all__ = [
    "AUTHOR_MASS",
    "create_polarization_measure",
    "DISTRIBUTION_READY",
    "EMD_POL_MEASURE",
    "EMDPolAdapter",
    "EMDPolConfig",
    "ESTEBAN_RAY_MEASURE",
    "EstebanRayAdapter",
    "EstebanRayConfig",
    "get_polarization_measure_names",
    "LIKERT5_POSITIONS",
    "MEASUREMENT_COMPUTED",
    "MEASUREMENT_NOT_COMPUTED",
    "MEC_MEASURE",
    "MECAdapter",
    "MECConfig",
    "NO_CLASSIFIABLE_OPINIONS",
    "OpinionDistribution",
    "PolarizationCalculationError",
    "PolarizationMeasure",
    "POLARIZATION_MEASURE_REGISTRY",
    "PolarizationMeasurement",
]
