from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from youtube_pipeline.activity_detection import ActivityDetectionRouteConfig
from youtube_pipeline.activity_signals import ActivitySignalDefinition
from youtube_pipeline.stance_classifier import CommentStanceClassifierConfig
from youtube_pipeline.polarization_signals import PolarizationTemporalSignalConfig
from youtube_pipeline.polarization_runtime_config import PolarizationRuntimeConfig
from youtube_pipeline.cleaning import CleaningConfig
from youtube_pipeline.cyclic_daily_signals import CyclicDailySignalConfig
from youtube_pipeline.cyclic_detection_connector import CyclicDetectionConnectorConfig
from youtube_pipeline.cyclic_ingestion import CyclicIngestionConfig
from youtube_pipeline.cyclic_orchestration import CyclicOrchestratorConfig
from youtube_pipeline.cyclic_stateful_adapter import CyclicStatefulAdapterConfig
from youtube_pipeline.daily_frequency_baseline import DailyFrequencyBaselineConfig
from youtube_pipeline.daily_rag_context_selection import DailyContextSelectionConfig
from youtube_pipeline.daily_rag_consumer import DailyRagConsumerConfig
from youtube_pipeline.daily_rag_sidecars import DailyRagSidecarBuildConfig
from youtube_pipeline.data_extraction import ExtractionConfig
from youtube_pipeline.detectors import ADWINConfig, KSWINConfig, PageHinkleyConfig, XiaoEMAConfig
from youtube_pipeline.detection_alerts import DetectionGapPolicyConfig
from youtube_pipeline.detection_association import DetectionAssociationConfig
from youtube_pipeline.candidate_evidence import CandidateEvidencePolicyConfig
from youtube_pipeline.candidate_rag import CandidateRagValidationConfig
from youtube_pipeline.experimental_input import ExperimentalInputPolicyConfig
from youtube_pipeline.prepared_replay import PreparedDatasetConfig, ReplayConfig
from youtube_pipeline.polarization_measures import (
    EMDPolConfig,
    EstebanRayConfig,
    MECConfig,
)
from youtube_pipeline.rag_consumer import RagConsumerConfig
from youtube_pipeline.rag_evidence import RagEvidenceBuildConfig
from youtube_pipeline.rag_generation_g1 import RagG1Config
from youtube_pipeline.rag_generation_g2 import RagG2Config
from youtube_pipeline.rag_generation_g2_hierarchical import RagG2HierarchicalConfig
from youtube_pipeline.rag_sidecars import RagSidecarBuildConfig
from youtube_pipeline.rag_validation import RagValidationPrepareConfig
from youtube_pipeline.storage import LocalFilesConfig
from youtube_pipeline.causal_replay_config import CausalEndToEndReplayConfig

from .models import (
    ArtifactsConfig,
    DataConfig,
    DetectionConfig,
    PolarizationConfig,
    RagConfig,
    RunConfig,
    RunIdentityConfig,
    SignalsConfig,
    SimulationConfig,
)

_ROOT_FIELDS = {
    "identity",
    "data",
    "simulation",
    "signals",
    "detection",
    "association",
    "polarization",
    "rag",
    "artifacts",
}
_DATA_FIELDS = {
    "youtube_api", "local_files", "prepared_dataset", "cleaning",
    "experimental_input",
}
_SIMULATION_FIELDS = {
    "ingestion", "orchestration", "stateful_adapter", "replay",
    "causal_end_to_end",
}
_SIGNALS_FIELDS = {"daily", "activity", "polarization"}
_DETECTION_FIELDS = {
    "activity_route",
    "routes",
    "gap_policy",
    "connector",
    "xiao_ema",
    "page_hinkley",
    "adwin",
    "kswin",
    "daily_frequency",
}
_POLARIZATION_FIELDS = {"measure_id", "esteban_ray", "emd_pol", "mec", "classifier", "runtime"}
_RAG_FIELDS = {
    "evidence",
    "sidecars",
    "consumer",
    "validation",
    "g1",
    "g2",
    "g2_hierarchical",
    "daily_sidecars",
    "daily_consumer",
    "daily_context_selection",
    "candidate_evidence",
    "candidate_validation",
}


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a JSON object.")
    return dict(value)


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {location} fields: {unknown}")


def _merge_explicit_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, override_value in overrides.items():
        if override_value is None:
            continue
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _merge_explicit_overrides(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def _matches_annotation(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    if annotation is None or annotation is type(None):
        return value is None

    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        return any(_matches_annotation(value, option) for option in get_args(annotation))
    if origin is dict:
        return isinstance(value, dict)
    if origin is list:
        return isinstance(value, list)
    if origin is tuple:
        return isinstance(value, tuple)
    if origin is set:
        return isinstance(value, set)

    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return True


def _validate_dataclass_field_types(instance: Any, location: str) -> None:
    if not is_dataclass(instance):
        raise TypeError(f"{location} must be a dataclass instance.")
    type_hints = get_type_hints(type(instance))
    for field in fields(instance):
        annotation = type_hints.get(field.name, Any)
        value = getattr(instance, field.name)
        if not _matches_annotation(value, annotation):
            raise TypeError(
                f"{location}.{field.name} has invalid type "
                f"{type(value).__name__}."
            )


def _build_component(
    config_type: type,
    payload: Any,
    location: str,
) -> Any:
    component_payload = _require_object(payload, location)
    instance = config_type.from_mapping(component_payload)
    _validate_dataclass_field_types(instance, location)
    return instance


def _build_identity(payload: Any) -> RunIdentityConfig:
    identity_payload = _require_object(payload, "identity")
    _reject_unknown_keys(identity_payload, {"run_id"}, "identity")
    if "run_id" not in identity_payload:
        raise ValueError("identity.run_id is required.")
    return RunIdentityConfig(run_id=identity_payload["run_id"])


def _build_artifacts(payload: Any) -> ArtifactsConfig:
    section = _require_object(payload, "artifacts")
    _reject_unknown_keys(
        section,
        {"run_mode", "trace_level"},
        "artifacts",
    )
    artifacts = ArtifactsConfig(**section)
    _validate_dataclass_field_types(artifacts, "artifacts")
    return artifacts


def _build_data(payload: Any) -> DataConfig:
    section = _require_object(payload, "data")
    _reject_unknown_keys(section, _DATA_FIELDS, "data")
    youtube_api = None
    local_files = None
    prepared_dataset = None
    cleaning = None
    experimental_input = None
    if "youtube_api" in section:
        component_payload = _require_object(
            section["youtube_api"],
            "data.youtube_api",
        )
        _reject_unknown_keys(
            component_payload,
            {field.name for field in fields(ExtractionConfig)},
            "data.youtube_api",
        )
        youtube_api = _build_component(
            ExtractionConfig,
            component_payload,
            "data.youtube_api",
        )
    if "local_files" in section:
        local_files = _build_component(
            LocalFilesConfig,
            section["local_files"],
            "data.local_files",
        )
    if "prepared_dataset" in section:
        prepared_dataset = _build_component(
            PreparedDatasetConfig,
            section["prepared_dataset"],
            "data.prepared_dataset",
        )
    if "cleaning" in section:
        cleaning = _build_component(
            CleaningConfig,
            section["cleaning"],
            "data.cleaning",
        )
    if "experimental_input" in section:
        experimental_input = _build_component(
            ExperimentalInputPolicyConfig,
            section["experimental_input"],
            "data.experimental_input",
        )
    return DataConfig(
        youtube_api=youtube_api,
        local_files=local_files,
        prepared_dataset=prepared_dataset,
        cleaning=cleaning,
        experimental_input=experimental_input,
    )


def _build_simulation(payload: Any) -> SimulationConfig:
    section = _require_object(payload, "simulation")
    _reject_unknown_keys(section, _SIMULATION_FIELDS, "simulation")
    return SimulationConfig(
        ingestion=(
            _build_component(
                CyclicIngestionConfig,
                section["ingestion"],
                "simulation.ingestion",
            )
            if "ingestion" in section
            else None
        ),
        orchestration=(
            _build_component(
                CyclicOrchestratorConfig,
                section["orchestration"],
                "simulation.orchestration",
            )
            if "orchestration" in section
            else None
        ),
        stateful_adapter=(
            _build_component(
                CyclicStatefulAdapterConfig,
                section["stateful_adapter"],
                "simulation.stateful_adapter",
            )
            if "stateful_adapter" in section
            else None
        ),
        replay=(
            _build_component(
                ReplayConfig,
                section["replay"],
                "simulation.replay",
            )
            if "replay" in section
            else None
        ),
        causal_end_to_end=(
            _build_component(
                CausalEndToEndReplayConfig,
                section["causal_end_to_end"],
                "simulation.causal_end_to_end",
            )
            if "causal_end_to_end" in section
            else None
        ),
    )


def _build_signals(payload: Any) -> SignalsConfig:
    section = _require_object(payload, "signals")
    _reject_unknown_keys(section, _SIGNALS_FIELDS, "signals")
    return SignalsConfig(
        daily=(
            _build_component(
                CyclicDailySignalConfig,
                section["daily"],
                "signals.daily",
            )
            if "daily" in section
            else None
        ),
        activity=(
            ActivitySignalDefinition(
                **_require_object(section["activity"], "signals.activity")
            )
            if "activity" in section
            else None
        ),
        polarization=(
            _build_component(PolarizationTemporalSignalConfig, section["polarization"], "signals.polarization")
            if "polarization" in section else None
        ),
    )


def _build_detection(payload: Any) -> DetectionConfig:
    section = _require_object(payload, "detection")
    _reject_unknown_keys(section, _DETECTION_FIELDS, "detection")
    activity_route = None
    if "activity_route" in section:
        route_payload = _require_object(
            section["activity_route"],
            "detection.activity_route",
        )
        _reject_unknown_keys(
            route_payload,
            {"signal_id", "detector_id"},
            "detection.activity_route",
        )
        activity_route = ActivityDetectionRouteConfig(**route_payload)
    routes = None
    if "routes" in section:
        route_values = section["routes"]
        if isinstance(route_values, Mapping):
            mapped_routes = []
            for signal_id, route_value in route_values.items():
                location = f"detection.routes.{signal_id}"
                route_payload = _require_object(route_value, location)
                _reject_unknown_keys(route_payload, {"detector_id"}, location)
                mapped_routes.append({
                    "signal_id": signal_id,
                    "detector_id": route_payload.get("detector_id"),
                })
            route_values = mapped_routes
        if not isinstance(route_values, (list, tuple)):
            raise TypeError("detection.routes must be an array or object.")
        parsed_routes = []
        for index, route_value in enumerate(route_values):
            location = f"detection.routes[{index}]"
            route_payload = _require_object(route_value, location)
            _reject_unknown_keys(route_payload, {"signal_id", "detector_id"}, location)
            parsed_routes.append(ActivityDetectionRouteConfig(**route_payload))
        routes = tuple(parsed_routes)
    return DetectionConfig(
        activity_route=activity_route,
        routes=routes,
        gap_policy=(
            _build_component(
                DetectionGapPolicyConfig,
                section["gap_policy"],
                "detection.gap_policy",
            )
            if "gap_policy" in section
            else None
        ),
        connector=(
            _build_component(
                CyclicDetectionConnectorConfig,
                section["connector"],
                "detection.connector",
            )
            if "connector" in section
            else None
        ),
        xiao_ema=(
            _build_component(
                XiaoEMAConfig,
                section["xiao_ema"],
                "detection.xiao_ema",
            )
            if "xiao_ema" in section
            else None
        ),
        page_hinkley=(
            _build_component(
                PageHinkleyConfig,
                section["page_hinkley"],
                "detection.page_hinkley",
            )
            if "page_hinkley" in section
            else None
        ),
        adwin=(
            _build_component(
                ADWINConfig,
                section["adwin"],
                "detection.adwin",
            )
            if "adwin" in section
            else None
        ),
        kswin=(
            _build_component(KSWINConfig, section["kswin"], "detection.kswin")
            if "kswin" in section else None
        ),
        daily_frequency=(
            _build_component(
                DailyFrequencyBaselineConfig,
                section["daily_frequency"],
                "detection.daily_frequency",
            )
            if "daily_frequency" in section
            else None
        )
    )


def _build_polarization(payload: Any) -> PolarizationConfig:
    section = _require_object(payload, "polarization")
    _reject_unknown_keys(section, _POLARIZATION_FIELDS, "polarization")
    measure_id = section.get("measure_id")
    if "measure_id" in section and not isinstance(measure_id, str):
        raise TypeError("polarization.measure_id must be a string.")
    return PolarizationConfig(
        measure_id=measure_id,
        classifier=(
            _build_component(CommentStanceClassifierConfig, section["classifier"], "polarization.classifier")
            if "classifier" in section else None
        ),
        runtime=(
            _build_component(PolarizationRuntimeConfig, section["runtime"], "polarization.runtime")
            if "runtime" in section else None
        ),
        esteban_ray=(
            _build_component(
                EstebanRayConfig,
                section["esteban_ray"],
                "polarization.esteban_ray",
            )
            if "esteban_ray" in section
            else None
        ),
        emd_pol=(
            _build_component(
                EMDPolConfig,
                section["emd_pol"],
                "polarization.emd_pol",
            )
            if "emd_pol" in section
            else None
        ),
        mec=(
            _build_component(
                MECConfig,
                section["mec"],
                "polarization.mec",
            )
            if "mec" in section
            else None
        ),
    )


def _build_rag(payload: Any) -> RagConfig:
    section = _require_object(payload, "rag")
    _reject_unknown_keys(section, _RAG_FIELDS, "rag")
    def build_component(
        field_name: str,
        config_type: type,
    ) -> Any:
        if field_name not in section:
            return None
        location = f"rag.{field_name}"
        component_payload = _require_object(section[field_name], location)
        _reject_unknown_keys(
            component_payload,
            {field.name for field in fields(config_type)},
            location,
        )
        return _build_component(
            config_type,
            component_payload,
            location,
        )

    return RagConfig(
        evidence=build_component("evidence", RagEvidenceBuildConfig),
        sidecars=build_component("sidecars", RagSidecarBuildConfig),
        consumer=build_component("consumer", RagConsumerConfig),
        validation=build_component("validation", RagValidationPrepareConfig),
        g1=build_component("g1", RagG1Config),
        g2=build_component("g2", RagG2Config),
        g2_hierarchical=build_component(
            "g2_hierarchical", RagG2HierarchicalConfig
        ),
        daily_sidecars=build_component(
            "daily_sidecars",
            DailyRagSidecarBuildConfig,
        ),
        daily_consumer=build_component(
            "daily_consumer",
            DailyRagConsumerConfig,
        ),
        daily_context_selection=build_component(
            "daily_context_selection",
            DailyContextSelectionConfig,
        ),
        candidate_evidence=build_component(
            "candidate_evidence",
            CandidateEvidencePolicyConfig,
        ),
        candidate_validation=build_component(
            "candidate_validation",
            CandidateRagValidationConfig,
        ),
    )


def run_config_from_mapping(
    payload: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Build a strict RunConfig using typed defaults, payload, then overrides."""

    root = _require_object(payload, "RunConfig")
    if overrides is not None:
        override_payload = _require_object(overrides, "overrides")
        root = _merge_explicit_overrides(root, override_payload)
    _reject_unknown_keys(root, _ROOT_FIELDS, "RunConfig")
    if "identity" not in root:
        raise ValueError("identity is required.")

    identity = _build_identity(root["identity"])
    return RunConfig(
        identity=identity,
        data=_build_data(root["data"]) if "data" in root else None,
        simulation=(
            _build_simulation(root["simulation"])
            if "simulation" in root
            else None
        ),
        signals=_build_signals(root["signals"]) if "signals" in root else None,
        detection=(
            _build_detection(root["detection"])
            if "detection" in root
            else None
        ),
        association=(
            _build_component(
                DetectionAssociationConfig,
                root["association"],
                "association",
            )
            if "association" in root
            else None
        ),
        polarization=(
            _build_polarization(root["polarization"])
            if "polarization" in root
            else None
        ),
        rag=(
            _build_rag(root["rag"])
            if "rag" in root
            else None
        ),
        artifacts=(
            _build_artifacts(root["artifacts"])
            if "artifacts" in root
            else None
        ),
    )


def load_run_config(
    config_file: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load one JSON file and resolve it into the current typed RunConfig model."""

    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Run config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Run config path must be a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in run config {path} at line {exc.lineno}, "
            f"column {exc.colno}."
        ) from exc
    return run_config_from_mapping(payload, overrides=overrides)


__all__ = ["load_run_config", "run_config_from_mapping"]
