from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from youtube_pipeline.activity_detection import ActivityDetectionRouteConfig
from youtube_pipeline.activity_signals import ActivitySignalDefinition
from youtube_pipeline.stance_classifier import CommentStanceClassifierConfig
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
from youtube_pipeline.detectors import (
    ADWINConfig,
    KSWINConfig,
    PageHinkleyConfig,
    XiaoEMAConfig,
    get_detector_names,
)
from youtube_pipeline.detection_alerts import DetectionGapPolicyConfig
from youtube_pipeline.detection_association import DetectionAssociationConfig
from youtube_pipeline.candidate_evidence import CandidateEvidencePolicyConfig
from youtube_pipeline.candidate_rag import CandidateRagValidationConfig
from youtube_pipeline.experimental_input import ExperimentalInputPolicyConfig
from youtube_pipeline.prepared_replay import PreparedDatasetConfig, ReplayConfig
from youtube_pipeline.polarization_signals import PolarizationTemporalSignalConfig
from youtube_pipeline.polarization_runtime_config import PolarizationRuntimeConfig
from youtube_pipeline.polarization_measures import (
    EMD_POL_MEASURE,
    ESTEBAN_RAY_MEASURE,
    MEC_MEASURE,
    EMDPolConfig,
    EstebanRayConfig,
    MECConfig,
    get_polarization_measure_names,
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


RUN_MODE_DEFAULT_TRACE_LEVEL = {
    "development": "minimal",
    "reference": "standard",
    "official": "full",
}
TRACE_LEVEL_ORDER = {
    "minimal": 0,
    "standard": 1,
    "full": 2,
}


def _require_optional_instance(
    field_name: str,
    value: Any,
    expected_type: type,
) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(
            f"{field_name} must be {expected_type.__name__} or None, "
            f"not {type(value).__name__}."
        )


@dataclass(frozen=True, slots=True)
class RunIdentityConfig:
    """Stable identity supplied by the caller for one configured execution."""

    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str):
            raise TypeError("run_id must be a string.")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty.")


@dataclass(frozen=True, slots=True)
class ArtifactsConfig:
    """Execution-level traceability policy without component artifact details."""

    run_mode: str = "development"
    trace_level: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_mode, str):
            raise TypeError("run_mode must be a string.")
        if self.run_mode not in RUN_MODE_DEFAULT_TRACE_LEVEL:
            supported = ", ".join(RUN_MODE_DEFAULT_TRACE_LEVEL)
            raise ValueError(
                f"Unsupported run_mode {self.run_mode!r}; expected one of: "
                f"{supported}."
            )
        level = self.trace_level
        if level is None:
            level = RUN_MODE_DEFAULT_TRACE_LEVEL[self.run_mode]
            object.__setattr__(self, "trace_level", level)
        if not isinstance(level, str):
            raise TypeError("trace_level must be a string or None.")
        if level not in TRACE_LEVEL_ORDER:
            supported = ", ".join(TRACE_LEVEL_ORDER)
            raise ValueError(
                f"Unsupported trace_level {level!r}; expected one of: {supported}."
            )
        minimum = RUN_MODE_DEFAULT_TRACE_LEVEL[self.run_mode]
        if TRACE_LEVEL_ORDER[level] < TRACE_LEVEL_ORDER[minimum]:
            raise ValueError(
                f"run_mode={self.run_mode!r} requires trace_level "
                f"{minimum!r} or higher."
            )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Composition of implemented data-source configurations."""

    youtube_api: ExtractionConfig | None = None
    local_files: LocalFilesConfig | None = None
    prepared_dataset: PreparedDatasetConfig | None = None
    cleaning: CleaningConfig | None = None
    experimental_input: ExperimentalInputPolicyConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_instance("youtube_api", self.youtube_api, ExtractionConfig)
        _require_optional_instance("local_files", self.local_files, LocalFilesConfig)
        _require_optional_instance(
            "prepared_dataset",
            self.prepared_dataset,
            PreparedDatasetConfig,
        )
        _require_optional_instance("cleaning", self.cleaning, CleaningConfig)
        _require_optional_instance(
            "experimental_input",
            self.experimental_input,
            ExperimentalInputPolicyConfig,
        )
        configured_sources = sum(
            source is not None
            for source in (
                self.youtube_api,
                self.local_files,
                self.prepared_dataset,
            )
        )
        if configured_sources > 1:
            raise ValueError("DataConfig must configure exactly one data source.")
        if configured_sources == 0 and self.cleaning is None:
            raise ValueError(
                "DataConfig must configure exactly one data source or at least "
                "one preparation stage."
            )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Composition of the existing cyclic simulation stage configurations."""

    ingestion: CyclicIngestionConfig | None = None
    orchestration: CyclicOrchestratorConfig | None = None
    stateful_adapter: CyclicStatefulAdapterConfig | None = None
    replay: ReplayConfig | None = None
    causal_end_to_end: CausalEndToEndReplayConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_instance("ingestion", self.ingestion, CyclicIngestionConfig)
        _require_optional_instance(
            "orchestration",
            self.orchestration,
            CyclicOrchestratorConfig,
        )
        _require_optional_instance(
            "stateful_adapter",
            self.stateful_adapter,
            CyclicStatefulAdapterConfig,
        )
        _require_optional_instance("replay", self.replay, ReplayConfig)
        _require_optional_instance(
            "causal_end_to_end",
            self.causal_end_to_end,
            CausalEndToEndReplayConfig,
        )
        if all(
            config is None
            for config in (
                self.ingestion,
                self.orchestration,
                self.stateful_adapter,
                self.replay,
                self.causal_end_to_end,
            )
        ):
            raise ValueError("SimulationConfig must configure at least one stage.")


@dataclass(frozen=True, slots=True)
class SignalsConfig:
    """Composition of implemented signal configurations."""

    daily: CyclicDailySignalConfig | None = None
    activity: ActivitySignalDefinition | None = None
    polarization: PolarizationTemporalSignalConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_instance("daily", self.daily, CyclicDailySignalConfig)
        _require_optional_instance(
            "activity",
            self.activity,
            ActivitySignalDefinition,
        )
        _require_optional_instance("polarization", self.polarization, PolarizationTemporalSignalConfig)
        if self.daily is None and self.activity is None and self.polarization is None:
            raise ValueError("SignalsConfig must configure at least one signal stage.")


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Composition of implemented detector configurations."""

    activity_route: ActivityDetectionRouteConfig | None = None
    routes: tuple[ActivityDetectionRouteConfig, ...] | None = None
    gap_policy: DetectionGapPolicyConfig | None = None
    connector: CyclicDetectionConnectorConfig | None = None
    xiao_ema: XiaoEMAConfig | None = None
    page_hinkley: PageHinkleyConfig | None = None
    adwin: ADWINConfig | None = None
    daily_frequency: DailyFrequencyBaselineConfig | None = None
    kswin: KSWINConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_instance(
            "activity_route",
            self.activity_route,
            ActivityDetectionRouteConfig,
        )
        if self.routes is not None:
            if not isinstance(self.routes, (list, tuple)):
                raise TypeError("routes must be a list or tuple of signal routes.")
            routes = tuple(self.routes)
            if not routes:
                raise ValueError("routes must be nonempty when provided.")
            if any(not isinstance(route, ActivityDetectionRouteConfig) for route in routes):
                raise TypeError("routes must contain ActivityDetectionRouteConfig values.")
            if len({route.signal_id for route in routes}) != len(routes):
                raise ValueError("routes must contain unique signal_id values.")
            object.__setattr__(self, "routes", routes)
        _require_optional_instance(
            "gap_policy", self.gap_policy, DetectionGapPolicyConfig
        )
        _require_optional_instance(
            "connector",
            self.connector,
            CyclicDetectionConnectorConfig,
        )
        _require_optional_instance(
            "xiao_ema",
            self.xiao_ema,
            XiaoEMAConfig,
        )
        _require_optional_instance(
            "page_hinkley",
            self.page_hinkley,
            PageHinkleyConfig,
        )
        _require_optional_instance("adwin", self.adwin, ADWINConfig)
        _require_optional_instance("kswin", self.kswin, KSWINConfig)
        _require_optional_instance(
            "daily_frequency",
            self.daily_frequency,
            DailyFrequencyBaselineConfig,
        )
        configured_routes = (
            ((self.activity_route,) if self.activity_route is not None else ())
            + (self.routes or ())
        )
        available_detectors = get_detector_names()
        configured_detectors = set()
        if self.xiao_ema is not None:
            configured_detectors.add("xiao_ema")
        if self.page_hinkley is not None:
            configured_detectors.add("page_hinkley")
        if self.adwin is not None:
            configured_detectors.add("adwin")
        if self.kswin is not None:
            configured_detectors.add("kswin")
        for route in configured_routes:
            if route.detector_id not in available_detectors:
                available = ", ".join(available_detectors)
                route_kind = (
                    "activity" if route is self.activity_route else "signal"
                )
                raise ValueError(
                    f"Unknown {route_kind} detector {route.detector_id!r}; "
                    f"available detectors: {available}."
                )
            if route.detector_id not in configured_detectors:
                raise ValueError(
                    "Each configured route requires a matching detector "
                    "configuration."
                )
        if (
            self.connector is None
            and self.xiao_ema is None
            and self.page_hinkley is None
            and self.adwin is None
            and self.kswin is None
            and self.daily_frequency is None
        ):
            raise ValueError("DetectionConfig must configure at least one detector.")


@dataclass(frozen=True, slots=True)
class PolarizationConfig:
    """Typed parameters; optional isolated selector, separate from temporal activation."""

    measure_id: str | None = None
    esteban_ray: EstebanRayConfig | None = None
    emd_pol: EMDPolConfig | None = None
    mec: MECConfig | None = None
    classifier: CommentStanceClassifierConfig | None = None
    runtime: PolarizationRuntimeConfig | None = None

    def __post_init__(self) -> None:
        if self.measure_id is not None and not isinstance(self.measure_id, str):
            raise TypeError("measure_id must be a string.")
        available_measures = get_polarization_measure_names()
        if self.measure_id is not None and self.measure_id not in available_measures:
            available = ", ".join(available_measures)
            raise ValueError(
                f"Unknown polarization measure {self.measure_id!r}; available "
                f"measures: {available}."
            )
        _require_optional_instance(
            "esteban_ray",
            self.esteban_ray,
            EstebanRayConfig,
        )
        _require_optional_instance("emd_pol", self.emd_pol, EMDPolConfig)
        _require_optional_instance("mec", self.mec, MECConfig)
        _require_optional_instance("classifier", self.classifier, CommentStanceClassifierConfig)
        _require_optional_instance("runtime", self.runtime, PolarizationRuntimeConfig)
        configurations = {
            ESTEBAN_RAY_MEASURE: self.esteban_ray,
            EMD_POL_MEASURE: self.emd_pol,
            MEC_MEASURE: self.mec,
        }
        if self.measure_id is None and self.classifier is None and self.runtime is None and not any(value is not None for value in configurations.values()):
            raise ValueError("PolarizationConfig requires measure parameter blocks.")
        if self.measure_id is not None and configurations[self.measure_id] is None:
            raise ValueError(
                "The selected polarization measure requires a matching "
                "configuration block."
            )

    @property
    def selected_config(self) -> EstebanRayConfig | EMDPolConfig | MECConfig:
        if self.measure_id is None:
            raise ValueError("An isolated measure_id is required for selected_config.")
        return self.config_for(self.measure_id)

    def config_for(self, measure_id: str) -> EstebanRayConfig | EMDPolConfig | MECConfig:
        if measure_id not in get_polarization_measure_names():
            raise ValueError(f"Unknown polarization measure {measure_id!r}.")
        selected = {
            ESTEBAN_RAY_MEASURE: self.esteban_ray,
            EMD_POL_MEASURE: self.emd_pol,
            MEC_MEASURE: self.mec,
        }[measure_id]
        if selected is None:
            raise ValueError(f"Enabled measure {measure_id!r} requires a matching configuration block.")
        return selected


@dataclass(frozen=True, slots=True)
class RagConfig:
    """Composition of implemented RAG-stage configurations."""

    evidence: RagEvidenceBuildConfig | None = None
    sidecars: RagSidecarBuildConfig | None = None
    consumer: RagConsumerConfig | None = None
    validation: RagValidationPrepareConfig | None = None
    g1: RagG1Config | None = None
    g2: RagG2Config | None = None
    g2_hierarchical: RagG2HierarchicalConfig | None = None
    daily_sidecars: DailyRagSidecarBuildConfig | None = None
    daily_consumer: DailyRagConsumerConfig | None = None
    daily_context_selection: DailyContextSelectionConfig | None = None
    candidate_evidence: CandidateEvidencePolicyConfig | None = None
    candidate_validation: CandidateRagValidationConfig | None = None

    def __post_init__(self) -> None:
        _require_optional_instance("evidence", self.evidence, RagEvidenceBuildConfig)
        _require_optional_instance("sidecars", self.sidecars, RagSidecarBuildConfig)
        _require_optional_instance("consumer", self.consumer, RagConsumerConfig)
        _require_optional_instance(
            "validation", self.validation, RagValidationPrepareConfig
        )
        _require_optional_instance("g1", self.g1, RagG1Config)
        _require_optional_instance("g2", self.g2, RagG2Config)
        _require_optional_instance(
            "g2_hierarchical", self.g2_hierarchical, RagG2HierarchicalConfig
        )
        _require_optional_instance(
            "daily_sidecars",
            self.daily_sidecars,
            DailyRagSidecarBuildConfig,
        )
        _require_optional_instance(
            "daily_consumer",
            self.daily_consumer,
            DailyRagConsumerConfig,
        )
        _require_optional_instance(
            "daily_context_selection",
            self.daily_context_selection,
            DailyContextSelectionConfig,
        )
        _require_optional_instance(
            "candidate_evidence",
            self.candidate_evidence,
            CandidateEvidencePolicyConfig,
        )
        _require_optional_instance(
            "candidate_validation",
            self.candidate_validation,
            CandidateRagValidationConfig,
        )
        if all(
            config is None
            for config in (
                self.evidence,
                self.sidecars,
                self.consumer,
                self.validation,
                self.g1,
                self.g2,
                self.g2_hierarchical,
                self.daily_sidecars,
                self.daily_consumer,
                self.daily_context_selection,
                self.candidate_evidence,
                self.candidate_validation,
            )
        ):
            raise ValueError("RagConfig must configure at least one RAG stage.")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Typed composition root for one execution without duplicating component fields."""

    identity: RunIdentityConfig
    data: DataConfig | None = None
    simulation: SimulationConfig | None = None
    signals: SignalsConfig | None = None
    detection: DetectionConfig | None = None
    polarization: PolarizationConfig | None = None
    rag: RagConfig | None = None
    artifacts: ArtifactsConfig | None = None
    association: DetectionAssociationConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentityConfig):
            raise TypeError("identity must be RunIdentityConfig.")
        _require_optional_instance("data", self.data, DataConfig)
        _require_optional_instance("simulation", self.simulation, SimulationConfig)
        _require_optional_instance("signals", self.signals, SignalsConfig)
        _require_optional_instance("detection", self.detection, DetectionConfig)
        _require_optional_instance(
            "association", self.association, DetectionAssociationConfig
        )
        _require_optional_instance(
            "polarization",
            self.polarization,
            PolarizationConfig,
        )
        _require_optional_instance("rag", self.rag, RagConfig)
        _require_optional_instance("artifacts", self.artifacts, ArtifactsConfig)
        temporal = self.signals.polarization if self.signals is not None else None
        if temporal is not None:
            if self.polarization is None:
                raise ValueError("Temporal polarization requires measure parameter configuration.")
            for measure_id in temporal.enabled_measures:
                self.polarization.config_for(measure_id)
        elif self.polarization is not None and self.polarization.measure_id is None and self.polarization.classifier is None:
            raise ValueError("polarization.measure_id is required for isolated measures.")
        runtime = self.polarization.runtime if self.polarization is not None else None
        if runtime is not None:
            if temporal is None:
                raise ValueError("Polarization runtime requires signals.polarization.")
            if self.polarization.classifier is None:
                raise ValueError("Polarization runtime requires polarization.classifier.")
            if self.data is None or self.data.local_files is None:
                raise ValueError("Polarization runtime requires data.local_files.")
            if temporal.proposition_id != self.polarization.classifier.proposition.proposition_id:
                raise ValueError("Temporal signal and classifier proposition_id must match.")
            if temporal.classification_run_id != self.identity.run_id:
                raise ValueError("classification_run_id must equal identity.run_id for an integrated runtime.")
        causal_replay = (
            self.simulation.causal_end_to_end
            if self.simulation is not None
            else None
        )
        if causal_replay is not None:
            if (
                self.data is None
                or self.data.local_files is None
                or self.data.experimental_input is None
            ):
                raise ValueError(
                    "Causal end-to-end replay requires local Silver files and "
                    "data.experimental_input."
                )
            if self.signals is None or self.signals.activity is None or temporal is None:
                raise ValueError(
                    "Causal end-to-end replay requires activity and polarization signals."
                )
            if self.detection is None or not (
                self.detection.routes or self.detection.activity_route
            ):
                raise ValueError("Causal end-to-end replay requires detector routes.")
            if self.association is None:
                raise ValueError("Causal end-to-end replay requires association config.")
            if (
                self.polarization is None
                or self.polarization.classifier is None
                or self.polarization.runtime is None
            ):
                raise ValueError(
                    "Causal end-to-end replay requires classifier and runtime lineage config."
                )
            if (
                self.rag is None
                or self.rag.candidate_evidence is None
                or self.rag.candidate_validation is None
            ):
                raise ValueError(
                    "Causal end-to-end replay requires candidate evidence and RAG config."
                )
            if temporal.classification_run_id != self.identity.run_id:
                raise ValueError(
                    "Causal replay classification_run_id must equal identity.run_id."
                )
            if temporal.proposition_id != self.polarization.classifier.proposition.proposition_id:
                raise ValueError(
                    "Causal replay temporal signal and classifier proposition must match."
                )
        routes = self.detection.routes if self.detection is not None else None
        if routes is not None:
            polarization_routes = tuple(
                route for route in routes
                if route.signal_id.startswith("polarization_")
            )
            if polarization_routes and temporal is None:
                raise ValueError(
                    "Polarization detection routes require signals.polarization."
                )
            if temporal is not None:
                expected = {
                    f"polarization_{measure_id}"
                    for measure_id in temporal.enabled_measures
                }
                unknown = sorted(
                    route.signal_id for route in polarization_routes
                    if route.signal_id not in expected
                )
                if unknown:
                    raise ValueError(
                        "Polarization detection routes reference disabled signals: "
                        f"{unknown}."
                    )
        if all(
            section is None
            for section in (
                self.data,
                self.simulation,
                self.signals,
                self.detection,
                self.association,
                self.polarization,
                self.rag,
            )
        ):
            raise ValueError("RunConfig must configure at least one execution section.")
