"""Typed composition models for pipeline run configuration."""

from youtube_pipeline.activity_detection import (
    ActivityDetectionRouteConfig,
    DetectionResult,
    SignalDetectionRouteConfig,
)
from youtube_pipeline.detectors import ADWINConfig, KSWINConfig, PageHinkleyConfig, XiaoEMAConfig
from youtube_pipeline.detection_alerts import DetectionAlert, DetectionGapPolicyConfig
from youtube_pipeline.experimental_input import ExperimentalInputPolicyConfig
from youtube_pipeline.stance_classifier import (
    CommentStanceClassifierConfig, ClassifierRetryPolicy, StanceProposition,
)
from youtube_pipeline.polarization_signals import PolarizationTemporalSignalConfig
from youtube_pipeline.polarization_measures import (
    EMDPolConfig,
    EstebanRayConfig,
    MECConfig,
)

from .loading import load_run_config, run_config_from_mapping
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
from .paths import ResolvedRunConfig, resolve_run_config, resolve_run_config_paths
from .serialization import (
    canonical_run_config_json,
    run_config_hash,
    run_config_to_mapping,
)

__all__ = [
    "ADWINConfig",
    "KSWINConfig",
    "CommentStanceClassifierConfig",
    "ClassifierRetryPolicy",
    "StanceProposition",
    "ActivityDetectionRouteConfig",
    "SignalDetectionRouteConfig",
    "ArtifactsConfig",
    "canonical_run_config_json",
    "DataConfig",
    "DetectionConfig",
    "DetectionAlert",
    "DetectionGapPolicyConfig",
    "EMDPolConfig",
    "ExperimentalInputPolicyConfig",
    "EstebanRayConfig",
    "DetectionResult",
    "load_run_config",
    "PageHinkleyConfig",
    "MECConfig",
    "PolarizationConfig",
    "PolarizationTemporalSignalConfig",
    "RagConfig",
    "ResolvedRunConfig",
    "resolve_run_config",
    "resolve_run_config_paths",
    "RunConfig",
    "run_config_hash",
    "RunIdentityConfig",
    "run_config_from_mapping",
    "run_config_to_mapping",
    "SignalsConfig",
    "SimulationConfig",
    "XiaoEMAConfig",
]
