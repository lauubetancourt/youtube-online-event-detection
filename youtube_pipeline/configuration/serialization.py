from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
from youtube_pipeline.prepared_replay import PreparedDatasetConfig, ReplayConfig
from youtube_pipeline.rag_consumer import RagConsumerConfig
from youtube_pipeline.rag_evidence import RagEvidenceBuildConfig
from youtube_pipeline.rag_generation_g1 import RagG1Config
from youtube_pipeline.rag_generation_g2 import RagG2Config
from youtube_pipeline.rag_generation_g2_hierarchical import RagG2HierarchicalConfig
from youtube_pipeline.rag_sidecars import RagSidecarBuildConfig
from youtube_pipeline.rag_validation import RagValidationPrepareConfig
from youtube_pipeline.storage import LocalFilesConfig

from .models import (
    DataConfig,
    DetectionConfig,
    PolarizationConfig,
    RunConfig,
    SignalsConfig,
    SimulationConfig,
)

_PATH_FIELDS_BY_TYPE: dict[type, frozenset[str]] = {
    CleaningConfig: frozenset({"input_path", "output_path"}),
    PreparedDatasetConfig: frozenset({"path"}),
    ReplayConfig: frozenset({"output_snapshots"}),
    ExtractionConfig: frozenset({"data_root", "metadata_path"}),
    LocalFilesConfig: frozenset(
        {"videos_path", "comments_path", "data_root"}
    ),
    CyclicIngestionConfig: frozenset({"input_path", "output_dir"}),
    CyclicOrchestratorConfig: frozenset({"simulation_dir"}),
    CyclicStatefulAdapterConfig: frozenset({"simulation_dir"}),
    CyclicDailySignalConfig: frozenset(
        {"simulation_dir", "canonical_dataset_path", "output_dir"}
    ),
    CyclicDetectionConnectorConfig: frozenset(
        {"simulation_dir", "canonical_dataset_path", "output_dir"}
    ),
    DailyFrequencyBaselineConfig: frozenset({"simulation_dir", "output_dir"}),
    DailyRagSidecarBuildConfig: frozenset(
        {
            "daily_events_path",
            "output_dir",
            "comments_path",
            "cycle_window_inventory_path",
            "daily_scores_path",
            "daily_detector_manifest_path",
            "cycle_signal_series_path",
            "cycle_stateful_context_path",
        }
    ),
    DailyRagConsumerConfig: frozenset({"sidecars_dir", "output_dir"}),
    DailyContextSelectionConfig: frozenset(
        {"consumer_dir", "sidecars_dir", "output_dir"}
    ),
    RagEvidenceBuildConfig: frozenset(
        {
            "trigger_comment_map_path",
            "output_dir",
            "comments_path",
            "snapshots_path",
        }
    ),
    RagSidecarBuildConfig: frozenset(
        {
            "trigger_comment_map_path",
            "output_dir",
            "comments_path",
            "snapshots_path",
        }
    ),
    RagConsumerConfig: frozenset({"sidecars_dir", "output_dir"}),
    RagValidationPrepareConfig: frozenset(
        {"evidence_packages_path", "output_dir"}
    ),
    RagG1Config: frozenset({"consumer_dir", "output_dir"}),
    RagG2Config: frozenset({"consumer_dir", "g1_dir", "output_dir"}),
    RagG2HierarchicalConfig: frozenset({"consumer_dir", "output_dir"}),
}

_OMITTED_NONE_FIELDS_BY_TYPE: dict[type, frozenset[str]] = {
    DataConfig: frozenset({"prepared_dataset", "experimental_input"}),
    DetectionConfig: frozenset(
        {"activity_route", "routes", "gap_policy", "xiao_ema", "page_hinkley", "adwin", "kswin"}
    ),
    PolarizationConfig: frozenset({"measure_id", "esteban_ray", "emd_pol", "mec", "classifier", "runtime"}),
    RunConfig: frozenset({"data", "polarization", "rag", "artifacts"}),
    SignalsConfig: frozenset({"daily", "activity", "polarization"}),
    SimulationConfig: frozenset({"replay"}),
}


def _normalized_logical_path(value: str | Path, path_base: Path | None) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() and path_base is not None:
        try:
            path = path.relative_to(path_base)
        except ValueError:
            try:
                path = path.resolve(strict=False).relative_to(
                    path_base.resolve(strict=False)
                )
            except ValueError:
                pass
    normalized = os.path.normpath(str(path))
    return Path(normalized).as_posix()


def _to_primitive(
    value: Any,
    *,
    path_base: Path | None,
    path_field: bool = False,
) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if path_field and value is not None:
            return _normalized_logical_path(value, path_base)
        return value
    if isinstance(value, Path):
        return _normalized_logical_path(value, path_base)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        path_fields = _PATH_FIELDS_BY_TYPE.get(type(value), frozenset())
        omitted_none_fields = _OMITTED_NONE_FIELDS_BY_TYPE.get(
            type(value),
            frozenset(),
        )
        return {
            field.name: _to_primitive(
                getattr(value, field.name),
                path_base=path_base,
                path_field=field.name in path_fields,
            )
            for field in fields(value)
            if not (
                field.name in omitted_none_fields
                and getattr(value, field.name) is None
            )
        }
    if isinstance(value, dict):
        return {
            str(key): _to_primitive(item, path_base=path_base)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item, path_base=path_base) for item in value]
    raise TypeError(f"Unsupported configuration value type: {type(value).__name__}")


def run_config_to_mapping(
    config: RunConfig,
    *,
    path_base: str | Path | None = None,
) -> dict[str, Any]:
    """Return all effective values as JSON-compatible primitives."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig.")
    normalized_base = (
        Path(path_base).expanduser().resolve(strict=False)
        if path_base is not None
        else None
    )
    result = _to_primitive(config, path_base=normalized_base)
    if not isinstance(result, dict):
        raise TypeError("RunConfig serialization must produce an object.")
    return result


def canonical_run_config_json(
    config: RunConfig,
    *,
    path_base: str | Path | None = None,
) -> str:
    """Serialize effective configuration deterministically without machine paths."""

    return json.dumps(
        run_config_to_mapping(config, path_base=path_base),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def run_config_hash(
    config: RunConfig,
    *,
    path_base: str | Path | None = None,
) -> str:
    """Return the SHA-256 digest of the canonical effective configuration."""

    canonical = canonical_run_config_json(config, path_base=path_base)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "canonical_run_config_json",
    "run_config_hash",
    "run_config_to_mapping",
]
