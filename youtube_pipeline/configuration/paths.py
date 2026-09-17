from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

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
    RagConfig,
    RunConfig,
    SignalsConfig,
    SimulationConfig,
)
from .serialization import canonical_run_config_json, run_config_hash


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    # Make paths absolute without dereferencing symlinks.  Legacy entrypoints
    # preserve the path spelling supplied by their callers (notably /var on
    # macOS), and canonicalizing it to /private/var changes their I/O contract.
    return path.absolute()


def _resolve_optional_path(
    value: str | Path | None,
    base_dir: Path,
) -> Path | None:
    return None if value is None else _resolve_path(value, base_dir)


def _resolve_youtube_api(
    config: ExtractionConfig,
    base_dir: Path,
) -> ExtractionConfig:
    return replace(
        config,
        data_root=str(_resolve_path(config.data_root, base_dir)),
        metadata_path=(
            None
            if config.metadata_path is None
            else str(_resolve_path(config.metadata_path, base_dir))
        ),
    )


def _resolve_local_files(
    config: LocalFilesConfig,
    base_dir: Path,
) -> LocalFilesConfig:
    return replace(
        config,
        videos_path=_resolve_path(config.videos_path, base_dir),
        comments_path=_resolve_path(config.comments_path, base_dir),
        data_root=_resolve_path(config.data_root, base_dir),
    )


def _resolve_cleaning(
    config: CleaningConfig,
    base_dir: Path,
) -> CleaningConfig:
    return replace(
        config,
        input_path=_resolve_path(config.input_path, base_dir),
        output_path=_resolve_path(config.output_path, base_dir),
    )


def _resolve_prepared_dataset(
    config: PreparedDatasetConfig,
    base_dir: Path,
) -> PreparedDatasetConfig:
    return replace(config, path=_resolve_path(config.path, base_dir))


def _resolve_replay(
    config: ReplayConfig,
    base_dir: Path,
) -> ReplayConfig:
    return replace(
        config,
        output_snapshots=_resolve_path(config.output_snapshots, base_dir),
    )


def _resolve_ingestion(
    config: CyclicIngestionConfig,
    base_dir: Path,
) -> CyclicIngestionConfig:
    return replace(
        config,
        input_path=_resolve_path(config.input_path, base_dir),
        output_dir=_resolve_path(config.output_dir, base_dir),
    )


def _resolve_orchestration(
    config: CyclicOrchestratorConfig,
    base_dir: Path,
) -> CyclicOrchestratorConfig:
    return replace(config, simulation_dir=_resolve_path(config.simulation_dir, base_dir))


def _resolve_stateful_adapter(
    config: CyclicStatefulAdapterConfig,
    base_dir: Path,
) -> CyclicStatefulAdapterConfig:
    return replace(config, simulation_dir=_resolve_path(config.simulation_dir, base_dir))


def _resolve_daily_signals(
    config: CyclicDailySignalConfig,
    base_dir: Path,
) -> CyclicDailySignalConfig:
    return replace(
        config,
        simulation_dir=_resolve_path(config.simulation_dir, base_dir),
        canonical_dataset_path=_resolve_path(config.canonical_dataset_path, base_dir),
        output_dir=_resolve_optional_path(config.output_dir, base_dir),
    )


def _resolve_daily_frequency(
    config: DailyFrequencyBaselineConfig,
    base_dir: Path,
) -> DailyFrequencyBaselineConfig:
    return replace(
        config,
        simulation_dir=_resolve_path(config.simulation_dir, base_dir),
        output_dir=_resolve_optional_path(config.output_dir, base_dir),
    )


def _resolve_detection_connector(
    config: CyclicDetectionConnectorConfig,
    base_dir: Path,
) -> CyclicDetectionConnectorConfig:
    return replace(
        config,
        simulation_dir=_resolve_path(config.simulation_dir, base_dir),
        canonical_dataset_path=_resolve_path(config.canonical_dataset_path, base_dir),
        output_dir=_resolve_optional_path(config.output_dir, base_dir),
    )


def _resolve_daily_rag_sidecars(
    config: DailyRagSidecarBuildConfig,
    base_dir: Path,
) -> DailyRagSidecarBuildConfig:
    return replace(
        config,
        daily_events_path=_resolve_path(config.daily_events_path, base_dir),
        output_dir=_resolve_path(config.output_dir, base_dir),
        comments_path=_resolve_path(config.comments_path, base_dir),
        cycle_window_inventory_path=_resolve_path(
            config.cycle_window_inventory_path,
            base_dir,
        ),
        daily_scores_path=_resolve_optional_path(config.daily_scores_path, base_dir),
        daily_detector_manifest_path=_resolve_optional_path(
            config.daily_detector_manifest_path,
            base_dir,
        ),
        cycle_signal_series_path=_resolve_optional_path(
            config.cycle_signal_series_path,
            base_dir,
        ),
        cycle_stateful_context_path=_resolve_optional_path(
            config.cycle_stateful_context_path,
            base_dir,
        ),
    )


def _resolve_rag_evidence(
    config: RagEvidenceBuildConfig,
    base_dir: Path,
) -> RagEvidenceBuildConfig:
    return replace(
        config,
        trigger_comment_map_path=str(
            _resolve_path(config.trigger_comment_map_path, base_dir)
        ),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
        comments_path=str(_resolve_path(config.comments_path, base_dir)),
        snapshots_path=(
            None
            if config.snapshots_path is None
            else str(_resolve_path(config.snapshots_path, base_dir))
        ),
    )


def _resolve_rag_sidecars(
    config: RagSidecarBuildConfig,
    base_dir: Path,
) -> RagSidecarBuildConfig:
    return replace(
        config,
        trigger_comment_map_path=str(
            _resolve_path(config.trigger_comment_map_path, base_dir)
        ),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
        comments_path=str(_resolve_path(config.comments_path, base_dir)),
        snapshots_path=(
            None
            if config.snapshots_path is None
            else str(_resolve_path(config.snapshots_path, base_dir))
        ),
    )


def _resolve_rag_consumer(
    config: RagConsumerConfig,
    base_dir: Path,
) -> RagConsumerConfig:
    return replace(
        config,
        sidecars_dir=str(_resolve_path(config.sidecars_dir, base_dir)),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
    )


def _resolve_rag_validation(
    config: RagValidationPrepareConfig,
    base_dir: Path,
) -> RagValidationPrepareConfig:
    return replace(
        config,
        evidence_packages_path=str(
            _resolve_path(config.evidence_packages_path, base_dir)
        ),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
    )


def _resolve_rag_g1(config: RagG1Config, base_dir: Path) -> RagG1Config:
    return replace(
        config,
        consumer_dir=str(_resolve_path(config.consumer_dir, base_dir)),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
    )


def _resolve_rag_g2(config: RagG2Config, base_dir: Path) -> RagG2Config:
    return replace(
        config,
        consumer_dir=str(_resolve_path(config.consumer_dir, base_dir)),
        g1_dir=str(_resolve_path(config.g1_dir, base_dir)),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
    )


def _resolve_rag_g2_hierarchical(
    config: RagG2HierarchicalConfig,
    base_dir: Path,
) -> RagG2HierarchicalConfig:
    return replace(
        config,
        consumer_dir=str(_resolve_path(config.consumer_dir, base_dir)),
        output_dir=str(_resolve_path(config.output_dir, base_dir)),
    )


def _resolve_daily_rag_consumer(
    config: DailyRagConsumerConfig,
    base_dir: Path,
) -> DailyRagConsumerConfig:
    config.validate()
    assert config.sidecars_dir is not None
    assert config.output_dir is not None
    return replace(
        config,
        sidecars_dir=_resolve_path(config.sidecars_dir, base_dir),
        output_dir=_resolve_path(config.output_dir, base_dir),
    )


def _resolve_daily_context_selection(
    config: DailyContextSelectionConfig,
    base_dir: Path,
) -> DailyContextSelectionConfig:
    config.validate()
    assert config.consumer_dir is not None
    assert config.sidecars_dir is not None
    assert config.output_dir is not None
    return replace(
        config,
        consumer_dir=_resolve_path(config.consumer_dir, base_dir),
        sidecars_dir=_resolve_path(config.sidecars_dir, base_dir),
        output_dir=_resolve_path(config.output_dir, base_dir),
    )


def resolve_run_config_paths(
    config: RunConfig,
    *,
    base_dir: str | Path,
) -> RunConfig:
    """Return a new RunConfig with known component paths resolved absolutely."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig.")
    base = Path(base_dir).expanduser().resolve(strict=False)

    data = config.data
    if data is not None:
        data = DataConfig(
            youtube_api=(
                _resolve_youtube_api(data.youtube_api, base)
                if data.youtube_api is not None
                else None
            ),
            local_files=(
                _resolve_local_files(data.local_files, base)
                if data.local_files is not None
                else None
            ),
            prepared_dataset=(
                _resolve_prepared_dataset(data.prepared_dataset, base)
                if data.prepared_dataset is not None
                else None
            ),
            cleaning=(
                _resolve_cleaning(data.cleaning, base)
                if data.cleaning is not None
                else None
            ),
        )

    simulation = config.simulation
    if simulation is not None:
        simulation = SimulationConfig(
            ingestion=(
                _resolve_ingestion(simulation.ingestion, base)
                if simulation.ingestion is not None
                else None
            ),
            orchestration=(
                _resolve_orchestration(simulation.orchestration, base)
                if simulation.orchestration is not None
                else None
            ),
            stateful_adapter=(
                _resolve_stateful_adapter(simulation.stateful_adapter, base)
                if simulation.stateful_adapter is not None
                else None
            ),
            replay=(
                _resolve_replay(simulation.replay, base)
                if simulation.replay is not None
                else None
            ),
        )

    signals = config.signals
    if signals is not None:
        signals = SignalsConfig(
            daily=(
                _resolve_daily_signals(signals.daily, base)
                if signals.daily is not None
                else None
            ),
            activity=signals.activity,
            polarization=signals.polarization,
        )

    detection = config.detection
    if detection is not None:
        detection = DetectionConfig(
            activity_route=detection.activity_route,
            connector=(
                _resolve_detection_connector(detection.connector, base)
                if detection.connector is not None
                else None
            ),
            xiao_ema=detection.xiao_ema,
            page_hinkley=detection.page_hinkley,
            adwin=detection.adwin,
            kswin=detection.kswin,
            daily_frequency=(
                _resolve_daily_frequency(detection.daily_frequency, base)
                if detection.daily_frequency is not None
                else None
            )
        )

    polarization = config.polarization

    rag = config.rag
    if rag is not None:
        rag = RagConfig(
            evidence=(
                _resolve_rag_evidence(rag.evidence, base)
                if rag.evidence is not None
                else None
            ),
            sidecars=(
                _resolve_rag_sidecars(rag.sidecars, base)
                if rag.sidecars is not None
                else None
            ),
            consumer=(
                _resolve_rag_consumer(rag.consumer, base)
                if rag.consumer is not None
                else None
            ),
            validation=(
                _resolve_rag_validation(rag.validation, base)
                if rag.validation is not None
                else None
            ),
            g1=(
                _resolve_rag_g1(rag.g1, base)
                if rag.g1 is not None
                else None
            ),
            g2=(
                _resolve_rag_g2(rag.g2, base)
                if rag.g2 is not None
                else None
            ),
            g2_hierarchical=(
                _resolve_rag_g2_hierarchical(rag.g2_hierarchical, base)
                if rag.g2_hierarchical is not None
                else None
            ),
            daily_sidecars=(
                _resolve_daily_rag_sidecars(rag.daily_sidecars, base)
                if rag.daily_sidecars is not None
                else None
            ),
            daily_consumer=(
                _resolve_daily_rag_consumer(rag.daily_consumer, base)
                if rag.daily_consumer is not None
                else None
            ),
            daily_context_selection=(
                _resolve_daily_context_selection(
                    rag.daily_context_selection,
                    base,
                )
                if rag.daily_context_selection is not None
                else None
            ),
        )

    return RunConfig(
        identity=config.identity,
        data=data,
        simulation=simulation,
        signals=signals,
        detection=detection,
        polarization=polarization,
        rag=rag,
        artifacts=config.artifacts,
    )


@dataclass(frozen=True, slots=True)
class ResolvedRunConfig:
    """Execution-ready paths plus stable canonical identity for one RunConfig."""

    config: RunConfig
    canonical_json: str
    config_hash: str


def resolve_run_config(
    config: RunConfig,
    *,
    base_dir: str | Path,
) -> ResolvedRunConfig:
    """Resolve physical paths and derive machine-independent canonical identity."""

    base = Path(base_dir).expanduser().resolve(strict=False)
    resolved = resolve_run_config_paths(config, base_dir=base)
    canonical = canonical_run_config_json(resolved, path_base=base)
    return ResolvedRunConfig(
        config=resolved,
        canonical_json=canonical,
        config_hash=run_config_hash(resolved, path_base=base),
    )


__all__ = ["ResolvedRunConfig", "resolve_run_config", "resolve_run_config_paths"]
