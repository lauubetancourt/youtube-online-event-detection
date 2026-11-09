from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from youtube_pipeline.configuration import (
    ResolvedRunConfig,
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.rag_consumer import (
    RagConsumerConfig,
    derive_rag_consumer_run_id,
)
from youtube_pipeline.rag_evidence import RagEvidenceBuildConfig, make_run_id
from youtube_pipeline.rag_generation_g1 import RagG1Config
from youtube_pipeline.rag_generation_g2 import RagG2Config
from youtube_pipeline.rag_generation_g2_hierarchical import RagG2HierarchicalConfig
from youtube_pipeline.rag_sidecars import (
    CONTEXT_SELECTION_MANIFEST_FILE,
    RagSidecarBuildConfig,
    derive_rag_sidecar_run_id,
)
from youtube_pipeline.rag_validation import (
    RagValidationPrepareConfig,
    derive_rag_validation_run_id,
)


_LEGACY_GLOBAL_RUN_ID = "legacy_non_daily_rag"


@dataclass(frozen=True, slots=True)
class _StageSpec:
    field_name: str
    legacy_section: str
    config_type: type


_STAGES = {
    "evidence": _StageSpec("evidence", "rag_evidence", RagEvidenceBuildConfig),
    "sidecars": _StageSpec("sidecars", "rag_sidecars", RagSidecarBuildConfig),
    "consumer": _StageSpec("consumer", "rag_consumer", RagConsumerConfig),
    "validation": _StageSpec(
        "validation", "rag_validation", RagValidationPrepareConfig
    ),
    "g1": _StageSpec("g1", "rag_generation_g1", RagG1Config),
    "g2": _StageSpec("g2", "rag_generation_g2", RagG2Config),
    "g2_hierarchical": _StageSpec(
        "g2_hierarchical",
        "rag_generation_g2_hierarchical",
        RagG2HierarchicalConfig,
    ),
}


def _read_json_object(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RAG config must be an object: {config_path}")
    return payload


def _merge_explicit(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_explicit(current, value)
        else:
            merged[key] = value
    return merged


def _is_run_profile(payload: dict[str, Any]) -> bool:
    return "identity" in payload or "rag" in payload


def _component(run: Any, field_name: str) -> Any:
    rag = run.rag
    component = None if rag is None else getattr(rag, field_name)
    if component is None:
        raise ValueError(f"RunConfig must include rag.{field_name}.")
    return component


def _with_component(run: Any, field_name: str, component: Any) -> Any:
    if run.rag is None:
        raise ValueError(f"RunConfig must include rag.{field_name}.")
    return replace(run, rag=replace(run.rag, **{field_name: component}))


def _physical_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _preserve_stage_identity(run: Any, field_name: str, base_dir: Path) -> Any:
    component = _component(run, field_name)
    if field_name == "evidence" and component.run_id is None:
        component = replace(
            component,
            run_id=make_run_id(
                detector_name=component.detector_name,
                trigger_comment_map_path=component.trigger_comment_map_path,
                comments_path=component.comments_path,
                snapshots_path=component.snapshots_path,
            ),
        )
    elif field_name == "sidecars" and component.run_id is None:
        component = replace(
            component,
            run_id=derive_rag_sidecar_run_id(component),
        )
    elif field_name == "consumer" and component.run_id is None:
        manifest_path = (
            _physical_path(component.sidecars_dir, base_dir)
            / CONTEXT_SELECTION_MANIFEST_FILE
        )
        if manifest_path.exists():
            manifest = _read_json_object(manifest_path)
            component = replace(
                component,
                run_id=derive_rag_consumer_run_id(component, manifest),
            )
    elif field_name == "validation" and component.validation_run_id is None:
        component = replace(
            component,
            validation_run_id=derive_rag_validation_run_id(component),
        )
    return _with_component(run, field_name, component)


def resolve_non_daily_rag_stage(
    stage: str,
    *,
    config_file: str | Path | None,
    overrides: dict[str, Any] | None,
    base_dir: str | Path,
) -> tuple[ResolvedRunConfig, Any]:
    """Resolve one non-daily RAG stage through the common RunConfig authority."""

    try:
        spec = _STAGES[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported non-daily RAG stage: {stage}") from exc

    base = Path(base_dir).expanduser().resolve(strict=False)
    explicit = dict(overrides or {})
    if stage == "g2_hierarchical" and "max_videos_per_event" in explicit:
        explicit.setdefault(
            "max_videos_per_event_batch",
            explicit["max_videos_per_event"],
        )
        explicit.pop("max_videos_per_event")
    if config_file is not None:
        payload = _read_json_object(config_file)
    else:
        payload = {}

    if _is_run_profile(payload):
        run = load_run_config(
            config_file,
            overrides={"rag": {spec.field_name: explicit}} if explicit else None,
        )
    else:
        legacy = payload.get(spec.legacy_section, payload)
        if not isinstance(legacy, dict):
            raise ValueError(f"{spec.legacy_section} config section must be an object.")
        logical = spec.config_type.from_mapping(_merge_explicit(legacy, explicit))
        run = run_config_from_mapping(
            {
                "identity": {"run_id": _LEGACY_GLOBAL_RUN_ID},
                "rag": {spec.field_name: asdict(logical)},
            }
        )

    run = _preserve_stage_identity(run, spec.field_name, base)
    resolved = resolve_run_config(run, base_dir=base)
    return resolved, _component(resolved.config, spec.field_name)


def resolve_rag_evidence_config(**kwargs: Any) -> tuple[ResolvedRunConfig, RagEvidenceBuildConfig]:
    return resolve_non_daily_rag_stage("evidence", **kwargs)


def resolve_rag_sidecar_config(**kwargs: Any) -> tuple[ResolvedRunConfig, RagSidecarBuildConfig]:
    return resolve_non_daily_rag_stage("sidecars", **kwargs)


def resolve_rag_consumer_config(**kwargs: Any) -> tuple[ResolvedRunConfig, RagConsumerConfig]:
    return resolve_non_daily_rag_stage("consumer", **kwargs)


def resolve_rag_validation_config(**kwargs: Any) -> tuple[ResolvedRunConfig, RagValidationPrepareConfig]:
    return resolve_non_daily_rag_stage("validation", **kwargs)


def resolve_rag_g1_config(**kwargs: Any) -> tuple[ResolvedRunConfig, RagG1Config]:
    return resolve_non_daily_rag_stage("g1", **kwargs)


def resolve_rag_g2_config(**kwargs: Any) -> tuple[ResolvedRunConfig, RagG2Config]:
    return resolve_non_daily_rag_stage("g2", **kwargs)


def resolve_rag_g2_hierarchical_config(
    **kwargs: Any,
) -> tuple[ResolvedRunConfig, RagG2HierarchicalConfig]:
    return resolve_non_daily_rag_stage("g2_hierarchical", **kwargs)


__all__ = [
    "resolve_non_daily_rag_stage",
    "resolve_rag_consumer_config",
    "resolve_rag_evidence_config",
    "resolve_rag_g1_config",
    "resolve_rag_g2_config",
    "resolve_rag_g2_hierarchical_config",
    "resolve_rag_sidecar_config",
    "resolve_rag_validation_config",
]
