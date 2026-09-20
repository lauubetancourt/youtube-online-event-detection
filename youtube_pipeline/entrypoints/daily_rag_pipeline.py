from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from youtube_pipeline.configuration import (
    RagConfig,
    ResolvedRunConfig,
    RunConfig,
    load_run_config,
    resolve_run_config,
)
from youtube_pipeline.daily_rag_consumer import (
    write_daily_rag_consumer_artifacts_from_config,
)
from youtube_pipeline.daily_rag_context_selection import (
    write_daily_context_selection_artifacts_from_config,
)
from youtube_pipeline.daily_rag_sidecars import (
    write_daily_rag_sidecar_artifacts_from_config,
)
from youtube_pipeline.entrypoints.common_cli import (
    CommonRunCliOptions,
    parse_common_run_args,
)
from youtube_pipeline.run_manifest import (
    validate_current_traceability_support,
    write_run_manifest,
)


DAILY_RAG_RUN_MANIFEST_FILE = "daily_rag_run_manifest.json"


def _resolved_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base_dir / value
    return value.resolve(strict=False)


def _validate_daily_rag_profile(config: RunConfig) -> None:
    rag = config.rag
    missing: list[str] = []
    if rag is None or rag.daily_sidecars is None:
        missing.append("rag.daily_sidecars")
    if rag is None or rag.daily_consumer is None:
        missing.append("rag.daily_consumer")
    if rag is None or rag.daily_context_selection is None:
        missing.append("rag.daily_context_selection")
    if missing:
        raise ValueError(
            "Daily RAG pipeline profile is missing required sections: "
            + ", ".join(missing)
        )

    assert rag is not None
    assert rag.daily_sidecars is not None
    assert rag.daily_consumer is not None
    assert rag.daily_context_selection is not None
    sidecars = rag.daily_sidecars
    consumer = rag.daily_consumer
    selection = rag.daily_context_selection
    sidecars.validate()
    consumer.validate()
    selection.validate()

    expected_links = {
        "rag.daily_consumer.sidecars_dir": (
            consumer.sidecars_dir,
            sidecars.output_dir,
        ),
        "rag.daily_context_selection.sidecars_dir": (
            selection.sidecars_dir,
            sidecars.output_dir,
        ),
        "rag.daily_context_selection.consumer_dir": (
            selection.consumer_dir,
            consumer.output_dir,
        ),
    }
    mismatched = [
        name
        for name, (actual, expected) in expected_links.items()
        if Path(actual).resolve(strict=False) != Path(expected).resolve(strict=False)
    ]
    if mismatched:
        raise ValueError(
            "Daily RAG stage paths do not form one chain: " + ", ".join(mismatched)
        )


def _with_output_root(config: RunConfig, *, output_root: Path) -> RunConfig:
    """Relocate only outputs and their downstream links, preserving stage IDs."""

    _validate_daily_rag_profile(config)
    assert config.rag is not None
    assert config.rag.daily_sidecars is not None
    assert config.rag.daily_consumer is not None
    assert config.rag.daily_context_selection is not None
    sidecars = config.rag.daily_sidecars
    consumer = config.rag.daily_consumer
    selection = config.rag.daily_context_selection

    output_dirs = {
        "rag.daily_sidecars.output_dir": Path(sidecars.output_dir).resolve(strict=False),
        "rag.daily_consumer.output_dir": Path(consumer.output_dir).resolve(strict=False),
        "rag.daily_context_selection.output_dir": Path(selection.output_dir).resolve(
            strict=False
        ),
    }
    previous_root = output_dirs["rag.daily_sidecars.output_dir"].parent
    outside_root = [
        name
        for name, path in output_dirs.items()
        if path.parent != previous_root
    ]
    if outside_root:
        raise ValueError(
            "--output-root requires sibling daily RAG output directories; "
            "mismatched fields: " + ", ".join(outside_root)
        )

    relocated_sidecars_dir = output_root / output_dirs[
        "rag.daily_sidecars.output_dir"
    ].name
    relocated_consumer_dir = output_root / output_dirs[
        "rag.daily_consumer.output_dir"
    ].name
    relocated_selection_dir = output_root / output_dirs[
        "rag.daily_context_selection.output_dir"
    ].name
    relocated_rag = RagConfig(
        daily_sidecars=replace(sidecars, output_dir=relocated_sidecars_dir),
        daily_consumer=replace(
            consumer,
            sidecars_dir=relocated_sidecars_dir,
            output_dir=relocated_consumer_dir,
        ),
        daily_context_selection=replace(
            selection,
            consumer_dir=relocated_consumer_dir,
            sidecars_dir=relocated_sidecars_dir,
            output_dir=relocated_selection_dir,
        ),
    )
    return RunConfig(
        identity=config.identity,
        data=config.data,
        simulation=config.simulation,
        signals=config.signals,
        detection=config.detection,
        association=config.association,
        rag=relocated_rag,
        artifacts=config.artifacts,
    )


def resolve_daily_rag_pipeline_run(
    options: CommonRunCliOptions,
    *,
    base_dir: str | Path,
) -> ResolvedRunConfig:
    """Resolve one guarded, non-generative daily RAG execution."""

    if options.execution_mode == "execute":
        raise ValueError(
            "The current daily RAG pipeline supports only --dry-run; "
            "generative or external execution is not implemented."
        )
    base = Path(base_dir).expanduser().resolve(strict=False)
    loaded = load_run_config(
        options.config_path,
        overrides=options.identity_overrides(),
    )
    resolved = resolve_run_config(loaded, base_dir=base)
    _validate_daily_rag_profile(resolved.config)
    if options.output_root is not None:
        effective = _with_output_root(
            resolved.config,
            output_root=_resolved_path(options.output_root, base_dir=base),
        )
        resolved = resolve_run_config(effective, base_dir=base)
        _validate_daily_rag_profile(resolved.config)
    return resolved


def run_daily_rag_pipeline(resolved: ResolvedRunConfig) -> dict[str, Any]:
    """Run sidecars, consumer, and deterministic selection without external calls."""

    validate_current_traceability_support(resolved)
    config = resolved.config
    _validate_daily_rag_profile(config)
    assert config.rag is not None
    assert config.rag.daily_sidecars is not None
    assert config.rag.daily_consumer is not None
    assert config.rag.daily_context_selection is not None

    stages = {
        "sidecars": write_daily_rag_sidecar_artifacts_from_config(
            config.rag.daily_sidecars
        ),
        "consumer": write_daily_rag_consumer_artifacts_from_config(
            config.rag.daily_consumer
        ),
        "context_selection": write_daily_context_selection_artifacts_from_config(
            config.rag.daily_context_selection
        ),
    }
    run_manifest = write_run_manifest(
        resolved,
        output_dir=Path(config.rag.daily_sidecars.output_dir).parent,
        execution_mode="dry_run",
        completed_stages=tuple(stages),
        filename=DAILY_RAG_RUN_MANIFEST_FILE,
    )
    return {
        "run_id": config.identity.run_id,
        "config_hash": resolved.config_hash,
        "execution_mode": "dry_run",
        "run_manifest": str(run_manifest),
        "stages": stages,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> None:
    options = parse_common_run_args(
        argv,
        description=(
            "Run daily RAG sidecars, non-generative consumer preparation, and "
            "deterministic context selection."
        ),
        prog="daily-rag-pipeline",
    )
    logging.basicConfig(level=getattr(logging, options.log_level))
    try:
        resolved = resolve_daily_rag_pipeline_run(
            options,
            base_dir=Path.cwd() if base_dir is None else base_dir,
        )
        summary = run_daily_rag_pipeline(resolved)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(f"daily-rag-pipeline: error: {exc}") from exc
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "DAILY_RAG_RUN_MANIFEST_FILE",
    "main",
    "resolve_daily_rag_pipeline_run",
    "run_daily_rag_pipeline",
]


if __name__ == "__main__":
    main()
