from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from youtube_pipeline.configuration import (
    ResolvedRunConfig,
    load_run_config,
    resolve_run_config,
    run_config_from_mapping,
)
from youtube_pipeline.data_extraction import (
    ExtractionConfig,
    run_extraction_pipeline,
)
from youtube_pipeline.run_manifest import (
    build_resolved_config_metadata,
    validate_current_traceability_support,
)


_LEGACY_IDENTITY = "legacy_youtube_extraction"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("YouTube extraction config must be a JSON object.")
    return payload


def _legacy_component_payload(
    payload: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    component = ExtractionConfig().as_dict()
    component.update(dict(payload))
    if overrides is not None:
        component.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )
    for field_name in ("query", "published_after", "published_before"):
        value = component.get(field_name)
        if isinstance(value, str) and not value.strip():
            component[field_name] = None
    return component


def _legacy_run_payload(component_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": {"run_id": _LEGACY_IDENTITY},
        "data": {"youtube_api": dict(component_payload)},
    }


def _component_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if overrides is None:
        return None
    return {"data": {"youtube_api": dict(overrides)}}


def load_legacy_youtube_extraction_config(
    config_file: str | Path | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExtractionConfig:
    """Preserve legacy defaults while using the strict common loader."""

    payload: dict[str, Any] = {}
    if config_file is not None:
        payload = _read_json_object(Path(config_file))
    component_payload = payload.get("youtube_extraction", payload)
    if not isinstance(component_payload, Mapping):
        raise TypeError("YouTube extraction config must be a JSON object.")
    run = run_config_from_mapping(
        _legacy_run_payload(
            _legacy_component_payload(component_payload, overrides)
        )
    )
    if run.data is None or run.data.youtube_api is None:
        raise ValueError("Legacy config did not resolve YouTube extraction.")
    return run.data.youtube_api


def resolve_youtube_extraction_config(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> ExtractionConfig:
    """Resolve either a RunConfig profile or the legacy component format."""

    resolved = resolve_youtube_extraction_run(
        config_file=config_file,
        overrides=overrides,
        base_dir=base_dir,
    )
    assert resolved.config.data is not None
    assert resolved.config.data.youtube_api is not None
    return resolved.config.data.youtube_api


def resolve_youtube_extraction_run(
    *,
    config_file: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
    base_dir: str | Path,
) -> ResolvedRunConfig:
    """Resolve acquisition while retaining its execution-level identity."""

    if config_file is not None:
        path = Path(config_file)
        payload = _read_json_object(path)
        is_run_profile = "identity" in payload or any(
            section in payload
            for section in ("data", "simulation", "signals", "detection", "rag")
        )
    else:
        path = None
        payload = {}
        is_run_profile = False

    if is_run_profile:
        run = load_run_config(
            path,
            overrides=_component_overrides(overrides),
        )
    else:
        component_payload = payload.get("youtube_extraction", payload)
        if not isinstance(component_payload, Mapping):
            raise TypeError("YouTube extraction config must be a JSON object.")
        run = run_config_from_mapping(
            _legacy_run_payload(
                _legacy_component_payload(component_payload, overrides)
            )
        )

    resolved = resolve_run_config(run, base_dir=base_dir)
    if (
        resolved.config.data is None
        or resolved.config.data.youtube_api is None
    ):
        raise ValueError("RunConfig must include data.youtube_api for this entrypoint.")
    return resolved


def resolve_youtube_api_key() -> str:
    """Resolve the infrastructure secret without adding it to RunConfig."""

    load_dotenv()
    value = os.environ.get("YOUTUBE_API_KEY")
    if value is None or not value.strip():
        raise RuntimeError(
            "No se encontro YOUTUBE_API_KEY. Define la variable en .env o entorno."
        )
    return value


def run_youtube_extraction(
    config: ExtractionConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve infrastructure credentials and call the domain component explicitly."""

    return run_extraction_pipeline(
        config,
        logger,
        api_key=resolve_youtube_api_key(),
    )


def _attach_resolved_config_to_metadata(
    metadata_path: str | Path,
    resolved: ResolvedRunConfig,
) -> None:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Acquisition run metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Acquisition run metadata must be a JSON object.")
    payload.update(build_resolved_config_metadata(resolved))
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_resolved_youtube_extraction(
    resolved: ResolvedRunConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run acquisition and enrich its existing primary metadata once."""

    if not isinstance(resolved, ResolvedRunConfig):
        raise TypeError("resolved must be ResolvedRunConfig.")
    validate_current_traceability_support(resolved)
    data = resolved.config.data
    if data is None or data.youtube_api is None:
        raise ValueError("RunConfig must include data.youtube_api for acquisition.")
    summary = run_youtube_extraction(data.youtube_api, logger)
    metadata_path = summary.get("run_metadata")
    if not isinstance(metadata_path, (str, Path)):
        raise ValueError("Acquisition summary must include run_metadata.")
    _attach_resolved_config_to_metadata(metadata_path, resolved)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract YouTube videos/comments and persist pipeline artifacts."
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--published-after", default=None)
    parser.add_argument("--published-before", default=None)
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--min-comments", type=int, default=None)
    parser.add_argument("--max-comments", type=int, default=None)
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument("--request-pause-seconds", type=float, default=None)
    parser.add_argument("--retry-attempts", type=int, default=None)
    parser.add_argument("--retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--quota-pause-seconds", type=float, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--metadata-path", default=None)
    legacy_group = parser.add_mutually_exclusive_group()
    legacy_group.add_argument(
        "--save-legacy-csv",
        dest="save_legacy_csv",
        action="store_true",
    )
    legacy_group.add_argument(
        "--no-save-legacy-csv",
        dest="save_legacy_csv",
        action="store_false",
    )
    parser.set_defaults(save_legacy_csv=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime la configuracion final sin llamar la API.",
    )
    return parser


def _setup_logger(log_level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("data_extraction")


def main(
    argv: Sequence[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logger = _setup_logger(args.log_level)
    overrides = {
        key: value
        for key, value in {
            "query": args.query,
            "published_after": args.published_after,
            "published_before": args.published_before,
            "min_views": args.min_views,
            "min_comments": args.min_comments,
            "max_comments": args.max_comments,
            "max_results": args.max_results,
            "request_timeout_seconds": args.request_timeout_seconds,
            "request_pause_seconds": args.request_pause_seconds,
            "retry_attempts": args.retry_attempts,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "quota_pause_seconds": args.quota_pause_seconds,
            "save_legacy_csv": args.save_legacy_csv,
            "data_root": args.data_root,
            "metadata_path": args.metadata_path,
        }.items()
        if value is not None
    }
    try:
        resolved = resolve_youtube_extraction_run(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd() if base_dir is None else base_dir,
        )
        assert resolved.config.data is not None
        assert resolved.config.data.youtube_api is not None
        config = resolved.config.data.youtube_api
        if args.dry_run:
            logger.info(
                "Dry run. Configuracion final: %s",
                json.dumps(config.as_dict(), ensure_ascii=False),
            )
            return
        summary = run_resolved_youtube_extraction(resolved, logger)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("quota_hit"):
        pause_s = float(config.quota_pause_seconds)
        if pause_s > 0:
            logger.warning(
                "Se detecto cuota agotada. Esperando %.1fs antes de finalizar.",
                pause_s,
            )
            time.sleep(pause_s)


__all__ = [
    "load_legacy_youtube_extraction_config",
    "main",
    "resolve_youtube_api_key",
    "resolve_youtube_extraction_config",
    "resolve_youtube_extraction_run",
    "run_resolved_youtube_extraction",
    "run_youtube_extraction",
]


if __name__ == "__main__":
    main()
