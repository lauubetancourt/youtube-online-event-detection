from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from youtube_pipeline.configuration import ArtifactsConfig, ResolvedRunConfig


RUN_MANIFEST_FILE = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = "1"
CURRENT_SUPPORTED_TRACEABILITY_POLICIES = frozenset(
    {("development", "minimal")}
)


def build_resolved_config_metadata(
    resolved: ResolvedRunConfig,
) -> dict[str, Any]:
    """Return the shared execution identity and effective traceability policy."""

    if not isinstance(resolved, ResolvedRunConfig):
        raise TypeError("resolved must be ResolvedRunConfig.")
    effective_config = json.loads(resolved.canonical_json)
    if not isinstance(effective_config, dict):
        raise TypeError("resolved_config must serialize to an object.")
    policy = resolved.config.artifacts or ArtifactsConfig()
    assert policy.trace_level is not None
    return {
        "run_id": resolved.config.identity.run_id,
        "run_mode": policy.run_mode,
        "trace_level": policy.trace_level,
        "config_hash": resolved.config_hash,
        "resolved_config": effective_config,
    }


def validate_current_traceability_support(
    resolved: ResolvedRunConfig,
) -> None:
    """Fail before execution when requested persistence is not implemented."""

    metadata = build_resolved_config_metadata(resolved)
    policy = (metadata["run_mode"], metadata["trace_level"])
    if policy not in CURRENT_SUPPORTED_TRACEABILITY_POLICIES:
        raise ValueError(
            "Current runners only implement run_mode='development' with "
            "trace_level='minimal'; requested "
            f"run_mode={policy[0]!r}, trace_level={policy[1]!r}."
        )


def build_run_manifest(
    resolved: ResolvedRunConfig,
    *,
    execution_mode: str,
    completed_stages: Sequence[str],
) -> dict[str, Any]:
    """Build one compact execution-level manifest from the resolved config."""

    traceability = build_resolved_config_metadata(resolved)
    if not isinstance(execution_mode, str) or not execution_mode.strip():
        raise ValueError("execution_mode must not be empty.")
    stage_names = list(completed_stages)
    if not stage_names or any(
        not isinstance(name, str) or not name.strip() for name in stage_names
    ):
        raise ValueError("completed_stages must contain non-empty stage names.")

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        **traceability,
        "status": "completed",
        "execution_mode": execution_mode,
        "completed_stages": stage_names,
    }


def write_run_manifest(
    resolved: ResolvedRunConfig,
    *,
    output_dir: str | Path,
    execution_mode: str,
    completed_stages: Sequence[str],
    filename: str = RUN_MANIFEST_FILE,
) -> Path:
    """Persist the single execution-level configuration manifest."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must not be empty.")
    if Path(filename).name != filename:
        raise ValueError("filename must be a file name without directories.")
    output = Path(output_dir) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        resolved,
        execution_mode=execution_mode,
        completed_stages=completed_stages,
    )
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "build_resolved_config_metadata",
    "build_run_manifest",
    "CURRENT_SUPPORTED_TRACEABILITY_POLICIES",
    "RUN_MANIFEST_FILE",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "validate_current_traceability_support",
    "write_run_manifest",
]
