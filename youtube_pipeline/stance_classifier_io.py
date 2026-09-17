"""Classification-only persistence/composition. No downstream execution handoff."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from youtube_pipeline.configuration.models import RunConfig
from youtube_pipeline.configuration.paths import resolve_run_config
from youtube_pipeline.signal_observations import to_json_mapping
from youtube_pipeline.stance_classifier import (
    OpenAICommentStanceClassifier, StanceCommentInput, StanceModelTransport,
)
from youtube_pipeline.stance_classifier_prompt import (
    REFERENCE_NOTEBOOK, REFERENCE_NOTEBOOK_SHA256, SYSTEM_INSTRUCTIONS, response_format,
)

STANCE_ARTIFACT = "comment_stances.jsonl"
CLASSIFIER_MANIFEST = "classifier_manifest.json"
CLASSIFIER_RESOLVED_CONFIG = "resolved_config.json"


def create_configured_stance_classifier(config: RunConfig, *,
                                        transport: StanceModelTransport | None = None) -> OpenAICommentStanceClassifier:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig.")
    if config.polarization is None or config.polarization.classifier is None:
        raise ValueError("A polarization.classifier configuration is required.")
    return OpenAICommentStanceClassifier(config.polarization.classifier, transport=transport)


@dataclass(frozen=True, slots=True)
class ClassifierRunArtifacts:
    classifier_run_id: str
    artifact_path: Path
    manifest_path: Path
    resolved_config_path: Path
    input_count: int


async def run_configured_comment_stance_classification(
    config: RunConfig, comments: Iterable[StanceCommentInput], *,
    output_dir: str | Path, base_dir: str | Path,
    transport: StanceModelTransport | None = None, input_ref: str | None = None,
) -> ClassifierRunArtifacts:
    """Explicitly invoked inference; callers/tests may inject a fake transport.

    Existing outputs fail before provider calls. Files are exclusively created;
    this does not claim atomic multi-file durability or resume across restarts.
    """
    resolved = resolve_run_config(config, base_dir=base_dir)
    classifier = create_configured_stance_classifier(resolved.config, transport=transport)
    root = Path(output_dir).absolute()
    paths = [root / name for name in (STANCE_ARTIFACT, CLASSIFIER_MANIFEST, CLASSIFIER_RESOLVED_CONFIG)]
    if any(path.exists() for path in paths):
        raise FileExistsError("Classifier outputs already exist; choose a fresh output directory.")
    root.mkdir(parents=True, exist_ok=True)
    # Open exclusive outputs before inference, including race protection and an
    # early check that the requested destination is actually writable.
    with paths[0].open("x", encoding="utf-8") as artifact:
        with paths[1].open("x", encoding="utf-8") as manifest_file:
            with paths[2].open("x", encoding="utf-8") as resolved_file:
                inputs = list(comments)
                await classifier.classify_batch(inputs)
                rows = "".join(json.dumps(record.to_mapping(), ensure_ascii=False, allow_nan=False)
                               + "\n" for record in classifier.records)
                artifact.write(rows)
                settings = classifier.config
                proposition = settings.proposition
                prompt = {"prompt_id": settings.prompt_id, "prompt_version": settings.prompt_version,
                          "system_instructions": SYSTEM_INSTRUCTIONS, "response_format": response_format()}
                prompt_hash = hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                manifest = {
                    "artifact_version": "stance_classifier_v1",
                    "execution_status": "executed", "methodological_status": "NOT_VALIDATED",
                    "classifier_run_id": resolved.config.identity.run_id,
                    "provider": "openai", "model": settings.model,
                    "prompt_id": settings.prompt_id, "prompt_version": settings.prompt_version,
                    "prompt_hash": prompt_hash, "prompt_artifact": prompt,
                    "reference_notebook": REFERENCE_NOTEBOOK,
                    "reference_notebook_sha256": REFERENCE_NOTEBOOK_SHA256,
                    "proposition": to_json_mapping(proposition),
                    "context_hash": hashlib.sha256(proposition.context.encode("utf-8")).hexdigest(),
                    "max_concurrency": settings.max_concurrency,
                    "retry_policy": to_json_mapping(settings.retry_policy),
                    "inference_parameters": {
                        "temperature": settings.temperature, "top_p": settings.top_p,
                        "max_completion_tokens": settings.max_completion_tokens,
                        "request_timeout_seconds": settings.request_timeout_seconds,
                        "max_input_characters": settings.max_input_characters,
                    },
                    "missing_parent_policy": "SKIPPED", "thread_context": "immediate_parent_only",
                    "token_usage": classifier.token_accounting,
                    "monetary_cost": None, "cost_tracking_status": "NOT_ESTIMATED",
                    "input_count": len(inputs), "result_count": len(classifier.records),
                    "counts_by_status": classifier.counts_by_status,
                    "input_ref": input_ref, "artifact_path": str(paths[0]),
                    "artifact_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest(),
                    "resolved_config_path": str(paths[2]), "config_hash": resolved.config_hash,
                    "ordering": "event_time_utc_then_comment_id",
                }
                resolved_file.write(resolved.canonical_json + "\n")
                manifest_file.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return ClassifierRunArtifacts(resolved.config.identity.run_id, *paths, len(inputs))
