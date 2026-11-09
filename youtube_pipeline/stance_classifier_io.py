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
from youtube_pipeline.experimental_input import (
    adapt_stance_records,
    prepare_causal_minimal_batch,
    read_input_table,
    source_sha256,
)
from youtube_pipeline.stance_classifier import (
    OpenAICommentStanceClassifier, StanceCommentInput, StanceModelTransport,
)
from youtube_pipeline.polarization_runtime_config import VideoScope, resolve_video_scope
from youtube_pipeline.storage import canonicalize_comment_event_time
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


@dataclass(frozen=True, slots=True)
class CsvStanceInputPlan:
    """In-memory classifier inputs whose identity remains the source CSV."""

    comments: tuple[StanceCommentInput, ...]
    authoritative_source_path: Path
    authoritative_source_hash: str
    input_policy_id: str
    source_rows: int
    roots: int
    replies: int
    replies_with_parent: int
    missing_parents: int
    expected_model_calls: int
    video_scope: VideoScope


def prepare_csv_stance_inputs(
    config: RunConfig,
    comments_path: str | Path,
) -> CsvStanceInputPlan:
    """Adapt one CSV source without replacing its authoritative identity.

    The source is canonicalized and adapted entirely in memory.  In particular,
    no transport JSONL is created and the returned path/hash always describe the
    CSV bytes that produced the classifier inputs.
    """

    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig.")
    if config.data is None or config.data.experimental_input is None:
        raise ValueError(
            "CSV stance input requires data.experimental_input (causal_minimal_v1)."
        )
    source = Path(comments_path).expanduser().absolute()
    dataframe = read_input_table(source)
    canonical = canonicalize_comment_event_time(dataframe)
    batch = prepare_causal_minimal_batch(
        canonical,
        config.data.experimental_input,
        source_path=source,
    )
    runtime = config.polarization.runtime if config.polarization is not None else None
    scope = resolve_video_scope(
        (record.video_id for record in batch.records),
        runtime.video_ids if runtime is not None else None,
    )
    effective_video_ids = set(scope.effective_video_ids)
    selected_records = tuple(
        record for record in batch.records if record.video_id in effective_video_ids
    )
    adapted = tuple(
        item for item in adapt_stance_records(batch.records)
        if item.video_id in effective_video_ids
    )
    comments = tuple(item.classifier_input for item in adapted)
    replies = sum(record.is_reply for record in selected_records)
    replies_with_parent = sum(
        item.classifier_input.parent_comment_id is not None
        and item.classifier_input.parent_text is not None
        and item.classifier_input.parent_event_time_utc is not None
        for item in adapted
    )
    expected_model_calls = sum(
        bool(comment.text.strip())
        and (
            comment.parent_comment_id is None
            or (
                comment.parent_text is not None
                and bool(comment.parent_text.strip())
                and comment.parent_event_time_utc is not None
                and comment.parent_event_time_utc <= comment.event_time_utc
            )
        )
        for comment in comments
    )
    return CsvStanceInputPlan(
        comments=comments,
        authoritative_source_path=source,
        authoritative_source_hash=batch.manifest["source_sha256"],
        input_policy_id=config.data.experimental_input.policy_id,
        source_rows=len(dataframe),
        roots=len(selected_records) - replies,
        replies=replies,
        replies_with_parent=replies_with_parent,
        missing_parents=replies - replies_with_parent,
        expected_model_calls=expected_model_calls,
        video_scope=scope,
    )


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
                    "topic_profile": (
                        to_json_mapping(resolved.config.topic_profile)
                        if resolved.config.topic_profile is not None
                        else None
                    ),
                    "topic_id": (
                        resolved.config.topic_profile.topic_id
                        if resolved.config.topic_profile is not None
                        else None
                    ),
                    "max_concurrency": settings.max_concurrency,
                    "retry_policy": to_json_mapping(settings.retry_policy),
                    "inference_parameters": {
                        "temperature": settings.temperature, "top_p": settings.top_p,
                        "max_completion_tokens": settings.max_completion_tokens,
                        "request_timeout_seconds": settings.request_timeout_seconds,
                        "max_input_characters": settings.max_input_characters,
                    },
                    "classification_policy": {
                        "policy_id": "comment_stance_likert5_structured_v1",
                        "missing_parent_policy": "SKIPPED",
                        "thread_context": "immediate_parent_only",
                        "text_authority": "original",
                    },
                    "missing_parent_policy": "SKIPPED", "thread_context": "immediate_parent_only",
                    "token_usage": classifier.token_accounting,
                    "monetary_cost": None, "cost_tracking_status": "NOT_ESTIMATED",
                    "input_count": len(inputs), "result_count": len(classifier.records),
                    "counts_by_status": classifier.counts_by_status,
                    "input_ref": input_ref, "artifact_path": str(paths[0]),
                    "input_policy_id": (
                        resolved.config.data.experimental_input.policy_id
                        if resolved.config.data is not None
                        and resolved.config.data.experimental_input is not None
                        else None
                    ),
                    "input_source_sha256": (
                        source_sha256(input_ref)
                        if input_ref is not None and Path(input_ref).exists()
                        else None
                    ),
                    "classified_comment_ids_sha256": hashlib.sha256(
                        "\n".join(sorted(item.comment_id for item in inputs)).encode("utf-8")
                    ).hexdigest(),
                    "artifact_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest(),
                    "resolved_config_path": str(paths[2]), "config_hash": resolved.config_hash,
                    "ordering": "event_time_utc_then_comment_id",
                }
                resolved_file.write(resolved.canonical_json + "\n")
                manifest_file.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return ClassifierRunArtifacts(resolved.config.identity.run_id, *paths, len(inputs))
