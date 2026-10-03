"""Guarded standalone classifier runner; no polarization/detector runtime."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Sequence

from youtube_pipeline.configuration import load_run_config, resolve_run_config
from youtube_pipeline.entrypoints.common_cli import build_common_run_parser
from youtube_pipeline.stance_classifier import StanceCommentInput
from youtube_pipeline.experimental_input import source_sha256
from youtube_pipeline.stance_classifier_io import (
    CLASSIFIER_MANIFEST,
    STANCE_ARTIFACT,
    prepare_csv_stance_inputs,
    run_configured_comment_stance_classification,
)


def _jsonl_summary(comments: Sequence[StanceCommentInput]) -> dict[str, int]:
    replies = sum(comment.parent_comment_id is not None for comment in comments)
    with_parent = sum(
        comment.parent_comment_id is not None
        and comment.parent_text is not None
        and comment.parent_event_time_utc is not None
        for comment in comments
    )
    expected_calls = sum(
        not comment.is_deleted
        and bool(comment.text.strip())
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
    return {
        "source_rows": len(comments),
        "roots": len(comments) - replies,
        "replies": replies,
        "replies_with_parent": with_parent,
        "missing_parents": replies - with_parent,
        "expected_model_calls": expected_calls,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_common_run_parser(description="Standalone individual stance classifier (not validated).")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--comments-path",
        type=Path,
        help=(
            "Authoritative CSV comment source; adapted in memory through "
            "causal_minimal_v1."
        ),
    )
    inputs.add_argument(
        "--input-jsonl",
        type=Path,
        help="Legacy StanceCommentInput JSONL (also its authoritative source).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Logical base for paths in the RunConfig. Defaults to the config "
            "directory for legacy compatibility."
        ),
    )
    args = parser.parse_args(argv)
    config = load_run_config(args.config_path, overrides=(
        {"identity": {"run_id": args.run_id}} if args.run_id is not None else None
    ))
    base_dir = args.config_path.parent if args.base_dir is None else args.base_dir
    resolved = resolve_run_config(config, base_dir=base_dir)
    if config.polarization is None or config.polarization.classifier is None:
        parser.error("polarization.classifier is required.")
    if args.comments_path is not None:
        try:
            plan = prepare_csv_stance_inputs(config, args.comments_path)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        comments = list(plan.comments)
        input_ref = str(plan.authoritative_source_path)
        input_hash = plan.authoritative_source_hash
        input_policy_id = plan.input_policy_id
        input_mode = "comments_csv"
        summary = {
            "source_rows": plan.source_rows,
            "roots": plan.roots,
            "replies": plan.replies,
            "replies_with_parent": plan.replies_with_parent,
            "missing_parents": plan.missing_parents,
            "expected_model_calls": plan.expected_model_calls,
        }
    else:
        assert args.input_jsonl is not None
        comments = [StanceCommentInput.from_mapping(json.loads(line))
                    for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        input_ref = str(args.input_jsonl.absolute())
        input_hash = source_sha256(args.input_jsonl)
        input_policy_id = (
            config.data.experimental_input.policy_id
            if config.data is not None and config.data.experimental_input is not None
            else None
        )
        input_mode = "stance_input_jsonl"
        summary = _jsonl_summary(comments)
    ids = [comment.comment_id for comment in comments]
    if len(set(ids)) != len(ids):
        parser.error("Duplicate comment_id in input.")
    output_paths = None
    if args.output_root is not None:
        output_root = args.output_root.absolute()
        output_paths = {
            "artifact_path": str(output_root / STANCE_ARTIFACT),
            "manifest_path": str(output_root / CLASSIFIER_MANIFEST),
        }
    if args.execution_mode != "execute":
        print(json.dumps({
            "execution_status": "prepared_not_executed",
            "input_mode": input_mode,
            "input_count": len(comments),
            "classifier_run_id": config.identity.run_id,
            "model": config.polarization.classifier.model,
            "config_hash": resolved.config_hash,
            "authoritative_source_path": input_ref,
            "authoritative_source_sha256": input_hash,
            "input_policy_id": input_policy_id,
            **summary,
            **(output_paths or {}),
        }))
        return
    if args.output_root is None:
        parser.error("--execute requires an explicit --output-root.")
    artifacts = asyncio.run(run_configured_comment_stance_classification(
        config, comments, output_dir=args.output_root, base_dir=base_dir,
        input_ref=input_ref,
    ))
    print(json.dumps({"execution_status": "executed", "classifier_run_id": artifacts.classifier_run_id,
                      "input_count": artifacts.input_count, "artifact_path": str(artifacts.artifact_path),
                      "manifest_path": str(artifacts.manifest_path)}))


if __name__ == "__main__":
    main()
