"""Guarded standalone JSONL classifier runner; no polarization/detector runtime."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Sequence

from youtube_pipeline.configuration import load_run_config, resolve_run_config
from youtube_pipeline.entrypoints.common_cli import build_common_run_parser
from youtube_pipeline.stance_classifier import StanceCommentInput
from youtube_pipeline.stance_classifier_io import run_configured_comment_stance_classification


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_common_run_parser(description="Standalone individual stance classifier (not validated).")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_run_config(args.config_path, overrides=(
        {"identity": {"run_id": args.run_id}} if args.run_id is not None else None
    ))
    resolved = resolve_run_config(config, base_dir=args.config_path.parent)
    if config.polarization is None or config.polarization.classifier is None:
        parser.error("polarization.classifier is required.")
    comments = [StanceCommentInput.from_mapping(json.loads(line))
                for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [comment.comment_id for comment in comments]
    if len(set(ids)) != len(ids):
        parser.error("Duplicate comment_id in input.")
    if args.execution_mode != "execute":
        print(json.dumps({"execution_status": "prepared_not_executed", "input_count": len(comments),
                          "classifier_run_id": config.identity.run_id,
                          "model": config.polarization.classifier.model,
                          "config_hash": resolved.config_hash}))
        return
    if args.output_root is None:
        parser.error("--execute requires an explicit --output-root.")
    artifacts = asyncio.run(run_configured_comment_stance_classification(
        config, comments, output_dir=args.output_root, base_dir=args.config_path.parent,
        input_ref=str(args.input_jsonl.absolute()),
    ))
    print(json.dumps({"execution_status": "executed", "classifier_run_id": artifacts.classifier_run_id,
                      "input_count": artifacts.input_count, "artifact_path": str(artifacts.artifact_path),
                      "manifest_path": str(artifacts.manifest_path)}))


if __name__ == "__main__":
    main()
