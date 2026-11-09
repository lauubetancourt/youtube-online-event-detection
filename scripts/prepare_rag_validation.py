#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.entrypoints.non_daily_rag import resolve_rag_validation_config
from youtube_pipeline.rag_validation import prepare_rag_validation_artifacts_from_config


def _read_json_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON config not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must contain an object: {p}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare contract-only RAG validation artifacts from event evidence "
            "packages. This does not run retrieval, embeddings, LLMs, or external APIs."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file for RAG validation preparation.",
    )
    parser.add_argument(
        "--evidence-packages-path",
        default=None,
        help="Path to event_evidence_packages.jsonl from the RAG evidence builder.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where validation preparation artifacts will be written.",
    )
    parser.add_argument(
        "--validation-run-id",
        default=None,
        help="Optional stable validation run ID. If omitted, one is derived from paths.",
    )
    parser.add_argument(
        "--validator",
        default=None,
        help="Validator label to store, for example manual_pending.",
    )
    parser.add_argument(
        "--query-language",
        default=None,
        help="Language code for future retrieval query placeholders.",
    )
    parser.add_argument(
        "--max-videos-per-event",
        type=int,
        default=None,
        help="Maximum video-specific query placeholders per event.",
    )
    parser.add_argument(
        "--validation-params-file",
        default=None,
        help="Optional JSON file with validation preparation parameters.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional note stored in rag_validation_manifest.json.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "evidence_packages_path": args.evidence_packages_path,
            "output_dir": args.output_dir,
            "validation_run_id": args.validation_run_id,
            "validator": args.validator,
            "query_language": args.query_language,
            "max_videos_per_event": args.max_videos_per_event,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    validation_params = _read_json_file(args.validation_params_file)
    if validation_params:
        overrides["validation_params"] = validation_params

    try:
        _, config = resolve_rag_validation_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = prepare_rag_validation_artifacts_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
