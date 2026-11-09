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

from youtube_pipeline.entrypoints.non_daily_rag import resolve_rag_sidecar_config
from youtube_pipeline.rag_sidecars import write_rag_sidecar_artifacts_from_config


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
            "Build non-invasive RAG sidecar artifacts for event evidence, full "
            "comment inventory, thread maps, and deterministic context units. "
            "This does not run retrieval, embeddings, LLMs, validation, or alter "
            "existing pipeline/PoC outputs."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file with a rag_sidecars section.",
    )
    parser.add_argument(
        "--trigger-comment-map-path",
        default=None,
        help="Existing trigger_comment_map.csv used to preserve PoC compatibility.",
    )
    parser.add_argument(
        "--comments-path",
        default=None,
        help="Gold comments table used as the all-comment evidence source.",
    )
    parser.add_argument(
        "--snapshots-path",
        default=None,
        help="Optional snapshots.csv used only for signal-count lineage.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where new sidecar artifacts will be written.",
    )
    parser.add_argument(
        "--detector-name",
        default=None,
        help="Detector name stored in event packages. Defaults to xiao_ema.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run ID. If omitted, one is derived from input paths.",
    )
    parser.add_argument(
        "--max-comments-per-context-unit",
        type=int,
        default=None,
        help="Maximum comments per deterministic context unit. Defaults to 25.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Optional JSON file stored as sidecar run parameters in the manifest.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional note stored in context_selection_manifest.json.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "trigger_comment_map_path": args.trigger_comment_map_path,
            "comments_path": args.comments_path,
            "snapshots_path": args.snapshots_path,
            "output_dir": args.output_dir,
            "detector_name": args.detector_name,
            "run_id": args.run_id,
            "max_comments_per_context_unit": args.max_comments_per_context_unit,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    params = _read_json_file(args.params_file)
    if params:
        overrides["params"] = params

    try:
        _, config = resolve_rag_sidecar_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = write_rag_sidecar_artifacts_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
