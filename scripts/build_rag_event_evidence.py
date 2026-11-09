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

from youtube_pipeline.entrypoints.non_daily_rag import resolve_rag_evidence_config
from youtube_pipeline.rag_evidence import write_rag_evidence_artifacts_from_config


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
            "Build non-invasive RAG event evidence artifacts from existing "
            "detection experiment outputs."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file for RAG evidence assembly.",
    )
    parser.add_argument(
        "--comments-path",
        default=None,
        help="Gold comments table used as the all-comments evidence source.",
    )
    parser.add_argument(
        "--trigger-comment-map-path",
        default=None,
        help="Existing exploratory trigger_comment_map.csv.",
    )
    parser.add_argument(
        "--snapshots-path",
        default=None,
        help="Optional snapshots.csv used to attach signal evidence.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where RAG evidence artifacts will be written.",
    )
    parser.add_argument(
        "--detector-name",
        default=None,
        help="Detector name to store in event candidates and run manifest.",
    )
    parser.add_argument(
        "--detector-config-file",
        default=None,
        help="Optional JSON file with detector parameters to store as metadata.",
    )
    parser.add_argument(
        "--monitoring-config-file",
        default=None,
        help="Optional JSON file with monitoring parameters to store as metadata.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run ID. If omitted, one is derived from input paths.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional human-readable note stored in run_manifest.json.",
    )
    parser.add_argument(
        "--snapshot-context",
        choices=["window", "anchor", "none"],
        default=None,
        help=(
            "How to link snapshots to events: all snapshots ending inside the "
            "evidence window, one trigger anchor snapshot, or none."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    overrides = {
        key: value
        for key, value in {
            "comments_path": args.comments_path,
            "trigger_comment_map_path": args.trigger_comment_map_path,
            "snapshots_path": args.snapshots_path,
            "output_dir": args.output_dir,
            "detector_name": args.detector_name,
            "run_id": args.run_id,
            "notes": args.notes,
            "snapshot_context": args.snapshot_context,
        }.items()
        if value is not None
    }
    detector_params = _read_json_file(args.detector_config_file)
    monitoring_params = _read_json_file(args.monitoring_config_file)
    if detector_params:
        overrides["detector_params"] = detector_params
    if monitoring_params:
        overrides["monitoring_params"] = monitoring_params

    try:
        _, config = resolve_rag_evidence_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = write_rag_evidence_artifacts_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
