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

from youtube_pipeline.entrypoints.non_daily_rag import resolve_rag_consumer_config
from youtube_pipeline.rag_consumer import write_rag_consumer_artifacts_from_config


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
            "Build non-generative RAG consumer payloads from approved RAG sidecars. "
            "This reads sidecars, infers temporal roles, selects context units "
            "deterministically, and writes structural validation stubs. It does "
            "not call LLMs, create embeddings, run retrieval, or modify pipeline "
            "outputs."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file with a rag_consumer section.",
    )
    parser.add_argument(
        "--sidecars-dir",
        default=None,
        help="Directory containing approved RAG sidecars.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where new non-generative consumer artifacts will be written.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable consumer run ID.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Optional JSON file stored as consumer run parameters in the manifest.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional note stored in rag_consumer_manifest.json.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "sidecars_dir": args.sidecars_dir,
            "output_dir": args.output_dir,
            "run_id": args.run_id,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    params = _read_json_file(args.params_file)
    if params:
        overrides["params"] = params

    try:
        _, config = resolve_rag_consumer_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = write_rag_consumer_artifacts_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
