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

from youtube_pipeline.entrypoints.non_daily_rag import resolve_rag_g1_config
from youtube_pipeline.rag_generation_g1 import run_rag_g1_validation_from_config


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
            "Run G-1 internal-only generative RAG validation for exactly one event. "
            "This uses approved non-generative consumer payloads and does not use "
            "news, Serper, embeddings, vectorstores, or external evidence."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file with a rag_generation_g1 section.",
    )
    parser.add_argument(
        "--consumer-dir",
        default=None,
        help="Directory containing rag_consumer artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where G-1 artifacts will be written.",
    )
    parser.add_argument(
        "--event-id",
        default=None,
        help="Approved event_id to validate. G-1 should be run for one event only.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Approved OpenAI model. Must remain gpt-5-mini for this G-1 run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature. Defaults to 0.",
    )
    parser.add_argument(
        "--max-approx-tokens",
        type=int,
        default=None,
        help="Approximate context-token limit. Defaults to 16000.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retry count for invalid JSON/schema/citations. Defaults to 1.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Optional JSON file stored as G-1 run parameters in the manifest.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional note stored in rag_generation_manifest.json.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "consumer_dir": args.consumer_dir,
            "output_dir": args.output_dir,
            "event_id": args.event_id,
            "model": args.model,
            "temperature": args.temperature,
            "max_approx_tokens": args.max_approx_tokens,
            "max_retries": args.max_retries,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    params = _read_json_file(args.params_file)
    if params:
        overrides["params"] = params

    try:
        _, config = resolve_rag_g1_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = run_rag_g1_validation_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
