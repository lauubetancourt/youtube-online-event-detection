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

from youtube_pipeline.entrypoints.non_daily_rag import resolve_rag_g2_config
from youtube_pipeline.rag_generation_g2 import run_rag_g2_validation_from_config


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
            "Run G-2 generative RAG validation for exactly one approved event using "
            "internal YouTube evidence plus Serper News evidence. This does not use "
            "embeddings, vectorstores, query expansion, multi-query retrieval, or "
            "pipeline-side modifications."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file with a rag_generation_g2 section.",
    )
    parser.add_argument(
        "--consumer-dir",
        default=None,
        help="Directory containing rag_consumer artifacts.",
    )
    parser.add_argument(
        "--g1-dir",
        default=None,
        help="Directory containing accepted G-1 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where G-2 artifacts will be written.",
    )
    parser.add_argument(
        "--event-id",
        default=None,
        help="Approved event_id to validate. The first G-2 run is limited to evt_34d7999bde8c.",
    )
    parser.add_argument(
        "--query-model",
        default=None,
        help="Approved OpenAI model for query generation. Must remain gpt-5-mini.",
    )
    parser.add_argument(
        "--validation-model",
        default=None,
        help="Approved OpenAI model for G-2 validation. Must remain gpt-5-mini.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Requested temperature. Defaults to 0; omitted for gpt-5-mini API calls.",
    )
    parser.add_argument(
        "--serper-num-results",
        type=int,
        default=None,
        help="Number of Serper news results. Defaults to 5.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retry count for invalid validation JSON/schema/citations. Defaults to 1.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Optional JSON file stored as G-2 run parameters in the manifest.",
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
            "g1_dir": args.g1_dir,
            "output_dir": args.output_dir,
            "event_id": args.event_id,
            "query_model": args.query_model,
            "validation_model": args.validation_model,
            "temperature": args.temperature,
            "serper_num_results": args.serper_num_results,
            "max_retries": args.max_retries,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    params = _read_json_file(args.params_file)
    if params:
        overrides["params"] = params

    try:
        _, config = resolve_rag_g2_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = run_rag_g2_validation_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
