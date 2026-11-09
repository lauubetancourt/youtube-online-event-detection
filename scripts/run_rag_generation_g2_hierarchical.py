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

from youtube_pipeline.entrypoints.non_daily_rag import (
    resolve_rag_g2_hierarchical_config,
)
from youtube_pipeline.rag_generation_g2_hierarchical import (
    plan_rag_g2_hierarchical_dry_run,
    run_rag_g2_hierarchical_from_config,
)


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
            "Plan or run hierarchical G-2 RAG validation. The "
            "external validation unit is event_id + video_id, followed by a "
            "deterministic event-level summary. Videos are processed in a "
            "deterministic batch when an event is large. This does not use embeddings, "
            "vectorstores, ChromaDB, query expansion, or multi-query retrieval "
            "beyond the approved one-query-per-video policy."
        )
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional JSON config file with a rag_generation_g2_hierarchical section.",
    )
    parser.add_argument(
        "--consumer-dir",
        default=None,
        help="Directory containing rag_consumer artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where hierarchical G-2 artifacts will be written.",
    )
    parser.add_argument(
        "--event-id",
        default=None,
        help=(
            "event_id to validate. Required for real execution. Omit it, or use "
            "__all__/all/*, with --dry-run to plan all events."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan events and deterministic video batches without OpenAI, Serper, "
            "API keys, output writes, or validations."
        ),
    )
    parser.add_argument(
        "--query-model",
        default=None,
        help="Approved OpenAI model for per-video query generation. Must remain gpt-5-mini.",
    )
    parser.add_argument(
        "--validation-model",
        default=None,
        help="Approved OpenAI model for per-video validation. Must remain gpt-5-mini.",
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
        help="Number of Serper news results per video query. Defaults to 5.",
    )
    parser.add_argument(
        "--max-videos-per-event",
        type=int,
        default=None,
        help="Deprecated alias for --max-videos-per-event-batch. Defaults to 5.",
    )
    parser.add_argument(
        "--max-videos-per-event-batch",
        type=int,
        default=None,
        help="Maximum videos to evaluate in the current deterministic batch. Defaults to 5.",
    )
    parser.add_argument(
        "--max-estimated-tokens-per-event-batch",
        type=int,
        default=None,
        help="Maximum estimated internal-context tokens in the current event batch.",
    )
    parser.add_argument(
        "--max-llm-calls-per-batch",
        type=int,
        default=None,
        help="Maximum estimated LLM calls in the current batch.",
    )
    parser.add_argument(
        "--max-serper-calls-per-batch",
        type=int,
        default=None,
        help="Maximum estimated Serper calls in the current batch.",
    )
    parser.add_argument(
        "--max-estimated-cost-usd-per-batch",
        type=float,
        default=None,
        help="Maximum estimated USD cost if pricing rates are provided in params.cost_estimation.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="Deterministic 1-based batch index for large events. Defaults to 1.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional stable batch_id stored in outputs and manifests.",
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
        help="Optional JSON file stored as hierarchical G-2 run parameters.",
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
            "query_model": args.query_model,
            "validation_model": args.validation_model,
            "temperature": args.temperature,
            "serper_num_results": args.serper_num_results,
            "max_videos_per_event": args.max_videos_per_event,
            "max_videos_per_event_batch": args.max_videos_per_event_batch,
            "max_estimated_tokens_per_event_batch": (
                args.max_estimated_tokens_per_event_batch
            ),
            "max_llm_calls_per_batch": args.max_llm_calls_per_batch,
            "max_serper_calls_per_batch": args.max_serper_calls_per_batch,
            "max_estimated_cost_usd_per_batch": (
                args.max_estimated_cost_usd_per_batch
            ),
            "batch_index": args.batch_index,
            "batch_id": args.batch_id,
            "max_retries": args.max_retries,
            "notes": args.notes,
        }.items()
        if value is not None
    }
    params = _read_json_file(args.params_file)
    if params:
        overrides["params"] = params

    try:
        _, config = resolve_rag_g2_hierarchical_config(
            config_file=args.config_file,
            overrides=overrides,
            base_dir=Path.cwd(),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        summary = plan_rag_g2_hierarchical_dry_run(config)
    else:
        summary = run_rag_g2_hierarchical_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
