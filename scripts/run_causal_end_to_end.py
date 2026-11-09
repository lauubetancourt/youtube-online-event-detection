#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.causal_end_to_end import run_causal_end_to_end_replay
from youtube_pipeline.configuration import load_run_config, resolve_run_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Silver-to-CandidateValidationResult causal historical replay. "
            "The default is prepare-only; --execute explicitly enables the run and "
            "its configured G-1/G-2 providers unless --skip-rag is supplied."
        )
    )
    parser.add_argument("--config", required=True, help="Resolved RunConfig source JSON.")
    parser.add_argument("--base-dir", default=".", help="Logical path base directory.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the replay, including configured G-1/G-2 calls for ready evidence.",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Stop after building local candidate evidence; do not call G-1/G-2 providers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible interrupted run. Completed candidates are reused; "
            "partial and error results are archived and retried."
        ),
    )
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute.")
    run = load_run_config(args.config)
    resolved = resolve_run_config(run, base_dir=args.base_dir)
    replay = (
        resolved.config.simulation.causal_end_to_end
        if resolved.config.simulation is not None
        else None
    )
    if replay is None:
        raise ValueError("RunConfig must include simulation.causal_end_to_end.")
    if not args.execute:
        print(json.dumps({
            "status": "prepared_not_executed",
            "run_id": resolved.config.identity.run_id,
            "config_hash": resolved.config_hash,
            "start_utc": replay.start_utc.isoformat(),
            "end_utc": replay.end_utc.isoformat(),
            "step": replay.step,
            "output_dir": str(replay.output_dir),
            "execute_rag": not args.skip_rag,
            "requires_execute": True,
        }, indent=2))
        return 0
    artifacts = run_causal_end_to_end_replay(
        run,
        base_dir=args.base_dir,
        execute_rag=not args.skip_rag,
        resume=args.resume,
        retry_rag_errors=True,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps({
        "status": "executed_successfully",
        "output_dir": str(artifacts.output_dir),
        "manifest": str(artifacts.manifest_path),
        "candidates": artifacts.candidates,
        "validations": artifacts.validations,
        "future_leak_count": artifacts.future_leak_count,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
