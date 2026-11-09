#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.candidate_rag import run_candidate_rag_validation
from youtube_pipeline.configuration import load_run_config, resolve_run_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run candidate evidence through the existing G-1 and hierarchical G-2 "
            "pipeline. Network-capable execution requires --execute."
        )
    )
    parser.add_argument("--config", required=True, help="Common RunConfig JSON file.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory used to resolve logical paths.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow configured G-1/G-2 provider calls.",
    )
    args = parser.parse_args()
    run = load_run_config(args.config)
    resolved = resolve_run_config(run, base_dir=Path(args.base_dir))
    if resolved.config.rag is None or resolved.config.rag.candidate_validation is None:
        raise ValueError("RunConfig must include rag.candidate_validation.")
    if not args.execute:
        component = resolved.config.rag.candidate_validation
        print(
            json.dumps(
                {
                    "status": "prepared_not_executed",
                    "candidate_id": component.candidate_id,
                    "candidates_path": component.candidates_path,
                    "evidence_dir": component.evidence_dir,
                    "output_dir": component.output_dir,
                    "requires_execute": True,
                },
                indent=2,
            )
        )
        return 0
    artifacts = run_candidate_rag_validation(
        resolved.config.rag.candidate_validation,
        execute=True,
        g2_gate=resolved.config.rag.g2_gate,
    )
    print(
        json.dumps(
            {
                "candidate_id": artifacts.candidate_id,
                "status": artifacts.status,
                "output_dir": artifacts.output_dir.as_posix(),
                "validation_results": artifacts.validation_results_path.as_posix(),
                "manifest": artifacts.manifest_path.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
