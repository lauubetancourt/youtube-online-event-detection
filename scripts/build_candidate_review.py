#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.candidate_review import build_candidate_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic CSV and Markdown candidate-review views from "
            "persisted run artifacts."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing event_candidates.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional destination; defaults to <run-dir>/candidate_review.",
    )
    args = parser.parse_args()
    artifacts = build_candidate_review(args.run_dir, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "status": "candidate_review_ready",
                "candidate_count": artifacts.candidate_count,
                "csv": str(artifacts.csv_path),
                "markdown": str(artifacts.markdown_path),
                "real_openai_calls": 0,
                "real_serper_calls": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
