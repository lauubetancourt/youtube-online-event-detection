#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.g1_gate_context_experiment import (
    G1GateContextExperimentConfig,
    run_g1_gate_context_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated G-1 gate/context-stability experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow capped real G-1 calls. G-2 and Serper are never invoked.",
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        parser.error("Config must be a JSON object.")
    if "budgets" in payload:
        payload["budgets"] = tuple(
            (str(item["name"]), int(item["estimated_tokens"]))
            for item in payload["budgets"]
        )
    config = G1GateContextExperimentConfig(**payload)
    artifacts = run_g1_gate_context_experiment(config, execute=args.execute)
    print(json.dumps(asdict(artifacts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
