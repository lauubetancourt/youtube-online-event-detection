#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.llm_token_context_audit import (
    LlmTokenContextAuditConfig,
    run_llm_token_context_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit persisted stance/G-1/G-2 usage without external calls."
    )
    parser.add_argument("--config", required=True, help="Audit-only JSON config.")
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    artifacts = run_llm_token_context_audit(LlmTokenContextAuditConfig(**payload))
    print(
        json.dumps(
            {
                "status": "executed",
                "output_dir": artifacts.output_dir.as_posix(),
                "summary": artifacts.summary_path.as_posix(),
                "manifest": artifacts.manifest_path.as_posix(),
                "real_model_calls": 0,
                "real_serper_requests": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
