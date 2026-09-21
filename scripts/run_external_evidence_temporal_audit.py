#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from youtube_pipeline.external_evidence_temporal_audit import (
    ExternalEvidenceTemporalAuditConfig,
    run_external_evidence_temporal_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit persisted G-2 publication times without external calls."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-rag-dir", action="append", default=[])
    parser.add_argument("--legacy-external-evidence")
    parser.add_argument("--legacy-event-candidates")
    parser.add_argument("--legacy-video-reports")
    parser.add_argument("--legacy-g2-manifest")
    args = parser.parse_args()
    artifacts = run_external_evidence_temporal_audit(
        ExternalEvidenceTemporalAuditConfig(
            output_dir=args.output_dir,
            candidate_rag_dirs=tuple(args.candidate_rag_dir),
            legacy_external_evidence_path=args.legacy_external_evidence,
            legacy_event_candidates_path=args.legacy_event_candidates,
            legacy_video_reports_path=args.legacy_video_reports,
            legacy_g2_manifest_path=args.legacy_g2_manifest,
        )
    )
    print(
        json.dumps(
            {
                "status": "executed",
                "candidate_count": artifacts.candidate_count,
                "source_count": artifacts.source_count,
                "source_audit": artifacts.source_audit_path.as_posix(),
                "candidate_summary": artifacts.candidate_summary_path.as_posix(),
                "manifest": artifacts.manifest_path.as_posix(),
                "model_calls": 0,
                "external_calls": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
