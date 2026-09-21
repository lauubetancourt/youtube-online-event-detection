# External evidence temporal audit

Status: deterministic audit tooling implemented; current empirical conclusion is
`MORE_EVIDENCE_REQUIRED`.

This audit keeps three clocks separate:

- candidate reference time: the time represented by the associated detections;
- candidate creation/evidence cutoff: when validation can causally consume data;
- external publication time: metadata returned for a retrieved source.

`EventCandidate` timing, real-world event timing, and external publication timing
are not interchangeable. A source may be after the first alert but still be
available before the evidence cutoff. Conversely, a page carrying an old
publication date is not proof that a historical search engine had indexed it at
that time.

## Current G-2 temporal behavior

Hierarchical G-2 anchors its search to the candidate/event reference date. The
search window is the configured number of calendar days before and after that
date; the current defaults are one day in each direction. Retrieval is one
primary news query per candidate/event and video.

The new candidate route additionally bounds the query end by the evidence cutoff
date and keeps only sources whose precise `published_at` is strictly before the
cutoff. Unknown, timezone-naive, date-only-on-the-cutoff-day, and at/after-cutoff
timestamps are excluded. The retained legacy route did not persist a causal
evidence cutoff and therefore cannot support a causal-availability conclusion.

The audit does not alter either behavior. It computes `published_at -
candidate_reference_time` only for timezone-aware timestamps and classifies
availability relative to the persisted cutoff. Calendar-date-only values retain
`DATE_ONLY`; same-day ordering and second-level lag stay unknown.

## Retained empirical sample

The reusable local sample is the legacy hierarchical G-2 batch at
`experiments/xiao/media/log_3/rag_generation_g2_hierarchical_batch_auto_lot1`.
It contains 10 legacy activity candidates and 166 retrieved sources, of which 77
were cited or used by G-2. All 166 publication values are Serper `date` metadata
at calendar-day precision. The legacy artifacts contain neither
`candidate_created_at_cutoff_utc` nor `evidence_cutoff_utc`.

At the only supported resolution:

- all retrieved: 53 pre-candidate day, 27 post-candidate day, 86 same-day unknown;
- used/cited by G-2: 26 pre-candidate day, 5 post-candidate day, 46 same-day unknown;
- exact lag distribution: unavailable (0 authoritative timestamps);
- causal availability: unknown for all 166 sources (missing cutoff).

These observations show that some relevant external material clearly predates
the candidate date, but they do not establish that external evidence usually was
causally available when the candidate could be validated. The current candidate
route would also pre-filter future records, so its retained evidence alone cannot
measure how much relevant future evidence was excluded.

## Running the deterministic audit

No `--execute` flag exists because this command cannot call Serper or a model.
For current candidate outputs, repeat `--candidate-rag-dir` as needed. Historical
inputs require the explicit compatibility triplet so missing cutoffs are visible.

```bash
.venv/bin/python scripts/run_external_evidence_temporal_audit.py \
  --output-dir outputs/external_evidence_temporal_audit \
  --legacy-external-evidence <dir>/rag_video_external_evidence.jsonl \
  --legacy-event-candidates <evidence-dir>/event_candidates.csv \
  --legacy-video-reports <dir>/rag_video_validation_reports.jsonl \
  --legacy-g2-manifest <dir>/rag_generation_manifest.json
```

Outputs are:

- `external_evidence_temporal_audit.csv`: one row per retrieved source;
- `external_evidence_candidate_summary.csv`: candidate-level aggregation;
- `external_evidence_temporal_audit_manifest.json`: hashes, policies, counts,
  lag summaries, reused-result counts, and limitations.

Removing future/unknown sources can change the G-2 conclusion. The audit marks
such cases `REQUIRES_G2_REEVALUATION`; it never simulates a replacement LLM
verdict.

## Decision boundary

No G-2 policy change follows from this sample. A subsequent evidence collection
run needs candidate-native artifacts with gate `OFF`, exact publication timestamps
where sources expose them, and the persisted evidence cutoff. Repeated monitoring,
delayed validation, and gate calibration remain outside this audit.
