# Candidate evidence policy

**Status:** IMPLEMENTED; TEMPORAL PARAMETERS ARE EXPERIMENTAL AND NOT CALIBRATED.

## Boundary and canonical source

The new evidence boundary is additive:

```text
AssociatedEventCandidate(READY_FOR_EVIDENCE)
→ candidate_evidence_v1
→ Silver + causal_minimal_v1
→ complete candidate-scoped inventory
→ deterministic thread/context sidecars
```

It does not modify detection, association, or candidate artifacts and does not run
the RAG consumer, G-1, G-2, external retrieval, ranking, or validation. Gold and all
legacy sidecars remain unchanged. Silver plus `causal_minimal_v1` is authoritative
because it is the same canonical comment universe used by the new activity and
stance branches: exact comment IDs are deduplicated, original text is authoritative,
and semantic filtering is disabled.

## Audit of the retrospective implementation

The existing retrospective route is:

```text
completed trigger
→ trigger_comment_map
→ XIAO-derived event_id and trigger/video scope
→ event_comment_inventory
→ event_video_map + event_thread_map
→ deterministic rag_context_units + comment map
→ event_evidence_packages + context_selection_manifest
→ non-generative rag_consumer payloads
→ later G-1 / G-2
```

Reusable behavior includes complete inventory before selection, closed temporal
bounds, roots/replies, no automatic inclusion of a parent outside the window,
deterministic thread/video blocks, comment-to-context lineage, and separate package,
inventory, video, thread, and context artifacts. Activity-specific behavior includes
the trigger-comment map, XIAO detector fields, Gold as the default comments source,
and the historical `event_id` formula.

The candidate route reuses the deterministic thread/context builders through an
explicit adapter, but keeps `candidate_id` as its canonical join key. A compatibility
view aliases `candidate_id → event_id` only inside a separate directory for the
unchanged non-generative RAG consumer. This alias never changes canonical artifacts
or legacy outputs.

## Temporal semantics

V1 selects `candidate.reference_time_utc` as the anchor because it is the latest
alert time in the finalized association. It represents the detected phenomenon more
directly than `created_at_cutoff_utc`, whose delay is partly caused by association
buffering. Candidate start/end remain alert extent, not evidence bounds.

Configured parameters are:

```text
requested_start = reference_time_utc - lookback
requested_end   = reference_time_utc + lookahead
```

Both durations are methodological/experimental, required in configuration, and
labelled `TEST_ONLY` or `DEVELOPMENT_NOT_CALIBRATED`. They are never derived from
signal windows, cadences, detector history, or association tolerance.

Comment inclusion is:

```text
video_id in candidate.associated_video_ids
AND evidence_window_start_utc <= event_time_utc <= evidence_window_end_utc
AND event_time_utc < evidence_cutoff_utc
```

The closed evidence window preserves the retrospective convention. The additional
strict cutoff guard preserves current causal semantics; a comment exactly at the
cutoff is not yet available.

## Immediate and delayed modes

`immediate_causal` is the V1 recommendation for online-like prototyping:

```text
evidence_cutoff = candidate.created_at_cutoff_utc
actual_end = min(requested_end, evidence_cutoff)
```

If the requested end reaches or exceeds the cutoff,
`window_clipped_by_causality=true` is persisted. Positive lookahead already available
by candidate creation can be included; unavailable future context cannot.

`delayed_enrichment` is supported but not recommended as the sole V1 mode. Its
`evidence_ready_at_utc` is the requested horizon and the final inventory is ready
only when the current cutoff is strictly later than that horizon. Before then the
package is explicitly `EVIDENCE_PENDING_HORIZON` and contains no final inventory.
This prevents post-candidate comments from being presented as trigger-time evidence.

## Video, comment, and parent policy

Candidate-associated videos are the only allowed video universe. Zero video IDs
produces `EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE`; the stage never searches all
YouTube or guesses a video. Multiple videos are supported.

Every canonical comment satisfying video, window, and cutoff rules enters the
inventory regardless of stance status, spam-like flags, short text, or uncertainty.
Detection lineage remains a narrow chain explaining the trigger; evidence inventory
is the complete in-scope contextual universe and may be broader.

Roots and replies are retained. A reply whose parent is outside the evidence window
keeps `parent_comment_id` and `parent_in_inventory=false`; the parent is not imported
as context-only evidence. Reply acquisition completeness remains open and is stated
as a manifest limitation.

Zero comments produces `NO_EVIDENCE_FOUND`, not fabricated evidence and not a
rejected candidate. Evidence presence never confirms an event.

## Artifacts and compatibility

Canonical candidate artifacts are:

```text
candidate_evidence_packages.jsonl
candidate_comment_inventory.csv
candidate_video_map.csv
candidate_thread_map.csv
candidate_rag_context_units.jsonl
candidate_context_unit_comment_map.csv
candidate_evidence_manifest.json
```

The manifest separates candidate-generation lineage from evidence-policy lineage,
records the Silver source path/hash and canonical comment-ID hash, explicit policy
parameters, counts, artifact hashes, causal rule, and downstream non-execution flags.
Package and inventory ordering are deterministic by `event_time_utc, comment_id`.

The generated `rag_compatibility/` directory contains the unchanged legacy filenames
and an explicit `candidate_id → event_id` alias. The existing non-generative RAG
consumer can read this view without schema changes. G-1/G-2 execution remains a
separate future phase.
