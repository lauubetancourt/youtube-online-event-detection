# Candidate evidence to RAG integration

## Boundary

The causal route is additive and opt-in:

```text
AssociatedEventCandidate
  -> candidate evidence package and complete Silver inventory
  -> explicit candidate_id/event_id compatibility view
  -> existing non-generative RAG consumer
  -> existing G-1
  -> existing hierarchical G-2
  -> CandidateValidationResult
```

Detection, association, candidate projection, and evidence construction are not
re-run by this stage. Inputs are the persisted `event_candidates.jsonl` and
candidate evidence directory.

## Reused retrospective implementation

The retrospective implementation remains authoritative for context-unit
construction, consumer payloads, G-1 labels and prompt, hierarchical G-2 video
validation, and deterministic candidate-level synthesis:

```text
legacy event
  -> event evidence package
  -> complete comment inventory and video/thread maps
  -> deterministic context units
  -> rag_validation_inputs / rag_context_payloads
  -> G-1 internal validation
  -> G-2 per-video external validation
  -> deterministic event summary
```

Legacy artifacts and consumers remain unchanged. The compatibility directory
uses `event_id := candidate_id` only as a transport alias. Canonical outputs
restore `candidate_id` and expose the alias as `compatibility_event_id`.

## Evidence and context coverage

The complete evidence inventory remains outside model context. The existing
consumer creates deterministic context units and currently selects all units.
The integration records inventory count, selected count, selected fraction, and
the unchanged G-1 capacity audit. If G-1 exceeds its token limit, its existing
behavior emits `insufficient_evidence` without an API call; the inventory is not
truncated.

## G-1 semantics

G-1 remains internal YouTube-community validation:

- model: `gpt-5-mini`;
- prompt: `rag_validation_prompt_v0.1`;
- existing validation labels and citation verification are preserved.

`confirmed` therefore means an internally supported conversational pattern,
not confirmation of an external public event.

## Hierarchical G-2 semantics

The existing hierarchical G-2 remains unchanged in its task:

- query and validation models: `gpt-5-mini`;
- prompts: `rag_video_news_query_prompt_v0.2` and
  `rag_video_validation_g2_prompt_v0.2`;
- provider: Serper News;
- validation unit: candidate/video;
- synthesis: deterministic candidate aggregation without transferring evidence
  between videos.

The adapter does not use stance propositions as news queries and does not expose
detector internals or polarization values as semantic evidence.

## External-web causal boundary

Legacy G-2 behavior remains unchanged when no cutoff is configured. Candidate
RAG always supplies the evidence cutoff. Under
`published_strictly_before_evidence_cutoff_v1`:

1. the provider date range is capped at the cutoff calendar date;
2. results with `published_at >= evidence_cutoff_utc` are excluded;
3. results with unknown/unparsable publication time are excluded;
4. only filtered results reach the G-2 validation payload;
5. exclusions and `future_leak_count` are recorded.

A historical web index cannot prove that an older page was retrievable at the
original cutoff. That residual limitation must remain visible in online-like
claims.

## Candidate validation result

`CandidateValidationResult` is downstream and never mutates the candidate. It
preserves candidate/evidence lineage, G-1 internal result, hierarchical G-2
summary, context coverage, and causal audits. Its synthesis keeps internal and
external statuses separate and creates no detector/RAG score.

External evidence absence retains the existing `insufficient_evidence`
semantics and is not proof that the candidate is false.

## Evidence-state gates

Only `EVIDENCE_READY` executes RAG.

| Evidence status | Behavior |
| --- | --- |
| `EVIDENCE_READY` | Build payloads and run G-1/G-2 when explicitly enabled. |
| `NO_EVIDENCE_FOUND` | Persist a result without model/provider execution. |
| `EVIDENCE_PENDING_HORIZON` | Defer RAG until delayed evidence is ready. |
| `EVIDENCE_BLOCKED_MISSING_VIDEO_LINEAGE` | Block RAG; never invent videos. |

The same path applies to `ACTIVITY`, `POLARIZATION`, and
`ACTIVITY_POLARIZATION`.

## Execution and artifacts

The CLI prepares by default and requires `--execute` for network-capable stages:

```text
python scripts/run_candidate_rag_validation.py --config RUN.json
python scripts/run_candidate_rag_validation.py --config RUN.json --execute
```

Outputs are `candidate_g1_results.jsonl`,
`candidate_g2_video_results.jsonl`, `candidate_g2_results.jsonl`,
`candidate_validation_results.jsonl`, and
`candidate_rag_validation_manifest.json`, plus unchanged stage artifacts under
G-1/G-2 subdirectories.

The manifest records models, prompt versions, policies, source hashes, context
coverage, token usage when returned, external-call metadata, candidate time,
evidence-ready time, and RAG execution time.

## Open decisions

- evidence-window, association, detector, and RAG calibration;
- candidate prioritization;
- RAG accuracy/prompt evaluation;
- historical web-index availability at the original cutoff;
- production durability and resumable multi-batch G-2 execution.

