# G1/G2 validation-state contract

## Scope

G1 and G2 answer independent questions:

- **G1 — internal validation:** does the available YouTube evidence support a
  coherent conversational pattern associated with the candidate?
- **G2 — external corroboration:** does sufficient, relevant and causally
  available external evidence corroborate the phenomenon/candidate?

No score or status is fused across these stages. A result has five separate
dimensions: evidence state, execution status, completeness status, validation
status, and reason code. No field may represent two of them.

## Three-state contract

Both stages emit only:

- `CONFIRMED`: the evidence for that stage is sufficient and supports its
  hypothesis.
- `INSUFFICIENT_EVIDENCE`: the stage completed its work and evaluated all
  available evidence, but that evidence does not support a sufficiently
  grounded conclusion.
- `NOT_CONFIRMED`: sufficient evidence exists to evaluate the hypothesis, but
  does not support it or substantively contradicts it.

`INSUFFICIENT_EVIDENCE` is valid only when completeness is `COMPLETE`.
Technical failures and partial batches have `validation_status=null`. This
null means that no valid semantic decision exists. It does not mean
`INSUFFICIENT_EVIDENCE`.

For G2, `CONFIRMED` and `NOT_CONFIRMED` require evaluable external evidence and
a valid external citation. Internal comments alone can never produce either
decisive G2 status. A complete G2 run with no adequate external evidence may
produce `INSUFFICIENT_EVIDENCE`; an incomplete search may not.

## Execution and completeness

`execution_status` permits only `EXECUTED`, `NOT_EXECUTED`, `NOT_ELIGIBLE`, and
`FAILED`. `EXECUTED` means that the stage produced a technically valid stage
artifact. It does not imply that every batch finished.

`completeness_status` permits only `COMPLETE` and `PARTIAL`. G2 is complete
only when all five conditions hold:

```text
summary_completeness == "complete"
event_batch_status == "complete"
next_batch_required == false
pending_video_ids == []
failed_video_ids == []
```

Every other combination is partial. A partial result never publishes an
authoritative semantic validation status.

## Reason codes

The stable reason codes are:

- `NO_EVIDENCE`
- `CONTEXT_LIMIT_EXCEEDED`
- `INVALID_QUERY`
- `INVALID_CITATION`
- `MISSING_PAYLOAD`
- `PROVIDER_ERROR`
- `TIMEOUT`
- `PENDING_BATCHES`
- `VIDEO_FAILURES`

The reason is null when no reason applies. When several G2 failures coexist,
the read normalizer uses this priority: missing payload, invalid query, invalid
citation, other video failure, then pending batches. Thus a failed video takes
precedence over a pending batch, and a known specific failure takes precedence
over generic `VIDEO_FAILURES`.

## Candidate validation result

New candidate results expose these fields at the top level and in the derived
synthesis:

```text
internal_validation_status
internal_execution_status
internal_completeness_status
internal_reason_code

external_validation_status
external_execution_status
external_completeness_status
external_reason_code
```

There is no global validation status and no score fusion. A legacy field named
`validation_status` or G2's historical `overall_validation_status` is not an
authoritative candidate-level decision.

Official human-readable example:

```text
Internal validation — G1:
CONFIRMED

External corroboration — G2:
INSUFFICIENT_EVIDENCE

Interpretation:
There is sufficient evidence of an internal conversational pattern on
YouTube, but insufficient external evidence to corroborate it.
```

## Legacy read compatibility

Legacy values are normalized only when read:

| Legacy value | Authoritative value |
|---|---|
| `confirmed` | `CONFIRMED` |
| `partially_confirmed` | `INSUFFICIENT_EVIDENCE` |
| `ambiguous` | `INSUFFICIENT_EVIDENCE` |
| `insufficient_evidence` | `INSUFFICIENT_EVIDENCE` |
| `not_confirmed` | `NOT_CONFIRMED` |

Unknown values fail explicitly. New model schemas and generated artifacts do
not emit legacy spellings.

Historical G1 reports with `generation_error=true` and a
`context_limit_exceeded` limitation normalize to `FAILED`, `PARTIAL`, null
validation, and `CONTEXT_LIMIT_EXCEEDED`. Historical partial G2 summaries
normalize to `EXECUTED`, `PARTIAL`, and null validation. Their reason is
derived from failed videos, pending batches, query records, and raw generation
records. Historical JSONL files are not rewritten.

`NO_EVIDENCE_FOUND` is a legitimate terminal evidence state. For both G1 and
G2 it maps to `NOT_ELIGIBLE`, `COMPLETE`, null validation, and `NO_EVIDENCE`.

## Resume limitation

A candidate that requires G2 is semantically terminal only when external
completeness is `COMPLETE`. The current resume implementation, however, can
restart a candidate directory but cannot continue pending G2 batches from the
saved batch boundary. Enforcing the semantic rule in resume now would archive
and rerun partial candidates from the beginning and could repeat the same
partial outcome. This phase therefore changes normalization, manifests, and
reporting only. Incremental batch continuation remains a separate design task.

## G2 query generation

The hierarchical G2 path uses `METADATA_FIRST_DETERMINISTIC`:

1. A non-empty, non-generic video title is sufficient.
2. The deterministic query starts with the title and may add channel and video
   date when present and not already represented.
3. Exact generic titles (`Video`, `En vivo`, `Live`, `Stream`, `Untitled`) or a
   missing title activate `LLM_FALLBACK`.
4. The fallback receives evidence from one video and returns only
   `{"query": "..."}`. It may extract observable entities, topics or phrases,
   but may not infer an event type absent from the evidence.

Every new query record persists `query_text`, `video_id`,
`query_generation_mode`, and `metadata_fields_used`. Fallback-only provenance
(`model_name`, `query_prompt_version`, and `generated_at_utc`) is assigned by
the runtime. `claim_verification_query` is neither requested nor persisted in
new query artifacts; extra legacy fields remain harmless when old records are
read.

The fallback keeps the existing query model, uses reasoning `low`, and caps
the structured output at 512 tokens. G1 and G2 Validation retain their existing
models and reasoning behavior.

## Traceability and compact prompts

Runtime-side inventories remain the authority for citation allowlists and
verification. Prompt payloads carry evidence text once:

- context units preserve IDs, membership and temporal/thread relationships;
- comment records preserve the text used by the model;
- repeated `context_text`, top-level duplicate ID lists and runtime-known
  provenance are omitted from the model schema;
- full source artifacts and runtime citation verification remain unchanged.

The causal cutoff, video isolation, external evidence filter, evidence policy,
hierarchical synthesis, candidate generation, signals, detectors and
association are unchanged.

## D1-C1-B-UP validation-state audit

The read-only audit covered 210 candidates. It made no OpenAI or Serper calls
and did not rewrite historical artifacts.

| Dimension | State | Count |
|---|---|---:|
| Evidence | `EVIDENCE_READY` | 199 |
| Evidence | `NO_EVIDENCE_FOUND` | 11 |
| G1 complete | `CONFIRMED` | 166 |
| G1 complete | `INSUFFICIENT_EVIDENCE` | 21 |
| G1 complete | `NOT_CONFIRMED` | 1 |
| G1 not evaluated | `CONTEXT_LIMIT_EXCEEDED` | 11 |
| G1 not eligible | `NO_EVIDENCE` | 11 |
| G2 complete | `CONFIRMED` | 6 |
| G2 complete | `INSUFFICIENT_EVIDENCE` | 131 |
| G2 complete | `NOT_CONFIRMED` | 0 |
| G2 partial | `PENDING_BATCHES` | 48 |
| G2 partial | `MISSING_PAYLOAD` | 1 |
| G2 partial | `INVALID_QUERY` | 12 |
| G2 partial | `VIDEO_FAILURES` | 1 |
| G2 not eligible | `NO_EVIDENCE` | 11 |

The `MISSING_PAYLOAD` case is the known fallback-query response that ended as
`incomplete` with `max_output_tokens`. The historical flow continued to video
validation, but the read normalizer does not count it as processed success.

## Earlier D1-C1-B structural payload audit

This audit did not call OpenAI or Serper. It reconstructed pre/post JSON payload
sizes from persisted pilot fixtures; token counts below use characters/4 only
as a relative structural estimate, not provider billing usage.

| Stage | Payloads | Before chars | After chars | Reduction |
|---|---:|---:|---:|---:|
| G1 | 13 | 154,883 | 125,275 | 19.12% |
| G2 Query | 26 | 133,956 | 70,332 | 47.50% |
| G2 Validation | 26 | 287,856 | 129,385 | 55.05% |

The 26 processed pilot video units had no populated title or channel metadata:

- deterministic metadata queries: 0/26 (0%);
- LLM fallback queries: 26/26 (100%).

Therefore the pilot itself demonstrates the fallback path, not query-call
avoidance. Avoided-call rates for Dataset 1 must be measured only after video
metadata is present in the input contract; no extrapolation is made here.
