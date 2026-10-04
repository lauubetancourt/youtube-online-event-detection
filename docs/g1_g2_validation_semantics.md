# G1/G2 validation semantics and query contract

## Scope

G1 and G2 answer independent questions:

- **G1 — internal validation:** does the available YouTube evidence support a
  coherent conversational pattern associated with the candidate?
- **G2 — external corroboration:** does sufficient, relevant and causally
  available external evidence corroborate the phenomenon/candidate?

No score or status is fused across these stages. New candidate results expose
`internal_validation_status` and `external_validation_status` as the two
authoritative fields. The former global `validation_status` projection is not
written in new candidate synthesis artifacts.

## Three-state contract

Both stages emit only:

- `CONFIRMED`: the evidence for that stage is sufficient and supports its
  hypothesis.
- `INSUFFICIENT_EVIDENCE`: the evidence does not permit a sufficiently
  supported decision; this includes partial, ambiguous, weak, missing or
  inconclusive evidence.
- `NOT_CONFIRMED`: sufficient evidence exists to evaluate the hypothesis, but
  does not support it or substantively contradicts it.

For G2, `CONFIRMED` and `NOT_CONFIRMED` require evaluable external evidence and
a valid external citation. Internal comments alone can never produce either
decisive G2 status. Absence of external evidence always maps to
`INSUFFICIENT_EVIDENCE`; it does not reject the internal candidate.

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

## D1-C1-B structural audit

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
