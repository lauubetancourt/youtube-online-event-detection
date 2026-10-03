# LLM token usage and context budget audit

Status: audit tooling and retained-artifact baseline complete. No prompt, budget,
selection, gate, or Serper behavior changed.

The audit separates three quantities that are not interchangeable:

- provider-reported tokens from persisted API responses;
- application estimates such as G-1's serialized-payload `characters / 4`;
- evidence coverage: inventory records represented in selected model context.

## Retained data baseline

- Stance: one 21-comment real smoke using `gpt-4.1-2025-04-14`.
  Provider usage is complete at run level (10,065 input, 236 output), but the old
  JSONL predates row-level token persistence. Per-comment billed usage and an
  exact root/reply comparison are therefore unavailable.
- G-1: 10 candidates have complete context payloads, but only 2 have real model
  responses with usage.
- G-2: 10 candidates, 40 executed video queries, 40 query-generation calls, and
  40 video-validation calls have provider usage. Candidate synthesis is
  deterministic and adds no model call.

The historical identities are legacy `event_id` values used only as explicit
candidate compatibility keys.

## Context coverage

The full evidence inventory contains 292 comments, 69 context units, 51
candidate-video memberships, and 289 candidate-thread memberships. The retained
G-1 payloads select all of them: comment, unit, video, and thread coverage are
100% for every candidate. No candidate was truncated.

Current selected-context estimates range from 4,610 to 14,461 tokens, with P50
5,544.5, P75 9,750.5, and P90 10,894.3. These are serialized-payload estimates,
not provider-token measurements. The two actual G-1 calls used 8,402 and 25,450
input tokens; the difference from their estimates was 3,234 and 10,989 tokens.
This delta includes prompt/schema overhead and estimator error and must not be
treated as a clean prompt-overhead measurement.

## Observed usage

The stance run averages 479.29 input, 11.24 output, and 490.52 total
provider-reported tokens per attempt at aggregate level. Character/4 reconstruction
estimates a 421.74-token root mean and 452-token reply mean. The approximately
30-token difference is descriptive only because there are just two replies and
no row-level provider usage.

G-1's two calls total 39,565 tokens. This sample is too small for budget or
stability conclusions.

G-2 totals 519,972 tokens:

- query generation: 153,156 tokens across 40 calls;
- video validation: 366,816 tokens across 40 calls;
- mean candidate cost: 51,997.2 tokens;
- 40 inferred Serper requests returned 166 sources; 77 were used/cited.

Cost grows with evaluated videos because the hierarchical route performs one
query-generation and one validation call per processed video. In this sample,
candidates with two processed videos average 40,085 tokens; four videos use
59,459 tokens; five videos average 56,709.5 tokens. These groups are small and
not a calibrated cost model.

## Gate counterfactual

Baseline remains `OFF`. Two descriptive, uncalibrated policies are materialized:

- confirmed or partially confirmed;
- confirmed, partially confirmed, or ambiguous.

Only two candidates have both G-1 and G-2 results, and both are G-1 `confirmed`
with external support found by G-2. Consequently both policies save zero calls,
requests, or tokens on the evaluable subset. Eight candidates remain
`NOT_EVALUABLE_MISSING_G1`; corpus-wide savings and methodological risk cannot be
estimated from the retained data.

## Budget candidates, not decisions

The audit publishes observed G-1 estimate percentiles as experiment inputs:

- P50: 5,544 estimated tokens;
- P75: 9,750;
- P90: 10,894;
- observed maximum: 14,461.

They are marked `AUDIT_DERIVED_NOT_CALIBRATED`. Applying them directly would
change context coverage and requires a stability experiment comparing G-1 status,
interpretation, confidence, and citations at different budgets.

## Reproducible execution

```bash
.venv/bin/python scripts/run_llm_token_context_audit.py \
  --config configs/audits/llm_token_context_audit.json
```

The command is deterministic and has no network or model execution path. It
produces per-comment stance rows, per-candidate/stage RAG rows, gate
counterfactuals, a summary, and a source-hashed manifest under
`outputs/llm_token_context_audit/`.

## Decision

No context-budget implementation change is justified yet. G-1 currently retains
full evidence, stance lacks historical row-level attribution, and gate savings
are observable for only 2/10 candidates. The clearest cost center is hierarchical
G-2, but choosing a gate or reducing per-video context without broader G-1/G-2
paired evidence would be premature. Deterministic token-aware packing, thread
integrity, temporal/video balance, ranking, and finally generative compression
remain candidate strategies for later experiments in that order.
