# Monitoring topic configuration and G-1→G-2 gate

## Monitoring topic

`TopicProfile` is the configuration authority for a monitored stance task. It
contains one `topic_id`, description, explicit Likert proposition, context, and the
approved `comment_stance_likert5` prompt identity/version. The JSON profile is
referenced by an experiment and expanded into the resolved `RunConfig`.

The profile is immutable. Its proposition/context are passed unchanged to the
classifier and recorded in the classifier and causal-replay manifests. Reusing
`comment_stances.jsonl` requires the same source hash, canonical comment IDs,
`topic_id`, complete proposition/context snapshot and context hash, prompt/model,
input policy, and classification policy.

## Actual G-1 output contract

The existing G-1 structured report exposes:

- `validation_status`: `confirmed`, `partially_confirmed`, `not_confirmed`,
  `ambiguous`, or `insufficient_evidence`;
- `event_interpretation`: internal reaction, possible external event, noise,
  disinformation, or unclear;
- `confidence_label`: `low`, `medium`, or `high`;
- summaries, cited comment/context-unit IDs, used comments, limitations, model and
  prompt lineage.

There is no separate evidence-sufficiency field. Sufficiency is represented by the
validation status and citation verification.

## Gate semantics

`G2GateDecision` answers only whether to spend the additional G-2 resources. It
does not confirm or reject the candidate. `OFF` always emits `RUN_G2` and is the
regression baseline. `PERMISSIVE` and `STRICT` require an explicit set of current
G-1 statuses; the repository deliberately does not assign definitive status sets to
those labels before calibration.

G-2 execution states are `EXECUTED`, `SKIPPED_BY_G1_GATE`, `NOT_ELIGIBLE`, and
`FAILED`. A skipped result retains G-1 and produces a valid
`CandidateValidationResult` without fabricating external evidence.

## Audit instrumentation

Stance rows record root/reply status and per-comment provider token usage when
reported. Candidate RAG manifests separate G-1/G-2 token usage and record inventory
versus selected comments, context units, videos, and threads. External evidence
projections retain candidate reference time, candidate creation cutoff, evidence
cutoff, source publication time, and lag from the candidate reference time.

No Serper temporal rule, repeated monitoring policy, context budget, selection
algorithm, detector parameter, association tolerance, or evidence window is changed
by this phase.
