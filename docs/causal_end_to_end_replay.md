# Causal end-to-end historical replay

The end-to-end runner simulates online availability over historical Silver
comments. It is an experimental replay, not a claim of production live
streaming and not a scientifically calibrated event-detection method.

## Implemented flow

```text
Silver comments
→ causal_minimal_v1
→ fixed causal clock
├→ activity window/cadence → configured detector
└→ causally released CommentStance
   → polarization window/cadence → configured detector
→ triggered DetectionAlert values
→ bounded temporal association
→ AssociatedEventCandidate
→ causal candidate evidence from the same Silver universe
→ existing G-1 and hierarchical G-2 compatibility path
→ CandidateValidationResult
```

`current_data_cutoff_utc` is the shared temporal authority. A comment, stance,
signal tick, alert, evidence record, or external record is not released before
its applicable strict cutoff. Precomputed stances are validated against the
configured model, prompt, proposition, canonical comment IDs, authors, and
event times, then released only with their source comments.

## Independent schedules

`simulation.causal_end_to_end.step`, `signals.activity.cadence`, and
`signals.polarization.cadence` are independent. Each replay cutoff processes all
signal ticks strictly before that cutoff; a coarse replay step therefore does
not discard intervening activity or polarization ticks. Activity and
polarization windows also remain independent.

All temporal, detector, association, evidence, and RAG choices remain
`TEST_ONLY` or `DEVELOPMENT_NOT_CALIBRATED`. The gap policy is explicitly
`preserve_state_v1` and is not calibrated.

## Execution and traceability

The opt-in entrypoint is:

```bash
python scripts/run_causal_end_to_end.py --config CONFIG.json --base-dir .
python scripts/run_causal_end_to_end.py --config CONFIG.json --base-dir . --execute
```

Without `--execute`, the command only prints the resolved plan. Execution keeps
the intermediate observations, detections, alerts, associations, candidates,
evidence sidecars, RAG results, resolved configuration, and aggregate
`causal_replay_manifest.json`.

The manifest distinguishes replay stepping from signal cadence; records stage
counts, routes, input/evidence policies, parameter status, artifacts, isolated
candidate failures, and all auditable future-leak counters. A run with zero
alerts or zero candidates is successful and produces empty downstream
artifacts.

## Deliberate limits

- No parameter calibration or detector comparison.
- No proposition discovery.
- No durable checkpoint, watermark, crash recovery, or late-arrival model.
- Historical web retrieval cannot prove that a page was indexed at the
  historical cutoff; published-time filtering remains explicit.
- Gold and all legacy retrospective/RAG paths remain unchanged.
