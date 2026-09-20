# Detection alerts and detector-gap policy

**Status:** IMPLEMENTED CONTRACT; PROVISIONAL, NOT-CALIBRATED GAP POLICY.

## Decision levels

The pipeline keeps four different objects:

```text
SignalObservation   numeric value or explicit absence at one signal tick
DetectionResult     detector decision for one available numeric observation
DetectionAlert      triggered result plus association-ready runtime lineage
EventCandidate      downstream event hypothesis, not created here
```

`DetectionResult` remains the complete audit record for both triggered and
non-triggered decisions. A `DetectionAlert` is a small, deterministic projection
created only when `triggered=true`; it is not a second statistical decision and is
not evidence that an external event occurred.

## Alert contract

The common envelope contains:

```text
alert_id                         deterministic
detection_result_id              deterministic source-result identity
detection_result_ref             artifact + result identity
run_id
config_hash
signal_id
signal_family                    ACTIVITY | POLARIZATION
detector_id
alert_time_utc                   originating observation_time_utc
data_cutoff_utc
triggered                        always true by invariant
quality                          propagated unchanged
support_count                    integer or null
```

The identity derives from run, signal, detector and observation time; no random UUID
or processing timestamp participates. Activity and polarization can therefore emit
alerts at different cadences without index alignment. Causality requires
`alert_time_utc < data_cutoff_utc`.

The persisted artifact is `detection_alerts.jsonl`. It does not copy detector
metadata, scores, values, comment text, evidence or association state. Those remain
in the referenced detection-results artifact and upstream lineage.

## Gap-policy authority

River defines how Page-Hinkley, ADWIN and KSWIN update when a numeric value is
provided. The pipeline—not River—defines whether two values separated by absent
signal ticks belong to one continuous detector stream. Gap policy is therefore a
methodological/runtime policy.

The four evaluated policies are:

| Policy | Behavior | Benefit | Risk |
|---|---|---|---|
| G0 / `preserve_state_v1` | Skip absence; keep state | Preserves evidence and current behavior; no arbitrary threshold | Collapses elapsed clock time between valid values |
| G1 | Reset after any gap | Never compares across missing intervals | Short or ordinary gaps repeatedly erase warmup/history |
| G2 | Reset after a clock-duration threshold | Distinguishes short and long gaps | Requires an unvalidated `Δ_gap`; results depend on signal cadence and threshold |
| G3 | Treat valid segments independently | Makes discontinuity explicit and auditable | Needs segment identity and a later rule for cross-segment interpretation |

V1 adopts G0 explicitly:

```text
gap_policy_id = preserve_state_v1
absent observation → no detector update
                    → no DetectionResult
                    → detector instance/state unchanged
                    → no alert
```

Its methodological status is `NOT_CALIBRATED`. No zero, interpolation, reset or
elapsed-time adjustment is introduced. Legacy configurations that omit the new block
retain exactly this historical behavior and are recorded as
`legacy_implicit_equivalent`; new experiments should configure it explicitly:

```json
{
  "detection": {
    "gap_policy": {"policy_id": "preserve_state_v1"}
  }
}
```

The explicit block participates in resolved configuration and `config_hash`.

## Detector implications

- **Page-Hinkley:** G0 preserves accumulated mean/deviation state. A post-gap level
  is treated as the next sample; G1/G2/G3 restart warmup and discard prior evidence.
- **ADWIN:** G0 preserves its adaptive observation window. Clock time is not part of
  that window; reset policies discard the pre-gap sample distribution.
- **KSWIN:** G0 preserves reference/recent scalar samples and seeded sampling state.
  It still compares scalar signal values, not opinion distributions. Reset policies
  construct a fresh River instance through the existing factory.

V1 performs no reset. If a future calibrated policy adopts G1, G2, or G3, the
runtime must replace the route's detector with a fresh instance built by the existing
factory and resolved detector config; it must not mutate River's private state or add
an artificial `reset()` API.

A **TEST_ONLY** sanity stream used 120 stable low values, a conceptual long gap and
120 stable high values. With deliberately sensitive test parameters, G0 produced one
post-gap trigger for each detector; recreating the detector at the boundary produced
zero. This demonstrates that gap policy materially affects results; it does not show
that either behavior is scientifically preferable.

## Activity versus polarization

Current activity observations always carry numeric values, including genuine zero
activity where defined; they do not emit the polarization absence statuses. Thus the
policy does not alter current activity behavior. It remains the compatible rule if a
future activity signal gains explicit absences.

Polarization can emit `NO_INPUT`, `NO_CLASSIFIABLE_OPINIONS`, and
`NO_REPRESENTABLE_AUTHORS`. These retain different diagnostic meanings, but V1 gives
them the same detector-state action: skip and preserve. Distinguishing their reset
behavior would require empirical evidence and is not encoded now.

The real six-tick smoke has three leading `NO_CLASSIFIABLE_OPINIONS` ticks followed
by three valid ticks for every measure. Because no detector history exists before
those leading gaps, G0, G1, G2 and G3 would all begin the first effective segment at
19:50 UTC. That artifact therefore supplies no evidence for a reset threshold.

## EventCandidate compatibility

The existing `EventCandidate` is activity-specific because it embeds
`ActivityObservation` and an evidence window. It can continue serving legacy paths.
Future association should consume `DetectionAlert` records, then create or adapt a
candidate after timestamp-based association. No `EventCandidate`, evidence package,
RAG artifact, candidate taxonomy or association tolerance is created in this phase.
