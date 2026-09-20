# Temporal detection association

**Status:** IMPLEMENTED; EXPERIMENTAL AND NOT CALIBRATED.

## Boundary

The association stage consumes only triggered neutral alerts:

```text
DetectionAlert(ACTIVITY) ─┐
                         ├─ bounded timestamp association ─ DetectionAssociation
DetectionAlert(POLARIZATION) ─┘
```

It does not inspect tick, observation or cycle indexes. Activity and polarization
may use different signal windows and cadences. Association is structural rather than
probabilistic: it does not combine detector scores, rank alerts or claim an external
event.

`DetectionAssociation` is warranted as a separate contract because it records a
buffering/finalization decision and can contain several alerts from either family.
Those concerns do not belong in `DetectionResult`, `DetectionAlert`, or the current
activity-specific `EventCandidate`.

## Policy V1

```text
policy_id = bounded_temporal_clusters_v1
membership boundary = closed (<= Δ)
anchor = earliest unassigned alert_time_utc
cluster = every unassigned alert with time <= anchor + Δ
finalize = cutoff > anchor + Δ, or explicit end-of-stream
```

The cluster span is therefore bounded by `Δ`. Membership never extends the boundary:

```text
A1 10:00, P1 10:04, A2 10:08, Δ=5min
→ [A1, P1]
→ [A2]
```

Although `P1` is close to `A2`, it cannot transitively join alerts whose distance
from the cluster anchor exceeds `Δ`. Every alert belongs to exactly one association.
Several alerts from the same family may share one cluster; their `signal_id` and
`detector_id` remain available through their alert records.

The parameter `Δ` is `METHODOLOGICAL / EXPERIMENTAL`. It is never derived
automatically from either branch's window or cadence. Supported configuration labels
are deliberately explicit:

```json
{
  "association": {
    "policy_id": "bounded_temporal_clusters_v1",
    "tolerance": "5min",
    "parameter_status": "DEVELOPMENT_NOT_CALIBRATED"
  }
}
```

Tests use `parameter_status=TEST_ONLY`. No value is recommended or calibrated here.

## Causal buffering

An immediate provisional output would minimize latency, but it would require
retrospective mutation or duplicate candidate suppression whenever a counterpart
arrived later. V1 instead uses a bounded logical buffer. This adds at most the
configured tolerance before ordinary finalization, but produces immutable,
reproducible associations and avoids provisional candidate revisions.

An alert may enter the buffer only when its own `data_cutoff_utc` is not later than
the current association cutoff. Cutoffs cannot regress. A cluster is not emitted when
the cutoff is exactly its closed boundary because an alert at that boundary may not
yet be causally available; finalization requires a strictly later cutoff.

At end-of-stream, every pending cluster is finalized as `ACTIVITY`, `POLARIZATION`,
or `ACTIVITY_POLARIZATION`, so unmatched alerts are not lost. Late arrivals that
would revise an already finalized boundary are rejected explicitly. Watermarks and
durable late-arrival handling remain part of deferred A9 work.

## Contract and time semantics

The association artifact stores only:

```text
association_id                     deterministic
run_id
policy_id
tolerance
parameter_status                  TEST_ONLY | DEVELOPMENT_NOT_CALIBRATED
association_type                  ACTIVITY | POLARIZATION | ACTIVITY_POLARIZATION
activity_alert_ids[]
polarization_alert_ids[]
start_time_utc                    minimum alert time
end_time_utc                      maximum alert time
reference_time_utc                end_time_utc
finalized_at_cutoff_utc
status                            FINALIZED
finalization_reason               TOLERANCE_ELAPSED | END_OF_STREAM
```

The ID hashes the run, policy, tolerance, closed-boundary semantics and sorted alert
IDs. It is independent of input ordering. V1 associates only alerts with the same
`run_id`. Source branch `config_hash` values may differ; they are retained through
the source alerts and summarized in the manifest rather than required to match.

The association interval is the extent of alert times. It is not an evidence window.
The future evidence layer may use a different interval and policy.

## Persistence and downstream boundary

The runtime writes:

```text
detection_associations.jsonl
detection_association_manifest.json
```

The manifest records policy parameters, closed-boundary and finalization rules,
input counts by family, association counts by type, source artifact references,
source config hashes, input-policy identity and causal cutoff. It explicitly records
that candidates, evidence, and RAG were not executed.

The existing `EventCandidate` embeds one `ActivityObservation`, one
`DetectionResult`, and an activity evidence window. It remains unchanged for legacy
paths. Finalized associations can now be projected one-to-one into the separate
neutral `AssociatedEventCandidate` contract, retaining all alert IDs while leaving
evidence-window policy downstream. See
[neutral event-candidate projection](event_candidate_fusion.md).
