# Neutral event-candidate projection

**Status:** IMPLEMENTED; READY FOR A FUTURE EVIDENCE POLICY.

## Boundary

The new causal route projects, without re-association:

```text
DetectionAssociation(status=FINALIZED)
→ association_to_candidate_v1
→ AssociatedEventCandidate(status=READY_FOR_EVIDENCE)
```

One finalized association produces exactly one candidate. A candidate is an event
hypothesis that still requires contextualization and validation; it is not a
confirmed event. Pending clusters cannot cross this boundary. The projection does
not compare timestamps, apply the association tolerance, find alerts, combine
detector scores, rank candidates, or retrieve evidence.

The legacy `EventCandidate` remains unchanged. It embeds an `ActivityObservation`, a
single `DetectionResult`, and an evidence interval, so it is not used to represent
polarization or joint associations. `AssociatedEventCandidate` is a separate opt-in
contract for the new route.

## Candidate types

The type is copied directly from the association:

```text
ACTIVITY              → activity alert IDs only
POLARIZATION          → polarization alert IDs only
ACTIVITY_POLARIZATION → at least one alert ID from each family
```

These categories are descriptive. A joint candidate is not assigned greater
confidence, probability, severity, or importance. An activity-only candidate means
only that no polarization alert was associated under that run's configuration; it
does not prove that polarization was absent. The inverse applies to a
polarization-only candidate.

## Identity, time, and lineage

`candidate_id` is a deterministic SHA-256-derived identifier over `run_id`,
`association_id`, candidate type, and projection-policy ID. It does not depend on
wall-clock write time. Incremental projection retains `promoted_association_ids`
in-run so a repeated association is not emitted twice. Durable state across process
restarts remains deferred under A9.

Candidate times are copied, not inferred:

```text
start_time_utc     = association.start_time_utc
end_time_utc       = association.end_time_utc
reference_time_utc = association.reference_time_utc
```

`created_at_cutoff_utc` must be at least the association's
`finalized_at_cutoff_utc`. Associations finalized at end-of-stream are valid inputs.
The candidate interval is the extent of its alerts and is not an evidence window.

The compact lineage is:

```text
candidate_id
→ association_id / association_ref
→ activity_alert_ids[] + polarization_alert_ids[]
→ DetectionAlert.detection_result_ref
→ observation/source artifacts
```

The candidate also records `run_id`, `config_hash`, `input_policy_id`, association
policy ID, association tolerance, and its explicit experimental parameter status.
The projector enforces a fixed association-policy signature within one execution;
different runs may use different tolerances without changing Python.

`associated_video_ids` is a zero-or-many lineage field. It is populated only when
the upstream runtime already has an authoritative mapping. An empty tuple means
video lineage has not yet been resolved; the projector never guesses a video and
does not scan comments. This supports multi-video candidates without prematurely
implementing evidence retrieval.

## Persistence

The runtime writes:

```text
event_candidates.jsonl
event_candidate_manifest.json
```

The candidate artifact contains no comments, evidence window, detector score,
combined score, confidence, ranking, validation label, or RAG output. The manifest
records the projection policy, counts by candidate type, the input and association
policy lineage, tolerance and parameter status, artifact hash, causal cutoff, and
explicit `false` flags for evidence retrieval and RAG execution.

## Existing sidecars

The current retrospective RAG sidecars use `event_id`, trigger/evidence-window
bounds, comment inventories, and event-to-video maps. They are activity-specific and
are intentionally unchanged. A future evidence-policy adapter can use
`candidate_id` as its join identity, derive a separate evidence interval, and build
comment/video/thread inventories from the canonical Silver universe. This requires
an adapter, not a change to the neutral candidate or to existing sidecars.

Changing detector parameters, signal windows/cadences, gap policy, or association
tolerance can change associations and candidate types between runs. That is an
expected experimental outcome, not a contract inconsistency.
