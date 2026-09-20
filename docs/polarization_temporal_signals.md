# Temporal polarization signal

**Status:** TECHNICAL_RUNTIME_INTEGRATION, not calibrated or scientifically validated.

```text
ordered CommentStance causal prefix
→ closed temporal window
→ last CLASSIFIED stance per author
→ one OpinionDistribution per tick
→ enabled formal-measure adapters
→ one SignalObservation per measure
```

The producer remains independently testable with synthetic stances. A dedicated
standalone runtime connects persisted classifier output to it and can optionally
route available scalar observations to configured neutral detectors. Candidates,
cross-branch association, and RAG remain disconnected. Activity's contracts and
behavior are unchanged. Measures continue to use the pinned `pol_measures` commit
`1efd4fe3f083e478cb91a6f3ca078cad01386a80`; no formulas are copied here.

## Configuration and composition

Example with **TEST_ONLY** durations/coverage and library technical parameter defaults:

```json
{
  "identity": {"run_id": "synthetic_polarization_01"},
  "signals": {
    "polarization": {
      "proposition_id": "given_proposition",
      "classification_run_id": "synthetic_classification_01",
      "window": "60s",
      "cadence": "30s",
      "start_time_utc": "2026-01-01T00:00:00+00:00",
      "end_time_utc": "2026-01-01T00:03:01+00:00",
      "enabled_measures": ["esteban_ray", "emd_pol", "mec"],
      "interval_policy": "closed",
      "distribution_policy_id": "likert5_author_latest_classified_v1"
    }
  },
  "polarization": {
    "esteban_ray": {},
    "emd_pol": {},
    "mec": {}
  }
}
```

`load_run_config` and `resolve_run_config` expose the effective temporal definition
and public measure parameters. `create_configured_polarization_signal(config)`
composes a standalone producer from the resolved `RunConfig`; no runner was added.
Temporal activation lives only in `signals.polarization.enabled_measures`; parameter
blocks live in `polarization`. Every enabled measure requires its typed parameter
block. Isolated `polarization.measure_id` remains backward compatible, and is
optional only when a temporal definition is present. A retained isolated selector
does not limit the temporal list. Configurations reject unknown fields.

`time_basis=event_time_utc`, `timezone=UTC`, and the author/latest-classified policy
are V1 invariants. Activity and polarization may have different durations and grids.
There is no default experimental window, cadence, or support threshold. An optional
positive `min_support_threshold` marks low support but does not change the value.
Durations must be positive and representable at Python datetime's microsecond
precision; UTC timestamps must be timezone-aware.

## Complete causal prefixes and ticks

```python
signal = create_configured_polarization_signal(resolved.config)
ticks = signal.advance(ordered_synthetic_stances, data_cutoff_utc=cutoff)
```

The caller provides all available records with `event_time_utc < cutoff`, in
nondecreasing `(event_time_utc, comment_id)` order. The producer validates rather
than sorts. Duplicate `(proposition_id, comment_id)`, order regression, cutoff
regression, and input preceding an already completed cutoff fail explicitly.
The next batch may contain records exactly at the preceding cutoff. The replay
assumes a synthetic classification is available with its comment, not that real
inference has zero latency. No observed/ingested timestamps or late-arrival policy
are introduced.

Ticks are `origin + k * cadence`, with the default UTC origin at Unix epoch. Coverage
is `[start_time_utc, end_time_utc)`. The first tick is the first grid point at or
after start; input before start may supply the first window's history. Callers must
provide that preceding causal history if they intend full initial coverage; an
empty prefix is not evidence of complete historical extraction.

Only `tick < cutoff` is emitted. At a tick `t` the closed signal window includes
exactly `t - window <= event_time_utc <= t`, restricted to the configured proposition.
All records exactly at `t` are incorporated before evaluation. Known records after
`t` cannot influence that tick. Ticks continue when no input arrives. Repeated
advancement to the same cutoff emits no duplicate ticks.

Validation and successful calculation commit producer state together. A measure
exception is propagated, not hidden as missing input; the producer can retry the
same batch because its state has not committed. Adapters themselves are stateless
formal-measure evaluations. This is not durable cross-restart idempotency.

## Author aggregation and quality

Only `CLASSIFIED` records with category in `{1,2,3,4,5}` and a nonempty author ID
contribute mass. For each author the maximum `(event_time_utc, comment_id)` wins;
equal timestamps use lexicographically greatest comment ID. Non-classified records
never erase an earlier eligible classified record. Authors disappear when no
eligible classified record remains in the current closed window.

The five complete bins are `(0, .25, .5, .75, 1)`; each category selects one bin.
Support and unique-author count are identical. Weights are bin counts divided by
represented authors; measures receive no absolute support or historical distribution.
Input count includes all same-proposition records in the window. Classified-comment
count includes missing-author and superseded classified comments before consolidation.

Exclusions record non-classified status counts, `missing_author` among classified
records, and `superseded_classified` among represented-author records. These are
diagnostics, not a second normalization denominator. Unknown author IDs are never
fabricated.

Observable reasons include `missing_author`, `uncertain_input`, `error_input`, and
`input_quality` (non-passed individual quality labels remain in observation metadata).
An explicitly configured support threshold may add `low_support`. With an available
value quality is `low_support` when that reason applies, otherwise `degraded` if any
reason exists, otherwise `passed`. Reasons remain sorted and complete, so precedence
does not hide simultaneous degradation. This is not model confidence. Distribution
quality and reasons propagate unchanged through measurement and signal observation;
there is no smoothing, attenuation, or invented score.

## Explicit absences and immutable artifacts

| Temporal status | Condition | Value / measure invocation |
| --- | --- | --- |
| `VALUE_AVAILABLE` | At least one represented author | Finite adapter value |
| `NO_INPUT` | No same-proposition records in window | `None`, no call |
| `NO_CLASSIFIABLE_OPINIONS` | Input exists, no CLASSIFIED comments | `None`, no call |
| `NO_REPRESENTABLE_AUTHORS` | CLASSIFIED comments exist, all lack author ID | `None`, no call |

No-value quality describes the corresponding observable absence; reasons preserve
any additional degradation. The existing empty `OpinionDistribution` contract uses
`status=no_classifiable_opinions`, zero support and `weights=None` for all empty-mass
cases. The temporal envelope distinguishes their causes without changing that contract.
No fake neutral distribution or polarization zero is created. Empty ticks have no
`PolarizationMeasurement` instances and still emit one absent observation per enabled
measure.

`SignalObservation` is a new frozen, slotted neutral artifact with `signal_id`, UTC
time, optional value, status, support, quality, and recursively immutable JSON metadata.
It does not replace `ActivityObservation`. Stable IDs are `polarization_esteban_ray`,
`polarization_emd_pol`, and `polarization_mec`.

`PolarizationTemporalTick` preserves window, cutoff, shared distribution, selected
comment IDs (ordered by author ID), measurements, and observations. `to_mapping()`
on ticks/observations produces JSON-compatible snapshots without modifying state.
The classification run supplies source-record lineage. The standalone runtime persists
the resolved config beside distributions and observations so grid, coverage, policy,
proposition and parameter identity remain reconstructable.

The active buffer drops records before the current window's left edge; author
selection is recomputed once per tick. An execution-level seen-identity set grows
with accepted input for duplicate validation. Optimization and durable checkpoints
remain deferred; no online-memory bound or restart equivalence is claimed.

## Standalone real-input and detector runtime

`scripts/run_polarization_runtime.py` composes the already-existing contracts without
joining the activity runtime:

```text
configured local comments
→ persisted CommentStance
→ persisted handoff reload
→ shared OpinionDistribution per tick
→ ER / EMDPol / MEC
→ persisted SignalObservation
→ configured neutral detector (optional)
→ persisted DetectionResult
```

Preparation is the default. Real inference requires both `--execute` and an explicit,
fresh `--output-root`. `polarization.runtime.video_ids` selects an explicit source
slice; temporal selection remains owned by `signals.polarization`. The source role is
`PROVISIONAL_RUNTIME_INPUT` for the legacy CSV profile. Opt-in Silver executions use
`causal_minimal_v1`, `silver_comments_v1`, and `CAUSAL_MINIMAL_V1`. Measure,
classifier, temporal-signal, and detector parameters remain in separate configuration
blocks.

Outputs are `comment_stances.jsonl`, `opinion_distributions.jsonl`,
`polarization_signal_observations.jsonl`, classifier/runtime manifests, and one
resolved config. When `detection.routes` contains polarization signals, the runtime
also persists `polarization_detection_results.jsonl` and
`polarization_detection_manifest.json`. Distribution rows retain selected
comment/author IDs and stance
categories, but not comment text. Coverage fractions are diagnostic metadata and do
not modify formal values.

Each configured route owns one fresh detector instance for the complete execution;
state is never shared across ER, EMDPol, and MEC, even when they use the same detector
type. Only `VALUE_AVAILABLE` observations with finite values are dispatched.
`NO_INPUT`, `NO_CLASSIFIABLE_OPINIONS`, and `NO_REPRESENTABLE_AUTHORS` produce no
`DetectionResult`, do not increment detector counters, and leave state unchanged.
No value is imputed and no reset is performed. The long-gap/reset policy remains
methodologically open.

The detector consumes the scalar level produced by a measure, not the underlying
opinion distribution or support. Signal window, signal cadence, and detector internal
history are separate concepts. Page-Hinkley, ADWIN, and KSWIN are technically
available for any configured polarization signal; none is selected as the final or
default detector. A zero-trigger execution is valid. The runtime performs no
association, candidate promotion, or RAG operation.

## Synthetic coverage and backlog

`tests/test_polarization_temporal_signals.py` contains synthetic fixtures covering
consensus ↔ bipolarization, author replacement/expiry, equal-time records, all excluded
statuses, missing authors, gaps, and equal distributions with different support.
Values are compared against the existing adapters, not ranked scientifically.

- RESOLVED: technical synthetic temporal polarization producer and config composition.
- RESOLVED: [KSWIN technical integration](kswin_integration.md), neutral PH/ADWIN/KSWIN
  routing, and the opt-in polarization runtime through `DetectionResult`.
- RESOLVED: [standalone classifier technical integration](stance_classifier.md) and
  its persisted handoff to the standalone temporal runtime.
- NEXT: define the detection-alert contract and methodological gap/reset policy.
- OPEN: classifier validation, proposition discovery, window/cadence
  selection, support threshold, detector gap/reset policy,
  activity/polarization timestamp association, measure/detector calibration.
- OPEN: `PREPROCESSING-EVAL`, dataset config consolidation, unified artifact traceability,
  output isolation, profile duplication, experimental fingerprint/config_hash semantics,
  EventCandidate runtime handoff.
- DEFERRED: A9 durability/late arrivals, retrospective modernization, daily-baseline and
  XIAO-specific studies. No downstream evidence/RAG work is performed here.
