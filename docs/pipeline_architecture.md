# End-to-end pipeline architecture

**Status:** CURRENT ARCHITECTURE
**Authority:** end-to-end layers, responsibilities, implemented connections, and
compatibility boundaries.
**Specialized contracts:** [data](data_contracts.md),
[activity and detection](activity_signal_semantics.md),
[event evidence and RAG preparation](rag_event_evidence_contract.md), and
[configuration](../configs/README.md).

## Purpose and limits

The prototype turns YouTube observations into statistically detected event
candidates and traceable evidence for posterior validation. It supports
retrospective and daily online-like execution. It does not yet provide durable
true-online ingestion.

A trigger means that one configured detector found a relevant change in one
configured numeric signal. Activity and polarization observations can use the same
neutral detector boundary. It does not prove that a real-world event occurred.
G-1 and G-2 are posterior validation stages and do not belong to statistical
detection.

## Conceptual architecture

```text
SOURCE / YOUTUBE
        ↓
ACQUISITION
        ↓
STORAGE / RAW DATA
        ↓
CLEANING + NORMALIZATION
        ↓
PREPARED DATASET
        ↓
REPLAY / CYCLIC SIMULATION
        ↓
ACTIVITY METRIC
        ↓
ACTIVITY SIGNAL
        ↓
ActivityObservation
        ↓
SIGNAL → DETECTOR ROUTING
        ↓
DETECTOR
        ↓
DetectionResult
        ↓
EVENT PROMOTION
        ↓
EventCandidate
        ↓
EVIDENCE ASSEMBLY
        ↓
RAG PREPARATION
        ↓
CONTEXT SELECTION
        ↓
G-1 / G-2 VALIDATION
```

This diagram is conceptual. The maturity table below states which handoffs are
active and which contracts exist but are not yet connected to the general runtime.

## Layer map

| Layer | Current implementation | Responsibility | Output or handoff |
|---|---|---|---|
| Source/acquisition | `youtube_pipeline/data_extraction.py` | Query YouTube endpoints, apply acquisition filters, and capture source records and run metadata | Video and comment dataframes |
| Storage | `youtube_pipeline/storage.py` | Normalize source timestamps and persist local Bronze JSONL and Silver Parquet | Persisted batch snapshot |
| Cleaning | `youtube_pipeline/cleaning.py` | Normalize text, flag/filter noise, handle orphan replies and temporal duplicates, and preserve canonical IDs/time | Prepared comments dataset |
| Prepared input | `PreparedDatasetConfig`, `LocalFilesConfig` | Declare which local dataset an execution consumes | Typed source configuration |
| Retrospective replay | `youtube_pipeline/replay.py`, `prepared_replay.py` | Emit historical records in event-time order | Incremental comment events |
| Cyclic simulation | `cyclic_ingestion.py`, `cyclic_orchestration.py`, `cyclic_stateful_adapter.py` | Reveal only records available before each simulated cutoff and maintain active windows | Cycle manifests and causal inventories |
| Activity signal | `activity_signals.py`, `cyclic_daily_signals.py` | Apply a metric over an explicit temporal definition | `ActivityObservation` or current daily signal rows |
| Signal routing | `activity_detection.py` | Associate one `signal_id` with one configured detector and validate the handoff | Detector-specific dispatch |
| Detection | `detectors.py`, `daily_frequency_baseline.py` | Maintain statistical state and decide whether a criterion is satisfied | `DetectionResult` for neutral detectors; daily score/event records for the baseline |
| Candidate promotion | `event_candidates.py` | Promote a triggered neutral result into an identifiable, traceable candidate | `EventCandidate` internal contract |
| Evidence | `rag_evidence.py`, `rag_sidecars.py`, `daily_rag_sidecars.py` | Build causal inventories and comment/video associations | Evidence packages and sidecars |
| RAG preparation | `rag_consumer.py`, `daily_rag_consumer.py` | Build context units and non-generative validation payloads | Validation inputs and capacity reports |
| Context selection | `daily_rag_context_selection.py` and retrospective consumer policy | Select traceable units under a budget | Selected comment/context-unit references |
| Validation | `rag_generation_g1.py`, `rag_generation_g2.py`, `rag_generation_g2_hierarchical.py` | Evaluate internal community evidence and external evidence | G-1/G-2 labels, rationale, and citations |

## Authority by layer

| Layer | Authority |
|---|---|
| Acquisition | Observations returned by the configured source and acquisition metadata |
| Normalization | Canonical timestamps, IDs, data types, deduplication, and prepared representation |
| Signal producer | `value`, `support_count`, and `quality` of each observation |
| Detector | `triggered`, optional `score`, and detector-specific `detector_metadata` |
| Candidate promotion | Candidate identity, causal evidence interval, lifecycle when applicable, and minimal lineage references |
| Evidence assembly | Complete comment/video inventories and causal associations |
| RAG preparation | Context units, chunking, ordering, budgets, and selection |
| G-1/G-2 | Validation labels, validation confidence, rationale, citations, and external evidence |

No downstream layer silently redefines an upstream authority. In particular, a
detector propagates observation `quality`; warmup and cooldown belong to
`detector_metadata`, not to signal quality.

## Source, storage, and prepared data

The acquisition layer captures fields returned by YouTube or an equivalent local
source. It does not assign event meaning. Storage maps source timestamps to
canonical UTC fields and persists Bronze/Silver materializations. Cleaning then
produces a prepared dataset while preserving source IDs and temporal lineage.

The current runtime can use `youtube_api`, `local_files`, or `prepared_dataset`
through `DataConfig`; the chosen path is external configuration. “Gold” is a
current local prepared dataset and compatibility path, not a domain concept or a
permanent location embedded in detection logic.

See [data contracts](data_contracts.md) for field-level rules.

## Retrospective replay and cyclic simulation

Retrospective replay emits a known corpus in `event_time_utc` order. The cyclic
simulation partitions that corpus into local-day cycles and enforces:

```text
event_time_utc < data_cutoff_utc
analysis_window_start_utc <= event_time_utc < analysis_window_end_utc
```

The simulation is online-like: each cycle sees only data available under the
simulated event-time cutoff. It is not true online ingestion. The source corpus
does not contain a distinct, durable `observed_at_utc` or `ingested_at_utc`, so the
prototype cannot yet model ingestion latency, watermarks, or late arrivals.

## Activity and detection architecture

### Metric and signal

An activity metric is a quantitative rule over available data. Current examples
are `comment_count` and `unique_author_count`.

An activity signal is a time series identified by more than its metric:

```text
metric + source + scope + unit + window + cadence
+ time_basis + timezone + interval_policy
```

Therefore `comment_count` over 120-second windows every 30 seconds and a daily
comment count are different signals.

### Neutral contracts

`ActivitySignalDefinition` records signal semantics. `ActivityObservation`
records one causal value and makes the signal producer authoritative for:

```text
value
support_count
quality
```

`ActivityDetectionRouteConfig` declares `signal_id → detector_id` without copying
signal or detector parameters. The detector receives only an observation, not
YouTube rows, paths, or `RunConfig`.

`DetectionResult` is the statistical decision for one observation:

```text
detector_id
signal_id
observation_time_utc
triggered
quality                 # propagated unchanged
score                   # optional and method-specific
detector_metadata       # method-specific state/evidence
```

`DetectionResult.triggered=true` does not confirm an event. The legacy
`EventCandidate` promotes one activity result with an evidence interval. The new
`AssociatedEventCandidate` instead projects one finalized cross-signal association
with alert-ID lineage and deliberately leaves evidence-window policy downstream.

### Polarization runtime boundary

The opt-in polarization branch now reaches the same neutral statistical decision
boundary without becoming part of the activity runtime:

```text
Silver → causal_minimal_v1 → StanceCommentInput → CommentStance
→ OpinionDistribution(t) → ER / EMDPol / MEC → SignalObservation(t)
→ configured Page-Hinkley / ADWIN / KSWIN → DetectionResult
→ DetectionAlert when triggered
→ DetectionAssociation after causal timestamp buffering
→ AssociatedEventCandidate after deterministic one-to-one projection
```

`detection.routes` supplies `signal_id → detector_id`; there is no measure-specific
branch or official detector assignment. `SignalObservation` enters detection through
the `NumericObservation` boundary rather than being relabeled as
`ActivityObservation`. One detector instance is maintained per signal route for the
execution. Observations without values are not dispatched and do not reset state;
V1 explicitly preserves detector state, while empirical calibration of long-gap
behavior remains open. Detection, association, and candidate projection remain
separate runtime boundaries; none retrieves evidence or validates an event.

Triggered results may now be projected into a neutral `DetectionAlert` carrying an
explicit `ACTIVITY` or `POLARIZATION` family, deterministic identity, support and
runtime lineage. Non-triggered results remain persisted but produce no alert. Alerts
now enter an optional bounded timestamp-association layer. That layer produces
`DetectionAssociation` and never assumes shared tick or cycle indexes. A finalized
association can then produce exactly one neutral `AssociatedEventCandidate`, without
reapplying Δ or combining scores. See
[detection alerts and gap policy](detection_alerts_and_gap_policy.md),
[temporal detection association](temporal_detection_association.md), and
[neutral event-candidate projection](event_candidate_fusion.md).

The three distinct temporal concepts remain independent: the signal window selects
comments, the signal cadence emits values, and a detector's internal window/history
operates over those emitted scalar values.

Both legacy and neutral candidate contracts are implemented and tested. The general
production path does not yet use the neutral candidate as the handoff to RAG.
Existing retrospective and daily event records remain the active compatibility
contracts.

See [activity signal semantics](activity_signal_semantics.md) for the normative A6
details.

## Formal polarization-measure boundary

The implemented non-temporal polarization boundary is independent of activity
detection:

```text
OpinionDistribution
→ PolarizationMeasure adapter
→ pinned pol_measures implementation
→ PolarizationMeasurement
```

`EstebanRayAdapter`, `EMDPolAdapter`, and `MECAdapter` receive the same complete
Likert-5 positions and normalized author-mass weights. They do not classify
comments, aggregate authors, create windows, or emit detector input. Selection
and public method parameters are typed under `RunConfig.polarization`. See
[formal polarization measures](polarization_measures.md).

A standalone [synthetic temporal polarization producer](polarization_temporal_signals.md)
now consumes ordered `CommentStance` causal prefixes, builds one closed-window author
distribution per configured tick, and evaluates the enabled measure list. It emits
new neutral `SignalObservation` values or explicit absences without replacing
`ActivityObservation`. Its grid is independent of activity, and it is not connected
to cyclic runtime, classifiers, detectors, association, candidates, or RAG.

PH, ADWIN and KSWIN now share a value-only `NumericObservationDetector` boundary
supporting both historical `ActivityObservation` and neutral `SignalObservation`.
The existing dispatcher skips explicit no-value observations without updating
state or producing a DetectionResult. Long-gap reset policy remains open. Standalone
synthetic polarization observations can be routed through this boundary in tests;
no polarization detector runtime is connected. See [KSWIN integration](kswin_integration.md).

## XIAO reference path

XIAO EMA is both `REFERENCE_DETECTOR` and `REGRESSION_ANCHOR`; it is not the final
detector by architectural commitment.

Neutral evaluation:

```text
comment_count_event_window_120s_step_30s
→ ActivityObservation
→ XiaoEMATriggerDetector.on_observation()
→ DetectionResult
```

Historical compatibility:

```text
XIAO
→ active trigger
→ cooldown
→ completed_triggers
→ trigger_comment_map
→ retrospective evidence
```

`on_event()` still drives that historical lifecycle. The `DetectionResult` returned
by each observation is not yet the persisted source of the historical trigger.
Comments collected internally during cooldown contain only time and text and are
not the definitive RAG evidence inventory. Retrospective evidence is reconstructed
outside the detector from the approved pre-trigger window and the prepared dataset.

## Daily baseline path

The daily path remains specialized:

```text
new_comment_count_local_day_daily
→ daily_frequency_baseline
→ trigger_candidate
→ daily_event_id
→ daily evidence sidecars
→ daily consumer and context selection
```

The baseline produces score rows and point candidates; it has no `OPEN/CLOSED`
lifecycle. Its statistical fields can be mapped conceptually to `DetectionResult`,
but that adapter is not implemented. The daily event currently combines decision,
candidate, and some lineage fields and remains the active compatible output.

## Retrospective and daily realizations

| Concern | Retrospective | Daily |
|---|---|---|
| Detector | XIAO EMA reference | Daily frequency baseline |
| Candidate shape | Trigger with optional cooldown/close lifecycle | Point candidate associated with one cycle |
| Candidate identity | Historical `event_id` assigned during evidence preparation | `daily_event_id` assigned by the baseline |
| Alert evidence | Approved pre-trigger event window | Comments new in the triggering cycle |
| Validation context | Event evidence window/context prepared downstream | Comments active in the analysis window |
| RAG identity | Retrospective evidence/stage IDs | `daily_rag_event_id` plus preserved `daily_event_id` |

The neutral contracts allow both shapes without forcing a common lifecycle or
changing historical identity formulas.

## Evidence and RAG boundary

Evidence answers: “Which source observations justify or reconstruct this
candidate?” It includes the causal window, comment inventory, video association,
and stable source references.

RAG answers: “How will available evidence be organized and selected for
validation?” It owns context units, chunking, ranking, token estimates, budgets,
queries, external evidence, prompts, model outputs, labels, confidence, rationale,
and citations.

Context selection may omit material because of a budget. It must not change the
complete evidence inventory. See the
[event evidence contract](rag_event_evidence_contract.md).

## G-1 and G-2

G-1 evaluates a candidate using internal evidence from the observed YouTube
community. G-2 evaluates external support. The hierarchical G-2 variant follows:

```text
event
→ associated video
→ external query/evidence
→ video-level assessment
→ event-level synthesis
```

Prompts, models, retrieval settings, labels, and artifact schemas remain owned by
their RAG-stage configurations and modules. They do not enter `DetectionResult` or
the core candidate contract.

## Configuration and traceability

One execution is composed by immutable `RunConfig` sections for identity, data,
simulation, signals, detection, RAG, and artifacts. The loader rejects unknown
keys, applies explicit overrides after profile values, resolves paths outside domain
modules, and serializes the effective configuration canonically.

Minimum reconstruction chain:

```text
dataset reference
+ resolved_config
→ config_hash
+ global run_id
→ result and stage artifacts
```

`RunConfig.identity.run_id` identifies the global execution. RAG evidence,
consumer, selection, query, and validation stages retain their own IDs because each
identifies a different transformation. The global ID provides context; it does not
replace stage identities or historical formulas.

Secrets remain external infrastructure and are excluded from `RunConfig`,
`resolved_config`, `config_hash`, and methodological manifests.

## Compatibility boundary

`run_pipeline.py`, stage wrappers, legacy loaders, historical event IDs, and RAG
artifact schemas remain when they have tested consumers. They translate or project
the current architecture; they must not introduce new methodological defaults.

Historical paths such as `data/gold` and `experiments/xiao/media/log_3` appear in
compatibility profiles and entrypoint shims. Domain modules receive resolved paths
and do not treat those locations as scientific authorities.

## Maturity status

| Component | Status | Current meaning |
|---|---|---|
| Acquisition | IMPLEMENTED_AND_ACTIVE | YouTube API and configured local inputs |
| Storage and timestamp normalization | IMPLEMENTED_AND_ACTIVE | Bronze/Silver persistence and canonical UTC fields |
| Cleaning | IMPLEMENTED_AND_ACTIVE | Prepared comments contract |
| Prepared dataset selection | IMPLEMENTED_AND_ACTIVE | Configurable local prepared source |
| Retrospective replay | IMPLEMENTED_AND_ACTIVE | Event-time replay of prepared data |
| Cyclic simulation | IMPLEMENTED_AND_ACTIVE | Deterministic daily online-like cutoffs |
| Comment-count signal | IMPLEMENTED_AND_ACTIVE | XIAO reference input |
| Unique-author signal | IMPLEMENTED_NOT_YET_CONNECTED | Implemented/tested experimental signal; not a default profile |
| `ActivityObservation` | IMPLEMENTED_AND_ACTIVE | Consumed by XIAO's neutral method |
| Signal→detector routing | IMPLEMENTED_AND_ACTIVE | Used by the technical cyclic activity runtime; compatibility paths remain separate |
| XIAO EMA | REFERENCE_COMPATIBILITY | Active reference detector and historical trigger lifecycle |
| `DetectionResult` | IMPLEMENTED_AND_ACTIVE | Persisted as neutral runtime envelopes; historical XIAO persistence remains separate |
| Legacy `EventCandidate` | IMPLEMENTED_NOT_YET_CONNECTED | Activity-specific internal contract kept unchanged |
| `AssociatedEventCandidate` | IMPLEMENTED_NOT_YET_CONNECTED | Neutral finalized-association projection; evidence policy remains downstream |
| Daily frequency baseline | IMPLEMENTED_AND_ACTIVE | Specialized daily point-candidate path |
| Retrospective RAG evidence/sidecars | IMPLEMENTED_AND_ACTIVE | Current reference contracts |
| Daily RAG sidecars/consumer/selection | IMPLEMENTED_AND_ACTIVE | Non-generative daily chain |
| G-1/G-2 and hierarchical G-2 | IMPLEMENTED_AND_ACTIVE | Local dry-run and configured external execution paths |
| River 0.26.1 | IMPLEMENTED_AND_AVAILABLE | Reproducible dependency encapsulated by neutral detector adapters |
| Page-Hinkley adapter | IMPLEMENTED_AND_ACTIVE | Selectable in the technical cyclic activity runtime; uncalibrated |
| ADWIN adapter | IMPLEMENTED_AND_ACTIVE | Selectable in the same technical cyclic activity runtime; uncalibrated |
| Temporal polarization producer | IMPLEMENTED_SYNTHETIC_ONLY | Standalone causal closed-window author distributions and configured formal measures; uncalibrated |
| Individual stance classifier | TECHNICALLY_INTEGRATED_NOT_VALIDATED | [Standalone classifier](stance_classifier.md), explicit proposition, strict stance-only output, mocked API tests; no temporal/detector runtime handoff |
| KSWIN adapter | TECHNICALLY_AVAILABLE_EXPERIMENTALLY_OPEN | Seeded scalar-distribution detection tested with activity and polarization; not a default/final polarization detector |
| True online ingestion | DEFERRED | Requires A9 contracts and operational state |
| Durable checkpoint/restore | DEFERRED | Current detector state is in-memory or report-oriented |
| Late-arrival handling/watermarks | DEFERRED | Requires distinct observation/ingestion time |

## Deferred to A9

The following are not current capabilities:

- `observed_at_utc` and `ingested_at_utc` as separate source facts;
- watermarks and explicit late-arrival policy;
- durable idempotency and atomic persistence;
- restart equivalence;
- durable checkpoints for signal and detector state;
- real online scheduling, polling, and recovery.

Their absence does not change the causal contract of retrospective or cyclic
executions. It limits claims about real online operation.
