# Data contracts and temporal invariants

**Status:** CURRENT CONTRACT
**Authority:** persisted data layers, canonical fields, identifiers, temporal
causality, deduplication, and prepared-dataset requirements.
**Architecture context:** [pipeline architecture](pipeline_architecture.md).

## Contract principles

- Preserve source identifiers and raw values needed for lineage.
- Normalize operational timestamps to timezone-aware UTC.
- Use Unix epoch seconds in canonical numeric timestamp fields.
- Keep legacy `*_unix_ms` aliases readable while retained artifacts require them;
  observed legacy values in this project contain seconds despite the suffix.
- Make deduplication and causal cutoffs explicit.
- Treat data paths as execution configuration, not domain constants.
- Do not rewrite historical datasets during architectural refactors without an
  approved migration.

## Source, normalized, and derived fields

The same row can carry values from different semantic categories:

| Category | Meaning | Examples |
|---|---|---|
| SOURCE | Returned directly by YouTube or the configured local equivalent | `comment_id`, `video_id`, comment text, source publication timestamp, `author_id`, likes |
| NORMALIZED | Source value represented in the canonical runtime contract | `event_time_utc`, `published_at_utc`, normalized IDs and data types |
| DERIVED | Value computed without a temporal series | `is_reply`, cleaned text, spam flags, calendar partitions |
| TEMPORAL_DERIVED | Value that depends on a window, cutoff, cadence, cycle, or prior state | new/active comment counts, deltas, signal observations, detector scores |

Acquisition owns source observations. Storage and cleaning own their canonical
representation. Signal and detector outputs are derived analytical records; they
must not be presented as source facts.

## Canonical time

| Entity | Canonical UTC field | Canonical numeric field | Legacy alias | Unit |
|---|---|---|---|---|
| Comment | `event_time_utc` | `event_time_unix_s` | `event_time_unix_ms` | seconds |
| Video | `published_at_utc` | `published_at_unix_s` | `published_at_unix_ms` | seconds |

`event_time_utc` is the time basis for current replay, cyclic simulation, activity
signals, and detection. It is not ingestion time. The current prepared corpus does
not provide distinct durable `observed_at_utc` or `ingested_at_utc` fields.

A source comment input may provide `published_at` instead of `event_time_utc`.
The comment input adapter validates an existing `event_time_utc` as authoritative,
or derives it in memory from `published_at` when the canonical field is absent.
Both inputs must include an explicit timezone or UTC offset; naive and invalid
timestamps are rejected. The adapter preserves `published_at` for lineage and does
not assign a default local timezone.

```text
published_at
→ canonical input normalization
→ event_time_utc
→ causal_minimal_v1
```

Temporal invariants:

- timestamps used by the runtime are timezone-aware;
- UTC is canonical for comparison and persistence;
- local-day simulation converts `America/Bogota` boundaries to UTC;
- a stage may consume only records allowed by its explicit cutoff;
- future observations must not leak into signal, detection, or alert evidence.

Current cyclic rules are:

```text
event_time_utc < data_cutoff_utc
collection_window_start_utc <= event_time_utc < collection_window_end_utc
analysis_window_start_utc <= event_time_utc < analysis_window_end_utc
```

Signal-specific interval policies are defined in
[activity signal semantics](activity_signal_semantics.md).

## Persisted data layers

These are local pipeline layers, not an instruction to version their contents in
Git.

| Layer | Current location/shape | Producer | Role |
|---|---|---|---|
| Legacy local input | `data/videos_preliminares.csv`, `data/comments.csv` | Earlier extraction/export flow | Compatibility source |
| Bronze videos/comments | JSONL under `data/bronze/` | `persist_batch_snapshot` | Source-oriented batch record |
| Silver videos/comments | Partitioned Parquet under `data/silver/` | `persist_batch_snapshot` | Normalized prepared input to cleaning |
| Prepared comments (“Gold” compatibility layout) | Commonly `data/gold/clean_comments.parquet` | Cleaning | Current analytical dataset for replay/simulation |
| Monitoring snapshots | CSV/Parquet selected by execution | Playback/monitoring | Window-level analytical output |
| Experiment/RAG artifacts | Under configured output roots, often `experiments/` | Detection, evidence, and RAG stages | Development/reference evidence |

`DataConfig` can select `youtube_api`, `local_files`, or `prepared_dataset`.
Therefore `data/gold/clean_comments.parquet` is a compatibility profile value, not
the universal prepared-dataset path.

Dataset storage, distribution, fingerprinting, backup, and recovery policy remain
outside this contract and belong to STAB-DATA-01.

## Comment contract

| Field | Role | Requirement |
|---|---|---|
| `comment_id` | Stable source/traceability key | Required by cyclic and evidence inventories |
| `video_id` | Join key to video metadata | Required by cyclic and RAG evidence paths |
| `author_id` | Author/channel reference | Optional at source; missingness affects unique-author signal quality |
| `text` | Raw comment text | Required by current XIAO compatibility path and RAG evidence |
| `reply_to_comment_id` | Parent reference | Optional; supports thread reconstruction |
| `published_at` | Source publication time | Optional source field; accepted as input when `event_time_utc` is absent and preserved for lineage |
| `event_time_utc` | Canonical event time | Required by prepared replay and temporal stages |
| `event_time_unix_s` | Canonical numeric event time | Required for new persisted artifacts when numeric time is needed |
| `text_clean` | Cleaned representation | Produced by cleaning when that stage is used |

Public outputs containing text or author references require a separate
minimization/anonymization decision. Internal traceability does not by itself
authorize publication of raw comments.

## Video contract

| Field | Role | Requirement |
|---|---|---|
| `video_id` | Stable join key | Required for comment/video evidence association |
| `published_at_utc` | Canonical publication time | Required in normalized video records |
| `published_at_unix_s` | Canonical numeric time | Used in new numeric artifacts |
| `title`, `channel_id`, `channel_title` | Source context | Optional by downstream contract |
| Platform counters | Mutable source observations | A single value is not a temporal rate |

View, like, or platform comment-count velocity requires repeated snapshots with an
explicit observation time. The current contract does not infer a temporal signal
from a single mutable counter.

## Deduplication and ordering

- `comment_id` is the primary identity for cyclic deduplication and evidence maps.
- A duplicated comment may appear in source material, but it is counted as newly
  observed only once in cyclic simulation.
- Cleaning may remove temporal text duplicates under its configured rules; it does
  not redefine the source `comment_id` contract.
- Replay and event-window signal producers require deterministic event-time order.
- Equal timestamps follow the signal-specific ordering protected by regression
  tests; they are not silently reordered by documentation convention.

## Experimental causal input policy

`causal_minimal_v1` is an opt-in runtime policy for new activity and stance
executions. Before the policy runs, the source/input adapter guarantees the
canonical `event_time_utc` field. The policy then applies only schema validation,
stable typing, UTC validation, exact `comment_id` deduplication,
deterministic `(event_time_utc, comment_id)` ordering, and source lineage. It
preserves original text as the semantic authority and does not remove short text,
emoji, replies, repeated text with distinct IDs, or spam-like observations.

The policy keeps an in-run `seen_comment_ids` set so the same rule works for batch
preparation and causal replay. Records at or beyond the active cutoff are withheld
before entering that state. Repeated IDs with different text, author, video,
event time, or parent are explicit conflicts rather than silently selected rows.

Gold remains unchanged and continues to serve all historical consumers. The new
policy is activated only through `data.experimental_input`; it creates no new
medallion layer or permanent materialized view. Activity and stance adapters share
the resulting comment-ID universe, and future evidence builders can select the
same configured source path/hash without changing existing RAG sidecars.

## Prepared dataset contract

A source comment dataset used by the experimental replay/cyclic paths may provide
either `event_time_utc` or timezone-aware `published_at`. After canonical input
normalization, the internal prepared contract must provide, at minimum:

```text
comment_id
video_id
event_time_utc
```

Additional consumers may require `text`, `author_id`, reply fields, or video
metadata. Configuration selects the dataset and maps component column names where
supported. Changing the dataset must not require editing domain modules.

The dataset does not select detector parameters automatically. Dataset reference,
signal definition, detector, and parameters are independent parts of execution
traceability.

## Analytical contracts above the dataset

The following contracts are implemented but are not persisted data-layer fields:

- `ActivitySignalDefinition`: semantic identity of an activity signal;
- `ActivityObservation`: causal signal value, support, and quality;
- `DetectionResult`: detector decision and method-specific evidence;
- `EventCandidate`: neutral promotion with minimal lineage and evidence interval.

The general runtime does not yet use `EventCandidate` as the common handoff to all
evidence paths. Historical retrospective and daily event records remain active
compatibility contracts. See
[activity signal semantics](activity_signal_semantics.md).

## Current evidence and RAG artifacts

Formal RAG artifacts do exist. The retrospective path currently produces, among
others:

- `event_candidates.csv`;
- `event_comment_map.csv` and/or complete event comment inventories;
- `event_video_map.csv`;
- `event_evidence_packages.jsonl`;
- context units and context-unit/comment maps;
- validation inputs and selected context payloads.

The daily path preserves `daily_event_id` and introduces a separate
`daily_rag_event_id` for the evidence-stage representation. It separates comments
new in the triggering cycle (`alert evidence`) from comments active in the analysis
window (`validation context`).

The complete artifact and ownership rules are defined in the
[RAG event evidence contract](rag_event_evidence_contract.md). Prompts, queries,
external evidence, token budgets, G-1/G-2 labels, and model outputs are RAG-stage
data, not fields of the prepared dataset or neutral candidate.

## Identity and lineage

Identifiers have different scopes and must not be collapsed:

| Identity | Meaning |
|---|---|
| `comment_id`, `video_id` | Source entities |
| global `run_id` | One configured pipeline execution |
| `config_hash` | Canonical effective configuration |
| `event_id` | Retrospective candidate/evidence identity under its historical formula |
| `daily_event_id` | Daily baseline point candidate |
| `daily_rag_event_id` | Daily evidence-stage representation |
| context/query/validation IDs | Stage-specific transformations |

Minimum execution lineage is:

```text
dataset reference
+ resolved_config
+ config_hash
+ global run_id
+ stage-specific IDs
→ produced result
```

Stage-specific IDs remain because they identify different transformations. They do
not replace the global execution identity, and the global identity does not replace
them.

## Compatibility rules

- Prefer canonical `*_unix_s`; read retained `*_unix_ms` aliases only for
  compatibility.
- Preserve historical event and stage-ID formulas while current artifacts depend
  on them.
- Preserve current RAG schemas through explicit adapters before changing them.
- Do not copy `RunConfig`, full datasets, prompts, or context payloads into the
  neutral candidate.
- Do not infer true-online guarantees from retrospective or cyclic event-time
  simulation.
