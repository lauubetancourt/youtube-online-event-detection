# Versionable execution configuration

**Status:** CURRENT CONFIGURATION AUTHORITY
**Architecture context:** [`docs/pipeline_architecture.md`](../docs/pipeline_architecture.md).

Files under `configs/` describe concrete executions or methodologies. They do not
store secrets, datasets, or generated outputs, and the project does not create one
profile per component.

## Configuration model

`RunConfig` is the immutable composition root for one execution:

```text
RunConfig
├── identity
├── data
├── simulation
├── signals
├── detection
├── association
├── polarization
├── rag
└── artifacts
```

Sections are optional when a run does not use them. Each section reuses the
component's existing typed configuration; `RunConfig` does not copy detector,
signal, RAG, storage, or replay parameters.

Components receive only the subconfiguration they need. Domain modules do not read
JSON, parse CLI arguments, inspect environment variables for methodological values,
or receive the complete `RunConfig` merely for convenience.

## Resolution and precedence

The common configuration path is:

```text
typed component defaults
→ versionable JSON profile
→ explicit supported overrides
→ strict validation
→ path resolution
→ resolved configuration
```

Unknown keys are rejected. Values are not silently coerced to unrelated types.
Explicit CLI overrides apply after file values, but the common CLI is limited to:

- `--config`;
- `--run-id`;
- `--output-root`;
- `--dry-run` or `--execute`;
- `--log-level`.

Methodological parameters remain in component configurations and profiles. Legacy
wrappers may translate older arguments into the same resolver, but they do not
introduce new defaults.

## Paths

Profiles store logical paths. `resolve_run_config` resolves them relative to the
profile/workspace boundary before components execute. Canonical serialization
normalizes known paths so a machine-specific absolute workspace path does not
change `config_hash`.

Historical locations such as:

- `data/gold/clean_comments.parquet`;
- `experiments/xiao/media/log_3/cyclic_ingestion_simulation`;

remain `LEGACY_COMPATIBILITY_DEFAULT` values in current profiles and entrypoint
shims. They are not embedded authorities in signal or detector logic. Another
prepared dataset can be selected by changing `DataConfig`/component paths, without
editing domain code.

Configuration declares where a dataset is. Dataset storage, fingerprinting,
distribution, and recovery belong to STAB-DATA-01.

## Effective configuration and hash

The resolver produces:

- a typed, path-resolved `RunConfig`;
- deterministic canonical JSON (`resolved_config`);
- a SHA-256 `config_hash` computed from effective values.

The canonical representation includes component defaults after resolution and
excludes machine-specific path prefixes where configured. A methodological change
changes the hash; merely running from another workspace does not.

The execution-level manifest records the global `run_id`, run mode, trace level,
`config_hash`, `resolved_config`, execution mode, and completed stages. It does not
copy the same configuration into every component artifact.

## Opt-in Silver causal input

New experimental activity or polarization runs may select the minimal causal
input policy without changing legacy defaults:

```json
{
  "data": {
    "experimental_input": {
      "policy_id": "causal_minimal_v1",
      "source": "silver",
      "exact_comment_id_dedup": true,
      "semantic_filtering": false
    }
  }
}
```

The configured comments path (and cyclic ingestion path where applicable) must
point to the existing Silver artifact. The policy preserves original text, short
comments, replies, emoji, and repeated text under distinct comment IDs. Gold and
its existing profiles remain unchanged.

## Signal-to-detector selection

When a neutral activity route is configured, the association is explicit:

```json
{
  "detection": {
    "activity_route": {
      "signal_id": "comment_count_event_window_120s_step_30s",
      "detector_id": "xiao_ema"
    },
    "xiao_ema": {
      "v_min": 46
    }
  }
}
```

`activity_route` contains references only. Signal semantics remain authoritative in
`ActivitySignalDefinition`; XIAO parameters remain authoritative in
`XiaoEMAConfig`. The route and loader are implemented and tested, but the integrated
cyclic compatibility profile has not yet adopted this route as its general runtime
dispatch.

Neutral multi-signal routes use `detection.routes`. They may be written as a list
of `{signal_id, detector_id}` objects or as a mapping keyed by signal ID. For
example, this technical smoke configuration routes three independently produced
polarization levels without assigning any detector as the methodological default:

```json
{
  "detection": {
    "gap_policy": {"policy_id": "preserve_state_v1"},
    "routes": {
      "polarization_mec": {"detector_id": "kswin"},
      "polarization_esteban_ray": {"detector_id": "adwin"},
      "polarization_emd_pol": {"detector_id": "page_hinkley"}
    },
    "page_hinkley": {},
    "adwin": {},
    "kswin": {"seed": 42}
  }
}
```

Routes, detector IDs, and detector parameters participate in canonical resolved
configuration and therefore in `config_hash`. Polarization detection remains
opt-in; absent values are skipped without updating or resetting detector state.
That gap behavior is the current technical policy, not a calibrated methodological
decision.

`preserve_state_v1` makes the existing absent-observation behavior explicit: no
detector update, no reset and no imputation. It is provisional and not calibrated.
Legacy profiles may omit it without changing behavior; new polarization experiments
should declare it so the policy participates in resolved configuration and
`config_hash`.

Timestamp association of triggered activity and polarization alerts is a separate,
opt-in stage. A development-only example is:

```json
{
  "association": {
    "policy_id": "bounded_temporal_clusters_v1",
    "tolerance": "5min",
    "parameter_status": "DEVELOPMENT_NOT_CALIBRATED"
  }
}
```

The tolerance is an experimental parameter and is not derived from signal windows or
cadences. Candidate projection consumes finalized associations without applying the
tolerance again.

Candidate evidence is a separate opt-in RAG configuration. Temporal values shown
here are illustrative only and must be labelled explicitly:

```json
{
  "rag": {
    "candidate_evidence": {
      "source_path": "data/silver/comments.parquet",
      "output_dir": "outputs/candidate_evidence",
      "policy_id": "candidate_evidence_v1",
      "time_anchor": "reference_time_utc",
      "lookback": "10min",
      "lookahead": "0s",
      "mode": "immediate_causal",
      "parameter_status": "DEVELOPMENT_NOT_CALIBRATED"
    }
  }
}
```

The source is fixed to Silver plus `causal_minimal_v1`; semantic filtering remains
disabled. Changing lookback, lookahead, mode, or parameter status changes resolved
configuration and `config_hash`. No duration in this example is recommended or
calibrated.

The development profile
`configs/development/polarization_detection_runtime_smoke.json` reads Silver through
`causal_minimal_v1`. Preparation is the default; execution still requires an
explicit output directory and `--execute`.

Page-Hinkley can be selected explicitly without changing the signal definition:

```json
{
  "detection": {
    "activity_route": {
      "signal_id": "comment_count_event_window_120s_step_30s",
      "detector_id": "page_hinkley"
    },
    "page_hinkley": {
      "min_instances": 30,
      "delta": 0.005,
      "threshold": 50.0,
      "alpha": 0.9999,
      "mode": "both"
    }
  }
}
```

ADWIN uses the same route without changing the signal's physical-time window:

```json
{
  "detection": {
    "activity_route": {
      "signal_id": "comment_count_event_window_120s_step_30s",
      "detector_id": "adwin"
    },
    "adwin": {
      "delta": 0.002,
      "clock": 32,
      "max_buckets": 5,
      "min_window_length": 5,
      "grace_period": 10
    }
  }
}
```

The Page-Hinkley and ADWIN values shown above mirror River's technical defaults;
they are not calibrated or methodologically validated for event detection. ADWIN's
adaptive window counts observations inside the detector and does not replace
`ActivitySignalDefinition.window`.

## Current profiles

### Development cyclic activity profiles

`development/cyclic_activity_page_hinkley.json` and
`development/cyclic_activity_adwin.json` execute the same neutral activity route
over the small synthetic fixture in `tests/fixtures/`. They differ only where the
selected detector and its typed configuration require it. The input role is
`PROVISIONAL_RUNTIME_INPUT`; these profiles validate plumbing, not preprocessing,
signal selection, calibration, or event-detection validity.

```bash
.venv/bin/python scripts/run_cyclic_pipeline.py \
  --config configs/development/cyclic_activity_page_hinkley.json \
  --output-root outputs/cyclic_activity_page_hinkley \
  --dry-run
```

Use `development/cyclic_activity_adwin.json` for the equivalent ADWIN run. Both
write `activity_detection_results.jsonl` below an
`activity_detection_runtime/` subdirectory and do not run the daily baseline.

The activity producer is also selected from JSON. The runtime registry currently
accepts these two implemented signal families (shown with their development
profile identities):

```text
comment_count_event_window_120s_step_30s
unique_author_count_event_window_120s_step_30s
```

To switch signals, keep `signals.activity.signal_id` and
`detection.activity_route.signal_id` equal and supply the matching signal semantics
(`metric="unique_authors"` and `unit="authors/window"` for the unique-author
signal). Window, cadence and time basis remain owned by `signals.activity`; the
connector does not override them. Missing `author_id` values remain in
`support_count`, do not contribute to the unique-author value, and produce
`quality="degraded_missing_author_id"`. These are experimental runtime signals,
not calibrated or methodologically validated choices.

### `compatibility/cyclic_current.json`

Represents the migrated cyclic compatibility execution:

```text
cyclic simulation
→ daily signals
→ XIAO connector compatibility configuration
→ daily_frequency_baseline
```

It preserves the current event-time, Bogotá timezone, XIAO, baseline, and guarded
dry-run parameters. Its historical data/output paths are compatibility values, not
permanent architecture.

Run it without overwriting the historical output tree:

```bash
.venv/bin/python scripts/run_cyclic_pipeline.py \
  --config configs/compatibility/cyclic_current.json \
  --output-root outputs/cyclic_current \
  --dry-run
```

### `development/polarization_runtime_smoke.json`

Selects one small real-comment slice and composes classification with the standalone
temporal polarization producer. Window/cadence and measure defaults are
`RUNTIME_SMOKE_NOT_CALIBRATED`; the input remains `PROVISIONAL_RUNTIME_INPUT`.
Preparation makes no model calls:

```bash
.venv/bin/python scripts/run_polarization_runtime.py \
  --config configs/development/polarization_runtime_smoke.json \
  --output-root outputs/polarization_runtime_smoke \
  --dry-run
```

Real inference additionally requires `--execute` and a fresh output directory.
Detector, association, candidate, and RAG stages are not part of this runner.

### `compatibility/daily_rag_current.json`

Represents the current non-generative daily RAG execution:

```text
daily events
→ daily sidecars
→ daily consumer
→ deterministic context selection
```

It preserves separate historical identities for sidecars (`drun_*`), consumer
(`dragconsumer_*`), and selection (`dragselect_*`). The global `run_id` supplies
execution context and does not replace these stage identities.

```bash
.venv/bin/python scripts/run_daily_rag_pipeline.py \
  --config configs/compatibility/daily_rag_current.json \
  --output-root outputs/daily_rag_current \
  --dry-run
```

Profiles represent real executions. A new profile should answer which execution or
methodology it preserves and why it needs versioning. Similar component-level files
should not proliferate.

## Traceability policy

Typed minimum levels are:

- `development` → `minimal`;
- `reference` → `standard`;
- `official` → `full`.

The current integrated runners implement only `development/minimal`. Requesting a
higher unimplemented policy fails before artifacts are created. The type model can
represent future policies without claiming their persistence behavior already
exists.

## Secrets and infrastructure

API keys and provider credentials are resolved at entrypoint/infrastructure
boundaries. They never belong in profiles, `RunConfig`, `resolved_config`,
`config_hash`, or methodological manifests.

Logging level and execution mode are operational controls. They are not detector or
signal parameters.

## Legacy compatibility

Stage-specific scripts and `run_pipeline.py` remain where their interfaces are
tested. They translate legacy inputs to the common resolver and may preserve old
path defaults. New methodological parameters must not be added independently to
those wrappers.

Removal requires all of the following:

```text
replacement working
+ compatibility tests passing
+ no required consumers
+ documentation updated
```

## Candidate evidence RAG integration

The causal candidate route is opt-in. Network-capable execution additionally
requires the CLI `--execute` guard:

```json
{
  "identity": {"run_id": "candidate-rag-development"},
  "rag": {
    "candidate_validation": {
      "candidates_path": "outputs/event_candidates.jsonl",
      "evidence_dir": "outputs/candidate_evidence",
      "output_dir": "outputs/candidate_rag_validation",
      "candidate_id": "candidate_example"
    }
  }
}
```

The adapter reuses the fixed G-1 and hierarchical G-2 model/prompt contracts.
Search-day and capacity parameters keep their existing experimental semantics;
this integration does not calibrate them. Candidate G-2 adds an explicit
publication-time cutoff while legacy G-2 behavior remains unchanged.

## Monitoring topic profiles

Stance-enabled causal experiments may reference an immutable topic profile instead
of repeating or editing proposition/context values in Python:

```text
configs/
├── topics/<topic_id>.json
└── experiments/<experiment>.json
```

The experiment uses `"topic_profile": "<topic_id>"`. The referenced file contains
`topic_id`, `topic_description`, and a `stance` object with `proposition_id`,
`proposition`, `context`, `prompt_id`, and `prompt_version`. Loading composes those
values into `polarization.classifier`; a conflicting inline classifier proposition
is rejected. The resolved config stores the complete profile, not only its filename,
so changes to the topic, proposition, context, or prompt change `config_hash`.

See [`topics/example_monitoring_topic.json`](topics/example_monitoring_topic.json).
One profile is frozen for the entire run. Historical configs without a topic profile
retain their legacy behavior; the causal end-to-end route requires one explicitly.

## G-1 to G-2 gate

Candidate RAG runs can add:

```json
{
  "rag": {
    "g2_gate": {
      "policy_id": "g1_gate_v1",
      "mode": "OFF",
      "parameter_status": "DEVELOPMENT_NOT_CALIBRATED"
    }
  }
}
```

`OFF` is the behavior-preserving baseline: every G-2-eligible candidate still runs
G-2 after G-1. `PERMISSIVE` and `STRICT` are available experiment labels, but neither
has a hard-coded scientific meaning. A non-`OFF` run must explicitly list
`run_g2_for_g1_statuses` using the existing G-1 status vocabulary. The decision and
G-2 execution status are persisted; a gate skip is not a validation failure.

## Causal end-to-end replay

An opt-in run may add `simulation.causal_end_to_end` with `start_utc`, `end_utc`,
`step`, `output_dir`, compatible precomputed `stance_artifact_path` and
`stance_manifest_path`, and `parameter_status` set to `TEST_ONLY` or
`DEVELOPMENT_NOT_CALIBRATED`. Replay step, activity cadence, and polarization
cadence are independent. The runner composes the existing `causal_minimal_v1`,
detector routes, association, candidate evidence, and candidate validation
blocks; it does not change legacy defaults.
