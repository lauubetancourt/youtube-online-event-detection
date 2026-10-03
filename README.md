# YouTube Event Detection Prototype

This repository contains the research prototype for detecting candidate events
from YouTube activity and preparing traceable evidence for posterior RAG
validation. It supports local prepared datasets, retrospective replay, a daily
cyclic simulation, activity signals, statistical detection, evidence assembly,
and internal/external validation stages.

The prototype separates four concerns:

```text
data and temporal simulation
→ activity signals and statistical detection
→ candidate evidence
→ RAG context and validation
```

The current implementation is research software. A detector trigger is an event
candidate, not confirmation that a real-world event occurred.

## Runtime

The active runtime is Python 3.14.3, pinned in `.python-version`.
`requirements-runtime.txt` is the dependency authority for the current pipeline,
maintained scripts, and test suite. `requirements.txt` is a legacy snapshot for
the previous environment and historical notebooks; it is not the active runtime
definition.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.0.1
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -p 'test_*.py'
```

River 0.26.1 is installed reproducibly. Neutral Page-Hinkley and ADWIN adapters
are available for composition, synthetic tests, and the technical cyclic activity
runtime. Their development profiles and technical defaults are not calibrated or
validated event-detection choices.

Formal polarization measures are available for isolated, validated
`OpinionDistribution` inputs through neutral adapters for Esteban-Ray, EMDPol,
and MEC. The alpha `pol_measures` dependency is pinned to its audited Git commit;
its library defaults are integration defaults, not calibrated project choices.
This layer does not yet classify comments, build temporal polarization signals,
or connect polarization values to detectors.

## Main execution paths

Versionable JSON profiles and the common resolver are the preferred interface for
integrated executions. The common CLI is intentionally small: `--config`,
`--run-id`, `--output-root`, execution mode, and `--log-level`.

Current cyclic compatibility flow:

```bash
.venv/bin/python scripts/run_cyclic_pipeline.py \
  --config configs/compatibility/cyclic_current.json \
  --output-root outputs/cyclic_current \
  --dry-run
```

Technical neutral-detector profiles:

```bash
.venv/bin/python scripts/run_cyclic_pipeline.py \
  --config configs/development/cyclic_activity_page_hinkley.json \
  --output-root outputs/cyclic_activity_page_hinkley \
  --dry-run
```

Replace the profile with `configs/development/cyclic_activity_adwin.json` to use
ADWIN without changing Python. Both profiles use a synthetic provisional input and
stop at persisted `DetectionResult` envelopes.

The same activity runtime selects either registered signal through
`signals.activity.signal_id`: the current explicit surface contains
`comment_count_event_window_120s_step_30s` and
`unique_author_count_event_window_120s_step_30s`. Both signals can be combined with
Page-Hinkley or ADWIN through JSON routing. This is technical runtime support, not
methodological validation of either signal or detector configuration.

Current non-generative daily RAG flow:

```bash
.venv/bin/python scripts/run_daily_rag_pipeline.py \
  --config configs/compatibility/daily_rag_current.json \
  --output-root outputs/daily_rag_current \
  --dry-run
```

`youtube_pipeline/run_pipeline.py` remains a compatibility facade for the
historical extraction, local storage, cleaning, and playback CLI. Stage-specific
scripts also remain available where their diagnostic or compatibility contracts
are still tested. They translate legacy arguments into the common configuration
model; they are not a second methodological authority.

## Architecture at a glance

```text
YouTube API or local files
→ Bronze/Silver persistence
→ cleaning and canonical timestamps
→ prepared dataset
→ retrospective replay or daily cyclic simulation
→ activity metric and activity signal
→ ActivityObservation
→ detector routing
→ DetectionResult
→ EventCandidate (internal contract; not yet the general runtime handoff)
→ evidence assembly
→ RAG sidecars and context selection
→ G-1 internal validation / G-2 external validation
```

The independent future polarization path currently stops at its formal measure:

```text
CommentStance (contract/design only)
→ OpinionDistribution
→ configured polarization measure
→ PolarizationMeasurement
```

XIAO EMA is the current reference detector and regression anchor. It is not a
commitment to the final detector. The daily frequency baseline remains a separate
implemented route and has not yet been migrated to the neutral
`DetectionResult → EventCandidate` handoff.

## Repository structure

- `youtube_pipeline/`: domain components, contracts, configuration, and runners.
- `scripts/`: maintained entrypoints plus historical audit/compatibility tools.
- `configs/`: versionable execution profiles; no secrets or datasets.
- `tests/`: synthetic unit, contract, regression, compatibility, and integration
  tests.
- `data/`: local acquisition and prepared datasets; not repository source.
- `experiments/`: local experimental outputs and retained evidence.
- `docs/`: current contracts, implementation references, audits, and historical
  decisions.

## Documentation map

The general documents explain and link. The specialized documents define their
contracts.

| Need | Authority |
|---|---|
| End-to-end architecture and implementation status | [`docs/pipeline_architecture.md`](docs/pipeline_architecture.md) |
| Data layers, canonical fields, time, IDs, and deduplication | [`docs/data_contracts.md`](docs/data_contracts.md) |
| Activity metrics, signals, observations, detection results, and candidates | [`docs/activity_signal_semantics.md`](docs/activity_signal_semantics.md) |
| Opinion distributions and formal polarization measures | [`docs/polarization_measures.md`](docs/polarization_measures.md) |
| Candidate/evidence/RAG preparation boundary | [`docs/rag_event_evidence_contract.md`](docs/rag_event_evidence_contract.md) |
| Profiles, `RunConfig`, resolved paths, hashes, and identities | [`configs/README.md`](configs/README.md) |
| Hierarchical G-2 implementation | [`docs/rag_g2_hierarchical_implementation.md`](docs/rag_g2_hierarchical_implementation.md) |
| External-evidence temporal audit | [`docs/external_evidence_temporal_audit.md`](docs/external_evidence_temporal_audit.md) |
| LLM token/context audit | [`docs/llm_token_context_audit.md`](docs/llm_token_context_audit.md) |
| G-1 gate/context stability experiment | [`docs/g1_gate_context_stability_experiment.md`](docs/g1_gate_context_stability_experiment.md) |
| Cyclic temporal semantics and implementation history | [`docs/cyclic_ingestion_simulation_design.md`](docs/cyclic_ingestion_simulation_design.md) |

Historical audits, regression reports, PoCs, and experiment registries retain the
facts and decisions of their stated cutoff. Their status headers link back to the
current architecture; they are evidence, not competing normative specifications.

## Secrets, data, and generated artifacts

API keys remain in environment variables such as `YOUTUBE_API_KEY`,
`OPENAI_API_KEY`, and `SERPER_API_KEY`. They are not part of `RunConfig`,
`resolved_config`, `config_hash`, or methodological manifests.

Real datasets, Bronze/Silver/Gold materializations, experiment outputs, caches,
and external API responses remain local unless a separate policy explicitly
selects a small artifact for versioning. Configuration profiles identify paths;
they do not version or distribute datasets.

## Compatibility boundary

Historical event IDs, RAG stage IDs, sidecar schemas, prompts, and artifact names
remain preserved where tests or retained evidence depend on them. Compatibility
does not make historical paths or schemas the final architecture. Removal requires
an explicit migration with equivalent tests and verified consumers.
