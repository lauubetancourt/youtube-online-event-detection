# KSWIN: technically available, experimentally open

KSWIN is not the official or default polarization detector. Its availability is
an integration result, not evidence of superiority or event-detection validity.
This phase uses synthetic scalar observations only: no classifier, corpus,
polarization runtime, association, candidate fusion, calibration, or RAG execution.

## Verified authority and semantics

River **0.26.1** installed source is authoritative. Its public API is:

```python
KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=None, window=None)
update(x)
drift_detected
```

The implementation mutates state and `update` returns `None`, despite the
docstring's return description. The adapter reads `drift_detected` after update.
Source: [River 0.26.1 KSWIN](https://github.com/online-ml/river/blob/0.26.1/river/drift/kswin.py).

KSWIN keeps a bounded deque of `window_size` scalar observations. Once full, it
draws `stat_size` historical values without replacement from the first
`window_size - stat_size` entries and compares them with the last `stat_size`
entries using SciPy's two-sample Kolmogorov–Smirnov test. In this version the
actual drift condition is **p_value <= alpha and KS statistic > 0.1**. This
code condition, rather than the explanatory formula in the docstring, governs
integration. Before a full window there is no test/drift. No extra warmup or
cooldown layer is introduced.

For `MEC(t1), MEC(t2), ...`, KSWIN analyzes the distribution over time of scalar
MEC values. It does **not** compare `OpinionDistribution(t1)` with
`OpinionDistribution(t2)`. The same applies to ER, EMDPol, activity or other finite
numeric observations; it does not inspect their metric identities.

Three distinct concepts must remain separate:

| Concept | Unit / responsibility |
| --- | --- |
| Signal window | Physical time; selects source comments to construct a scalar |
| Signal cadence | Physical time; emission grid for that scalar |
| KSWIN internal window_size | Number of scalar updates; detector-owned history |

Skipped absences do not count as KSWIN samples. Therefore its internal history
does not imply a fixed physical-time duration, especially across gaps.

## Configuration and validation

`KSWINConfig` is frozen and slotted. It is the sole parameter authority; registry
and factories contain no second set of defaults. Library defaults are technical,
not recommended, calibrated, optimal or methodologically validated.

| Parameter | Library default | Classification |
| --- | --- | --- |
| alpha | 0.005 | METHODOLOGICAL: statistical sensitivity |
| window_size | 100 | METHODOLOGICAL: detector observation history |
| stat_size | 30 | METHODOLOGICAL: recent/reference sample size |
| seed | None | TECHNICAL: sampling reproducibility |

The default values themselves are `LIBRARY_DEFAULT`. Alpha must be finite and in
`[0,1]`, including the endpoints accepted by River. Sizes must be positive integers;
seed is an integer (negative values allowed) or None. Booleans are not numbers for
configuration. Unknown fields and non-finite/invalid numeric types are rejected.

An additional **computational precondition derived from the actual update code**
is `window_size >= 2 * stat_size`: the historical population must contain at least
`stat_size` distinct sample slots. River's constructor only checks the weaker
`window_size >= stat_size` and otherwise crashes when the first test samples too
many values. Positive sample sizes are needed for nonempty KS inputs. These checks
prevent invalid execution; they are not empirical methodological constraints.

Public constructor `window` is initial data/state, not a tunable parameter, and
is intentionally excluded from config. History enters only through observations,
not a hidden preloaded buffer that bypasses causal input handling.

Example technical JSON (seed is an explicit reproducibility choice):

```json
{
  "identity": {"run_id": "synthetic_kswin"},
  "detection": {
    "activity_route": {
      "signal_id": "polarization_mec",
      "detector_id": "kswin"
    },
    "kswin": {
      "alpha": 0.005,
      "window_size": 100,
      "stat_size": 30,
      "seed": 42
    }
  }
}
```

This composes an isolated route; it does not activate a polarization runtime.
Loading, path resolution, canonical serialization and config_hash preserve every
effective parameter. Changed parameters or seed change the current hash. Its
existing run_id participation is unchanged. Unconfigured KSWIN blocks are omitted
from legacy serialization, preserving existing effective configurations.

## Shared observation boundary and absence policy

`NumericObservation` is the minimal compatible union of `ActivityObservation`
and `SignalObservation`. `NumericObservationDetector` defines the same single
`on_observation(...) -> DetectionResult` operation for PH, ADWIN and KSWIN. The
historical `ActivityObservationDetector` and XIAO lifecycle remain supported;
XIAO is not generalized to arbitrary polarization observations in this phase.

`dispatch_activity_observation` retains its compatibility name and route model.
An identity bridge handles activity's nested signal ID and the neutral observation's
flat ID. Dispatch has no detector-specific or measure-specific branching.
Available observations must have finite real values; PH, ADWIN and KSWIN share
this check. They update only with `observation.value`, never support, quality,
opinion vectors, proposition text or context. The source contracts are unchanged.

For `NO_INPUT`, `NO_CLASSIFIABLE_OPINIONS` and `NO_REPRESENTABLE_AUTHORS`, dispatch
still validates route identity and detector availability, then returns **None**
without calling an adapter. No zero, NaN or missing value is supplied to River;
neither counters nor a pending post-drift reset advance. Direct adapters reject
absent observations before touching state.

`TEMPORAL-DETECTOR-GAP-POLICY = OPEN`: this technical skip behavior does not decide
whether a sufficiently long gap should eventually reset a detector. No gap
threshold, interpolation, timestamp ordering policy or reset-on-gap is introduced.
Temporal producers remain responsible for causal ordered emission.

## Result mapping, public metadata and lifecycle

Mapping is `detector_id=kswin`, source signal ID and observation UTC time,
`triggered=bool(drift_detected)`, literal source quality, and **score=None**.
No p-value or internal statistic is presented as a cross-detector score or confidence.

Small immutable JSON metadata contains:

- `observations_processed`: adapter-owned lifetime successful updates, gaps excluded;
- `segment_observations`: public River `n`, reset with River's segment;
- `window_length`: length only of public River `window`, never buffer contents;
- resolved `alpha`, `window_size`, `stat_size` and `seed`.

River also exposes `p_value`; it starts at zero before any test, so it is not
included as misleading decision metadata. No private River attributes, RNG
objects, raw detector, `__dict__`, or history are exported.

On drift, River keeps the latest `stat_size` values in its public deque and marks
drift. At the **next valid update**, it automatically clears the entire deque,
resets `n`/p_value/drift and reinitializes its RNG using seed, then incorporates
the new value. Thus the truncated recent history is not carried into the next
segment. KSWINAdapter adds no manual reset, XIAO lifecycle or cooldown.

With an explicit integer seed, the same ordered update stream/config in the same
runtime is `DETERMINISTIC_WITH_SEED`, including across algorithm resets. With
seed=None, new instances use system-initialized randomness: `NON_DETERMINISTIC_RISK`.
Reproducible evaluations must choose and record seed, alongside dependency versions.

## Verification and compatibility limits

Synthetic tests cover stable temporal values, shifted mean and a change from
concentration around 0.5 to alternating 0.2/0.8 with the same mean. Sensitivity
and sample sizes used to obtain synthetic triggers are **TEST_ONLY**. They are
not event-detection recommendations and no claims about ADWIN/PH failure are made.
Adapter triggers and public lifecycle state are checked against direct calls to
the installed River implementation using the same seed.

The existing standalone polarization producer emits all three measures into the
same generic dispatcher in tests. Its scalar values and quality are unchanged;
absence ticks generate no decisions. There is no complete polarization detector
runtime connection, persistence stage or classifier in this phase.

KSWIN activity DetectionResults can use existing generic candidate promotion in
an isolated test. EventCandidate currently still requires ActivityObservation;
promotion of SignalObservation is **not** claimed or implemented. No candidate
contract changes, automatic promotion, fusion or downstream association occur.

The only cyclic-entrypoint change preserves the typed kswin block when relocating
existing output configuration. No profile or default detector is changed, and no
polarization route is connected to that entrypoint.

## Backlog and hypotheses

- RESOLVED: PH, ADWIN and KSWIN technical integration; synthetic temporal polarization.
- TECHNICALLY_AVAILABLE / EXPERIMENTALLY_OPEN: all three neutral numeric detectors.
- OPEN: classifier/validation, proposition discovery, polarization detector runtime,
  detector gap/reset policy, preprocessing, timestamp-based alert association,
  EventCandidate fusion, calibration, and existing configuration/traceability debts.
- NEXT: `POLARIZATION-CLASSIFIER-IMPLEMENTATION`.

These hypotheses remain open, never encoded as routing restrictions:

- H1: ADWIN may be well suited for activity.
- H2: KSWIN may add information for scalar polarization signals.
- H3: Page-Hinkley remains a useful level-shift baseline.
