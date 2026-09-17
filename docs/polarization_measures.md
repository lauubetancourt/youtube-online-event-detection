# Opinion distribution to formal polarization measure

**Status:** IMPLEMENTED TECHNICAL CONTRACT  
**Methodological status:** NOT CALIBRATED  
**Dependency authority:** `pol_measures` `main` commit
`1efd4fe3f083e478cb91a6f3ca078cad01386a80`, package version `0.1.0a2`.

## Boundary

```text
classification                 # not implemented here
→ distribution building        # standalone synthetic temporal producer available
→ OpinionDistribution
→ configured formal measure
→ PolarizationMeasurement
→ temporal polarization signal # standalone synthetic integration; not in cyclic runtime
→ change detector              # deferred
```

The implementation delegates Esteban-Ray, EMDPol, and MEC mathematics to the
pinned dependency. The local layer validates the approved common distribution,
selects an adapter, calls the public library API, and maps its scalar result to a
neutral immutable record. It contains no copied formulas.

## Input contract

A ready `OpinionDistribution` uses the approved author-mass policy and always
contains:

```text
positions = [0.00, 0.25, 0.50, 0.75, 1.00]
bin_counts = five non-negative integer author counts
weights   = five finite non-negative values whose sum is 1
support_count = unique_author_count >= 1
status = ready
```

`bin_counts` sums to `support_count`, and every weight must equal its bin count
divided by that support. Input/classified-comment counts and exclusion counts
remain attached for construction audit, but they are not sent to the measures.

Its proposition, classification run, distribution policy, support, and quality
remain traceable. A distribution with `status=no_classifiable_opinions` has zero
support and `weights=None`; adapters do not call the dependency and return an
explicit non-computed result instead of artificial polarization zero.

## Neutral output

`PolarizationMeasurement` records:

```text
measure_id
value or None
effective public parameters
status
input quality and quality reasons
support_count and unique_author_count
proposition_id
distribution_policy_id
classification_run_id
```

Quality is propagated literally. A non-finite library result raises an explicit
calculation error; it is not repaired or converted into a quality label.

## Registry and configuration

The explicit registry contains:

```text
esteban_ray
emd_pol
mec
```

Example isolated configuration:

```json
{
  "identity": {"run_id": "mec_isolated_01"},
  "polarization": {
    "measure_id": "mec",
    "mec": {
      "alpha": 2.0,
      "beta": 1.15
    }
  }
}
```

The selected measure must have a matching typed configuration block. Esteban-Ray
exposes `alpha` and `K`; MEC exposes `alpha` and `beta`; EMDPol exposes no public
constructor parameters in the pinned version. Every current default comes from
the library and is technical, not recommended or calibrated.

## Normalization

The pipeline contract owns normalization. Adapters pass the same positions and
weights to every measure with:

```text
normalize_weights=False
normalize_positions=False
```

This prevents the dependency from silently changing or repairing the resolved
distribution. EMDPol is specifically the repository's polarization measure, not
a generic Earth Mover's Distance adapter. Its implementation relies on ordered,
equally spaced bin indices, which is why empty Likert bins remain present.

## Deferred work

The formal adapters themselves do not provide comment classification, proposition
discovery, temporal aggregation, detector integration, calibration, preprocessing
evaluation, candidates, or RAG. A separate [synthetic temporal producer](polarization_temporal_signals.md)
now builds windowed distributions and emits neutral signal observations using these
unchanged adapters. It is not connected to the cyclic runtime or any detector.
