# G-1 gate and context-stability experiment

Status: executed exploratory experiment; no runtime policy changed.

The experiment pairs ten retained legacy candidates with real G-1 and historical
G-2 results. Eight missing FULL-context G-1 results were generated; the two
compatible retained results were reused. No G-2 model call or Serper request was
made. `OFF` remains the production baseline, while `STRICT` and `PERMISSIVE` are
counterfactual and `NOT_CALIBRATED`.

All ten FULL-context G-1 results were `confirmed`. Therefore the two candidate
gate policies route exactly the same ten candidates as `OFF`: this sample provides
no evidence of resource savings and cannot expose the risk of filtering a candidate
that later receives external support. The gate finding is `GATE_INCONCLUSIVE`.

Context stability was evaluated on four size-stratified candidates using audit-
derived B50, B75, and B90 estimates plus FULL. Reduced contexts use existing
context-unit order and deterministic whole-unit first-fit packing; comments are
never split and no semantic ranking is introduced. B50 and B75 changed confidence
and cited support in some cases. B90 preserved status and confidence in this small
sample, but the maximum candidate retained 62.1% of comments and only partial
citation overlap with FULL. The context finding is `MORE_EVIDENCE_REQUIRED`, not a
new budget recommendation.

The estimator remains rounded serialized characters divided by four. Provider
usage is persisted separately; the experiment must not describe the estimator as
exact tokenization. Full data, source hashes, individual decisions, coverage, and
usage are under `outputs/g1_gate_context_experiment/`.

Run the deterministic analysis/reuse path:

```bash
.venv/bin/python scripts/run_g1_gate_context_experiment.py \
  --config configs/audits/g1_gate_context_experiment.json
```

`--execute` permits only the capped G-1 calls defined by the config. The harness
has no G-2 or Serper execution path and uses `max_retries=0` for call control.
