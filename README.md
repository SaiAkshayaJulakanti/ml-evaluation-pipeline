# Evaluation protocol, baseline, search and model selection

- Day 1 (Mon 19 Oct 2026): evaluation protocol and baseline.
- Day 2 (Tue 20 Oct 2026): leakage-safe pipelines and bounded hyperparameter search. See [Day 2](#day-2-leakage-safe-pipelines-and-bounded-search) below.
- Day 3 (Wed 21 Oct 2026): error analysis and model selection. See [Day 3](#day-3-error-analysis-and-model-selection) below.
- Day 4 (Thu 22 Oct 2026): experiment logs and reproducibility. See [Day 4](#day-4-experiment-logs-and-reproducibility) below.
- Day 5 (Fri 23 Oct 2026): model validation capstone. See [Day 5](#day-5-model-validation-capstone) below.

Picks a dataset and target, defines a leakage-safe validation policy based on
the data structure (stratified / group / time-aware), fixes the metrics, and
records baseline scores that later models must beat.

Dataset: **simulated** hourly energy-meter data (20 buildings, 150 days),
target = consumption anomaly (3.1% of hours). It is simulated because the
sandbox could not download real data; swap `evalproto/data.py::load_dataset`
for real data and the rest of the protocol applies unchanged.

Full write-up of the decisions: [docs/SPLIT_POLICY.md](docs/SPLIT_POLICY.md).

## Layout

```
evalproto/
  data.py       simulated energy-meter data + causal features
  splits.py     strategy selection, hold-out/CV splitters, leakage checks
  metrics.py    metric set (primary: PR-AUC)
  baseline.py   dummy floors, rule baseline, logistic regression -> results/baseline.json
  search.py     pipelines + bounded seeded random search -> results/search.json
  analysis.py   OOF error analysis, thresholds, slices, bootstrap, selection -> results/model_selection.json
  tracking.py   experiment log (data hash, params, code revision, metrics, artifacts) + re-run check
  inference.py  raw readings -> validated features -> score -> alert; model package save/load
  monitoring.py PSI drift, alert-rate, input-quality and label signals
  final.py      release gates, one-time hold-out evaluation + ledger, packaging, CLI
tests/          test_protocol.py (22, Day 1), test_search.py (17, Day 2), test_analysis.py (30, Day 3), test_tracking.py (35, Day 4), test_inference.py (24), test_monitoring.py (21), test_final.py (22), all Day 5
docs/           SPLIT_POLICY.md (Day 1), SEARCH_POLICY.md (Day 2), MODEL_SELECTION.md (Day 3), REPRODUCIBILITY.md (Day 4), FINAL_REPORT.md and MODEL_CARD.md (Day 5)
models/         final/package.json (metadata; model.joblib is git-ignored, rebuild with `final build`)
results/        baseline.json, search.json, model_selection.json (generated), experiments.jsonl (run log, committed)
```

## Run

```bash
pip install -r requirements.txt           # numpy, pandas, scikit-learn
python -m evalproto.baseline               # prints table, writes results/baseline.json
python -m evalproto.search                 # Day 2: ~1 min, writes results/search.json
python -m evalproto.analysis               # Day 3: ~25 s, needs search.json, writes results/model_selection.json
python -m evalproto.tracking run           # Day 4: log the selected experiment (~5 s)
python -m evalproto.tracking verify        # Day 4: re-run and compare within tolerance (exit 0 = consistent)
python -m evalproto.final build               # Day 5: rebuild models/final (needs search.json + model_selection.json)
python -m evalproto.final predict raw.csv    # Day 5: alerts from raw readings
python -m evalproto.final monitor raw.csv    # Day 5: drift / quality / label signals
python -m unittest discover -s tests -v    # 171 tests, ~1.5 min (pytest also works)
python -m evalproto.data                   # optional: writes data/energy_meter.csv (git-ignored)
```

Tested with Python 3.13, scikit-learn 1.9.1. No network needed.

## Decisions in one paragraph

The data is time-ordered with 20 buildings and a rare target, so validation is
time-aware: a locked hold-out of the last 20% of hours plus 5-fold
expanding-window CV on the rest, with a 24 h gap between train and test
(matching the lag window and event length). Splits cut on unique timestamps so
buildings measured in the same hour stay together. Buildings appear on both
sides on purpose (forecasting known sites). Code falls back to group-aware or
stratified splitting when there is no time column. Primary metric is PR-AUC;
recall is the key secondary metric; accuracy is shown only to prove it misleads.

## Baseline (hold-out: 13,720 rows, 375 positives)

| Model | PR-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Majority class | 0.027 | 0.000 | 0.000 | 0.973 |
| Rule: >50% off same hour last week | 0.524 | 0.965 | 0.497 | 0.972 |
| Logistic regression | 0.609 | 0.979 | 0.545 | 0.977 |

CV PR-AUC (5 expanding-window folds): majority 0.033 ± 0.002, rule 0.522 ± 0.020, logistic regression 0.590 ± 0.016.

## Evidence of negative / boundary behaviour

Covered in `tests/test_protocol.py`:

- overlapping rows, shared groups, training data later than test data, and a training period inside the 24 h gap each raise `LeakageError`; a gap of exactly `gap + 1` steps is accepted (boundary)
- single-class target and a class with 5 rows raise `SplitPolicyError`; exactly 10 rows is accepted (boundary)
- time hold-out whose test period has too few positives is rejected; a time range too short for the gap is rejected
- features are causal: changing future readings leaves earlier feature rows unchanged
- majority-class model scores 97% accuracy with 0 recall (the accuracy trap)
- metrics reject single-class `y_true` and NaN scores

## Known limitations

- Simulated data: scores do not transfer to real meters, only the protocol does.
- Known buildings only: performance on a new building is not measured.
- 150 days, one season: no yearly seasonality.
- Anomalies contaminate the lag features 24 h / 168 h later, which caps precision (about 0.5).
- Decision threshold fixed at 0.5.
- Stratified and group branches are tested on synthetic arrays only.

---

# Day 2: leakage-safe pipelines and bounded search

Full write-up: [docs/SEARCH_POLICY.md](docs/SEARCH_POLICY.md).

**What it does.** `python -m evalproto.search` runs a seeded random search for two
model families (logistic regression, gradient boosting). Every candidate is a
`Pipeline(scale -> classifier)`, scored by mean PR-AUC over time-aware CV folds
of the **training portion only**. The best config per family is refit on all
training rows and scored on the locked hold-out once. The model family is chosen
by CV score, never by hold-out.

**Decisions.**
- The hold-out is not an argument to the search function, so it cannot be used by accident.
- Scaling is inside the pipeline, so it is fit per fold on training rows only.
- Reproducible: seeded sampler and models, single process; two full runs give identical JSON.
- Bounded: at most 50 trials per call (hard cap, error above it), a wall-clock budget per family that stops cleanly with partial results, and an optional cap on rows per fit.

**Results (seed 42, 12 trials per family).**

| Family | Best CV PR-AUC | Hold-out PR-AUC | Hold-out recall | Hold-out precision |
|---|---|---|---|---|
| Logistic regression | 0.586 | 0.609 | 0.979 | 0.545 |
| Gradient boosting (selected) | 0.991 | 0.997 | 0.989 | 0.976 |

Tuning within a family moved CV PR-AUC by under 0.015; switching family is what
gave the jump. A shuffled-label control scores near chance, so it is not leakage.

**Evidence of negative / boundary behaviour** (`tests/test_search.py`):
- search never fits on a hold-out row; scaler is fit on each fold's training rows only
- 51 trials rejected, exactly 50 accepted, 0 trials and a non-positive time budget rejected
- time budget exhausted mid-search returns partial results flagged `stopped_early`; exhausted before the first trial returns no best config
- `max_rows` caps rows per fit; unknown model family rejected
- same seed gives identical trials, different seed gives different configs
- shuffled labels collapse to chance (a leaking search would still score high)

**Known limitations.** Simulated data (the 0.99 will not transfer to real
meters); only 12 random candidates; no nested CV, so CV is slightly optimistic
(the hold-out agrees); the time budget is checked between candidates, not inside one.

---

# Day 3: error analysis and model selection

Full write-up with every number: [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md).

**What it does.** `python -m evalproto.analysis` compares four candidates (rule,
logistic regression, class-weighted boosting, unweighted boosting) on several
kinds of evidence instead of one score, then applies a written selection rule.
Everything that decides is computed from out-of-fold predictions on the training
portion; the hold-out only confirms.

**Evidence examined:** fold-to-fold stability and overfit gap (train vs CV score);
class imbalance (lift over the no-skill floor, effect of class weighting); threshold
trade-off (precision, recall and alerts per 1,000 hours); error slices by building,
time of day, weekday/weekend, temperature, anomaly type and position in the event;
where false alarms come from; and a paired bootstrap (resampling days) on the hold-out.

**Decision:** gradient boosting without class weights, alerting at score >= 0.303
(threshold chosen as the highest that still catches 95% of anomalies out-of-fold).

| Model | CV PR-AUC | Overfit gap | Hold-out precision | Hold-out recall | False alarms in hold-out |
|---|---|---|---|---|---|
| Rule | 0.519 | -0.001 | 0.496 | 0.979 | 373 |
| Logistic regression | 0.586 | -0.002 | 0.599 | 0.963 | 242 |
| Boosting, weighted | 0.991 | 0.009 | 0.978 | 0.965 | 8 |
| Boosting, unweighted (selected) | 0.990 | 0.010 | 0.979 | 0.981 | 8 |

**Main findings.**
- Boosting beats the rule and logistic regression decisively (paired bootstrap on the hold-out: +0.47 and +0.39 PR-AUC, intervals far from zero).
- Weighted and unweighted boosting are tied (difference +0.000, interval includes zero), so the simpler unweighted model wins by the written tie rule, not by a 0.001 edge.
- 92% of the rule's false alarms and 89% of logistic regression's fire exactly 24 or 168 hours after a real anomaly (about 15 times the chance rate). That explains the precision of about 0.5 on Day 1.
- Weakest slices for the selected model: work hours (recall 0.92 vs 0.97 at night), weekends (0.92 vs 0.96) and warm hours (0.93 vs 0.96).

**Evidence of negative / boundary behaviour** (`tests/test_analysis.py`):
- a slice with 19 anomalies is flagged `insufficient_positives` and gets no metrics, exactly 20 is scored; an all-positive slice is flagged
- a recall target of 0, negative or above 1 is rejected; 1.0 is accepted; a target with no positives to reach is rejected
- bootstrap rejects fewer than 10 days and all-single-class resamples; identical models give a zero difference whose interval covers 0
- selection: an overfit model is excluded even if it has the best CV score, a weak slice excludes a model, no eligible model raises an error, a lead exactly at the tie margin is a tie and just beyond it is not
- missing `results/search.json` gives a clear error telling you to run Day 2 first
- the tests were mutation-checked: reversing the threshold direction, dropping the 168 h lag, removing the small-slice guard and disabling the tie rule each make tests fail

**Data change.** The dataset now also exposes slice-only metadata (anomaly type and
position in the event). It is never a model input, and features, labels, time and groups
were verified identical to Day 1, so earlier results are unchanged.

**Known limitations.** Simulated data (absolute scores will not transfer, and the slice
and error findings must be re-measured on real data); the overfit and slice gates did
not separate these four models and their limits are judgement calls; the 95% recall
target is an assumption about cost; the work-hours explanation is an untested
hypothesis; new-building performance is still not measured.

---

# Day 4: experiment logs and reproducibility

Full write-up: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

**What it does.** `python -m evalproto.tracking run` runs the selected experiment (Day 3 winner,
gradient boosting without class weights) and appends one JSON line to `results/experiments.jsonl`:
SHA-256 fingerprint of the dataset, preprocessing steps, hyper-parameters, seed and split, git commit
(plus a dirty flag and a hash of the source), library versions, metrics, and hashes of the input/output
files. `python -m evalproto.tracking verify` re-runs the latest logged experiment and compares.

**Decisions.**
- The configuration must match exactly (data hash, params, seed, split); otherwise the runs are reported as not comparable instead of passing or failing.
- Dataset identity is a hash of the features rounded to 4 decimals (the exact hash differs between machines by float noise and is only recorded).
- Metrics must agree within stated tolerances: PR-AUC 0.002, precision/recall 0.01, threshold 0.05 (widened from 0.02 after measuring 0.024 on another machine), error counts 2.
- The log is append-only JSON Lines; a corrupt line raises an error.

**Result.** Same machine: every metric identical (difference 0), also with one thread. Different machine
(Python 3.13 Linux vs 3.14 Windows): consistent within tolerance, CV PR-AUC 0.98981 vs 0.98938, hold-out
PR-AUC 0.99770 vs 0.99713, recall 0.981 vs 0.976, false negatives 7 vs 9 (exactly at the tolerance).

**Evidence of negative / boundary behaviour** (`tests/test_tracking.py`):
- a changed cell, label, row order or column changes the data hash (float noise of 1e-13 does not); a changed hyper-parameter or data makes the re-run `ComparabilityError`
- a logged metric edited by 0.05 fails `verify`; exactly at tolerance passes, just beyond fails; NaN fails; a missing metric, empty log, unknown run and corrupt log line raise errors
- outside a git repo the revision is `unknown` and the source hash still works; the log never rewrites earlier lines
- mutation-checked: ignoring tolerance, skipping the configuration check, letting NaN pass, ignoring labels in the hash and overwriting the log each make tests fail

**Known limitations.** Tolerance is a judgement measured in two environments (the threshold one was widened after the fact, the false-negative difference sat at its limit); the log has
no signature (an edited number is caught, a deleted line is not); simulated data is regenerated, so
real data would need a stored or versioned copy; only the selected experiment is logged; no model file
is stored (it refits in about 5 s).

---

# Day 5: model validation capstone

Full write-ups: [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) (comparison report) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

**What it does.** `python -m evalproto.final evaluate` runs the one formal hold-out evaluation of the Day 3 winner
(boosting without class weights, threshold 0.303), checks release gates that are declared in code before
scoring, writes the experiment record, a ledger entry and the model package. `inference.py` turns raw hourly
readings into alerts; `monitoring.py` computes drift, alert-rate, input-quality and label signals.

**Decisions.**
- A configuration can be formally evaluated once: the ledger refuses a second evaluation before scoring anything (`HoldoutAlreadyUsedError`); a changed configuration is allowed but is logged as another look.
- Gates (PR-AUC, precision, recall at least 0.90; CV-to-hold-out drop at most 0.02; worst building recall at least 0.80, buildings with fewer than 10 anomalies not judged) are fixed in code and pinned by a test.
- The package is the evaluated model (training portion only). The binary is git-ignored; `package.json` holds the metadata, hash, versions and monitoring reference, and `final build` recreates the model.
- Feature drift alone can only raise the monitoring status to `warn`; `alert` needs an output-side signal.

**Result.** Hold-out PR-AUC 0.9977, precision 0.979, recall 0.981 (8 false alarms, 7 misses of 375), worst
judged building 0.867; all five gates passed, release = yes (on simulated data).

**Evidence of negative / boundary behaviour** (`tests/test_final.py`, `test_inference.py`, `test_monitoring.py`):
- each gate fails alone when its metric is just beyond the limit, passes exactly at it; NaN or no judged building fails; random scores and a single decision stump are not released
- a second evaluation of the same configuration is refused and writes nothing; tests never touch the real ledger
- features from raw readings equal the training features; 168 rows give no score, 169 give one; missing columns, NaN, bad values, duplicates and hour gaps are rejected
- altered model file, missing files, wrong scikit-learn version and changed feature list are rejected on load
- PSI and alert-rate status boundaries (exactly at the limit is not flagged); too few rows or expected alerts gives `not_evaluable`; drifted input raises an alert
- mutation-checked: reversed gate, disabled ledger, no gap check, wrong lag, no file hash check, uncapped feature drift, scoring rows without history and judging tiny slices each make tests fail
- bugs found by the tests and fixed: CSV round-trip float noise falsely flagged out-of-range values; a float edge made a gate at exactly 0.02 fail; the alert-rate signal was too noisy on small windows

**Known limitations.** Simulated data, so absolute numbers will not transfer; the hold-out was also scored for
confirmation on Days 2 to 4; gates and monitoring thresholds are judgement calls; known buildings only; one season; a
pickle package must only be loaded from a trusted source. See the model card for the full list.
