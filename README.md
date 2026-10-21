# Evaluation protocol, baseline and search

- Day 1 (Mon 19 Oct 2026): evaluation protocol and baseline.
- Day 2 (Tue 20 Oct 2026): leakage-safe pipelines and bounded hyperparameter search. See [Day 2](#day-2-leakage-safe-pipelines-and-bounded-search) below.

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
tests/          test_protocol.py (22, Day 1), test_search.py (17, Day 2)
docs/           SPLIT_POLICY.md (Day 1), SEARCH_POLICY.md (Day 2)
results/        baseline.json, search.json (generated)
```

## Run

```bash
pip install -r requirements.txt           # numpy, pandas, scikit-learn
python -m evalproto.baseline               # prints table, writes results/baseline.json
python -m evalproto.search                 # Day 2: ~1 min, writes results/search.json
python -m unittest discover -s tests -v    # 39 tests, ~1 min (pytest also works)
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
