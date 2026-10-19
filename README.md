# Evaluation protocol and baseline

Day 1 (Mon 19 Oct 2026): evaluation protocol and baseline.

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
tests/          happy-path and negative/boundary tests (22)
docs/           split policy, metric choice, baseline, limitations
results/        baseline.json (generated)
```

## Run

```bash
pip install -r requirements.txt           # numpy, pandas, scikit-learn
python -m evalproto.baseline               # prints table, writes results/baseline.json
python -m unittest discover -s tests -v    # 22 tests (pytest also works)
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
