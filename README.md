# Evaluation protocol and baseline

Day 1 (Mon 19 Oct 2026): evaluation protocol and baseline.

Picks a dataset and target, defines a leakage-safe validation policy based on
the data structure (stratified / group / time-aware), fixes the metrics, and
records baseline scores that later models must beat.

Full write-up of the decisions: [docs/SPLIT_POLICY.md](docs/SPLIT_POLICY.md).

## Layout

```
evalproto/
  data.py       dataset + target (WDBC breast cancer, malignant = 1)
  splits.py     strategy selection, hold-out/CV splitters, leakage checks
  metrics.py    metric set (primary: PR-AUC)
  baseline.py   dummy + logistic regression baselines -> results/baseline.json
tests/          happy-path and negative/boundary tests
docs/           split policy, metric choice, baseline, limitations
results/        baseline.json (generated)
```

## Run

```bash
pip install -r requirements.txt           # numpy, pandas, scikit-learn
python -m evalproto.baseline               # prints table, writes results/baseline.json
python -m unittest discover -s tests -v    # 16 tests (pytest also works)
```

Tested with Python 3.13, scikit-learn 1.9.1. No network needed (dataset ships with scikit-learn).

## Decisions in one paragraph

Rows are independent and untimed, but the target is imbalanced (37% positive),
so splits are stratified: a locked 20% hold-out plus 5x3 repeated stratified CV
on the rest. Code picks group-aware or time-aware splitting automatically if a
group or time column is supplied. Primary metric is PR-AUC; recall is the key
secondary metric; accuracy is reported only to show it is misleading.

## Baseline (hold-out, 114 rows)

| Model | PR-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Majority class | 0.368 | 0.000 | 0.000 | 0.632 |
| Logistic regression | 0.994 | 0.929 | 0.975 | 0.965 |

CV PR-AUC of logistic regression: 0.993 ± 0.005 (15 folds).

## Evidence of negative / boundary behaviour

Covered in `tests/test_protocol.py`:

- overlapping train/test rows, shared groups, and training data later than test data each raise `LeakageError`
- single-class target and a class with 5 rows raise `SplitPolicyError`; exactly 10 rows is accepted (boundary)
- empty test set is rejected
- metrics reject single-class `y_true` and NaN scores
- majority-class model scores 63% accuracy but 0 recall (the accuracy trap)

## Known limitations

- No patient id in the bundled data, so patient-level leakage cannot be ruled out.
- Hold-out has only 42 positives; use CV for decisions.
- Baseline is near ceiling, so gains will be small relative to fold noise.
- Decision threshold fixed at 0.5.
- Group and time branches are tested on synthetic data only.
