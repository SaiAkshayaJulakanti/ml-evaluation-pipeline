# Split policy, metric selection and baseline

Day 1 deliverable. Everything below is enforced in code (`evalproto/splits.py`,
`evalproto/metrics.py`) and covered by tests.

## 1. Dataset and target

| Item | Decision |
|---|---|
| Dataset | Wisconsin Diagnostic Breast Cancer (WDBC), bundled with scikit-learn (offline, reproducible) |
| Rows / features | 569 rows, 30 numeric features |
| Target | `malignant` (1) vs benign (0). sklearn's encoding is flipped so the positive class is the one that matters |
| Class balance | 37.3% positive (212 / 569). Moderately imbalanced |

## 2. Data structure, and what it implies

| Question | Answer for WDBC | Consequence |
|---|---|---|
| Repeated entities (patient, device, site)? | No group column is available | Rows treated as independent, no group split |
| Temporal order? | No timestamp | Random split is allowed, no time split |
| Imbalanced target? | Yes (63/37) | Stratify on the target everywhere |
| Small dataset? | Yes (569 rows) | Repeated CV for stable estimates, one locked hold-out |

The strategy is chosen by code, not by hand (`choose_strategy`), first match wins:

1. `time` column present -> time-aware (train strictly before test, `TimeSeriesSplit`)
2. `groups` column present -> group-aware (`GroupKFold` / `StratifiedGroupKFold`)
3. otherwise -> stratified (`RepeatedStratifiedKFold`, `train_test_split(stratify=y)`)

This keeps the policy reusable: the energy / sensor data from later days will
have timestamps and devices, so it will automatically take branches 1 and 2.

## 3. Split policy

1. **Locked hold-out**: 20% (114 rows), stratified, `random_state=42`. Used once per model for the final number. Never used for tuning or model selection.
2. **Development CV**: 5-fold x 3 repeats stratified CV on the remaining 80% (455 rows) only. All tuning and comparison happens here.
3. **Leakage guard**: every split passes `assert_no_leakage` (no shared rows, no shared groups, no training data later than test data). A violation raises `LeakageError`, it is not a warning.
4. **Preprocessing inside the fold**: scaling lives in a `Pipeline`, so it is fit on training folds only.
5. **Reproducibility**: fixed seed (`SEED = 42`); the split is deterministic (tested).
6. **Guard rails**: single-class targets, or a class with fewer than 10 rows (or fewer than `n_splits`), raise `SplitPolicyError` instead of producing an unstable estimate.

## 4. Metric selection

| Role | Metric | Why |
|---|---|---|
| Primary | PR-AUC (average precision) | Threshold-free, focuses on the positive class, informative under imbalance |
| Secondary | Recall | A missed malignant case is the costly error |
| Secondary | Precision, F1 | Cost of false alarms; threshold fixed at 0.5 for now |
| Secondary | ROC-AUC, balanced accuracy | Ranking quality, class-balanced view |
| Reported as a trap | Accuracy | Majority-class predictor already scores 63% |

## 5. Baselines and results

Run with `python -m evalproto.baseline` (writes `results/baseline.json`).

| Model | CV PR-AUC (mean ± sd, 15 folds) | Hold-out PR-AUC | Hold-out recall | Hold-out precision | Hold-out accuracy |
|---|---|---|---|---|---|
| Majority class (floor) | 0.374 ± 0.000 | 0.368 | 0.000 | 0.000 | 0.632 |
| Prior (floor) | 0.374 ± 0.000 | 0.368 | 0.000 | 0.000 | 0.632 |
| Scaled logistic regression | 0.993 ± 0.005 | 0.994 | 0.929 | 0.975 | 0.965 |

Reading the table:

- The dummy floor shows why accuracy is a trap: 63% accuracy with zero recall.
- PR-AUC of a no-skill model equals the positive rate (0.37), which is the floor any real model must beat.
- CV and hold-out agree (0.993 vs 0.994), so there is no sign of the CV being optimistic or of a lucky hold-out.
- **Target to beat for later days: CV PR-AUC 0.993 and recall 0.95.**

## 6. Known limitations

- **No patient / group id** in the bundled copy, so independence of rows from the same patient cannot be verified. If ids exist in the source data, switch to the group branch.
- **Small hold-out**: 114 rows with 42 positives. Recall 0.929 means 39 of 42 found, so the hold-out recall has a wide confidence interval (a single case moves it by about 2.4 points). Use CV for decisions, hold-out only as a final check.
- **Near-ceiling baseline**: logistic regression is already about 0.99 PR-AUC, so there is little headroom; improvements will be hard to distinguish from fold noise (CV sd is 0.005).
- **Threshold is fixed at 0.5**, not tuned to a recall target. Threshold selection must be done inside CV, not on the hold-out.
- **Time-aware and group-aware branches are tested on synthetic data only**, since WDBC has neither structure.
