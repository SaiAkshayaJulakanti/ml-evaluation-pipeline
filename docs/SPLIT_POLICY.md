# Split policy, metric selection and baseline

Day 1 deliverable. Everything below is enforced in code (`evalproto/splits.py`,
`evalproto/metrics.py`) and covered by tests.

## 1. Dataset and target

| Item | Decision |
|---|---|
| Dataset | **Simulated** hourly energy-meter data: 20 buildings x 150 days, deterministic (seed 42). Simulated because the sandbox cannot download real data |
| Rows / features | 68,640 rows (first 168 h per building dropped for lag windows), 10 numeric features |
| Target | `anomaly` = hour inside an injected consumption event (spike 1.5-2.5x, or drop to 0.2-0.5x, lasting 1-4 h) |
| Class balance | 3.1% positive. Rare target |

Features (all causal, tested): temperature, building size, hour-of-day sin/cos, weekend flag, log consumption, and the log ratio of consumption to the same hour yesterday / last week (signed and absolute).

## 2. Data structure, and what it implies

| Question | Answer | Consequence |
|---|---|---|
| Temporal order? | Yes, hourly index | Time-aware splits. A random split would train on the future |
| Repeated entities? | Yes, 20 buildings | Same buildings on both sides **by design**: the goal is forecasting known sites. New-building generalisation is not evaluated (see limitations) |
| Lagged features / persistent events? | 24 h and 168 h lags, events last up to 4 h | A 24 h gap between train and test |
| Rare target? | 3.1% | PR-AUC and recall, never accuracy |
| Distribution shift over time? | Temperature drifts from 25 C to 7 C | Hold-out is the final 20% of time, so it measures performance on a colder period |

The strategy is chosen by code, not by hand (`choose_strategy`), first match wins:

1. `time` column present -> time-aware (train strictly before test, with a gap)
2. `groups` column present -> group-aware (`GroupKFold` / `StratifiedGroupKFold`)
3. otherwise -> stratified (`RepeatedStratifiedKFold`, `train_test_split(stratify=y)`)

This dataset takes branch 1. Branches 2 and 3 are kept and tested on synthetic arrays so the same code serves other datasets.

## 3. Split policy

1. **Locked hold-out**: the last 20% of unique hours (hours 2914-3599, 13,720 rows, 375 positives). Used once per model for the final number. Never used for tuning or model selection.
2. **Gap**: 24 hours removed from the end of the training period (train ends at hour 2889, test starts at hour 2914). It covers the 24 h lag window and event persistence.
3. **Development CV**: 5-fold expanding-window CV on the first 80% only. Fold k trains on chunks 0..k and validates on chunk k+1, with the same 24 h gap. All tuning and comparison happens here.
4. **Cuts are made on unique timestamps**, so the 20 buildings measured in the same hour always land on the same side.
5. **Leakage guard**: every split passes `assert_no_leakage` (no shared rows, no shared groups for group splits, test starts more than `gap` steps after train ends). A violation raises `LeakageError`, it is not a warning.
6. **Causal features**: a test perturbs future readings and checks that no earlier feature row changes.
7. **Preprocessing inside the fold**: scaling lives in a `Pipeline`, so it is fit on training folds only.
8. **Guard rails**: single-class targets, too-rare classes, a validation fold with fewer than 10 of either class, or a time range too short for the gap raise `SplitPolicyError` instead of producing an unstable estimate.

## 4. Metric selection

| Role | Metric | Why |
|---|---|---|
| Primary | PR-AUC (average precision) | Threshold-free, focuses on the rare positive class |
| Secondary | Recall | A missed anomaly is the costly error |
| Secondary | Precision, F1 | False alarms waste attention; threshold fixed at 0.5 for now |
| Secondary | ROC-AUC, balanced accuracy | Ranking quality, class-balanced view |
| Reported as a trap | Accuracy | Predicting "never anomalous" already scores about 97% |

Because the positive rate differs by period (3.2% in training, 2.7% in the hold-out), compare PR-AUC against the no-skill floor of the same period, not across periods.

## 5. Baselines and results

Run with `python -m evalproto.baseline` (writes `results/baseline.json`).

| Model | CV PR-AUC (mean ± sd, 5 folds) | Hold-out PR-AUC | Hold-out recall | Hold-out precision | Hold-out accuracy |
|---|---|---|---|---|---|
| Majority class (floor) | 0.033 ± 0.002 | 0.027 | 0.000 | 0.000 | 0.973 |
| Prior (floor) | 0.033 ± 0.002 | 0.027 | 0.000 | 0.000 | 0.973 |
| Rule: deviates >50% from same hour last week | 0.522 ± 0.020 | 0.524 | 0.965 | 0.497 | 0.972 |
| Logistic regression (scaled, class-balanced) | 0.590 ± 0.016 | 0.609 | 0.979 | 0.545 | 0.977 |

Reading the table:

- Accuracy is a trap: the do-nothing model gets 97.3% accuracy and finds no anomaly.
- The no-skill PR-AUC equals the positive rate (about 0.03), which is the floor.
- The untrained rule already reaches 0.52 PR-AUC, so a learned model must clearly beat it to justify itself. Logistic regression is ahead by about 0.07 in CV (4 fold standard deviations), and the hold-out agrees.
- CV and hold-out agree (0.590 vs 0.609), so there is no sign of optimistic CV.
- Precision is about 0.5 at the default threshold: roughly one alert in two is false, even though recall is above 0.96.
- **Targets to beat on later days: CV PR-AUC 0.590 (learned) and 0.522 (rule); hold-out recall 0.979.**

## 6. Known limitations

- **The data is simulated.** Labels are injected events, so these scores say nothing about real meters. The protocol, not the numbers, is what carries over. Re-run on real data when available.
- **Known buildings only.** Evaluation measures forecasting the future of buildings seen in training. Performance on a brand-new building is not measured; that would need a group-aware hold-out.
- **One season, 150 days.** Yearly seasonality cannot be evaluated, and the hold-out is a colder period than most of the training data.
- **Label contamination in lags.** An anomaly also distorts the ratio features of the same hour 24 h and 168 h later, creating false alarms. This caps precision for ratio-based models.
- **Threshold fixed at 0.5**, not tuned to a recall or alert-rate target. Threshold selection must be done inside CV, never on the hold-out.
- **Group and stratified branches are tested on synthetic arrays only**, since this dataset takes the time branch.
