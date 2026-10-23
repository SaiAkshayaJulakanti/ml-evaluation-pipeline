# Day 5: comparison report and final validation

Code: `evalproto/final.py`, `inference.py`, `monitoring.py`. Tests: `tests/test_final.py`, `test_inference.py`, `test_monitoring.py`. Experiment record: `results/experiments.jsonl` (`run-002-6d28e1`), `results/final_evaluation.json`, `results/holdout_ledger.json`. Model card: [MODEL_CARD.md](MODEL_CARD.md).

## 1. Decision

**Release candidate: gradient boosting without class weights (`hgb_unweighted`), alert threshold 0.303.** It passed all five release gates on the one formal hold-out evaluation. Release here means "passes validation on the simulated data"; it is not approval to run on real meters (section 8).

## 2. What was compared

Four candidates, all judged on training-side evidence (time-ordered cross-validation, out-of-fold scores) so the hold-out never chose between them:

| Candidate | What it is |
|---|---|
| `rule` | No training: alert when consumption is more than 50% away from the same hour last week |
| `logreg` | Scaled, class-weighted logistic regression, tuned on Day 2 |
| `hgb` | Boosting with class weighting, tuned on Day 2 |
| `hgb_unweighted` | The same boosting without class weighting |

## 3. Comparison table

| | rule | logreg | hgb | hgb_unweighted |
|---|---|---|---|---|
| CV PR-AUC (mean, sd over 4 folds) | 0.519 (0.022) | 0.586 (0.002) | 0.991 (0.005) | 0.990 (0.003) |
| Overfit gap (train minus CV PR-AUC) | -0.001 | -0.002 | 0.009 | 0.010 |
| Threshold at 95% out-of-fold recall | 0.482 | 0.683 | 0.873 | 0.303 |
| Out-of-fold precision at that threshold | 0.507 | 0.594 | 0.980 | 0.969 |
| Hold-out precision (Day 3 confirmation) | 0.496 | 0.599 | 0.978 | 0.979 |
| Hold-out recall (Day 3 confirmation) | 0.979 | 0.963 | 0.965 | 0.981 |
| Hold-out false alarms / misses | 373 / 8 | 242 / 14 | 8 / 13 | 8 / 7 |

Reading: boosting beats the rule and logistic regression by about +0.47 and +0.39 PR-AUC (paired bootstrap, intervals far from zero). The two boosting variants are statistically tied (difference +0.000), so the written tie rule picks the one with fewer assumptions: no class weights, and a threshold on a probability-like scale. The hold-out columns are the Day 3 confirmation scores and were not used to choose. Full evidence: [MODEL_SELECTION.md](MODEL_SELECTION.md).

## 4. The formal hold-out evaluation (once)

Gates were written into `evalproto/final.py` before scoring (a test pins their values). The evaluation scored the hold-out once through `tracking.run_experiment`, which also wrote the experiment record.

| Gate | Value | Limit | Result |
|---|---|---|---|
| Hold-out PR-AUC | 0.9977 | at least 0.90 | pass |
| Hold-out precision | 0.979 | at least 0.90 | pass |
| Hold-out recall | 0.981 | at least 0.90 | pass |
| CV PR-AUC minus hold-out PR-AUC | -0.008 | at most 0.02 | pass |
| Worst building recall (18 buildings with at least 10 anomalies) | 0.867 | at least 0.80 | pass |

Counts at the threshold: 368 of 375 anomalies caught, 8 false alarms in 13,345 normal building-hours, 27.4 alerts per 1,000 building-hours (true anomaly rate 27.3).

**One-time rule.** `results/holdout_ledger.json` records the configuration (`6d28e1a05f466d11`). A second formal evaluation of the same configuration raises `HoldoutAlreadyUsedError` before anything is scored; a changed configuration is allowed but becomes a second ledger entry, so the number of looks at the hold-out stays visible. Reproducibility checks use `python -m evalproto.tracking verify` instead.

**Negative controls.** Random scores fail the gates; a single decision stump (hold-out PR-AUC 0.48) is evaluated end to end, fails the gates and is marked `release: false` in its package.

## 5. Slices

| Slice | Anomalies | Recall | Precision |
|---|---|---|---|
| Night / evening | 218 | 0.991 | 0.982 |
| Work hours | 157 | 0.968 | 0.974 |
| Weekday | 251 | 0.980 | 0.980 |
| Weekend | 124 | 0.984 | 0.976 |
| Building 18 (lowest judged) | 15 | 0.867 | 1.000 |
| Buildings 0 and 14 | 2 and 7 | not judged | not judged |

Slices with fewer than 10 anomalies are reported but not scored. The Day 3 weak spots (work hours, weekends) are not visible on the hold-out; with 375 anomalies a 4 to 5 point recall difference is within noise.

## 6. Inference pipeline

`evalproto/inference.py` turns raw readings into alerts and refuses input it cannot score correctly. Evidence (`tests/test_inference.py`):

- **Training/serving parity:** features computed from raw readings equal the training features to 1e-12 on all 68,640 rows, and scores equal the model's scores on the training features.
- **Causal:** changing future consumption does not change earlier scores.
- **Boundary:** 168 rows give no score, 169 rows give exactly one; history is counted per building.
- **Rejected input:** missing column, NaN, infinity, text, zero or negative consumption, non-positive size, duplicate hour, a gap in hours, fractional or negative hour, empty input.
- **Range warnings:** rows outside the training range are scored with a warning; exactly at the edge is not flagged. A found bug: values that went through CSV changed at the 1e-14 level and flagged the largest building (472 rows), so the check has a 1e-9 relative slack, with tests for both sides.
- **Package:** altered model file, missing files, wrong scikit-learn version and a changed feature list are all rejected on load.

## 7. Reproducibility and environments

| | Value |
|---|---|
| Experiment record | `run-002-6d28e1` (stage `final_evaluation`), code commit `6db466c8`, clean tree, Python 3.13.16 / numpy 2.5.3 / scikit-learn 1.9.1 |
| Dataset identity | `c54ee06d2aa26aec…` (features rounded to 4 decimals) |
| Cross-environment | Day 4 re-run on Python 3.14 / Windows stayed within the stated tolerances (recall 0.976, 9 misses instead of 7). The package is rebuilt per machine with `build` because a saved pickle depends on library versions |

## 8. Limitations and what would change the decision

- All numbers come from simulated data; real data would very likely score lower and must go through Days 1 to 5 again.
- The hold-out was scored on Days 2 to 4 for confirmation as well; the ledger starts on Day 5.
- Gates and the 95% recall target are judgement calls with no real cost of errors behind them.
- Known buildings only; no unseen-building test; one hold-out period and one season.
- Monitoring thresholds (PSI 0.10/0.25, ratio 1.5/2.5, 0.90 label floor) are conventions, not measured on real drift.
- The decision would change if: a gate failed on re-evaluation with new data, labelled production data show recall below 0.90, or a group-aware test on unseen buildings failed.

## 9. Index of deliverables

| Deliverable | Where |
|---|---|
| Comparison report | this file |
| Code | `evalproto/final.py`, `inference.py`, `monitoring.py` (+ `tracking.py` changes) |
| Tests | `tests/test_final.py`, `test_inference.py`, `test_monitoring.py` (171 tests in total) |
| Experiment record | `results/experiments.jsonl`, `results/final_evaluation.json`, `results/holdout_ledger.json` |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |
| Package metadata | `models/final/package.json` (binary rebuilt with `python -m evalproto.final build`) |
