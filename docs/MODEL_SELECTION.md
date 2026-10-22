# Day 3: error analysis and model selection

Code: `evalproto/analysis.py`. Tests: `tests/test_analysis.py` (30). Output: `results/model_selection.json`.
Run: `python -m evalproto.analysis` (needs `results/search.json` from Day 2; about 25 s).

## 1. Decision

**Selected model: gradient boosting without class weighting (`hgb_unweighted`), alerting at score >= 0.303.**

This is not "the highest score wins". The choice rests on seven pieces of measured evidence below.
The most important caveat is in section 9: the data is simulated, so absolute numbers will not transfer.

## 2. Candidates and how they were judged

| Candidate | What it is |
|---|---|
| `rule` | Untrained: alert when consumption is more than 50% off the same hour last week |
| `logreg` | Scaled, class-weighted logistic regression, tuned on Day 2 (C = 21.4) |
| `hgb` | Gradient boosting, class-weighted, tuned on Day 2 |
| `hgb_unweighted` | The same tuned boosting with class weighting switched off. Added to isolate the class-imbalance effect |

**What decides and what only confirms.** Everything that decides the model (fold scores, overfit gap, slices, the alert threshold, error analysis) is computed from **out-of-fold (OOF) predictions on the training portion only**, using the Day 1 time-aware folds with the 24 h gap. The locked hold-out is scored once per model afterwards, to confirm the decision, never to make it. Caveat: with an expanding window the earliest 20% of the training period is never validated, so OOF covers the later 80% of it.

## 3. Evidence 1: a single score hides the picture, so here are several

| Model | CV PR-AUC (mean ± sd, 4 folds) | Worst fold | Train PR-AUC | Overfit gap | Lift over no-skill |
|---|---|---|---|---|---|
| rule | 0.519 ± 0.022 | 0.490 | 0.519 | -0.001 | 15.9x |
| logreg | 0.586 ± 0.002 | 0.584 | 0.584 | -0.002 | 18.0x |
| hgb | 0.991 ± 0.005 | 0.985 | 1.000 | 0.009 | 30.4x |
| hgb_unweighted | 0.990 ± 0.003 | 0.986 | 1.000 | 0.010 | 30.3x |

- **Overfit:** the two boosting models score 1.000 on their own training data and 0.99 on unseen folds, a gap of about 0.01. That is small, and the worst fold (0.985) is still far above everything else, so there is no sign of a model that only memorised. The gate for rejection is a gap above 0.05; nothing came close.
- **Stability:** the fold scores of the boosting models stay within 0.985 to 0.996, so the result is not one lucky fold.
- **Lift:** PR-AUC divided by the anomaly rate (3.2%). Boosting is about 30 times better than chance at ranking anomalies; the rule is about 16 times.

## 4. Evidence 2: class imbalance

Anomalies are 3.2% of training hours (2.7% in the hold-out), so accuracy is useless (a model that never alerts scores 97%) and PR-AUC is judged against a floor of about 0.03.

- **Class weighting made no measurable difference to ranking quality.** CV PR-AUC 0.991 versus 0.990, and the paired hold-out difference is +0.000 with a 95% interval of [-0.0003, +0.001], which includes zero (section 8).
- **It does change the score scale.** The weighted model needs an alert threshold of 0.873 to reach 95% recall; the unweighted one needs 0.303. Weighted scores are pushed towards 1 and do not behave like probabilities. The unweighted model is easier to explain and to re-tune.

## 5. Evidence 3: threshold trade-off

The default 0.5 is arbitrary. The operating threshold was instead set on the OOF scores as the **highest threshold that still catches at least 95% of anomalies** (fewest alerts for that recall).

| Model | Threshold | OOF precision at 95% recall | Hold-out precision | Hold-out recall | False alarms in hold-out (13,720 h) | Missed anomalies |
|---|---|---|---|---|---|---|
| rule | 0.482 | 0.507 | 0.496 | 0.979 | 373 | 8 |
| logreg | 0.683 | 0.594 | 0.599 | 0.963 | 242 | 14 |
| hgb | 0.873 | 0.980 | 0.978 | 0.965 | 8 | 13 |
| hgb_unweighted | 0.303 | 0.969 | 0.979 | 0.981 | 8 | 7 |

The rule and logistic regression raise 44 to 54 alerts per 1,000 hours in the hold-out, and 40% to 50% of those alerts are false. Boosting raises about 27 per 1,000, almost exactly the true anomaly rate (27.3 per 1,000), with 8 false alarms in 13,720 hours.

For the selected model the threshold barely matters, because the scores separate the classes well. Hold-out values at fixed thresholds (information only, the threshold was not chosen from them):

| Threshold | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| Precision | 0.964 | 0.976 | 0.979 | 0.979 | 0.981 | 0.989 | 0.989 | 0.995 | 0.994 |
| Recall | 0.989 | 0.984 | 0.981 | 0.979 | 0.979 | 0.976 | 0.971 | 0.968 | 0.955 |
| Alerts per 1,000 h | 28.1 | 27.6 | 27.4 | 27.3 | 27.3 | 27.0 | 26.8 | 26.6 | 26.2 |

Raising the threshold from 0.1 to 0.9 trades about 3.4 points of recall for about 3 points of precision. Which end is right depends on the relative cost of a missed anomaly versus a false alarm, which is a business decision. The 95% recall target is an assumption stated in the code (`TARGET_RECALL`).

## 6. Evidence 4: error slices (selected model, OOF, at threshold 0.303)

| Slice | Anomalies | Recall | Precision |
|---|---|---|---|
| Night / evening hours | 826 | 0.972 | 0.979 |
| Work hours (08:00-18:00) | 594 | 0.919 | 0.955 |
| Weekday | 1,018 | 0.962 | 0.972 |
| Weekend | 402 | 0.920 | 0.961 |
| Cold third of temperatures | 493 | 0.959 | 0.977 |
| Mild third | 492 | 0.959 | 0.955 |
| Warm third | 435 | 0.929 | 0.976 |
| Spikes | 970 | 0.957 | n/a |
| Drops | 450 | 0.936 | n/a |
| First hour of an event | 546 | 0.940 | n/a |
| Fourth hour of an event | 161 | 0.988 | n/a |

- **Weakest areas:** work hours (0.919), weekends (0.920) and warm hours (0.929). The overall recall hides a 5-point gap between work hours and night. Hypothesis, not tested: normal consumption itself jumps about 3.3 times at the start of a working day, so an anomaly there looks more like ordinary change.
- **Per building:** all 20 buildings have 41 to 94 anomalies and recall between 0.896 and 0.989. With samples of that size, random variation alone is about 3 points either side, so this spread is mostly noise and no building is a clear outlier.
- **Guard:** a slice with fewer than 20 anomalies is reported as `insufficient_positives` with no metrics, so small samples are not read as findings. No slice here triggered it, but the behaviour is tested.

## 7. Evidence 5: where the errors come from

For each false alarm, the analysis checks whether the same building had a real anomaly exactly 24 or 168 hours earlier. By chance 6.1% of normal hours satisfy that.

| Model | False alarms (OOF) | Share that follow a real event 24 h or 168 h earlier | Times the chance rate |
|---|---|---|---|
| rule | 1,312 | 92.0% | 15.0x |
| logreg | 922 | 88.6% | 14.5x |
| hgb_unweighted | 43 | 79.1% | 12.9x |
| hgb | 28 | 75.0% | 12.3x |

This explains the Day 1 result that precision was stuck near 0.5: nearly every false alarm is an echo of an earlier anomaly, because the "same hour yesterday / last week" comparison is contaminated by that event. Boosting removes 97% to 98% of those false alarms (43 or 28 against 922 and 1,312) by combining the two lags and the hour of day, but the few it still makes are of the same kind. Misses: at the chosen threshold the selected model missed 7 of 375 hold-out anomalies.

## 8. Evidence 6: is the lead bigger than noise?

Paired bootstrap on the hold-out, resampling whole days (300 resamples; rows within a day are correlated). Difference in PR-AUC, selected model minus the other:

| Comparison | Difference | 95% interval | Conclusion |
|---|---|---|---|
| hgb_unweighted - rule | +0.474 | [+0.417, +0.545] | Clearly better |
| hgb_unweighted - logreg | +0.387 | [+0.315, +0.446] | Clearly better |
| hgb_unweighted - hgb | +0.000 | [-0.0003, +0.001] | No difference |

## 9. Selection rule and why this model

The rule is in code (`select_model`) and tested:

1. **Gates:** overfit gap above 0.05, or worst-building recall below 0.80, makes a model ineligible. All four models passed, so the gates did not separate them here; they exist to stop a future model that overfits or fails a building.
2. **Rank** eligible models by mean CV PR-AUC. Best: `hgb` (0.9909).
3. **Tie rule:** any eligible model within one CV-fold standard deviation (0.0047) of the best is treated as tied, and among tied models the one with the fewest modelling assumptions wins. `hgb_unweighted` (0.9898) is inside that margin, so it ties with `hgb`.
4. Models are never ranked by hold-out score.

Why `hgb_unweighted`: boosting beats the rule and logistic regression decisively on every measure (CV, stability, false alarms, bootstrap). Between the two boosting variants the evidence says there is no difference in quality, so the simpler one is chosen: no class weights to justify, and a threshold (0.30) that is on a probability-like scale. As a confirmation only, its hold-out result (precision 0.979, recall 0.981) is at least as good as the weighted variant's (0.978, 0.965).

## 10. Known limitations

- **Simulated data.** Anomalies are clean multipliers with 5% noise, so a tree separates them almost perfectly. Real meter data will score far lower, and the slice and error findings (work hours, echo false alarms) must be re-measured on it. The method carries over, the numbers do not.
- **The gates did not discriminate** between these four models. Their thresholds (0.05 gap, 0.80 recall) are judgement calls, not derived values.
- **The tie margin** (one fold standard deviation, from 4 folds) is a rough noise estimate.
- **Hypotheses are untested.** The work-hours explanation in section 6 is a reasonable guess, not a measured cause.
- **The 95% recall target and the 0.5 default** are assumptions about cost. A different cost ratio gives a different threshold.
- **Known buildings only.** Performance on a new building is still not measured; that needs a group-aware hold-out.
- **OOF covers the later 80% of the training period**, and slices are measured there, not on the earliest data.
- **Bootstrap resamples days of one hold-out period**, so it captures day-to-day noise but not changes between seasons.
