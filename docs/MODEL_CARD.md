# Model card: building energy-meter anomaly detector

Package: `models/final/` (built by `python -m evalproto.final evaluate` or `build`). Evaluation record: `results/final_evaluation.json`, run `run-002-6d28e1` in `results/experiments.jsonl`. Comparison report: [FINAL_REPORT.md](FINAL_REPORT.md).

## 1. Model details

| Item | Value |
|---|---|
| Task | Flag building-hours whose consumption is an anomaly (a 1 to 4 hour spike of 1.5x to 2.5x or drop to 0.2x to 0.5x of the expected level) |
| Model | `StandardScaler` + `HistGradientBoostingClassifier`, no class weighting (`hgb_unweighted`), hyper-parameters from the Day 2 search (learning rate 0.235, max depth 5, 25 leaves, min 123 samples per leaf, L2 0.0014) |
| Output | Score between 0 and 1 per building-hour, and an alert when the score is at least **0.303** |
| Threshold rule | Highest threshold that catches at least 95% of anomalies on out-of-fold training scores (fixed on Day 3, not tuned on the hold-out) |
| Inputs | Raw hourly readings: building, hour index, temperature, building size, consumption. Features are built inside the package (24 h and 168 h log ratios, hour of day, weekend, temperature, size, log consumption) |
| Trained on | 54,440 building-hours, hours 168 to 2,889 of a simulated 20-building, 150-day data set. Not retrained on the hold-out |
| Version | config `6d28e1a05f466d11`, code commit `6db466c8`, Python 3.13.16, scikit-learn 1.9.1 |

## 2. Intended use

- Triage aid: rank or flag building-hours for a person to look at, for buildings that already have at least 168 hours (7 days) of consecutive hourly readings.
- Both over-consumption (spikes) and under-consumption (drops, for example a stopped meter) are covered.

**Out of scope, do not use for:**
- Real meters without re-validation. The model was trained and tested on **simulated** data (see section 8).
- Buildings with less than 168 hours of history (the package returns `insufficient_history`, no score), buildings never seen in training, data with missing hours (rejected), or readings finer than hourly.
- Automatic actions (shutting equipment, billing). The false-alarm and miss rates below are for the simulated data only.

## 3. Evaluation data and method

Hold-out = the last 20% of unique hours (13,720 building-hours, 375 anomalies, 2.7% base rate), separated from training by a 24 hour gap. All choices (model family, hyper-parameters, threshold) were made on training data with time-ordered cross-validation; see [MODEL_SELECTION.md](MODEL_SELECTION.md). The hold-out was formally evaluated once for the final candidate, against gates declared before scoring ([FINAL_REPORT.md](FINAL_REPORT.md) section 4).

## 4. Performance (hold-out, threshold 0.303)

| Metric | Value | Release gate |
|---|---|---|
| PR-AUC (no-skill floor 0.027) | 0.9977 | at least 0.90 |
| Precision | 0.979 | at least 0.90 |
| Recall | 0.981 | at least 0.90 |
| CV PR-AUC minus hold-out PR-AUC | -0.008 | at most 0.02 |
| Worst building recall (buildings with at least 10 anomalies) | 0.867 (building 18, 13 of 15) | at least 0.80 |
| False alarms / misses | 8 / 7 | |
| Alerts | 27.4 per 1,000 building-hours (true rate 27.3) | |

All gates passed. On another machine (Python 3.14, Windows) a re-run gave recall 0.976, precision 0.979, 9 misses instead of 7: expect differences of that size between environments ([REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

**Slices (hold-out recall / precision):** night and evening 0.991 / 0.982; work hours 0.968 / 0.974; weekday 0.980 / 0.980; weekend 0.984 / 0.976. Per building recall ranges from 0.867 to 1.000 for the 18 buildings with enough anomalies; buildings 0 (2 anomalies) and 14 (7) are too small to judge and are not scored. On Day 3 out-of-fold data the weak spots were work hours (0.92), weekends (0.92) and warm hours (0.93); they were not weak on the hold-out, so those differences are small enough to be noise at this sample size.

## 5. Limitations

1. **Simulated data.** Anomalies are clean multipliers on a smooth signal, which a tree separates almost perfectly. Real meters have noisier anomalies, so real scores will be lower. Only the method and the code carry over, not the numbers.
2. **The hold-out is not untouched.** It was also scored for confirmation on Days 2 to 4. No decision used those numbers, and the ledger blocks a second formal evaluation of the same configuration, but the ledger cannot see the earlier looks.
3. **Known buildings only.** No test on an unseen building; a new building needs 168 hours of history and its performance is unmeasured.
4. **One season, one hold-out period.** 150 days; the hold-out is the colder end of a simulated cooling trend, so seasonal behaviour is untested.
5. **Gates and thresholds are judgement.** The gate values (0.90, 0.02, 0.80), the 95% recall target and the cost trade-off behind them are assumptions, not derived from a real cost of a missed anomaly versus a false alarm.
6. **Echo false alarms.** Most remaining false alarms fire 24 or 168 hours after a real event, because the lag features contain that event. Two consecutive anomalies one day or one week apart are the hardest case.
7. **Temperature and building size may not be needed.** Both drift or change without hurting results in the monitoring demo (section 6). Not tested by removing them.
8. **Scores are not probabilities.** The threshold is on the model's score scale; do not read 0.3 as a 30% chance.
9. **Loading a package runs pickle.** Load only packages you built yourself. The package checks its file hash and the scikit-learn version, but the hash is stored next to the file it protects.
10. **Not refit on all data.** The shipped model is the evaluated one (training portion only). Refitting on everything gives a different, unevaluated model.

## 6. Monitoring signals

Computed by `evalproto/monitoring.py` (`python -m evalproto.final monitor FILE.csv`). Reference values (feature bins, score bins, expected alert rate 3.2% of building-hours) are stored in `package.json`.

| Signal | How it is measured | Warn / alert | Needs labels | What to do |
|---|---|---|---|---|
| Input quality | share of rows outside the training range, and share without 168 h history | out-of-range above 1% / 5%; no-history above 30% (warn only) | no | Alert: check units, sensors, new building types. High no-history is normal right after start-up |
| Score drift | PSI of the score against out-of-fold training scores, fixed bins | PSI above 0.10 / 0.25 | no | Output distribution changed: look at inputs, then labels |
| Alert rate | observed alert share divided by expected (ratio; needs at least 20 expected alerts) | ratio above 1.5 or below 0.67 / above 2.5 or below 0.4 | no | Alert storm or a silent model. Compare with the true anomaly rate if known |
| Feature drift | PSI per feature against training data | PSI above 0.10 / 0.25; **capped at warn overall** | no | Leading indicator. Investigate; act when an output-side signal agrees |
| Label performance | recall and precision on confirmed anomalies (at least 20 needed) | below 0.90 either | yes | Alert: review, retrain candidate |

Demo (`python -m evalproto.final sample`, then `monitor`): the hold-out window gives **WARN** (temperature PSI 5.4 because of the seasonal drop; score drift 0.002, alert rate ratio 0.81, recall 0.984, precision 0.984). A copy with temperature +12 and building size x3 gives **ALERT** from input quality (35% out of range). Its recall stayed 0.979 and precision fell to 0.944, so label performance alone would not have caught it: drift signals and label signals catch different failures.

**Retraining triggers (proposed, not automated):** label performance alert for two consecutive windows; score drift alert together with an alert-rate alert; any new building type or meter unit; a new season not covered by training data. Retraining must go through the same protocol: time-aware split, new hold-out, gates, new ledger entry.

## 7. How to use

```bash
python -m evalproto.final build                       # build models/final from training data (the binary is not committed)
python -m evalproto.final predict raw.csv out.csv     # columns: building,hour,temperature,building_size,consumption
python -m evalproto.final monitor raw.csv             # add an `anomaly` column to include label performance
```

```python
from evalproto.inference import load_package
result = load_package("models/final").predict(raw_dataframe)   # building, hour, score, alert, status, warning
```

Bad input raises `InputError` with the reason (missing column, NaN or non-positive values, duplicates, hour gaps).

## 8. Data note

The data set is simulated (seed 42, `evalproto/data.py`) because the build environment had no access to real meter data. Swap `load_dataset` for real data and re-run Days 1 to 5; every threshold and gate in this card must then be re-derived.
