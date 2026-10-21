# Day 2: leakage-safe pipelines and bounded hyperparameter search

Code: `evalproto/search.py`. Tests: `tests/test_search.py` (17). Output: `results/search.json`.

## 1. What problem this solves

Tuning hyperparameters means trying many configurations and keeping the best. Two things can go wrong:

1. **Leakage into the search.** If the search sees the hold-out, or fits preprocessing on data it later validates on, the "best" score is optimistic.
2. **Unbounded or irreproducible search.** A search that runs forever, or gives a different answer each run, cannot be reviewed.

## 2. Design

| Rule | How it is enforced |
|---|---|
| Search evaluates only training folds | `run_search` receives only `X_train, y_train, time_train`. The hold-out is never an argument, so it cannot be scored. Test: a spy on `Pipeline.fit` confirms no hold-out row is ever fitted on |
| Preprocessing cannot leak | Scaling is a step inside a `Pipeline` (`scale -> clf`), so it is fit on each fold's training rows only. Test: a spy scaler records the sizes it is fit on |
| Folds respect time | Same expanding-window folds and 24 h gap as Day 1 (`splits.cv_splits`). Test: every validation fold starts more than 24 h after its training rows |
| Reproducible | Candidates come from a seeded `ParameterSampler`; models take the same seed; `n_jobs=1`. Tests: same seed gives identical trials, different seed gives different configs. Two full runs of `python -m evalproto.search` produce identical JSON apart from wall-clock time |
| Bounded: trials | Hard cap of 50 candidates per call. Above it raises `SearchBudgetError`; exactly 50 is accepted (boundary test) |
| Bounded: time | Wall-clock budget per model family (default 240 s). When exceeded the search stops cleanly and returns the trials finished so far, flagged `stopped_early`. If no trial finished, `best_params` is `None` |
| Bounded: data | Optional `max_rows` caps rows used per fit (keeps the most recent rows) |
| Model choice never uses the hold-out | The winner is picked by mean CV PR-AUC. The hold-out is scored once per family, after the final refit, for reporting only |

## 3. Search spaces

| Family | Parameters (distribution) |
|---|---|
| Logistic regression | `C` log-uniform 1e-3..1e2; class-balanced |
| Histogram gradient boosting | learning rate log-uniform 0.02..0.3; max depth 2..6; max leaf nodes 8..39; min samples per leaf 20..199; L2 log-uniform 1e-3..10; 150 iterations; class-balanced |

12 random candidates per family, 4 expanding-window folds each, scored by PR-AUC (the Day 1 primary metric).

## 4. Results (seed 42)

| Family | Best CV PR-AUC | Trial range (min / median / max) | Hold-out PR-AUC | Hold-out recall | Hold-out precision |
|---|---|---|---|---|---|
| Logistic regression | 0.586 | 0.578 / 0.585 / 0.586 | 0.609 | 0.979 | 0.545 |
| Gradient boosting | **0.991** | 0.979 / 0.989 / 0.991 | **0.997** | 0.989 | 0.976 |

Selected by CV score: gradient boosting. Search took about 3 s (logistic regression) and 60 s (boosting) on 2 CPU cores.

Compared with Day 1: logistic regression is unchanged (0.590 before, 0.586 now; the small difference is 4 folds instead of 5). The new best CV PR-AUC to beat on later days is **0.991**.

## 5. How to read these results

- **The big gain came from the model family, not from tuning.** Within a family the best and worst candidates differ by under 0.015 PR-AUC. Tuning is cheap here, but choosing a model that can threshold on the deviation features is what mattered.
- **It is not leakage.** CV (0.991) and hold-out (0.997) agree, the hold-out was never in the search, features are causal (Day 1 test), and a negative control passes: with shuffled training labels the same search scores near the base rate instead of 0.99 (`test_shuffled_labels_collapse_to_chance`).
- **It is also a property of the simulation.** Anomalies are injected as a clean 1.5x-2.5x or 0.2x-0.5x multiplier with 5% noise, which a tree can separate almost perfectly. Real meter data will not be this clean, so expect real scores to be much lower.

## 6. Known limitations

- **Simulated data**: the absolute scores (especially 0.99) do not transfer. The process does.
- **Random search over 12 candidates** is small on purpose (bounded). The trial range suggests more candidates would add little here, but that is not proven for other data.
- **No nested CV.** The best CV score is slightly optimistic because it is also the selection criterion. The untouched hold-out is the unbiased check; it agrees closely.
- **Time budget is checked between candidates**, not inside one, so a single slow candidate can overrun the budget by up to its own run time.
- **`max_rows` keeps the most recent rows**, which suits time data but is not a random subsample.
- **Single process** (`n_jobs=1`) for reproducibility, so it does not use all cores.
- Threshold is still the default 0.5; threshold tuning is not part of this search.
