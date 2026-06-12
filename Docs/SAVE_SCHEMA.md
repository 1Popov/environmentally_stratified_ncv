# SAVE_SCHEMA.md — `.mat` artifact contract for the 3-stage pipeline

This document specifies the standardized variables saved by each pipeline stage
(`Discovery.m` → `Tuning.m` → `Production.m`) into the run's `.mat` files under
`Results/`. The contract is enforced by the `save(...)` list in each driver
and by the `buildMetadata.m` helper that produces the `meta` struct shared by
every stage.

The goal is **full provenance + zero ad-hoc lookups**: a downstream analyst
opening a single Production `.mat` should be able to recover the entire run
context (strategy, smoke-vs-full, hyperparams, host, code git SHA, upstream
stage outputs) without consulting external state.

---

## 1. Common provenance — `meta` struct

Built by `buildMetadata(strategy, stage, TestMode, upstreamFile, excludeCoords)` and
present in **every** `.mat`. All fields except `upstreamMeta` are written by
`buildMetadata.m`; `upstreamMeta` is attached afterward by `Tuning.m` /
`Production.m` (Discovery's `meta` has none).

| Field                  | Type    | Source                                | Meaning |
|------------------------|---------|---------------------------------------|---------|
| `timestamp`            | char    | `datetime('now')` at save time        | ISO 8601 local time of this `save()` call |
| `strategy`             | string  | hard-coded in Discovery driver, chained in Tuning/Production | One of `RandomCV`, `RandomCV_Weighted`, `GeoClusterCV`, `OFCV`, `EOFCV`, `HybridCV` |
| `stage`                | string  | hard-coded per driver                 | `Discovery`, `Tuning`, or `Production` |
| `TestMode`             | logical | from driver input arg                 | `true` = smoke-test budget; `false` = full-budget |
| `excludeCoords`        | logical | from driver input arg                 | `true` if the run used the 33-feature coord-removal variant (LAT/LON excluded); `false` for the canonical 35-feature set |
| `upstreamFile`         | string  | path arg to Tuning/Production         | Empty for Discovery; Discovery `.mat` path for Tuning; Tuning `.mat` path for Production |
| `hostName`             | string  | `getenv('COMPUTERNAME')`              | Machine that ran it |
| `userName`             | string  | `getenv('USERNAME')`                  | OS user |
| `matlab_version`       | string  | `version()`                           | MATLAB release |
| `python_version`       | string  | `pyenv().Version`                     | Python interpreter version |
| `python_executable`    | string  | `pyenv().Executable`                  | Full path to `python.exe` used |
| `lightgbm_version`     | string  | `pyrun("import lightgbm; ...")`       | LightGBM version (best effort) |
| `optuna_version`       | string  | `pyrun("import optuna; ...")`         | Optuna version (best effort) |
| `code_git_commit`      | string  | `git rev-parse HEAD` at save time     | Repo HEAD SHA (best effort) |
| `code_git_dirty`       | logical | `git status --porcelain` non-empty    | `true` if working tree had uncommitted changes |
| `upstreamMeta`         | struct  | Tuning/Production only                | The previous stage's `meta` verbatim — gives the full lineage chain |

Best-effort fields fall back to `"(unknown)"` strings if the underlying query
fails (e.g. `pyenv` inactive, not a git repo, `system()` blocked).

### Lineage chain example

```matlab
load('Results/Results_<stamp>_Production.mat');

fprintf('Strategy:       %s\n', meta.strategy);
fprintf('Test mode:      %d\n', meta.TestMode);
fprintf('Code SHA:       %s (dirty=%d)\n', meta.code_git_commit, meta.code_git_dirty);
fprintf('Tuning ran at:  %s\n', meta.upstreamMeta.timestamp);
fprintf('Discovery ran:  %s\n', meta.upstreamMeta.upstreamMeta.timestamp);
fprintf('Python:         %s\n', meta.python_version);
fprintf('LightGBM:       %s\n', meta.lightgbm_version);
```

---

## 2. Stage-specific top-level fields

### 2.1 Discovery_*.mat — `Results_<stamp>_Discovery.mat`

Produced by `Discovery.m` (single driver, strategy parameterized via `meta.strategy`). Save list is identical for every strategy.

| Field                       | Type           | Meaning |
|-----------------------------|----------------|---------|
| `allFoldResults`            | 1×9 cell       | Per outer-fold record: `.foldIndex`, `.metrics`, `.bestHyperparams`, `.bayesoptResults.TrialsHistory`, `.featureImportance`, `.modelString`, `.learningCurves` (which holds `bestT`, `train_rmse_history`, `valid_rmse_history`) |
| `devSet`                    | table          | Development set (~239k rows, 1458 profiles) after preprocessing |
| `holdoutSet`                | table          | Globally consistent hold-out (~28k rows, 169 profiles) |
| `cyclicNames`               | string array   | Cyclic predictor names. Already cos/sin-encoded upstream in featureTable; `rangeNormalizePair` uses the list to exclude them from min-max scaling. |
| `selectedPredictorNames`    | string array   | 35-predictor working set (paper Table 1) |
| `targetName`                | string         | `"OXYGEN"` |
| `totalTime`                 | double         | Wall-clock seconds for the Discovery NCV loop |
| `nFolds`               | int            | `=9` (`K_out`) |
| `MaxObjectiveEvaluations`   | int            | Optuna trial budget per fold |
| `logContent`                | char array     | Full stdout of the run (captured via `diary`) |
| `meta`                      | struct         | See §1 |
| `config`                    | struct         | Full output of `getExperimentConfig(strategy)` (TestMode read internally via `isTestMode()`) |
| `optVars`                   | optimizableVariable array | Hyperparameter search bounds (paper Table 2) |
| `testMode`              | logical        | Mirror of `meta.TestMode` for top-level access (saved lowercase by Discovery.m save list) |

### 2.2 Tuning.mat — `Results_<stamp>_Tuning.mat`

Produced by single `Tuning.m`, consumed by `Production.m`.

| Field                       | Type           | Meaning |
|-----------------------------|----------------|---------|
| `discoveryRunFolder`        | char           | Folder of the upstream Discovery `.mat` |
| `devSet`, `holdoutSet`, `cyclicNames`, `targetName` | (forwarded from Discovery) | — |
| `recommendedPredictorNames` | string array   | Top-N features selected by aggregate importance (currently N = NumberFeatures = all 35) |
| `finalHyperparams`          | table          | Best per-fold hyperparam table picked for Production |
| `results`                   | struct         | `.PerformanceSummary_NCV.<metric>.{mean,std,values}` |
| `allFoldResults`            | 1×9 cell       | Forwarded from Discovery (Production reads `.learningCurves.bestT` etc.) |
| `nFolds`               | int            | `=9` |
| `featureImportanceTable`    | table          | Raw per-feature mean+std importance across folds |
| `sortedTable`               | table          | `featureImportanceTable` sorted descending by mean importance |
| `dataFilePath`              | char           | Path to the upstream Discovery `.mat` (alias for `meta.upstreamFile`) |
| `meta`                      | struct         | See §1; `meta.upstreamMeta` chains to Discovery's `meta` |
| `bestFoldIndex`             | int            | The fold whose hyperparams Production should use |
| `metricNames`               | cell array     | Field names of the per-fold metrics struct (RMSE_raw, MAE_raw, …) |
| `allBestHyperparams`        | 1×9 cell       | Per-fold best hyperparam tables (Production uses `bestFoldIndex` to pick; downstream ensemble analysis can use all 9) |
| `CP_Quantile`               | struct         | LACP conformal calibration carried to Production. `.Available` (logical); when `true` also `.Alpha`, `.NormalizedScore`, `.SigmaDepthGrid`, `.SigmaValues`, `.WindowSize`, `.nCal`. Calibrated in Tuning §4.5 from aggregated out-of-fold predictions via `conformalCalibrate.m` (shared with `conformalCoverage.m` so the live and reproduction paths cannot diverge); falls back to `struct('Available', false)` when fewer than 100 valid OOF rows are available. |

### 2.3 Production_*.mat — `Results_<stamp>_Production.mat`

Produced by `Production.m` (single driver, strategy parameterized via `meta.strategy`). Save list is identical for every strategy.

| Field                       | Type           | Meaning |
|-----------------------------|----------------|---------|
| `results`                   | struct         | `.FinalHyperparameters`, `.FinalModelString`, `.FinalFeatureImportance`, `.HoldoutPerformance` and `.TrainingPerformance` (each: RMSE_raw/MAE_raw/R2/Bias/RMSE_scaled/MAE_scaled/Huber_scaled **+ RMSE_raw_unweighted + RMSE_raw_depthweighted** — the dual-weighting hold-out/training RMSE, both computed via `calculateRegressionMetrics`), `.IndexAgreement.{Training,Holdout,Labels}`, `.ReliabilityDiagnostics.{GeneralizationGap,InflationFactor,RelativeGap}` (gap pairs NCV with the hold-out at the SAME per-strategy training weight), `.ProductionLogContent`, `.PerformanceSummary_NCV` (forwarded from Tuning) |
| `devSet`, `holdoutSet`      | tables         | Forwarded from Tuning (full row state at training time) |
| `recommendedPredictorNames` | string array   | Forwarded from Tuning |
| `targetName`                | string         | `"OXYGEN"` |
| `C_final`, `S_final`        | structs        | Per-column center + scale used by `rangeNormalizePair` to map predictions back to raw units. Reproducing predictions on new data requires these |
| `finalHyperparams`          | table          | Top-level mirror of `results.FinalHyperparameters` (avoids one struct-field hop) |
| `bestT_final`               | int            | `num_boost_round` used to train the final model (after `best_iteration` backstop) |
| `bestFoldIndex`             | int            | Which outer fold's hyperparams + bestT were used |
| `Yholdout_hat_raw`          | double vector  | Model predictions on hold-out, raw units (µmol kg⁻¹) |
| `Yholdout_true_raw`         | double vector  | Hold-out truth values, raw units |
| `Ydev_hat_raw`              | double vector  | Model predictions on training set (for resubstitution diagnostics) |
| `Ydev_true_raw`             | double vector  | Training truth values |
| `tuningFilePath`            | char           | Path to the upstream Tuning `.mat` (alias for `meta.upstreamFile`) |
| `meta`                      | struct         | See §1; chained via `meta.upstreamMeta.upstreamMeta` back to Discovery |

### 2.4 Models artefacts — `Models/Model_<Strategy>.mat`

Separate from the active-pipeline schema above. These are the frozen
trained-model files shipped with the repo. They make `reproduce_models.m`
the single ground-truth source for every documented paper metric (the
spatial-leakage / adversarial-validation diagnostics are computed separately
and are not part of this release) without re-training from scratch (see Path A
in `README.md`): the
hold-out metrics are recomputed live from `FinalModelString`, while the
nested-CV quantities — per-fold metrics (`PerformanceSummary_NCV`), per-fold
hyperparameters (`NCV_FoldHyperparams`) and per-fold feature importance
(`NCV_FoldFeatureImportance`) — are carried as stored original-run ground truth
because a final model cannot regenerate its own outer folds. The stored
arrays are byte-identical to the locked Production+Tuning runs they were
extracted from; `reproduce_models.m` computes all means, SDs, CV%,
confidence intervals and normalisations live from them (no stored summary
statistic is printed without live recomputation). The frozen `Models/` are
reconstructed from the original run artefacts by `extract_models_from_results.m`
in one pass (model string, normalization, hold-out, NCV, hyperparameters, feature
importance, OOF), gated on a byte-exact per-fold RMSE check.

| Field                       | Type           | Meaning |
|-----------------------------|----------------|---------|
| `strategy`                  | string         | Pipeline/paper order: `RandomCV`, `RandomCV_Weighted`, `GeoClusterCV`, `OFCV`, `EOFCV`, `HybridCV`, then the two coordinate-removal controls `RandomCV_NoLatLon`, `RandomCV_Weighted_NoLatLon`. |
| `FinalModelString`          | char array     | The trained LightGBM booster serialized via `model.model_to_string()`. Round-trips through `py.lgbm_wrapper.predict_from_model`. |
| `recommendedPredictorNames` | string array   | The 35-feature (or 33-feature, for the no-LatLon variants) working set used at training time. |
| `cyclicNames`               | string array   | Cyclic predictor names. Already cos/sin-encoded upstream in featureTable; `rangeNormalizePair` uses the list to exclude them from min-max scaling. |
| `targetName`                | string         | `"OXYGEN"`. |
| `C_final`, `S_final`        | structs        | Per-column centre + scale applied to the hold-out before prediction. |
| `weightFn`                  | string         | Training weight function recorded for provenance: `"DepthWeightsUniform"` for the RandomCV-family (`RandomCV`, `RandomCV_NoLatLon`), `"DepthWeights"` for `RandomCV_Weighted`, the two `*_Weighted_NoLatLon`/stratified variants, and the four stratified strategies. This is the weight used at TRAINING time and for the NCV metric; it does NOT determine which hold-out RMSE to read (both are stored — see below). |
| `HoldoutPerformance`        | struct         | Frozen hold-out metrics. Hold-out RMSE is reported under BOTH weightings: `RMSE_raw_unweighted` (every sample equal) and `RMSE_raw_depthweighted` (oxycline-emphasis; always ≥ the unweighted value). `reproduce_models.m` re-scores `FinalModelString` on the hold-out via `calculateRegressionMetrics.m` under each weighting and verifies BOTH against the stored fields at 1e-3 µmol kg⁻¹. `RMSE_raw` is retained as the training-weight value for back-compat (equals `RMSE_raw_unweighted` for the RandomCV-family, `RMSE_raw_depthweighted` for the depth-weighted strategies). Written by `Production.m` in the live pipeline and carried on the frozen `Models/` artefacts. Note: frozen Models may also carry a vestigial `Huber_raw` field the active pipeline no longer writes — ignore it. |
| `PerformanceSummary_NCV`    | struct         | Stored nested-CV summary: one sub-struct per metric (`RMSE_raw`, `MAE_raw`, `Bias`, `R2`, …), each with `.mean`, `.std`, and `.values` (the K=9 per-fold values). This is the canonical NCV ground truth carried so `reproduce_models.m` can report NCV alongside the recomputed hold-out metrics. The 95% confidence interval is **not** stored — `reproduce_models.m` derives it live from `RMSE_raw.values` via the Nadeau-Bengio variance correction (`computeNadeauBengioCI.m`); the simple t/√N interval used in early logs is not used. Extracted from the locked original Production artefacts (per-fold values are byte-identical to those runs). |
| `FinalHyperparameters`      | table          | Top-level mirror of `results.FinalHyperparameters` so consumers can read the tuned hyperparameter row without a deep struct walk. The final-model hyperparameter row printed in `reproduce_models.m` §11. |
| `NCV_FoldHyperparams`       | K×P double     | Per-fold tuned hyperparameters across the K=9 outer folds (P parameters); column order in `NCV_HyperparamNames`. `reproduce_models.m` §10 computes the Mean and CV% across folds live from this. Extracted from `allFoldResults{k}.bestHyperparams`. |
| `NCV_HyperparamNames`       | 1×P string     | Column order for `NCV_FoldHyperparams` (e.g. `max_depth`, `num_leaves`, `learning_rate`, …). |
| `NCV_FoldFeatureImportance` | K×F double     | Per-fold raw LightGBM **gain** importance across the K=9 folds (F features), in the **original predictor order** (`FeatureNames`). `reproduce_models.m` §12 normalises each fold to % of that fold's total, then takes the cross-fold mean (NCV %) and CV% — fold-scale-invariant, matching the paper's per-fold importance-stability convention. Extracted from `allFoldResults{k}.featureImportance`; the per-fold mean equals the producer's `featureImportanceTable.MeanImportance` (verified, max\|Δ\|=0). |
| `FinalFeatureImportance`    | 1×F double     | Final-model raw gain importance, **re-ordered onto `FeatureNames`** at extraction time (the source `results.FinalFeatureImportance` is in `recommendedPredictorNames` / importance-sorted order — a different axis from the per-fold array; both are stored on one shared axis so a feature row cannot be mislabelled). `reproduce_models.m` §12 prints it normalised to %. |
| `FeatureNames`              | 1×F string     | Shared feature-name axis for both `NCV_FoldFeatureImportance` and `FinalFeatureImportance` (original predictor order, from `featureImportanceTable.FeatureName`). |
| `IndexAgreement`            | struct         | `.Training`, `.Holdout`, `.Labels` — Willmott Index of Agreement per depth stratum, cached at training time for downstream plotting. `reproduce_models.m` does NOT read this cached field; it recomputes IoA live on the hold-out (same bins, range-labelled) so the reported values cannot drift from the shipped model. Note: the frozen `Models/Model_*.mat` artefacts carry the older `'Abyssal (151-200m)'` label for the deepest stratum; current `Production.m` writes `'Deeper (151-200m)'` (more oceanographically accurate). The depth-range bin definitions are byte-identical — only the cached display label differs. |
| `sourceFile`                | string         | Path to the original Production `.mat` the model was built from; pure provenance. |
| `OOF_Yhat`, `OOF_Ytrue`, `OOF_Depth` | N_dev×1 double | Out-of-fold prediction / truth / depth over the development set — every dev row predicted by the outer fold that held it out. **Six main strategies only** (the two coordinate-removal controls carry no OOF). `conformalCoverage.m` calibrates the locally-adaptive 95% interval on these and `reproduce_models.m` §9b reports its empirical hold-out coverage. |
| `OOF_FoldID`                | N_dev×1 double | Outer-fold id (1..K) of every dev row, recorded during OOF reconstruction so per-fold diagnostics can group OOF residuals by fold without re-deriving the partition. **Six main strategies only.** Not read by `reproduce_models.m`; reserved for a future per-fold spatial-leakage diagnostic. |

The Models/ directory ships separately from the active `Results/`
runtime output and is not regenerated by the pipeline; it is a frozen
artefact set that grows only on a re-run.

---

## 3. Backwards compatibility

Older `.mat` files written before this schema do not have `meta`, `config`,
`optVars`, `bestT_final`, `Yholdout_hat_raw`, etc. The pipeline gracefully
handles those:

- `Tuning.m` wraps its meta-chaining with
  `if exist('meta', 'var') && isstruct(meta)` and falls back to a sentinel
  struct so it does not crash on a pre-schema Discovery `.mat`.
- `Production.m` instead hard-fails on a missing or invalid `meta`
  (`Production:MissingMeta`) — by the time a run reaches Production the
  strategy identity must be recoverable, so it refuses to guess.
- Production's `best_iteration` backstop recomputes `bestT_final` as
  `argmin(valid_rmse_history)` whenever the curve is available, so the
  final-model boost-round count is always the true optimum regardless of
  which iteration convention the upstream Discovery `.mat` stored.
- Downstream analysis code should similarly defensively-check field
  existence: `if isfield(loadedData, 'meta') ...`.

---

## 4. Quick reference for downstream consumers

### Re-running paper Hold-out metrics from a Production `.mat`

```matlab
load('Results/Results_<stamp>_Production.mat');
% Yholdout_hat_raw is saved UNCLAMPED for downstream analysis;
% results.HoldoutPerformance.RMSE_raw is a sample-weighted RMSE on
% the clamped predictions (max(Yhat, 0)). To reproduce the stored
% metric exactly, derive the per-row weights from the strategy used:
config = getExperimentConfig(meta.strategy);
W = config.WeightFcn(holdoutSet.DEPTH);
W = max(W, eps);
Yhat_clamped = max(Yholdout_hat_raw, 0);
RMSE_holdout = sqrt(sum(W .* (Yhat_clamped - Yholdout_true_raw).^2) / sum(W));
% Equals results.HoldoutPerformance.RMSE_raw within float rounding.
% Hold-out RMSE is reported under BOTH weightings:
%   results.HoldoutPerformance.RMSE_raw_unweighted    (DepthWeightsUniform)
%   results.HoldoutPerformance.RMSE_raw_depthweighted (DepthWeights)
% Substitute the chosen weight function above to reproduce either one.
```

### Reproducing the trained model on new data

```matlab
load('Results/Results_<stamp>_Production.mat');
% Apply training-time normalisation via C_final, S_final (NOT learning mode):
[Xnew_norm, ~, ~, ~] = rangeNormalizePair(newTable, [], cyclicNames, C_final, S_final);
yhat_norm = py.lgbm_wrapper.predict_from_model(results.FinalModelString, ...
                                              Xnew_norm{:, recommendedPredictorNames});
% Production clamps predictions to non-negative oxygen; mirror that here:
yhat_raw  = max(double(yhat_norm) .* S_final.(targetName) + C_final.(targetName), 0);
```

### Plotting predicted vs observed (hold-out)

```matlab
load('Results/Results_<stamp>_Production.mat');
scatter(Yholdout_true_raw, Yholdout_hat_raw, '.');
xlabel('Observed DO (µmol kg^{-1})');
ylabel('Predicted DO (µmol kg^{-1})');
refline(1, 0);
title(sprintf('%s — Hold-out RMSE = %.2f', meta.strategy, ...
              results.HoldoutPerformance.RMSE_raw));
```

### Inspecting the per-trial hyperparameter search for a Discovery fold

```matlab
load('Results/Results_<stamp>_Discovery.mat');
fold = 5;
% Full per-trial history (every Optuna trial, not just the winning row):
trials = allFoldResults{fold}.bayesoptResults.TrialsHistory;
disp(trials);
% optVars carries the search bounds used to generate these trials:
disp(optVars);
% The single best-trial hyperparameter row (what Tuning forwards):
disp(allFoldResults{fold}.bestHyperparams);
```
