function Production(tuningFilePath)
% Production - Final stage of the NCV pipeline.
%
% Loads a Tuning .mat, trains the production LightGBM model on the
% full development set with the tuning-selected hyperparameters,
% evaluates the trained model on the global hold-out, computes
% reliability diagnostics (generalization gap, inflation factor,
% relative gap) and the depth-stratified Index of Agreement, and
% persists the trained model + metrics into a Production .mat under
% Results/.
%
% Production also verifies the coord-mode contract (meta.excludeCoords
% expects 33 or 35 predictors); a mismatch aborts with
% Production:CoordModeMismatch.
%
% Input:
%   tuningFilePath - Path to the Tuning .mat produced by Tuning.m.
%
% Output:
%   None (void). Persists the trained final model, hold-out metrics, reliability
%   diagnostics, and LACP conformal coverage into a Production .mat under
%   Results/ (see Docs/SAVE_SCHEMA.md for the variable list).

%% 1. Setup
tic;

% Temporary log for this run; flushed into results.ProductionLogContent at save time.
tempLogFile = [tempname, '.log'];
diary(tempLogFile);

fprintf('=== Black Sea DO Production Script ===\n');
fprintf('Loading data from tuning file: %s\n\n', tuningFilePath);

%% 2. Load Tuning Results

try
    load(tuningFilePath, 'devSet', 'holdoutSet', 'cyclicNames', ...
                          'targetName', 'recommendedPredictorNames', 'finalHyperparams', ...
                          'results', 'allFoldResults', 'meta', 'bestFoldIndex', 'CP_Quantile');
    fprintf('Tuning data loaded successfully.\n');
catch ME
    error('Failed to load tuning data file: %s\n%s', tuningFilePath, ME.message);
end

% Fail loud if the Tuning artefact lacks meta (load() silently skips missing
% requested vars, which would otherwise crash later on an opaque field read).
if ~exist('meta', 'var') || ~isstruct(meta)
    error('Production:MissingMeta', ...
          ['Tuning artefact does not carry a meta struct. ' ...
           'The strategy identity cannot be recovered; refusing to ' ...
           'fall back to a default to prevent silent config mismatch. ' ...
           'Source: %s.'], tuningFilePath);
end

% Coord-mode contract: reject the load if the feature count contradicts
% meta.excludeCoords (35 canonical / 33 coord-removal).
if isfield(meta, 'excludeCoords') && meta.excludeCoords
    expectedNumFeatures = 33;
else
    expectedNumFeatures = 35;
end
assert(numel(recommendedPredictorNames) == expectedNumFeatures, ...
    'Production:CoordModeMismatch', ...
    'Feature-count mismatch: meta.excludeCoords expects %d predictors, got %d. ', ...
    expectedNumFeatures, numel(recommendedPredictorNames));

fprintf('Production ready with %d recommended features.\n\n', length(recommendedPredictorNames));

%% 3. Final Model Training and Evaluation
fprintf('\nTraining Final Model with Pre-Validated Hyperparameters and Evaluating on Hold-Out Set...\n');
results.FinalHyperparameters = finalHyperparams;

% --- Identify the Optimal Number of Boosting Rounds ---
% bestFoldIndex is selected and validated by Tuning; use it directly.
bestT_final = allFoldResults{bestFoldIndex}.learningCurves.bestT;

% Best-iteration backstop: recompute bestT_final as the argmin of the
% validation curve when available; no-op if it matches the stored bestT.
if isfield(allFoldResults{bestFoldIndex}.learningCurves, 'valid_rmse_history') && ...
        ~isempty(allFoldResults{bestFoldIndex}.learningCurves.valid_rmse_history)
    [~, correctedBestT] = min(allFoldResults{bestFoldIndex}.learningCurves.valid_rmse_history);
    if correctedBestT ~= bestT_final
        fprintf('   best_iteration backstop: bestT %d -> %d (argmin of valid_rmse_history).\n', ...
                bestT_final, correctedBestT);
        bestT_final = correctedBestT;
    end
end

fprintf('   Using optimal boosting rounds (%d) from discovery fold %d.\n', bestT_final, bestFoldIndex);

% --- Per-strategy RandomSeed + WeightFcn (recovered from meta.strategy) ---
prodConfig = getExperimentConfig(meta.strategy);
randomSeed = prodConfig.RandomSeed;
WeightFcn  = prodConfig.WeightFcn;

% --- Train Final Model (returns model as a string) ---
[TdevN, TholdoutN, C_final, S_final] = rangeNormalizePair(devSet, holdoutSet, cyclicNames);
pyFinalParams = py.dict(table2struct(finalHyperparams));

pyTrainResults = py.lgbm_wrapper.train_final_model(pyFinalParams, ...
    TdevN{:, recommendedPredictorNames}, TdevN{:, targetName}, ...
    WeightFcn(devSet.DEPTH), int32(bestT_final), int32(randomSeed));
trainResults = struct(pyTrainResults);

if ~strcmp(char(trainResults.status), 'ok')
    error('Final Python model training failed: %s', char(trainResults.status));
end
fprintf('   Final model trained successfully in memory.\n');

results.FinalModelString = char(trainResults.model_string);

% --- Evaluate on Hold-Out (Test) Set ---
XholdoutN = TholdoutN{:, recommendedPredictorNames};
YholdoutTrueN = TholdoutN{:, targetName};
Wholdout = WeightFcn(holdoutSet.DEPTH);

% Pass the model string directly for prediction
Yholdout_hat_N_py = py.lgbm_wrapper.predict_from_model(trainResults.model_string, XholdoutN);
Yholdout_hat_N = double(Yholdout_hat_N_py); Yholdout_hat_N = Yholdout_hat_N(:);
% Saved UNCLAMPED for residual analysis; calculateRegressionMetrics clamps
% (max(Yhat,0)) internally before computing metrics.
Yholdout_hat_raw = Yholdout_hat_N .* S_final.(targetName) + C_final.(targetName);
Yholdout_true_raw = holdoutSet{:, targetName};
holdoutMetrics = calculateRegressionMetrics(Yholdout_true_raw, Yholdout_hat_raw, YholdoutTrueN, Yholdout_hat_N, Wholdout, finalHyperparams.huber_delta);

% --- Dual-weighting hold-out RMSE ---
% Hold-out RMSE is sample-weighted; report under both weightings (RMSE_raw
% above is the training-weight value).
hoMetrics_unweighted    = calculateRegressionMetrics(Yholdout_true_raw, Yholdout_hat_raw, YholdoutTrueN, Yholdout_hat_N, DepthWeightsUniform(holdoutSet.DEPTH), finalHyperparams.huber_delta);
hoMetrics_depthweighted = calculateRegressionMetrics(Yholdout_true_raw, Yholdout_hat_raw, YholdoutTrueN, Yholdout_hat_N, DepthWeights(holdoutSet.DEPTH),        finalHyperparams.huber_delta);
holdoutMetrics.RMSE_raw_unweighted    = hoMetrics_unweighted.RMSE_raw;
holdoutMetrics.RMSE_raw_depthweighted = hoMetrics_depthweighted.RMSE_raw;
results.HoldoutPerformance = holdoutMetrics;

% --- Evaluate on Development (Training) Set for Resubstitution Error ---
XdevN = TdevN{:, recommendedPredictorNames};
YdevTrueN = TdevN{:, targetName};
Wdev = WeightFcn(devSet.DEPTH);

% Pass the model string directly for prediction
Ydev_hat_N_py = py.lgbm_wrapper.predict_from_model(trainResults.model_string, XdevN);
Ydev_hat_N = double(Ydev_hat_N_py); Ydev_hat_N = Ydev_hat_N(:);
Ydev_hat_raw = Ydev_hat_N .* S_final.(targetName) + C_final.(targetName);
Ydev_true_raw = devSet{:, targetName};
devMetrics = calculateRegressionMetrics(Ydev_true_raw, Ydev_hat_raw, YdevTrueN, Ydev_hat_N, Wdev, finalHyperparams.huber_delta);

% Dual-weighting training RMSE (same pattern as the hold-out block above).
devMetrics_unweighted    = calculateRegressionMetrics(Ydev_true_raw, Ydev_hat_raw, YdevTrueN, Ydev_hat_N, DepthWeightsUniform(devSet.DEPTH), finalHyperparams.huber_delta);
devMetrics_depthweighted = calculateRegressionMetrics(Ydev_true_raw, Ydev_hat_raw, YdevTrueN, Ydev_hat_N, DepthWeights(devSet.DEPTH),        finalHyperparams.huber_delta);
devMetrics.RMSE_raw_unweighted    = devMetrics_unweighted.RMSE_raw;
devMetrics.RMSE_raw_depthweighted = devMetrics_depthweighted.RMSE_raw;
results.TrainingPerformance = devMetrics;

% --- Display Final Comparison Table ---
fprintf('\n\n=== FINAL MODEL PERFORMANCE SUMMARY ===\n');
fprintf('+---------------------+----------------+----------------+\n');
fprintf('| Metric              |  Training Set  |   Hold-Out Set |\n');
fprintf('+---------------------+----------------+----------------+\n');
fprintf('| --- Scaled Units ---|                |                |\n');
fprintf('| RMSE (scaled)       | %14.4f | %14.4f |\n', devMetrics.RMSE_scaled, holdoutMetrics.RMSE_scaled);
fprintf('| MAE (scaled)        | %14.4f | %14.4f |\n', devMetrics.MAE_scaled, holdoutMetrics.MAE_scaled);
fprintf('| Huber Loss (scaled) | %14.4f | %14.4f |\n', devMetrics.Huber_scaled, holdoutMetrics.Huber_scaled);
fprintf('+---------------------+----------------+----------------+\n');
fprintf('| --- Raw Units ---   |                |                |\n');
fprintf('| RMSE (unweighted)   | %14.4f | %14.4f |\n', devMetrics.RMSE_raw_unweighted, holdoutMetrics.RMSE_raw_unweighted);
fprintf('| RMSE (depth-wt)     | %14.4f | %14.4f |\n', devMetrics.RMSE_raw_depthweighted, holdoutMetrics.RMSE_raw_depthweighted);
fprintf('| MAE (raw)           | %14.4f | %14.4f |\n', devMetrics.MAE_raw, holdoutMetrics.MAE_raw);
fprintf('+---------------------+----------------+----------------+\n');
fprintf('| --- Other ---       |                |                |\n');
fprintf('| R²                  | %14.4f | %14.4f |\n', devMetrics.R2, holdoutMetrics.R2);
fprintf('| Bias                | %+14.4f | %+14.4f |\n', devMetrics.Bias, holdoutMetrics.Bias);
fprintf('+---------------------+----------------+----------------+\n\n');

%% 4. Calculate and Display Index of Agreement by Depth
% Willmott's Index of Agreement (d), computed per depth stratum.

% --- Define Depth Strata (canonical source: getDepthStrata.m) ---
depthStrata = getDepthStrata();
numStrata = size(depthStrata, 1);
d_train = zeros(numStrata, 1);
d_holdout = zeros(numStrata, 1);

for i = 1:numStrata
    lowerBound = depthStrata{i, 2}(1);
    upperBound = depthStrata{i, 2}(2);
    train_mask   = (devSet.DEPTH     >= lowerBound & devSet.DEPTH     <= upperBound);
    holdout_mask = (holdoutSet.DEPTH >= lowerBound & holdoutSet.DEPTH <= upperBound);
    d_train(i)   = calculateWillmottIndex(Ydev_hat_raw(train_mask),    Ydev_true_raw(train_mask));
    d_holdout(i) = calculateWillmottIndex(Yholdout_hat_raw(holdout_mask), Yholdout_true_raw(holdout_mask));
end

% Round to 4dp for table display; helper returns full precision.
results.IndexAgreement.Training = round(d_train, 4);
results.IndexAgreement.Holdout  = round(d_holdout, 4);
results.IndexAgreement.Labels   = depthStrata(:, 1);

% --- Display Depth-Stratified Index of Agreement Table ---
fprintf('=== INDEX OF AGREEMENT (d) BY DEPTH RANGE ===\n');
fprintf('+-----------------------+----------------+----------------+\n');
fprintf('| Depth Range           |  Training Set  |   Hold-Out Set |\n');
fprintf('+-----------------------+----------------+----------------+\n');
for i = 1:numStrata
    fprintf('| %-21s | %14.4f | %14.4f |\n', depthStrata{i, 1}, d_train(i), d_holdout(i));
end
fprintf('+-----------------------+----------------+----------------+\n\n');

%% 5. Reliability diagnostics
% NCV and hold-out are compared at the SAME per-strategy training weight
% (depth-weighted for the stratified strategies, unweighted for RandomCV).
% Uses Tuning's NaN-safe NCV mean (omitnan).
ncv_rmse = results.PerformanceSummary_NCV.RMSE_raw.mean;
ho_rmse  = results.HoldoutPerformance.RMSE_raw;
gen_gap  = ho_rmse - ncv_rmse;
infl_fac = ho_rmse / ncv_rmse;
rel_gap  = gen_gap / ncv_rmse * 100;
fprintf('Reliability diagnostics (NCV vs hold-out at the per-strategy training weight):\n');
fprintf('  NCV mean RMSE   = %.2f µmol kg⁻¹\n', ncv_rmse);
fprintf('  Hold-out RMSE   = %.2f µmol kg⁻¹\n', ho_rmse);
fprintf('  Generalization gap = %+.2f µmol kg⁻¹\n', gen_gap);
fprintf('  Inflation factor   = %.2fx\n', infl_fac);
fprintf('  Relative gap       = %+.1f%%\n\n', rel_gap);
results.ReliabilityDiagnostics.GeneralizationGap = gen_gap;
results.ReliabilityDiagnostics.InflationFactor   = infl_fac;
results.ReliabilityDiagnostics.RelativeGap       = rel_gap;

%% 5b. Conformal-prediction coverage (LACP) on the hold-out
% Apply the Tuning-time depth-modulated 95% interval (CP_Quantile) to the
% hold-out; coverage near 95% = structurally honest CV, under-coverage flags
% optimism/leakage. Skipped if the Tuning artefact carries no OOF calibration.
if exist('CP_Quantile', 'var') && isstruct(CP_Quantile) && ...
        isfield(CP_Quantile, 'Available') && CP_Quantile.Available
    % Rebuild the depth->sigma interpolant from the stored raw vectors (depth grid
    % + sigma values). Same construction as Tuning; 'nearest' extrapolation outside
    % the calibration depth range matches the calibration side.
    sigmaInterp = griddedInterpolant(CP_Quantile.SigmaDepthGrid, CP_Quantile.SigmaValues, 'linear', 'nearest');
    halfWidth = CP_Quantile.NormalizedScore * sigmaInterp(holdoutSet.DEPTH);
    halfWidth = halfWidth(:);
    ho_abs_err = abs(Yholdout_true_raw - max(Yholdout_hat_raw, 0));
    covered    = ho_abs_err <= halfWidth;
    cp_target  = (1 - CP_Quantile.Alpha) * 100;

    cpRes = struct();
    cpRes.TargetCoverage   = cp_target;
    cpRes.Coverage_Holdout = mean(covered) * 100;
    cpRes.AvgWidth_Holdout = mean(2 * halfWidth);
    cpRes.nCal             = CP_Quantile.nCal;

    fprintf('=== CONFORMAL PREDICTION COVERAGE (%.0f%% LACP, OOF-calibrated) ===\n', cp_target);
    fprintf('+-----------------------+-------------------+------------------------------+\n');
    fprintf('| Depth Range           | Hold-Out Coverage | Avg Interval Width (µmol/kg) |\n');
    fprintf('+-----------------------+-------------------+------------------------------+\n');
    cpBands = depthStrata;                 % reuse the IoA depth strata defined in §4
    nB = size(cpBands, 1);
    cpRes.Coverage_byDepth = nan(nB, 1);
    cpRes.AvgWidth_byDepth = nan(nB, 1);
    cpRes.Labels = cpBands(:, 1);
    for i = 1:nB
        m = (holdoutSet.DEPTH >= cpBands{i,2}(1) & holdoutSet.DEPTH <= cpBands{i,2}(2));
        if any(m)
            cpRes.Coverage_byDepth(i) = mean(covered(m)) * 100;
            cpRes.AvgWidth_byDepth(i) = mean(2 * halfWidth(m));
            fprintf('| %-21s | %16.2f%% | %28.4f |\n', cpBands{i,1}, cpRes.Coverage_byDepth(i), cpRes.AvgWidth_byDepth(i));
        else
            fprintf('| %-21s | %17s | %28s |\n', cpBands{i,1}, 'N/A', 'N/A');
        end
    end
    fprintf('+-----------------------+-------------------+------------------------------+\n');
    cov_gap = cpRes.Coverage_Holdout - cp_target;
    fprintf('  Global coverage %.2f%% (target %.0f%%, gap %+.2f%%); calibration N=%d OOF.\n', ...
            cpRes.Coverage_Holdout, cp_target, cov_gap, cpRes.nCal);
    if cpRes.Coverage_Holdout < 90.0
        fprintf('  [WARN] Under-coverage: OOF errors were optimistically small — sign of leakage/optimism bias.\n\n');
    elseif cpRes.Coverage_Holdout >= 93.0 && cpRes.Coverage_Holdout <= 97.0
        fprintf('  [OK] Valid coverage: the OOF-calibrated interval transfers to the independent hold-out.\n\n');
    else
        fprintf('  [INFO] Coverage off-target — check covariate shift or fold variance.\n\n');
    end
    results.ConformalPrediction = cpRes;
else
    fprintf('Conformal coverage: no LACP calibration in the Tuning artefact (pre-OOF run) — skipped.\n\n');
end

%% 6. Save Final Production Results
fprintf('Saving complete production run results to file...\n');

% Recreate Results/ if an external process removed it mid-run.
if ~isfolder('Results'), mkdir('Results'); fprintf('   - Recreated missing Results/ directory.\n'); end

% Extract timestamp from tuning filename so the Production .mat lands
% under a matching Results_<timestamp>_Production.mat name.
[~, tuningFileName, ~] = fileparts(tuningFilePath);
timestampCell = extractBetween(tuningFileName, 'Results_', '_Tuning');
if isempty(timestampCell) || isempty(timestampCell{1})
    error('Could not extract timestamp from tuning filename: %s\nExpected the Results_<timestamp>_Tuning.mat naming convention.', tuningFileName);
end
timestamp = timestampCell{1};

productionResultsFile = fullfile('Results', sprintf('Results_%s_Production.mat', timestamp));

results.FinalFeatureImportance = double(trainResults.feature_importance);

% Capture the production log content before saving
diary off;
if exist(tempLogFile, 'file')
    results.ProductionLogContent = fileread(tempLogFile);
    delete(tempLogFile);
end

% Chain reference to upstream Tuning meta (guaranteed present by the
% Production:MissingMeta guard above).
upstreamMeta = meta;
if isfield(meta, 'excludeCoords')
    excludeCoords = meta.excludeCoords;
else
    excludeCoords = false;  % default when the Tuning artefact omits the flag
end
meta = buildMetadata(meta.strategy, 'Production', meta.TestMode, tuningFilePath, excludeCoords);
meta.upstreamMeta = upstreamMeta;

save(productionResultsFile, ...
    'results', ...
    'devSet', 'holdoutSet', ...
    'recommendedPredictorNames', 'targetName', ...
    'C_final', 'S_final', ...
    'finalHyperparams', 'bestT_final', 'bestFoldIndex', ...
    'Yholdout_hat_raw', 'Yholdout_true_raw', ...
    'Ydev_hat_raw', 'Ydev_true_raw', ...
    'tuningFilePath', 'meta', ...
    '-v7.3');
fprintf('   Results saved to: %s\n', productionResultsFile);

totalProdTime = toc;
fprintf('\n=== Production Script Complete. Total Time: %.1f mins ===\n\n', totalProdTime/60);

end