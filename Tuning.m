function tuningResultsFile = Tuning(dataFilePath)
% Tuning - Middle stage of the NCV pipeline.
%
% Loads a Discovery .mat, summarises per-fold performance (mean and SD
% of every metric across the K_out outer folds), selects the best
% fold's hyperparameters (argmin of RMSE_raw across folds), computes
% the Nadeau-Bengio 95% CI on the NCV mean RMSE, ranks features by
% mean importance for diagnostic display, and persists everything
% Production needs into a Tuning .mat. No model training happens here.
%
% Input:
%   dataFilePath      - Path to the Discovery .mat produced by Discovery.m.
%
% Output:
%   tuningResultsFile - Path to the Tuning .mat written under Results/.
%                       Consumed by Production.m.

%% 1. Configuration
tic;

% Discovery folder, saved for downstream tooling (Production doesn't read it;
% see Docs/SAVE_SCHEMA.md §2.2).
[discoveryRunFolder, ~, ~] = fileparts(dataFilePath);

fprintf('=== Black Sea DO Tuning Script ===\n');
fprintf('Loading data from discovery run: %s\n\n', discoveryRunFolder);

%% 2. Load and Summarize Discovery Performance

try
    load(dataFilePath, 'allFoldResults', 'devSet', 'holdoutSet', 'cyclicNames', ...
                       'selectedPredictorNames', 'targetName', 'nFolds', 'meta');
    fprintf('Discovery data loaded successfully.\n');
catch ME
    error('Failed to load discovery data file: %s\n%s', dataFilePath, ME.message);
end

NumberFeatures = numel(selectedPredictorNames);

fprintf('\n--- Summary of Nested CV Performance (from Discovery Run) ---\n');
% Failed folds are empty-struct sentinels; find the first successful one.
firstSuccess = find(cellfun(@(s) isstruct(s) && isfield(s, 'metrics'), allFoldResults), 1);
if isempty(firstSuccess)
    error('Tuning:NoSuccessfulFolds', ...
          'All %d Discovery outer folds failed (empty .metrics across the board). Source: %s.', ...
          nFolds, dataFilePath);
end
metricNames = fieldnames(allFoldResults{firstSuccess}.metrics);
outerMetrics = NaN(nFolds, length(metricNames));
allBestHyperparams = cell(nFolds, 1);
for k = 1:nFolds
    if isstruct(allFoldResults{k}) && ~isempty(fieldnames(allFoldResults{k}))
        outerMetrics(k,:) = cell2mat(struct2cell(allFoldResults{k}.metrics))';
        allBestHyperparams{k} = allFoldResults{k}.bestHyperparams;
    end
end

results.PerformanceSummary_NCV = struct();
for m = 1:length(metricNames)
    values = outerMetrics(:, m);
    results.PerformanceSummary_NCV.(metricNames{m}) = struct(...
        'mean', mean(values, 'omitnan'), ...
        'std', std(values, 'omitnan'), ...
        'values', values);
    fprintf('   - NCV Mean %-12s: %.4f (± %.4f)\n', metricNames{m}, ...
        results.PerformanceSummary_NCV.(metricNames{m}).mean, ...
        results.PerformanceSummary_NCV.(metricNames{m}).std);
end
fprintf('\n');

%% 3. Detailed Feature Importance Analysis & Suggestion
fprintf('--- Analyzing Feature Importance Across %d Folds ---\n', nFolds);

% Aggregate feature importance from all folds
allFeatureImportances = NaN(NumberFeatures, nFolds);
for k = 1:nFolds
    if isstruct(allFoldResults{k}) && isfield(allFoldResults{k}, 'featureImportance')
        allFeatureImportances(:, k) = allFoldResults{k}.featureImportance;
    end
end

featureImportanceTable = table;
featureImportanceTable.FeatureName = selectedPredictorNames(:);
featureImportanceTable.MeanImportance = mean(allFeatureImportances, 2, 'omitnan');
featureImportanceTable.StdImportance = std(allFeatureImportances, 0, 2, 'omitnan');

% Normalize mean importance for easier interpretation
totalImportance = sum(featureImportanceTable.MeanImportance);
if totalImportance > 0
    featureImportanceTable.MeanImportance_Normalized = 100 * featureImportanceTable.MeanImportance / totalImportance;
else
    featureImportanceTable.MeanImportance_Normalized = zeros(NumberFeatures, 1);
end

% Sort by importance (diagnostic display only — feature set is fixed upstream).
[sortedTable, ~] = sortrows(featureImportanceTable, 'MeanImportance_Normalized', 'descend');
fprintf('Feature importance ranking across %d folds (diagnostic — feature set is fixed):\n', nFolds);
disp(sortedTable);

% Forward the Discovery feature set unchanged — Tuning does no feature
% selection (the working set is locked in getWorkingFeatures.m).
recommendedPredictorNames = selectedPredictorNames;
fprintf('Feature set forwarded from Discovery: %d predictors.\n\n', length(recommendedPredictorNames));

%% 4. Select Best Hyperparameters from Discovery
fprintf('--- Selecting Best Hyperparameters from Discovery Folds ---\n');

rmseValues = results.PerformanceSummary_NCV.RMSE_raw.values;
if all(isnan(rmseValues))
    error('All NCV folds failed (RMSE_raw all NaN). Cannot select best fold.');
end
[~, bestFoldIndex] = min(rmseValues);

% Nadeau-Bengio (2003) 95% CI on NCV mean RMSE; equal-split shortcut form.
results.PerformanceSummary_NCV.RMSE_raw.NB_CI95 = computeNadeauBengioCI(rmseValues);
fprintf('   - NB-corrected 95%% CI on NCV mean RMSE: [%.2f, %.2f]\n', ...
    results.PerformanceSummary_NCV.RMSE_raw.NB_CI95(1), ...
    results.PerformanceSummary_NCV.RMSE_raw.NB_CI95(2));

% Extract the corresponding hyperparameter table for that fold
finalHyperparams = allBestHyperparams{bestFoldIndex};

fprintf('> Best performance found in Fold %d. Using its hyperparameters for the final model.\n', bestFoldIndex);
fprintf('Final hyperparameters selected:\n');
disp(finalHyperparams);
fprintf('\n');

%% 4.5 Locally-Adaptive Conformal Prediction (LACP) calibration
% Aggregate OOF predictions (each dev row held out by one fold) and calibrate a
% depth-modulated 95% interval via conformalCalibrate.m (shared with
% conformalCoverage.m so the live and reproduction paths cannot diverge).
fprintf('--- Calibrating Locally-Adaptive Conformal Prediction (LACP) ---\n');
CP_Quantile = struct('Available', false);
try
    n_dev    = height(devSet);
    OOF_Yhat = NaN(n_dev, 1);
    OOF_Ytrue = NaN(n_dev, 1);
    for k = 1:nFolds
        fk = allFoldResults{k};
        if isstruct(fk) && isfield(fk, 'OOF_Indices') && ~isempty(fk.OOF_Indices)
            idx = fk.OOF_Indices;
            if max(idx) > n_dev
                error('Tuning:OOFIndexOverflow', ...
                      'Fold %d OOF index exceeds devSet height — clear Results/TemporaryResults and re-run.', k);
            end
            OOF_Yhat(idx)  = fk.OOF_Predictions;
            OOF_Ytrue(idx) = fk.OOF_TrueValues;
        end
    end
    validOOF = ~isnan(OOF_Yhat) & ~isnan(OOF_Ytrue);
    if nnz(validOOF) < 100
        warning('Tuning:InsufficientOOF', ...
                'Only %d valid OOF predictions; conformal calibration skipped (need >=100).', nnz(validOOF));
    else
        % Store sigma(depth) as raw vectors (grid + values) — plain doubles
        % round-trip cleanly; Production rebuilds the interpolant from them.
        cal = conformalCalibrate(OOF_Yhat, OOF_Ytrue, devSet.DEPTH, 0.05);
        CP_Quantile = struct( ...
            'Available',       true, ...
            'Alpha',           cal.Alpha, ...
            'NormalizedScore', cal.NormalizedQuantile, ...
            'SigmaDepthGrid',  cal.SigmaDepthGrid, ...
            'SigmaValues',     cal.SigmaValues, ...
            'WindowSize',      cal.WindowSize, ...
            'nCal',            cal.nCal);
        fprintf('   LACP calibrated: N=%d valid OOF | target %.0f%% | normalized quantile=%.4f\n', ...
                cal.nCal, (1 - cal.Alpha) * 100, CP_Quantile.NormalizedScore);
    end
catch cpME
    warning('Tuning:ConformalCalibrationFailed', ...
            'LACP calibration failed (%s); Production will skip conformal reporting.', cpME.message);
    CP_Quantile = struct('Available', false);
end
fprintf('\n');

%% 5. Tuning Analysis Complete - Save Results for Production
fprintf('--- Tuning Analysis Complete ---\n');
fprintf('Recommended features and the single best hyperparameter set have been determined.\n');
fprintf('Variables available for production:\n');
fprintf('  - recommendedPredictorNames (%d features)\n', length(recommendedPredictorNames));
fprintf('  - finalHyperparams (the single best hyperparameter set)\n');
fprintf('  - All discovery data and performance metrics\n\n');

% Extract timestamp from discovery filename so the Tuning .mat lands
% under a matching Results_<timestamp>_Tuning.mat name.
[~, discoveryFileName, ~] = fileparts(dataFilePath);
timestampCell = extractBetween(discoveryFileName, 'Results_', '_Discovery');
if isempty(timestampCell) || isempty(timestampCell{1})
    error('Could not extract timestamp from discovery filename: %s\nExpected the Results_<timestamp>_Discovery.mat naming convention.', discoveryFileName);
end
timestamp = timestampCell{1};

tuningResultsFile = fullfile('Results', sprintf('Results_%s_Tuning.mat', timestamp));

% Build standardized run metadata. Chain reference to upstream Discovery
% meta if it's present in the loaded .mat (see SAVE_SCHEMA.md).
if exist('meta', 'var') && isstruct(meta)
    upstreamMeta = meta;
    stratName    = upstreamMeta.strategy;
    testMode     = upstreamMeta.TestMode;
    if isfield(upstreamMeta, 'excludeCoords')
        excludeCoords = upstreamMeta.excludeCoords;
    else
        excludeCoords = false;  % default when the Discovery artefact omits the flag
    end
else
    upstreamMeta  = struct('note', 'no upstream meta found in Discovery .mat');
    stratName     = "unknown";
    testMode      = false;
    excludeCoords = false;
end
meta = buildMetadata(stratName, 'Tuning', testMode, dataFilePath, excludeCoords);
meta.upstreamMeta = upstreamMeta;

if ~isfolder('Results'), mkdir('Results'); end  % recreate if removed mid-run
save(tuningResultsFile, ...
    'discoveryRunFolder', 'devSet', 'holdoutSet', 'cyclicNames', 'targetName', ...
    'recommendedPredictorNames', 'finalHyperparams', 'results', ...
    'allFoldResults', 'nFolds', 'featureImportanceTable', 'sortedTable', 'dataFilePath', ...
    'meta', 'bestFoldIndex', 'metricNames', 'allBestHyperparams', ...
    'CP_Quantile', ...
    '-v7.3');

totalTuningTime = toc;
fprintf('Tuning analysis saved to: %s\n', tuningResultsFile);
fprintf('=== Tuning Script Complete. Total Time: %.1f mins ===\n\n', totalTuningTime/60);

end