function finalResultsFile = Discovery(strategy, excludeCoords)
% Discovery — strategy-parameterized nested-CV discovery driver.
%
% First and most expensive stage of the pipeline (Discovery -> Tuning ->
% Production): runs the full nested cross-validation that produces the paper's
% NCV performance estimate and the per-fold hyperparameters Tuning selects from.
%
% Algorithm:
%   1. Resolve the strategy's knobs from getExperimentConfig, open a parpool,
%      and seed the global stream rng(config.RandomSeed, 'twister').
%   2. Carve a quarantined hold-out: partition featureTable with CV_Hybrid's
%      first fold on a TARGET-BLIND copy (OXYGEN stripped) so the split cannot
%      peek at the label; the complement is the development set.
%   3. Preprocess the dev set: drop missing rows, optionally subsample in smoke
%      mode, then shuffle row order (featureTable ships BuoyName-sorted).
%   4. Select the working features via getWorkingFeatures (35 canonical / 33
%      coord-removal) and build the outer-CV partition via config.OuterCVFcn.
%   5. For each outer fold (parfor), in Python via lgbm_wrapper: run the Optuna
%      inner-CV search over getHyperparameterSpace, train the fold model at the
%      best iteration, predict the held-out fold, and record metrics, feature
%      importance, learning curves, and the out-of-fold (OOF) predictions Tuning
%      later uses for conformal calibration.
%   6. Each fold is checkpointed to Results/TemporaryResults/<strategy>/ so an
%      interrupted run RESUMES rather than recomputing; on completion the folds
%      are consolidated into one Discovery .mat and the temp files are cleared.
%
% Inputs:
%   strategy      One of "RandomCV", "RandomCV_Weighted", "GeoClusterCV",
%                 "OFCV", "EOFCV", "HybridCV". Drives partitioner + weights
%                 + budget via getExperimentConfig. The global TEST/SMOKE-TEST
%                 toggle is in isTestMode.m (single file edit).
%   excludeCoords (Optional, default false) When true, runs the
%                 33-feature coord-removal variant (LAT/LON excluded);
%                 when false, the canonical 35-feature set. Propagates to
%                 getWorkingFeatures and is recorded in meta.excludeCoords.
%
% Output:
%   finalResultsFile - Path to the Results/Results_<runId>_Discovery.mat written
%                      on success (per-fold results, dev/hold-out sets, feature
%                      and target names, config, meta, search space; -v7.3).
%                      Consumed by Tuning.m; see Docs/SAVE_SCHEMA.md for the
%                      full variable list.

if nargin < 2, excludeCoords = false; end

tic;

% py.dict -> struct conversions trip MATLAB:structOnObject; silence it.
prevWarn   = warning('off', 'MATLAB:structOnObject');
restoreWarn = onCleanup(@() warning(prevWarn));

testMode                = isTestMode();
config                  = getExperimentConfig(strategy);
nAtomicBlocks           = config.nAtomicBlocks;
ParProc                 = config.ParProc;
nHoldoutStrata          = config.nHoldoutStrata;
MaxObjectiveEvaluations = config.MaxObjectiveEvaluations;   % hoisted for the parfor body
SubsetSize              = config.SubsetSize;
nFolds                  = config.nFolds;
WeightFcn               = config.WeightFcn;
InnerCVFcn              = config.InnerCVFcn;
OuterCVFcn              = config.OuterCVFcn;
RandomSeed              = config.RandomSeed;
NumBoostRounds          = config.NumBoostRounds;

if isempty(gcp('nocreate'))
    parpool(ParProc);
end

rng(config.RandomSeed, 'twister');

%% 1. Results directory + run-id (resume-aware)
% Coord-mode-aware temp dir so the coord-removal variant (33 features) and the
% canonical run (35 features) of the same strategy never share a fold cache.
stratDir = char(strategy);
if excludeCoords, stratDir = [stratDir '_NoLatLon']; end
resultsDir = fullfile('Results', 'TemporaryResults', stratDir);
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
logFilePath = fullfile(resultsDir, 'discovery_log.txt');

matFiles = dir(fullfile(resultsDir, 'OuterFold_*.mat'));
if isempty(matFiles)
    fprintf('--- No completed folds found. Starting a fresh discovery run. ---\n');
    entries = dir(resultsDir);
    nonDotEntries = entries(~ismember({entries.name}, {'.', '..'}));
    if ~isempty(nonDotEntries)
        diary off;
        delete(fullfile(resultsDir, '*'));
        fprintf('   - Cleared temporary directory of orphaned files from a previous failed run.\n');
    end

    runIdFile = fullfile(resultsDir, 'run_id.txt');
    timestamp = char(datetime('now', 'Format', 'dd_MM_yyyy_HH-mm'));

    fileID = -1;
    for attempt = 1:5
        [fileID, errmsg] = fopen(runIdFile, 'w');
        if fileID ~= -1, break; end
        fprintf('   - Warning: Failed to create run_id.txt on attempt %d/5. Retrying in 0.5s...\n', attempt);
        pause(0.5);
    end
    if fileID == -1
        error('CRITICAL: Unable to create run ID file after 5 attempts.\nPath: %s\nLast System Error: %s', runIdFile, errmsg);
    end
    fprintf(fileID, '%s\n', timestamp);
    fclose(fileID);

    diary(logFilePath);
    fprintf('   - New Run ID created: %s\n', timestamp);
else
    fprintf('--- Found %d completed folds. Resuming discovery run. ---\n', numel(matFiles));
    runIdFile = fullfile(resultsDir, 'run_id.txt');
    if exist(runIdFile, 'file')
        timestamp = strtrim(fileread(runIdFile));
    else
        timestamp = char(datetime('now', 'Format', 'dd_MM_yyyy_HH-mm'));
        warning('MATLAB:ResumeWarning','Fold results found but run_id.txt was missing. Using new timestamp.');
    end
    diary(logFilePath);
    fprintf('   - Resuming with existing Run ID: %s\n', timestamp);
end

if testMode, modeStr = 'quick'; else, modeStr = 'full'; end
runId = sprintf('%s_%s_%s', char(strategy), modeStr, timestamp);
finalResultsFile = fullfile('Results', sprintf('Results_%s_Discovery.mat', runId));

fprintf('   - Temporary log active at: %s\n', logFilePath);
fprintf('   - Final results will be archived with timestamp: %s\n\n', timestamp);

%% 2. Verify Python
fprintf('--- Verifying Python Environment ---\n');
pyrun("import lightgbm as lgb; import sys;");
pyrun("print(f'Using LightGBM version: {lgb.__version__}')");
pyrun([ ...
    "try:" ...
    "    lgb.LGBMRegressor(device='gpu');" ...
    "    print('[OK] LightGBM GPU device successfully accessed.')" ...
    "except Exception:" ...
    "    print('[WARN] LightGBM GPU device check FAILED.', file=sys.stderr)" ...
    "    print('   This pipeline trains with device=gpu (set in lgbm_wrapper.py);', file=sys.stderr)" ...
    "    print('   a GPU-enabled LightGBM build is required or training will error.', file=sys.stderr)" ...
    ]);
fprintf('------------------------------------\n\n');

%% 3. Load data + select hold-out via CV_Hybrid
fprintf('1. Loading Raw Data & Selecting Hold-Out Set...\n');
load(fullfile('DataFiles', 'featureTable.mat'), 'featureTable');
fprintf('   - Loaded %d raw observations.\n', height(featureTable));

% Target-blind hold-out: strip OXYGEN before partitioning.
[c_full, ~] = CV_Hybrid(featureTable(:, setdiff(featureTable.Properties.VariableNames, 'OXYGEN', 'stable')), nHoldoutStrata, nAtomicBlocks);
holdoutMask = test(c_full, 1);
devSet_raw     = featureTable(~holdoutMask, :);
holdoutSet_raw = featureTable( holdoutMask, :);

%% 4. Preprocess
fprintf('2. Applying Preprocessing to Data Splits...\n');
nDevProfilesRaw  = height(unique([devSet_raw.LATITUDE     devSet_raw.LONGITUDE],     'rows'));
nHoldProfilesRaw = height(unique([holdoutSet_raw.LATITUDE holdoutSet_raw.LONGITUDE], 'rows'));
fprintf('   - Before cleaning, Development Set has %d samples (%d profiles).\n', ...
    height(devSet_raw), nDevProfilesRaw);
devSet = rmmissing(devSet_raw);
fprintf('   - After cleaning, Development Set has %d samples.\n', height(devSet));

if testMode && SubsetSize < 1
    fprintf('   Test mode: subsetting development set to %.1f%%.\n', SubsetSize*100);
    SubIdx = randperm(height(devSet), floor(SubsetSize*height(devSet)));
    devSet = devSet(SubIdx,:);
end

% Decorrelate row order (featureTable is BuoyName-sorted upstream).
devSet = devSet(randperm(height(devSet)), :);
fprintf('   > Final Development Set for this run: %d samples (split = %d profiles).\n', ...
    height(devSet), nDevProfilesRaw);

fprintf('   - Before cleaning, Hold-Out Set has %d samples (%d profiles).\n', ...
    height(holdoutSet_raw), nHoldProfilesRaw);
holdoutSet = rmmissing(holdoutSet_raw);
fprintf('   > Final Hold-Out Set for this run: %d samples (split = %d profiles) (quarantined).\n\n', ...
    height(holdoutSet), nHoldProfilesRaw);

%% 5. Features
[selectedPredictorNames, targetName, cyclicNames] = getWorkingFeatures(devSet, excludeCoords);
fprintf('   Features selected: %d\n\n', length(selectedPredictorNames));

%% 6. Outer CV + fold-resume scan
fprintf('3. Initializing and checking for existing results...\n');
outerCV = OuterCVFcn(devSet, nFolds, nAtomicBlocks);
allFoldResults = cell(nFolds, 1);
foldDone = false(nFolds, 1);   % pre-allocated mask; foldsToRun derived once after the scan

fprintf('   Scanning for completed fold results in: %s\n', resultsDir);
for k = 1:nFolds
    foldResultFile = fullfile(resultsDir, sprintf('OuterFold_%d_of_%d_results.mat', k, nFolds));
    if exist(foldResultFile, 'file')
        try
            fprintf('   - Fold %d of %d... Found. Loading and skipping.\n', k, nFolds);
            loadedData = load(foldResultFile);
            assert(loadedData.data.foldIndex == k, ...
                'Discovery:FoldIndexMismatch', ...
                ['Resume-scan loaded fold %d from disk but the file ' ...
                 'records foldIndex=%d. Aborting to prevent a silent ' ...
                 'fold mix-up; clear Results/TemporaryResults/%s/ to ' ...
                 'start a fresh discovery run.'], ...
                k, loadedData.data.foldIndex, char(strategy));
            % Coord-mode guard: reject a stale fold whose feature count does not
            % match this run (e.g. 33-feature coord-removal temp files resumed
            % into a 35-feature canonical run). The error is caught below, so the
            % fold is re-queued and recomputed with the correct feature set.
            if isfield(loadedData.data, 'featureImportance') && ...
                    numel(loadedData.data.featureImportance) ~= numel(selectedPredictorNames)
                error('Discovery:FeatureCountMismatch', ...
                    ['Fold %d on disk carries %d-feature data but this run uses %d ' ...
                     '(stale coord-mode temp files); re-running this fold.'], ...
                    k, numel(loadedData.data.featureImportance), numel(selectedPredictorNames));
            end
            allFoldResults{k} = loadedData.data;
            foldDone(k) = true;
        catch loadME
            warning('Could not load existing file for Fold %d. Adding to queue. Error: %s', k, loadME.message);
            % foldDone(k) stays false -> queued by find(~foldDone) below
        end
    else
        fprintf('   - Fold %d of %d... MISSING. Adding to run queue.\n', k, nFolds);
        % foldDone(k) stays false -> queued
    end
end
foldsToRun = reshape(find(~foldDone), 1, []);   % row vector for the parfor loop counter

%% 7. Hyperparameter search space
optVars = getHyperparameterSpace(config.LearningRate);
fprintf('   Bayesian hyperparameter search configured (bounds in getHyperparameterSpace.m).\n\n');

%% 8. Nested CV main loop
fprintf('=== Starting Nested Cross-Validation (%d Outer Folds) ===\n\n', nFolds);
startTime = tic;
tempResults = cell(1, numel(foldsToRun));

parfor i = 1:numel(foldsToRun)
    setenv('OMP_NUM_THREADS', '1');
    warning('off', 'MATLAB:structOnObject');  % per-worker: client suppression doesn't reach parfor workers
    getPythonEnvironment();
    if count(py.sys.path, pwd) == 0
        insert(py.sys.path, int32(0), pwd);
    end

    k = foldsToRun(i);
    foldStartTime = tic;
    fprintf('--- Outer Fold %d of %d (Job %d of %d): Running Bayesian optimization... ---\n', k, nFolds, i, numel(foldsToRun)); %#ok<PFBNS>

    trainIdx = training(outerCV, k);
    testIdx  = test(outerCV, k);
    Ttrain = devSet(trainIdx, :); %#ok<PFBNS>  % devSet broadcast: each worker slices the full table
    Ttest  = devSet(testIdx,  :);

    [TtrainN, TtestN, C_outer, S_outer] = rangeNormalizePair(Ttrain, Ttest, cyclicNames);

    XtrainN_obj = TtrainN{:, selectedPredictorNames};
    YtrainN_obj = TtrainN{:, targetName};
    Wtrain_obj  = WeightFcn(Ttrain.DEPTH); %#ok<PFBNS>

    innerCV = InnerCVFcn(Ttrain); %#ok<PFBNS>
    cv_folds_list = py.list();
    for i_cv = 1:innerCV.NumTestSets
        cv_folds_list.append(py.tuple({py.list(find(innerCV.training(i_cv))), ...
                                       py.list(find(innerCV.test(i_cv)))}));
    end

    optVarsStruct = struct();
    for i_vars = 1:length(optVars)
        v = optVars(i_vars);
        optVarsStruct.(v.Name) = v.Range;
    end
    pyOptVars = py.dict(optVarsStruct);

    fprintf('   Calling Python to run full optimization loop for Fold %d...\n', k);
    pyResults = py.lgbm_wrapper.run_bayes_opt_in_python( ...
        XtrainN_obj, YtrainN_obj, Wtrain_obj, cv_folds_list, pyOptVars, ...
        int32(MaxObjectiveEvaluations), int32(k), int32(RandomSeed), int32(NumBoostRounds));
    results = struct(pyResults);

    if ~strcmp(char(results.status), 'ok')
        warning('Python bayesopt failed for fold %d: %s', k, char(results.status));
        tempResults{i} = struct();
        continue;
    end

    bestP_struct = struct(results.best_params);
    fields = fieldnames(bestP_struct);
    for j = 1:length(fields)
        bestP_struct.(fields{j}) = double(bestP_struct.(fields{j}));
    end
    bestP = struct2table(bestP_struct);

    bestT = double(results.best_iteration);
    bestTrialUserData = struct();
    bestTrialUserData.bestT = bestT;
    bestTrialUserData.train_rmse_history = cellfun(@double, cell(results.train_rmse_history));
    bestTrialUserData.valid_rmse_history = cellfun(@double, cell(results.valid_rmse_history));

    fprintf('   Training outer fold model with optimal parameters (Fold %d)...\n', k);
    pyFinalParams = py.dict(table2struct(bestP));
    pyTrainResults = py.lgbm_wrapper.train_final_model(pyFinalParams, ...
        XtrainN_obj, YtrainN_obj, Wtrain_obj, int32(bestT), int32(RandomSeed));
    trainResults = struct(pyTrainResults);

    if ~strcmp(char(trainResults.status), 'ok')
        warning('Python final training failed for fold %d: %s', k, char(trainResults.status));
        tempResults{i} = struct();
        continue;
    end

    fprintf('   Model for fold %d trained successfully in memory.\n', k);

    XtestN = TtestN{:, selectedPredictorNames};
    YtestN = TtestN{:, targetName};
    Wtest  = WeightFcn(Ttest.DEPTH);

    YhatN_py = py.lgbm_wrapper.predict_from_model(trainResults.model_string, XtestN);
    YhatN = double(YhatN_py); YhatN = YhatN(:);
    YhatRaw  = YhatN .* S_outer.(targetName) + C_outer.(targetName);
    YtrueRaw = Ttest{:, targetName};

    foldMetrics = calculateRegressionMetrics(YtrueRaw, YhatRaw, YtestN, YhatN, Wtest, bestP.huber_delta);
    featureImportance = double(trainResults.feature_importance);

    fprintf('   [OK] Captured learning curve data with %d points.\n', length(bestTrialUserData.valid_rmse_history));

    % OOF predictions (global row indices) -> Tuning aggregates them for
    % conformal calibration; each dev row is held out by exactly one fold.
    oofIndices = find(testIdx);

    bayesoptResults = struct('TrialsHistory', struct(results.trials_history));
    foldResult = struct('foldIndex', k, 'metrics', foldMetrics, ...
        'bestHyperparams', bestP, 'bayesoptResults', bayesoptResults, ...
        'featureImportance', featureImportance(:)', ...
        'modelString', char(trainResults.model_string), ...
        'learningCurves', bestTrialUserData, ...
        'OOF_Indices', oofIndices(:), ...
        'OOF_Predictions', max(YhatRaw, 0), ...
        'OOF_TrueValues', YtrueRaw(:));

    tempResults{i} = foldResult;

    foldTimeMinutes = toc(foldStartTime)/60;
    fprintf('\n--- Fold %d Performance Summary ---\n', k);
    fprintf('+--------------------+---------------+\n');
    fprintf('| Metric             | Value         |\n');
    fprintf('+--------------------+---------------+\n');
    fprintf('| Huber Loss (scaled)| %-13.4f |\n', foldResult.metrics.Huber_scaled);
    fprintf('|--------------------+---------------|\n');
    fprintf('| RMSE (scaled)      | %-13.4f |\n', foldResult.metrics.RMSE_scaled);
    fprintf('| MAE (scaled)       | %-13.4f |\n', foldResult.metrics.MAE_scaled);
    fprintf('| RMSE (raw)         | %-13.4f |\n', foldResult.metrics.RMSE_raw);
    fprintf('| MAE (raw)          | %-13.4f |\n', foldResult.metrics.MAE_raw);
    fprintf('| R²                 | %-13.4f |\n', foldResult.metrics.R2);
    fprintf('| Bias               | %+-13.4f |\n', foldResult.metrics.Bias);
    fprintf('+--------------------+---------------+\n');
    fprintf('| Fold Time (mins)   | %-13.1f |\n', foldTimeMinutes);
    fprintf('+--------------------+---------------+\n\n');

    try
        foldResultFile = fullfile(resultsDir, sprintf('OuterFold_%d_of_%d_results.mat', k, nFolds));
        parsave(foldResultFile, foldResult);
    catch saveME
        warning('Could not save intermediate results for fold %d. Error: %s', k, saveME.message);
    end
end

fprintf('--- Consolidating parallel results into final structure... ---\n');
for i = 1:numel(foldsToRun)
    k = foldsToRun(i);
    if ~isempty(tempResults{i})
        allFoldResults{k} = tempResults{i};
        fprintf('   - Mapping result from job %d to final position for fold %d.\n', i, k);
    end
end

%% 9. Finalize, save, clean
totalTime = toc(startTime);
fprintf('\n=== Computation Phase Complete. Total Time: %.1f hours ===\n\n', totalTime/3600);
fprintf('--- Starting Finalization: Saving All Artifacts to a Single File ---\n');

diary off;
try
    logContent = fileread(logFilePath);
    fprintf('1. Log file content captured.\n');

    meta = buildMetadata(char(strategy), 'Discovery', testMode, '', excludeCoords);
    if ~isfolder('Results'), mkdir('Results'); end
    % MaxObjectiveEvaluations mirrored at top level for config-agnostic loaders.
    save(finalResultsFile, ...
        'allFoldResults', 'devSet', 'holdoutSet', 'cyclicNames', ...
        'selectedPredictorNames', 'targetName', 'totalTime', 'nFolds', ...
        'MaxObjectiveEvaluations', 'logContent', ...
        'meta', 'config', 'optVars', 'testMode', ...
        '-v7.3');
    fprintf('2. Saving all consolidated data to final results file...\n');
    fprintf('   [OK] Final results artifact saved to:\n      %s\n', finalResultsFile);
catch finalME
    error('CRITICAL: Failed to save the final consolidated results file. Temporary data is in %s. Error: %s', ...
        resultsDir, getReport(finalME, 'basic'));
end

fprintf('3. Cleaning up temporary files...\n');
try
    delete(fullfile(resultsDir, '*'));
    fprintf('   [OK] All temporary files have been removed.\n');
catch cleanME
    warning('Could not remove temporary files. Manual cleanup needed in:\n   %s\nError: %s', ...
        resultsDir, getReport(cleanME, 'basic'));
end

fprintf('\n=== Discovery Run Fully Complete ===\n\n');
end


function parsave(fname, data)
save(fname, 'data');
end
