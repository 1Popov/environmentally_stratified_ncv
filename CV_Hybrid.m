function [c, observationFoldIDs] = CV_Hybrid(featureTable, nFolds, M)
% CV_Hybrid - Environmentally stratified cvpartition built from a hybrid
% fingerprint that combines PCA-on-vertical-profiles structural modes
% with scalar physics-informed indices.
%
% Pipeline: build M cosine-weighted atomic geospatial blocks; PCA the
% interpolated DENSITY / SALINITY / TEMP profiles to numPCs=5 dominant
% modes; compute the same five scalar indices as CV_OceanFingerprint;
% aggregate to per-block fingerprints; z-score the scalar block and
% rescale so its total variance matches the PCA block (equal-influence
% balancing); k-means cluster blocks into nFolds environmental strata
% on the balanced hybrid fingerprint; shuffled round-robin distribute
% blocks within each stratum to nFolds folds.
%
% Inputs:
%   featureTable - Black Sea DOXY feature table. Must contain LATITUDE,
%                  LONGITUDE, DEPTH, DENSITY, SALINITY, TEMP, plus the
%                  per-profile scalar columns SALINITY2, TEMP2, TEMP100,
%                  DENSITY2, DENSITY200, MLD.
%   nFolds       - Number of outer-CV folds.
%   M            - Number of atomic geospatial blocks (canonical: 750,
%                  set by getExperimentConfig.nAtomicBlocks).
%
% Outputs:
%   c                  - cvpartition('CustomPartition', testSets) object.
%   observationFoldIDs - Per-row fold ID (1..nFolds) over featureTable.

%% 1. Build M atomic geospatial blocks (shared helper, see buildAtomicBlocks.m).
fprintf('Building %d atomic blocks via buildAtomicBlocks helper...\n', M);
[profileData, profileIdx] = buildAtomicBlocks(featureTable, M);
profileIDs  = profileData.ProfileID;
numProfiles = height(profileData);

%% 2. Structural fingerprint via PCA on interpolated vertical profiles.
% Resampling grid matches CV_EOF.m; the interpolation policy deliberately does
% not (linear/no-extrap here vs nearest/extrap there) -- see the note at the
% interp1 call below.
fprintf('Creating structural fingerprint via PCA...\n');
standardDepths    = (0:5:200)';                  % 5-m grid, 0-200 m -> 41 standard depths
varsToInterpolate = {'DENSITY', 'SALINITY', 'TEMP'};
numStandardDepths = length(standardDepths);
numVars           = length(varsToInterpolate);
numPCs            = 5;

interpolatedProfiles = NaN(numStandardDepths, numVars, numProfiles);
for k = 1:numProfiles
    isCurrentProfile = (profileIdx == k);
    profileTable = featureTable(isCurrentProfile, :);
    for v = 1:numVars
        varName = varsToInterpolate{v};
        depths = profileTable.DEPTH;
        values = profileTable.(varName);
        hasData = ~isnan(depths) & ~isnan(values);
        if ~any(hasData), continue; end
        [uniqueDepths, ~, groupIdx] = unique(depths(hasData));
        uniqueValues = accumarray(groupIdx, values(hasData), [], @mean);
        if length(uniqueDepths) >= 2
            % Resample each profile onto the common depth grid by linear
            % interpolation within the observed range (no extrapolation), which drops
            % partial-coverage profiles from the PCA basis. Acceptable here because the
            % structural PCA is only one of two balanced fingerprint components: the
            % per-profile scalar indices (step 3) still represent the dropped profiles'
            % blocks, and block-averaging plus mean-fill absorb the residual PCA NaNs.
            % CV_EOF, whose only signal is PCA, uses 'nearest','extrap' to keep coverage.
            interpolatedProfiles(:, v, k) = interp1(uniqueDepths, uniqueValues, standardDepths, 'linear');
        end
    end
end

pcaInputMatrix = reshape(interpolatedProfiles, [numStandardDepths * numVars, numProfiles])';
isCompleteProfile = all(~isnan(pcaInputMatrix), 2);
if ~any(isCompleteProfile)
    error('No complete profiles were found after interpolation. Cannot perform PCA.');
end

[~, score, ~, ~, explained] = pca(pcaInputMatrix(isCompleteProfile, :));
numPCs = min(numPCs, length(explained));
fprintf('   PCA complete. First %d components explain %.2f%% of variance.\n', numPCs, sum(explained(1:numPCs)));

pca_fingerprint = NaN(numProfiles, numPCs);
pca_fingerprint(isCompleteProfile, :) = score(:, 1:numPCs);
pcVarNames = arrayfun(@(i) ['PC' num2str(i)], 1:numPCs, 'UniformOutput', false);
pcaFingerprintTable = array2table(pca_fingerprint, 'VariableNames', pcVarNames);
pcaFingerprintTable.ProfileID = profileIDs;

%% 3. Scalar fingerprint from physics-informed indices (matches CV_OceanFingerprint).
fprintf('Creating scalar fingerprint from physical indices...\n');
mean_SALINITY2  = accumarray(profileIdx, featureTable.SALINITY2,  [], @(x) mean(x, 'omitnan'));
mean_TEMP100    = accumarray(profileIdx, featureTable.TEMP100,    [], @(x) mean(x, 'omitnan'));
mean_DENSITY200 = accumarray(profileIdx, featureTable.DENSITY200, [], @(x) mean(x, 'omitnan'));
mean_DENSITY2   = accumarray(profileIdx, featureTable.DENSITY2,   [], @(x) mean(x, 'omitnan'));
mean_TEMP2      = accumarray(profileIdx, featureTable.TEMP2,      [], @(x) mean(x, 'omitnan'));
mean_MLD        = accumarray(profileIdx, featureTable.MLD,        [], @(x) mean(x, 'omitnan'));

stratification_Index  = mean_DENSITY200 - mean_DENSITY2;
surface_Heating_Index = mean_TEMP2 - mean_TEMP100;

scalarFingerprintTable = table(profileIDs, mean_SALINITY2, mean_TEMP100, stratification_Index, surface_Heating_Index, mean_MLD, ...
    'VariableNames', {'ProfileID', 'Mean_SALINITY2', 'Mean_TEMP100', 'Stratification_Index', 'Surface_Heating_Index', 'Mean_MLD'});

%% 4. Combine fingerprints and aggregate to block level.
fprintf('Combining fingerprints and aggregating to block level...\n');
hybridProfileTable = join(pcaFingerprintTable, scalarFingerprintTable, 'Keys', 'ProfileID');
tempBlockData      = join(profileData, hybridProfileTable, 'Keys', 'ProfileID');

hybridFeatureNames = [pcVarNames, {'Mean_SALINITY2', 'Mean_TEMP100', 'Stratification_Index', 'Surface_Heating_Index', 'Mean_MLD'}];

blockProfiles = groupsummary(tempBlockData, 'BlockID', 'mean', hybridFeatureNames);
blockProfiles.Properties.VariableNames = strrep(blockProfiles.Properties.VariableNames, 'mean_', '');
blockProfiles = removevars(blockProfiles, 'GroupCount');

%% 5. Stratify blocks via k-means on the equal-influence balanced fingerprint.
fprintf('Stratifying blocks using k-means on the hybrid fingerprint...\n');
profilingVars = blockProfiles{:, 2:end};

% Mean-fill any NaN columns (blocks with no complete profile).
for col = 1:size(profilingVars, 2)
    colData = profilingVars(:, col);
    isNan = isnan(colData);
    if any(isNan)
        profilingVars(isNan, col) = mean(colData, 'omitnan');
    end
end

% Equal-influence balancing: z-score the scalar block, then rescale so
% its total variance matches the (raw) PCA block. Without rescaling, the
% z-scored scalars carry one unit of variance per column while the PCA
% block carries the natural eigenvalue hierarchy, and one set would
% dominate the downstream k-means.
pca_scores     = profilingVars(:, 1:numPCs);
scalar_indices = profilingVars(:, (numPCs+1):end);

scalar_indices_z        = zscore(scalar_indices);
total_variance_pca      = sum(var(pca_scores));
total_variance_scalars  = sum(var(scalar_indices_z));  % equals size(scalar_indices_z, 2) since var(z-score)=1; explicit sum is symmetric with the PCA side.
balancing_factor        = sqrt(total_variance_pca / total_variance_scalars);
scalar_indices_balanced = scalar_indices_z * balancing_factor;

hybridFingerprint = [pca_scores, scalar_indices_balanced];

fprintf('   Clustering %d blocks into %d strata based on the %d-dimensional fingerprint...\n', ...
    size(hybridFingerprint, 1), nFolds, size(hybridFingerprint, 2));
rng(42, 'twister');  % LOCKED seed -- see Docs/REPRODUCIBILITY.md before changing.
stratumIDs = kmeans(hybridFingerprint, nFolds, 'MaxIter', 1000, 'Replicates', 10);
blockProfiles.StratumID = stratumIDs;

% Stratum size balance check: round-robin (step 6) assigns blocks one per
% stratum per pass; a stratum with fewer than nFolds blocks means some
% folds miss that environmental regime.
stratumSizes = histcounts(stratumIDs, 0.5:1:(nFolds+0.5));
if any(stratumSizes < nFolds)
    smallStrata = find(stratumSizes < nFolds);
    warning('CV_Hybrid:undersized_stratum', ...
        ['%d stratum/strata have < nFolds=%d blocks (smallest = %d blocks). ' ...
         'Round-robin may unbalance fold composition for these regimes.'], ...
        numel(smallStrata), nFolds, min(stratumSizes));
end
fprintf('   Stratum sizes: min=%d, median=%g, max=%d  (target >= %d per stratum)\n', ...
        min(stratumSizes), median(stratumSizes), max(stratumSizes), nFolds);

%% 6. Distribute blocks to nFolds folds via shuffled round-robin (shared helper).
fprintf('Mapping fold IDs to observations via assignBlocksToFolds helper...\n');
[c, observationFoldIDs] = assignBlocksToFolds(blockProfiles, profileData, profileIdx, nFolds, M, featureTable);
profileBlockIDs = profileData.BlockID;  % needed by the KL-divergence diagnostic below

% --- Diagnostic: final fold distribution report (counts + percentages) ---
fprintf('\n--- FINAL FOLD DISTRIBUTION REPORT ---\n');
totalObs = length(observationFoldIDs);
fprintf(' Fold | Observations | Percentage \n');
fprintf('----------------------------------\n');
for f = 1:nFolds
    foldCount = sum(observationFoldIDs == f);
    foldPct   = (foldCount / totalObs) * 100;
    fprintf('  %2d  | %12d | %8.2f%%\n', f, foldCount, foldPct);
end
fprintf('----------------------------------\n\n');

% --- Diagnostic: per-fold environmental representativeness (KL divergence) ---
% Balanced fold COUNT does not guarantee balanced env DISTRIBUTION per fold.
% KL divergence from per-fold to global env-feature distribution quantifies
% the residual drift. Low KL (< 0.05) = fold is env-representative.
compute_fold_env_kl(hybridFingerprint, profileBlockIDs, profileIdx, observationFoldIDs, nFolds);

fprintf('--- CV_Hybrid complete. ---\n');

end


function compute_fold_env_kl(hybridFingerprint, profileBlockIDs, profileIdx, observationFoldIDs, nFolds)
% compute_fold_env_kl - Per-fold KL divergence vs global env distribution.
%
% Maps block-level fingerprint -> per-observation fingerprint via two index
% steps (obs -> profile -> block -> fingerprint row), then bins each env
% feature into 10 equal-population global-quantile bins and computes
% KL(fold || global) per feature, averaged across features. Equal-population
% global bins make KL = 0 the perfectly-representative baseline regardless
% of feature distribution shape.

obsBlockIDs    = profileBlockIDs(profileIdx);
obsFingerprint = hybridFingerprint(obsBlockIDs, :);

nFeatures = size(obsFingerprint, 2);
nBins     = 10;
eps_      = 1e-9;

% Pre-compute global per-feature edges + reference distribution.
edgesPerFeature   = cell(1, nFeatures);
globalDistPerFeat = cell(1, nFeatures);
for f = 1:nFeatures
    edges      = quantile(obsFingerprint(:, f), linspace(0, 1, nBins+1));
    edges(1)   = -Inf;
    edges(end) =  Inf;
    edgesPerFeature{f}   = edges;
    globalDistPerFeat{f} = histcounts(obsFingerprint(:, f), edges, ...
                                       'Normalization', 'probability');
end

fprintf('\n--- FOLD ENV REPRESENTATIVENESS REPORT (KL vs global env distribution) ---\n');
fprintf(' Fold | mean KL  | interpretation \n');
fprintf('---------------------------------------------\n');
for k = 1:nFolds
    foldMask = (observationFoldIDs == k);
    klFold   = 0;
    for f = 1:nFeatures
        foldDist = histcounts(obsFingerprint(foldMask, f), edgesPerFeature{f}, ...
                              'Normalization', 'probability');
        klFold = klFold + sum(foldDist .* log((foldDist + eps_) ./ ...
                                              (globalDistPerFeat{f} + eps_)));
    end
    klFold = klFold / nFeatures;

    if     klFold < 0.05, label = 'excellent';
    elseif klFold < 0.10, label = 'good';
    elseif klFold < 0.20, label = 'acceptable';
    else,                 label = 'WARNING - investigate';
    end
    fprintf('  %2d  | %7.4f | %s\n', k, klFold, label);
end
fprintf('---------------------------------------------\n\n');
end
