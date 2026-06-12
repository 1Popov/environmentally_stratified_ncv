function [c, observationFoldIDs, explainedVar] = CV_EOF(featureTable, nFolds, M)
% CV_EOF - Environmentally stratified cvpartition built from PCA on
% vertical profiles.
%
% Pipeline: build M cosine-weighted atomic geospatial blocks; interpolate
% DENSITY / SALINITY / TEMP profiles onto a 5-m, 0-200 m standard grid;
% PCA extracts the first numPCs=5 dominant structural modes; aggregate to
% per-block fingerprints; k-means cluster blocks into nFolds environmental
% strata using raw (un-z-scored) PCA scores so the natural variance
% hierarchy weights the clustering; shuffled round-robin distribute blocks
% within each stratum to nFolds folds.
%
% Inputs:
%   featureTable - Black Sea DOXY feature table. Must contain LATITUDE,
%                  LONGITUDE, DEPTH, DENSITY, SALINITY, TEMP columns.
%   nFolds       - Number of outer-CV folds.
%   M            - Number of atomic geospatial blocks (canonical: 750,
%                  set by getExperimentConfig.nAtomicBlocks).
%
% Outputs:
%   c                  - cvpartition('CustomPartition', testSets) object.
%   observationFoldIDs - Per-row fold ID (1..nFolds) over featureTable.

%% 1. Build M atomic geospatial blocks (shared helper, see buildAtomicBlocks.m).
[profileData, profileIdx] = buildAtomicBlocks(featureTable, M);
profileIDs  = profileData.ProfileID;
numProfiles = height(profileData);

%% 2. Interpolate vertical profiles onto a standard depth grid (nearest-neighbour, constant-value extrapolation).
standardDepths = (0:5:200)';
varsToInterpolate = {'DENSITY', 'SALINITY', 'TEMP'};
numStandardDepths = length(standardDepths);
numVars = length(varsToInterpolate);

interpolatedProfiles = NaN(numStandardDepths, numVars, numProfiles);
fprintf('Interpolating %d profiles onto a standard depth grid (nearest-neighbour, constant-value extrapolation)...\n', numProfiles);
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
            % Resample each profile onto the common depth grid by nearest-neighbour
            % interpolation with constant-value extrapolation beyond the observed
            % range, so that profiles with partial depth coverage still contribute a
            % complete feature vector to the EOF basis.
            % nearest+extrap (not linear) is used because the PCA scores are the SOLE
            % fingerprint here: retaining partial profiles matters more than per-sample
            % fidelity. CV_Hybrid uses 'linear' without extrapolation instead, since its
            % scalar physics indices carry the profiles that strict interpolation drops.
            interpolatedProfiles(:, v, k) = interp1(uniqueDepths, uniqueValues, standardDepths, 'nearest', 'extrap');
        end
    end
end
fprintf('   Interpolation complete.\n');

%% 3. PCA on stacked vertical profiles; first numPCs columns become the fingerprint.
numPCs = 5;

% Reshape 3D -> 2D: each row = one profile, columns = (depth x variable) features.
numFeatures = numStandardDepths * numVars;
pcaInputMatrix = reshape(interpolatedProfiles, [numFeatures, numProfiles])';

isCompleteProfile = all(~isnan(pcaInputMatrix), 2);
if ~any(isCompleteProfile)
    error('No complete profiles were found after interpolation. Cannot perform PCA.');
end

fprintf('Performing PCA on %d complete profiles...\n', sum(isCompleteProfile));
[~, score, ~, ~, explained] = pca(pcaInputMatrix(isCompleteProfile, :));
numPCs = min(numPCs, length(explained));
explainedVar = sum(explained(1:numPCs));   % cumulative percent variance of first numPCs EOFs (exposed for reproduce_models)

fprintf('   PCA complete. First %d components explain %.2f%% of variance.\n', numPCs, explainedVar);

fingerprint = NaN(numProfiles, numPCs);
fingerprint(isCompleteProfile, :) = score(:, 1:numPCs);

pcVarNames = cell(1, numPCs);
for i = 1:numPCs
    pcVarNames{i} = ['PC' num2str(i)];
end

fingerprintTable = array2table(fingerprint, 'VariableNames', pcVarNames);
fingerprintTable.ProfileID = profileIDs;

%% 4. Aggregate per-profile fingerprints to block level.
tempBlockData = join(profileData, fingerprintTable, 'Keys', 'ProfileID');

fprintf('Averaging per-block PCA fingerprints across %d atomic blocks...\n', M);
blockProfiles = groupsummary(tempBlockData, 'BlockID', 'mean', pcVarNames);

% Strip 'mean_' prefix and drop auto-added GroupCount so the column layout
% matches CV_OceanFingerprint and CV_Hybrid (precondition for any shared
% downstream slicing).
blockProfiles.Properties.VariableNames = strrep(blockProfiles.Properties.VariableNames, 'mean_', '');
blockProfiles = removevars(blockProfiles, 'GroupCount');
fprintf('   Block-level fingerprint created.\n');

%% 5. Stratify blocks via k-means on the raw PCA fingerprint.
% No z-scoring: the natural variance hierarchy of PC scores (PC1 highest,
% PC5 lowest) weights the clustering so dominant modes dominate the
% stratification.
profilingVars = blockProfiles{:, pcVarNames};

% Mean-fill any NaN columns (blocks with no complete profile).
for col = 1:size(profilingVars, 2)
    colData = profilingVars(:, col);
    isNan = isnan(colData);
    if any(isNan)
        profilingVars(isNan, col) = mean(colData, 'omitnan');
    end
end

fprintf('   Clustering %d blocks using variance-weighted PCA fingerprint...\n', size(profilingVars, 1));
rng(42, 'twister');  % LOCKED seed -- see Docs/REPRODUCIBILITY.md before changing.
stratumIDs = kmeans(profilingVars, nFolds, 'MaxIter', 1000, 'Replicates', 10);
blockProfiles.StratumID = stratumIDs;
fprintf('   Stratification complete.\n');

%% 6. Distribute blocks to nFolds folds via shuffled round-robin (shared helper).
[c, observationFoldIDs] = assignBlocksToFolds(blockProfiles, profileData, profileIdx, nFolds, M, featureTable);
fprintf('--- CV_EOF complete. ---\n');

end
