function [c, observationFoldIDs] = CV_OceanFingerprint(featureTable, nFolds, M)
% CV_OceanFingerprint - Environmentally stratified cvpartition built from
% scalar physics-informed indices.
%
% Pipeline: build M cosine-weighted atomic geospatial blocks; compute
% per-profile means of SALINITY2, TEMP100, MLD plus two derived indices
% (stratification = DENSITY200 - DENSITY2; surface-heating = TEMP2 -
% TEMP100); aggregate to per-block fingerprints; z-score normalise;
% k-means cluster blocks into nFolds environmental strata; shuffled
% round-robin distribute blocks within each stratum to nFolds folds.
%
% Inputs:
%   featureTable - Black Sea DOXY feature table. Must contain LATITUDE,
%                  LONGITUDE, SALINITY2, TEMP2, TEMP100, DENSITY2,
%                  DENSITY200, MLD columns.
%   nFolds       - Number of outer-CV folds.
%   M            - Number of atomic geospatial blocks (canonical: 750,
%                  set by getExperimentConfig.nAtomicBlocks).
%
% Outputs:
%   c                  - cvpartition('CustomPartition', testSets) object.
%   observationFoldIDs - Per-row fold ID (1..nFolds) over featureTable.

%% 1. Build M atomic geospatial blocks (shared helper, see buildAtomicBlocks.m).
[profileData, profileIdx] = buildAtomicBlocks(featureTable, M);
profileIDs = profileData.ProfileID;

%% 2. Per-profile summary statistics + derived indices.
mean_SALINITY2  = accumarray(profileIdx, featureTable.SALINITY2,  [], @(x) mean(x, 'omitnan'));
mean_TEMP100    = accumarray(profileIdx, featureTable.TEMP100,    [], @(x) mean(x, 'omitnan'));
mean_DENSITY200 = accumarray(profileIdx, featureTable.DENSITY200, [], @(x) mean(x, 'omitnan'));
mean_DENSITY2   = accumarray(profileIdx, featureTable.DENSITY2,   [], @(x) mean(x, 'omitnan'));
mean_TEMP2      = accumarray(profileIdx, featureTable.TEMP2,      [], @(x) mean(x, 'omitnan'));
mean_MLD        = accumarray(profileIdx, featureTable.MLD,        [], @(x) mean(x, 'omitnan'));

stratification_Index  = mean_DENSITY200 - mean_DENSITY2;
surface_Heating_Index = mean_TEMP2 - mean_TEMP100;

profileSummary = table(profileIDs, mean_SALINITY2, mean_TEMP100, stratification_Index, surface_Heating_Index, mean_MLD, ...
    'VariableNames', {'ProfileID', 'Mean_SALINITY2', 'Mean_TEMP100', 'Stratification_Index', 'Surface_Heating_Index', 'Mean_MLD'});

%% 3. Aggregate per-profile fingerprints to block level.
tempBlockData = join(profileData, profileSummary, 'Keys', 'ProfileID');

blockProfiles = groupsummary(tempBlockData, 'BlockID', 'mean', ...
    {'Mean_SALINITY2', 'Mean_TEMP100', 'Stratification_Index', 'Surface_Heating_Index', 'Mean_MLD'});

blockProfiles.Properties.VariableNames = strrep(blockProfiles.Properties.VariableNames, 'mean_', '');
blockProfiles = removevars(blockProfiles, 'GroupCount');

%% 4. Stratify blocks via k-means on the z-scored fingerprint.
profilingVars = blockProfiles{:, 2:end};

% Mean-fill any NaN columns (blocks with no valid profile).
for col = 1:size(profilingVars, 2)
    colData = profilingVars(:, col);
    isNan = isnan(colData);
    if any(isNan)
        profilingVars(isNan, col) = mean(colData, 'omitnan');
    end
end

profilingVars_z = zscore(profilingVars);

rng(42, 'twister');  % LOCKED seed -- see Docs/REPRODUCIBILITY.md before changing.
stratumIDs = kmeans(profilingVars_z, nFolds, 'MaxIter', 1000, 'Replicates', 10);
blockProfiles.StratumID = stratumIDs;

%% 5. Distribute blocks to nFolds folds via shuffled round-robin (shared helper).
[c, observationFoldIDs] = assignBlocksToFolds(blockProfiles, profileData, profileIdx, nFolds, M, featureTable);
fprintf('--- CV_OceanFingerprint complete. ---\n');

end
