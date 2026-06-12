function [profileData, profileIdx] = buildAtomicBlocks(featureTable, M)
% buildAtomicBlocks - Cosine-weighted k-means atomic geospatial blocks.
%
% Shared by the three environmental CV partitioners
% (CV_OceanFingerprint, CV_EOF, CV_Hybrid). Produces M atomic blocks from
% the unique (LATITUDE, LONGITUDE) profile coordinates by clustering on a
% cosine-pre-scaled longitude axis so Euclidean distance in the scaled
% space approximates physical distance on the sphere.
%
% The inner k-means uses rng(42, 'twister'); this seed is intentionally
% decoupled from config.RandomSeed because changing it would alter every
% environmental fold assignment.
%
% Inputs:
%   featureTable - Black Sea DOXY feature table; must contain LATITUDE
%                  and LONGITUDE columns.
%   M            - Number of atomic blocks (canonical value: 750, set by
%                  getExperimentConfig.nAtomicBlocks).
%
% Outputs:
%   profileData  - Table of unique profiles with LATITUDE, LONGITUDE,
%                  ProfileID, BlockID columns.
%   profileIdx   - Vector mapping each row of featureTable to its
%                  ProfileID (1..numProfiles).

[uniqueCoords, ~, profileIdx] = unique([featureTable.LATITUDE, featureTable.LONGITUDE], 'rows');
numProfiles = size(uniqueCoords, 1);
profileIDs  = (1:numProfiles)';

profileData = table(uniqueCoords(:,1), uniqueCoords(:,2), profileIDs, ...
    'VariableNames', {'LATITUDE', 'LONGITUDE', 'ProfileID'});

rng(42, 'twister');  % LOCKED seed -- see Docs/REPRODUCIBILITY.md before changing.
lat      = profileData.LATITUDE;
lonW     = profileData.LONGITUDE .* cosd(lat);
blockIDs = kmeans([lat, lonW], M, 'MaxIter', 1000, 'Replicates', 10);
profileData.BlockID = blockIDs;

end
