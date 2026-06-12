function [c, observationFoldIDs] = assignBlocksToFolds(blockProfiles, profileData, profileIdx, N, M, featureTable)
% assignBlocksToFolds - Shuffled round-robin distribution of atomic blocks
% to N folds within each environmental stratum, followed by cvpartition
% construction.
%
% Shared by the three environmental CV partitioners
% (CV_OceanFingerprint, CV_EOF, CV_Hybrid). Requires blockProfiles to
% have a StratumID column populated by the caller's upstream k-means
% clustering. Uses a local RandStream('mlfg6331_64', 'Seed', 42) so
% shuffling is deterministic and decoupled from the global generator.
%
% Inputs:
%   blockProfiles      - Table with at least BlockID and StratumID
%                        columns; one row per atomic block.
%   profileData        - Output of buildAtomicBlocks (unique profiles
%                        with BlockID column).
%   profileIdx         - Output of buildAtomicBlocks (row-to-profile
%                        index over featureTable).
%   N                  - Number of folds.
%   M                  - Number of atomic blocks (canonical: 750).
%   featureTable       - The full feature table (needed for fold-mask
%                        construction).
%
% Outputs:
%   c                  - cvpartition('CustomPartition', testSets) object.
%   observationFoldIDs - Per-row fold ID (1..N) over featureTable.

% Deterministic within-stratum shuffle, decoupled from the global stream.
% LOCKED seed/generator -- see Docs/REPRODUCIBILITY.md before changing.
shuffleStream = RandStream('mlfg6331_64', 'Seed', 42);

stratifiedBlockOrder = zeros(size(blockProfiles, 1), 1);
cursor = 0;
for s = 1:N
    stratumBlockIDs  = blockProfiles.BlockID(blockProfiles.StratumID == s);
    shuffledIdx      = randperm(shuffleStream, length(stratumBlockIDs));
    shuffledBlockIDs = stratumBlockIDs(shuffledIdx);
    n_s = length(stratumBlockIDs);
    stratifiedBlockOrder(cursor + (1:n_s)) = shuffledBlockIDs;
    cursor = cursor + n_s;
end

foldAssignments = mod(0:length(stratifiedBlockOrder)-1, N) + 1;

blockFoldIDs                       = zeros(M, 1);
blockFoldIDs(stratifiedBlockOrder) = foldAssignments;

profileBlockIDs    = profileData.BlockID;
profileFoldIDs     = blockFoldIDs(profileBlockIDs);
observationFoldIDs = profileFoldIDs(profileIdx);

testSets = false(height(featureTable), N);
for i = 1:N
    testSets(:, i) = (observationFoldIDs == i);
end
c = cvpartition('CustomPartition', testSets);

end
