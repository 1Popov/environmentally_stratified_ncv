function [c, observationFoldIDs] = CV_GeoCluster(tbl, nFolds)
%CV_GeoCluster - Geographic-block cvpartition from k-means on (lat, lon*cosd(lat)).
%
% Clusters profiles by weighted geographic coordinates (LATITUDE,
% LONGITUDE*cosd(LATITUDE)) into nFolds k-means clusters, assigning one whole
% cluster per outer fold. The cosine weighting makes Euclidean distance in the
% scaled space approximate physical distance on the sphere.
%
% Inputs:
%   tbl    - Black Sea DOXY feature table. Must contain LATITUDE, LONGITUDE.
%   nFolds - Number of outer-CV folds (k-means cluster count).
%
% Outputs:
%   c                  - cvpartition('CustomPartition', testSets) object.
%   observationFoldIDs - Per-row fold ID (1..nFolds), derived from testSets
%                        (not raw k-means labels) so it is robust to whatever
%                        label values k-means assigns; rows in no cluster -> 0.

% Uses the mlfg6331_64 generator with seed 0 to match the kmeans 'Options'
% substream below. The generator family and seed are fixed for
% reproducibility: changing either alters every GeoCluster fold assignment.
% See Docs/REPRODUCIBILITY.md for the full RNG-site inventory.
rng(0,'mlfg6331_64');                               % seed global stream

lat  = tbl.LATITUDE;
lon  = tbl.LONGITUDE;
lonW = lon .* cosd(lat);                            % weight longitude

coords = [lat, lonW];

options = statset('UseParallel', 0, 'UseSubstreams', 1);  % substream-coupled to the locked seed above

[idx, ~] = kmeans(coords, nFolds, 'Replicates', 10, ...
    'Start', 'cluster', 'Distance', 'sqeuclidean', 'Options', options);

uniClust = unique(idx);
testSets = false(height(tbl), nFolds);

for k = 1:nFolds
    testSets(:, k) = idx == uniClust(k);            % one whole cluster per column
end

c = cvpartition("CustomPartition", testSets);
% Per-row fold ID derived from testSets (NOT raw kmeans idx, which would
% conflate label values with fold-position indices if k-means returned
% labels other than 1..nFolds).
[~, observationFoldIDs] = max(testSets, [], 2);
observationFoldIDs(~any(testSets, 2)) = 0;
fprintf('--- CV_GeoCluster complete. ---\n');
end
