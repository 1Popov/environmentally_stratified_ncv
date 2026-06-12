function [c, observationFoldIDs] = CV_RandomKFold(featureTable, nFolds)
% CV_RandomKFold - Standard k-fold random partition baseline.
%
% Wraps MATLAB's cvpartition KFold mode to produce nFolds folds drawn
% uniformly at random over the rows of featureTable. Used as the
% random-CV baseline by every Discovery strategy whose OuterCVFcn /
% InnerCVFcn slot points at this function (the two RandomCV variants,
% plus the inner loop for every strategy where seasonal-mirror partitioning
% does not apply).
%
% Inputs:
%   featureTable - Black Sea DOXY feature table; row count drives the
%                  partition size.
%   nFolds       - Number of folds.
%
% Outputs:
%   c                  - cvpartition('KFold', ...) object.
%   observationFoldIDs - Per-row fold ID (1..nFolds), parallel-symmetric
%                        with the other CV_*.m partitioners' second
%                        return.

h = height(featureTable);
c = cvpartition(h, "KFold", nFolds);
observationFoldIDs = zeros(h, 1);
for k = 1:c.NumTestSets
    observationFoldIDs(test(c, k)) = k;
end
fprintf('--- CV_RandomKFold complete. ---\n');
end