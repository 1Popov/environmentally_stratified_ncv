function c = CV_SeasonalMirror(tbl)
% CV_SeasonalMirror — paired-season inner CV.
% Maps each month m to its mirrored partner via mod(6 - m, 6) + 1, yielding
% six month-pair folds (Jan/Jul, Feb/Aug, Mar/Sep, Apr/Oct, May/Nov, Jun/Dec).
% Empty fold columns are pruned so strategies that subset by season do not
% produce all-zero partitions; a warning fires when fewer than 6 folds
% survive so a downstream consumer notices the reduced inner-CV budget.
%
% Input:
%   tbl - Training table for one outer fold. Must contain a TIME datetime
%         column (the only field read; the month drives the pairing).
%
% Output:
%   c   - cvpartition('CustomPartition', testSets) with up to six month-pair
%         inner folds (fewer if some pairs are unrepresented).
assert(ismember('TIME', tbl.Properties.VariableNames), ...
    'Table must contain a TIME datetime column.');

months = month(tbl.TIME);           % numeric 1-12
idx    = mod(6 - months, 6) + 1;    % Jan/Jul -> fold 6, ..., Jun/Dec -> fold 1

testSetsInner = false(height(tbl), 6);
for k = 1:6
    testSetsInner(:, k) = (idx == k);
end

nNonEmpty = sum(any(testSetsInner, 1));
if nNonEmpty < 6
    warning('CV_SeasonalMirror:UnderfilledFolds', ...
        ['Only %d of 6 seasonal folds have data; %d empty fold(s) pruned. ' ...
         'Inner-CV budget effectively reduced.'], ...
        nNonEmpty, 6 - nNonEmpty);
end
testSetsInner = testSetsInner(:, any(testSetsInner, 1));
c = cvpartition("CustomPartition", testSetsInner);

fprintf('--- CV_SeasonalMirror complete. ---\n');
end