function strata = getDepthStrata()
% getDepthStrata - Canonical depth-stratum definition for the DOXY pipeline.
%
% Single source of truth for the depth strata used in every depth-resolved
% report (Index of Agreement, conformal coverage, depth-binned hold-out error).
% Returned as a 5x2 cell: column 1 = display label, column 2 = [lower, upper]
% depth bounds in metres (inclusive on both ends). Centralised so the live
% pipeline (Production.m) and the reproduction/diagnostic path
% (reproduce_models.m, conformalCoverage.m) cannot drift apart.
%
% Output:
%   strata - 5x2 cell, fixed row order All / Surface / Mid-Layer / Deep /
%            Deeper. Numeric bands: cell2mat(strata(:,2)); labels: strata(:,1)'.
%
% Note: the deepest stratum is 'Deeper (151-200m)'. The frozen
% Models/Model_*.mat carry the older 'Abyssal (151-200m)' label in their cached
% IndexAgreement field; reproduce_models.m recomputes IoA live with these
% labels, so the published values use this definition (see SAVE_SCHEMA.md).

strata = {
    'All Depths (0-200m)',  [0,   200];
    'Surface (0-50m)',      [0,    50];
    'Mid-Layer (51-100m)',  [51,  100];
    'Deep (101-150m)',      [101, 150];
    'Deeper (151-200m)',    [151, 200]
    };

end
