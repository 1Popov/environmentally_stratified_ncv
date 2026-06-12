function [selectedPredictorNames, targetName, cyclicNames] = getWorkingFeatures(featureTable, excludeCoords)
% getWorkingFeatures - Centralized feature, target, and cyclic-name selection.
%
% Returns the canonical 35-predictor working set used by every Discovery
% run in this codebase, the regression target name, and the list of
% cyclic predictors. Cyclic features are already cosine/sine-encoded
% upstream in featureTable (DAY1/DAY2/VELOCITY_DIR_COS/VELOCITY_DIR_SIN);
% rangeNormalizePair.m only ensures they are excluded from min-max scaling.
%
% Selection is index-based on the predictor pool starting at 'TIME' (see
% WorkingFeatureSelection below) so the 35-feature set is recovered
% byte-identically run-to-run.
%
% Inputs:
%   featureTable  - The raw featureTable loaded from DataFiles/featureTable.mat.
%                   The function expects 'TIME' to be the first predictor
%                   column and 'OXYGEN' to be present as the target.
%   excludeCoords - (Optional, default false) When true, returns the
%                   33-feature coord-removal variant (LAT/LON excluded);
%                   when false, the canonical 35-feature set.
%
% Outputs:
%   selectedPredictorNames - String array of 35 predictor names, grouped into
%                            categories: 6 spatiotemporal-context + 3 in-situ
%                            + 18 vertical-structure + 5 dynamics + 3 derived
%                            depth metrics.
%   targetName             - String "OXYGEN" (the BGC-Argo DO target,
%                            µmol kg⁻¹).
%   cyclicNames            - String array of cyclic predictors. Already
%                            cos/sin-encoded in featureTable; the name list
%                            tells rangeNormalizePair to skip them.

if nargin < 2, excludeCoords = false; end

targetName  = "OXYGEN";
cyclicNames = ["DAY1", "DAY2", "VELOCITY_DIR_COS", "VELOCITY_DIR_SIN"];

% --- Build the predictor pool starting at 'TIME' (excludes BuoyName etc.) ---
allPreds = featureTable.Properties.VariableNames( ...
                find(strcmp(featureTable.Properties.VariableNames, 'TIME')) : end);
allPreds = setdiff(allPreds, targetName, 'stable');

% --- Canonical 35-feature working set ---
% Column indices into allPreds. Five feature groups are documented in the
% docstring above. To verify the selected names interactively, evaluate
% allPreds(WorkingFeatureSelection) at the MATLAB prompt with featureTable
% already loaded.
if excludeCoords
    % 33-feature coord-removal variant (LAT/LON excluded).
    WorkingFeatureSelection = [1, 4:30, 33, 36, 49:51];
else
    % canonical 35-feature working set.
    WorkingFeatureSelection = [1:30, 33, 36, 49:51];
end

selectedPredictorNames = allPreds(WorkingFeatureSelection);

% Defensive: verify selection matches either the canonical 35-feature set
% or the 33-feature coordinate-removal variant (no LAT/LON).
expectedFeatures = ["TIME","LATITUDE","LONGITUDE","DEPTH","TEMP","SALINITY","DENSITY", ...
    "TEMP2","TEMP20","TEMP50","TEMP100","TEMP150","TEMP200", ...
    "SALINITY2","SALINITY20","SALINITY50","SALINITY100","SALINITY150","SALINITY200", ...
    "DENSITY2","DENSITY20","DENSITY50","DENSITY100","DENSITY150","DENSITY200", ...
    "DAY1","DAY2","UO","VO","VELOCITY_MAG","SSH","MLD", ...
    "FRACTIONAL_DEPTH","SIGNED_MLD_DIST","RELATIVE_DEPTH"];
expected33Features = setdiff(expectedFeatures, ["LATITUDE","LONGITUDE"], 'stable');

n = numel(selectedPredictorNames);
assert(ismember(n, [33, 35]), ...
    'getWorkingFeatures:wrongCount', ...
    ['Expected 35 (canonical) or 33 (coord-removal variant, no LAT/LON) ' ...
     'features, got %d. Check featureTable column ordering or WorkingFeatureSelection indices.'], n);

if n == 35
    expected = expectedFeatures;
else
    expected = expected33Features;
end
assert(isequal(sort(string(selectedPredictorNames)), sort(expected)), ...
    'getWorkingFeatures:nameMismatch', ...
    'Selected features do not match the expected %d-feature list.\nMissing: %s\nExtra: %s', n, ...
    strjoin(setdiff(expected, string(selectedPredictorNames)), ', '), ...
    strjoin(setdiff(string(selectedPredictorNames), expected), ', '));

end
