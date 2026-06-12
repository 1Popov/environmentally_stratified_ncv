function f = DepthWeights(x)
%DepthWeights Depth-dependent sample weights from the oxygen variability profile.
%   Assigns weights proportional to the standard deviation of oxygen at each
%   depth (from a pre-computed profile), scaled to a user-defined maximum.
%
%   Algorithm:
%     Per-depth O2 variability sigma(z) is loaded from DepthWeights.mat
%     (fields: depths, stdOxy) and mapped linearly to a weight
%         w(z) = 1 + (max_weight - 1) * sigma(z) / max(sigma)
%     with max_weight = 5, so the depth of highest O2 variability is weighted
%     5x the least variable. Depths outside the profile range get weight 1.
%     The interpolant is cached in a persistent (read .mat / build once).
%
%   Input:
%     x - Sample depths (numeric, any shape); only the values are used.
%   Output:
%     f - Column vector of weights, size [numel(x), 1].

% --- Weighting hyperparameter ---
% Maximum weight, applied at the depth of highest oxygen variability; all
% other weights scale linearly from a baseline of 1.0 up to this ceiling.
max_weight = 5.0;

% --- Data Loading (persistent for efficiency) ---
persistent variability_interpolant max_std_dev;

if isempty(variability_interpolant)
    try
        matPath = fullfile(fileparts(mfilename('fullpath')), 'DepthWeights.mat');
        data = load(matPath);
        variability_interpolant = griddedInterpolant(data.depths, data.stdOxy, 'linear', 'none');
        max_std_dev = max(data.stdOxy);
    catch ME
        error('Could not load DepthWeights.mat (expected next to DepthWeights.m). It ships vendored with the repo. Error: %s', ME.message);
    end
end

% --- Weight Calculation ---

% 1. Find the standard deviation for the input depths using the interpolant.
std_at_x = variability_interpolant(x);
std_at_x(isnan(std_at_x)) = 0; % Handle depths outside the data range.

% 2. Define the baseline and the maximum weight to be added.
baseline_weight = 1.0;
max_added_weight = max_weight - baseline_weight;

% 3. Create a scaling factor to map the max standard deviation to the max added weight.
if max_std_dev > 0
    scaling_factor = max_added_weight / max_std_dev;
else
    scaling_factor = 0;
end

% 4. The final weight is the baseline plus the scaled variability.
f = baseline_weight + (scaling_factor * std_at_x);

end