function f = DepthWeightsUniform(x)
% DepthWeightsUniform - Uniform-weight stub for unweighted strategies.
%
% Returns a column vector of ones with the same length as x. Used by
% getExperimentConfig as the WeightFcn for the Random CV (unweighted)
% strategy so the Huber loss falls back to the plain (uniform-weighted)
% form. The variability-driven weighting curve lives in DepthWeights.m
% and is selected by every other strategy.
%
% Input:
%   x - Sample-depth vector. Values are ignored (only numel(x) matters).
%
% Output:
%   f - Column vector of ones, size [numel(x), 1].

f = ones(numel(x), 1);

end