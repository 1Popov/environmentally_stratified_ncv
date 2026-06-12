function optVars = getHyperparameterSpace(LearningRate)
% getHyperparameterSpace - Centralized LightGBM hyperparameter search bounds.
%
% Returns the array of optimizableVariable objects defining the Optuna
% inner-CV search space. Bounds are shared identically across all six CV
% strategies configured in getExperimentConfig.m. Holding the search
% space constant isolates the validation-design variable from the
% optimization-budget variable when comparing per-strategy outcomes.
%
% Parameter order is stable so downstream code that iterates over optVars
% (e.g., the dict serialization in Discovery.m) can rely on it.
%
% Input:
%   LearningRate - Two-element vector [min, max] for the LR bound.
%                  Source: getExperimentConfig().LearningRate (smoke-test
%                  mode narrows the grid; full-budget mode uses [0.0005, 0.02]).
%
% Output:
%   optVars - 1x12 array of optimizableVariable objects.
%
% Note: although optimizableVariable is native to MATLAB's bayesopt, the
% Discovery driver serializes this array into the opt_vars_dict Python
% dict consumed by run_bayes_opt_in_python() inside lgbm_wrapper.py. The
% optimizableVariable API is kept here because it documents bounds +
% transforms + integer-ness in one declarative line.

% NOTE: do NOT insert blank lines or comment-only lines between the
% optimizableVariable(...) entries below — MATLAB interprets those as
% row separators inside an array literal and the resulting vertcat will
% fail with a dimension-mismatch error. Keep all 12 entries contiguous.
optVars = [
    optimizableVariable('huber_delta', [0.9, 1.0]), ...
    optimizableVariable('learning_rate', LearningRate, 'Transform', 'log'), ...
    optimizableVariable('num_leaves', [31, 512], 'Type', 'integer', 'Transform', 'log'), ...
    optimizableVariable('max_depth', [10, 50], 'Type', 'integer'), ...
    optimizableVariable('min_child_samples', [5, 50], 'Type', 'integer'), ...
    optimizableVariable('reg_alpha', [1e-3, 20], 'Transform', 'log'), ...
    optimizableVariable('reg_lambda', [1e-3, 20], 'Transform', 'log'), ...
    optimizableVariable('min_sum_hessian_in_leaf', [1e-3, 10], 'Transform', 'log'), ...
    optimizableVariable('min_split_gain', [1e-8, 0.1], 'Transform', 'log'), ...
    optimizableVariable('feature_fraction', [0.6, 0.8]), ...
    optimizableVariable('bagging_fraction', [0.4, 0.9]), ...
    optimizableVariable('bagging_freq', [1, 7], 'Type', 'integer')
    ];

end
