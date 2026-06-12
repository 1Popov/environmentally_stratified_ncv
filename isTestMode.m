function tf = isTestMode()
% isTestMode - Global SMOKE-TEST toggle for the entire pipeline.
%
% Returns true  -> smoke-test budget: 15 Optuna trials per outer fold, a 20%
%                  subset of the development set (SubsetSize = 0.20), narrow
%                  learning-rate grid [0.04, 0.05], 500-boost-round ceiling
%                  (early stopping cuts well before it). Discovery -> Tuning ->
%                  Production cycle finishes in tens of minutes per strategy on
%                  a 4070-Ti-class GPU.
% Returns false -> full-budget run: per-strategy Optuna trials (50-150) and
%                  num_boost_round ceiling (300 for the RandomCV family, 5000
%                  for the stratified strategies), log-spaced learning-rate
%                  grid [0.0005, 0.02]. Hours to days per strategy.
%
% TO TOGGLE: edit the single value on the assignment below. No other file
% change is needed. The toggle is consumed by:
%   - getExperimentConfig.m (overrides MaxObjectiveEvaluations /
%     LearningRate / NumBoostRounds when true)
%   - Discovery.m (tags the run as 'quick' vs 'full' in the runId)
%   - buildMetadata.m (records the toggle into meta.TestMode for provenance)

tf = false;   % false = full/paper budget (shipped default); true = fast smoke

end
