function config = getExperimentConfig(strategy)
% getExperimentConfig — single source of truth for per-strategy pipeline knobs.
% The TEST/SMOKE-TEST toggle is in isTestMode.m (single file edit).
%
% Returned fields:
%   nHoldoutStrata, DepthsPerPoint, nAtomicBlocks, RandomSeed   shared constants
%   ParProc                                          parfor workers
%   MaxObjectiveEvaluations                          Optuna inner-CV trial budget
%   NumBoostRounds                                   lgb.cv ceiling (LR-aware early stopping; per-strategy)
%   SubsetSize, LearningRate                         smoke-mode-overridable knobs
%   nFolds = nHoldoutStrata - 1
%   InnerCVFcn(table) -> cvpartition                                 per-strategy inner-CV partitioner
%   OuterCVFcn(devSet, nFolds, nAtomicBlocks) -> cvpartition         per-strategy outer-CV partitioner
%   WeightFcn(depth) -> column vector of weights                     per-strategy sample weighting

% --- Shared constants ---
config.nHoldoutStrata = 10;
config.DepthsPerPoint = 201;
config.nAtomicBlocks  = 750;
config.RandomSeed     = 42;

% --- Shared full-budget knobs (overridden by isTestMode block below) ---
% The trained models under Models/ are unaffected by these search-budget
% knobs: reproduce_models.m consumes the budget at Discovery time only,
% never at prediction time.
config.ParProc                 = 3;
% Search budget (MaxObjectiveEvaluations) and the num_boost_round ceiling
% (NumBoostRounds) are PER-STRATEGY — set in the switch below. The ceiling is
% load-bearing for the RandomCV family: i.i.d. inner folds give a monotone
% validation curve, so early stopping never fires and the cap alone fixes the
% final tree count (300). The stratified strategies early-stop well below
% their 5000 safety ceiling.

% --- Per-strategy partitioners, weighting, and search budget ---
% The search budget is set per strategy below; the inner-CV trial counts
% differ because each partitioner exposes a different inner-fold structure.
switch string(strategy)
    case "RandomCV"
        config.OuterCVFcn = @(devSet, nFolds, ~) CV_RandomKFold(devSet, nFolds);
        config.InnerCVFcn = @(Ttrain) CV_RandomKFold(Ttrain, 6);
        config.WeightFcn  = @DepthWeightsUniform;
        config.MaxObjectiveEvaluations = 50;
        config.NumBoostRounds          = 300;
    case "RandomCV_Weighted"
        config.OuterCVFcn = @(devSet, nFolds, ~) CV_RandomKFold(devSet, nFolds);
        config.InnerCVFcn = @(Ttrain) CV_RandomKFold(Ttrain, 6);
        config.WeightFcn  = @DepthWeights;
        config.MaxObjectiveEvaluations = 50;
        config.NumBoostRounds          = 300;
    case "GeoClusterCV"
        config.OuterCVFcn = @(devSet, nFolds, ~) CV_GeoCluster(devSet, nFolds);
        config.InnerCVFcn = @CV_SeasonalMirror;
        config.WeightFcn  = @DepthWeights;
        config.MaxObjectiveEvaluations = 80;
        config.NumBoostRounds          = 5000;
    case "OFCV"
        config.OuterCVFcn = @(devSet, nFolds, nAtomicBlocks) CV_OceanFingerprint(devSet, nFolds, nAtomicBlocks);
        config.InnerCVFcn = @CV_SeasonalMirror;
        config.WeightFcn  = @DepthWeights;
        config.MaxObjectiveEvaluations = 120;
        config.NumBoostRounds          = 5000;
    case "EOFCV"
        config.OuterCVFcn = @(devSet, nFolds, nAtomicBlocks) CV_EOF(devSet, nFolds, nAtomicBlocks);
        config.InnerCVFcn = @CV_SeasonalMirror;
        config.WeightFcn  = @DepthWeights;
        config.MaxObjectiveEvaluations = 50;
        config.NumBoostRounds          = 5000;
    case "HybridCV"
        config.OuterCVFcn = @(devSet, nFolds, nAtomicBlocks) CV_Hybrid(devSet, nFolds, nAtomicBlocks);
        config.InnerCVFcn = @CV_SeasonalMirror;
        config.WeightFcn  = @DepthWeights;
        config.MaxObjectiveEvaluations = 150;
        config.NumBoostRounds          = 5000;
    otherwise
        error('getExperimentConfig:UnknownStrategy', ...
              'Unknown strategy "%s". Valid: RandomCV, RandomCV_Weighted, GeoClusterCV, OFCV, EOFCV, HybridCV.', ...
              strategy);
end

% --- Smoke-mode budget overrides ---
if isTestMode()
    config.MaxObjectiveEvaluations = 15;
    config.NumBoostRounds          = 500;
    config.SubsetSize              = 0.20;
    config.LearningRate            = [0.04, 0.05];
else
    config.SubsetSize              = 1.0;
    config.LearningRate            = [0.0005, 0.02];
end

% Off-by-one: nHoldoutStrata partitions are built for the global hold-out
% split; the first partition becomes the quarantined hold-out, the
% remaining nHoldoutStrata - 1 partitions are the outer-CV folds.
config.nFolds = config.nHoldoutStrata - 1;

end
