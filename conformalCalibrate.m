function cal = conformalCalibrate(oofYhat, oofYtrue, oofDepth, alpha)
%conformalCalibrate Depth-local LACP calibration from out-of-fold residuals.
%
%   SINGLE source of truth for the locally-adaptive conformal-prediction (LACP)
%   calibration. Shared by:
%     - conformalCoverage.m  (which then APPLIES it to an independent hold-out), and
%     - Tuning.m             (which STORES it as CP_Quantile for Production to apply).
%   Keeping the calibration in one place guarantees the live pipeline
%   (Tuning -> Production) and the reproduction path (reproduce_models ->
%   conformalCoverage) cannot silently diverge.
%
%   From the valid out-of-fold residuals e_i = |ytrue_i - yhat_i| it computes:
%     - a depth-local error scale sigma(z): residuals sorted by depth, robust
%       moving-median over a window = max(100, 5% of N), plus a beta penalty
%       (10th-percentile residual) to avoid division by ~0;
%     - normalized non-conformity scores s_i = e_i / sigma(depth_i);
%     - the finite-sample-corrected conformal quantile q at level (n+1)(1-alpha)/n.
%
%   Inputs:
%     oofYhat, oofYtrue, oofDepth  OOF prediction / truth / depth over the dev
%                                  set (raw units; predictions clamped upstream).
%     alpha                        (Optional) miscoverage level; default 0.05.
%
%   Output struct cal:
%     .Alpha               nominal miscoverage level
%     .nCal                number of valid OOF calibration points
%     .NormalizedQuantile  the conformal quantile q
%     .SigmaDepthGrid      unique sorted calibration depths (column vector)
%     .SigmaValues         sigma(z) at those depths (column vector) — feed both to
%                          griddedInterpolant(grid, vals, 'linear', 'nearest') to
%                          evaluate sigma at any query depth
%     .WindowSize          movmedian window used
%
%   References:
%     Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
%       Random World. Springer. (conformal-prediction framework)
%     Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.
%       (2018). Distribution-Free Predictive Inference for Regression. JASA,
%       113(523), 1094-1111. (split conformal + locally-weighted residuals)
%
%   No randomness, no model training — a deterministic function of its inputs.

if nargin < 4 || isempty(alpha), alpha = 0.05; end

oofYhat = oofYhat(:); oofYtrue = oofYtrue(:); oofDepth = oofDepth(:);

%% 1. Valid OOF residuals
valid = ~isnan(oofYhat) & ~isnan(oofYtrue);
e     = abs(oofYtrue(valid) - oofYhat(valid));
zc    = oofDepth(valid);
nCal  = numel(e);
if nCal < 2
    error('conformalCalibrate:TooFewOOF', 'Only %d valid OOF points; cannot calibrate.', nCal);
end

%% 2. Depth-local error scale sigma(z)
[zSorted, sIdx] = sort(zc);
eSorted    = e(sIdx);
windowSize = max(100, floor(nCal * 0.05));
sigma_heuristic = smoothdata(eSorted, 'movmedian', windowSize);
beta_penalty = prctile(e, 10);
if beta_penalty == 0, beta_penalty = 1e-4; end
sigma_profile = sigma_heuristic + beta_penalty;

%% 3-4. Normalized scores + finite-sample-corrected conformal quantile
normalized_scores = eSorted ./ sigma_profile;
q_level = min(max((nCal + 1) * (1 - alpha) / nCal, 0), 1);

% Depth -> sigma support (collapse duplicate depths first).
[uz, ui] = unique(zSorted);

cal = struct();
cal.Alpha              = alpha;
cal.nCal               = nCal;
cal.NormalizedQuantile = quantile(normalized_scores, q_level);
cal.SigmaDepthGrid     = uz(:);
cal.SigmaValues        = sigma_profile(ui);
cal.WindowSize         = windowSize;
end
