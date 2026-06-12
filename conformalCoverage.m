function cp = conformalCoverage(oofYhat, oofYtrue, oofDepth, hoYhat, hoYtrue, hoDepth, alpha)
%conformalCoverage Locally-Adaptive Conformal Prediction (LACP) coverage on a hold-out.
%
%   Single source of truth for the conformal-prediction diagnostic, used both
%   by the live pipeline (Production) and by reproduce_models. Calibrates a
%   depth-modulated prediction interval from out-of-fold (OOF) residuals and
%   measures its empirical coverage and mean width on an independent hold-out.
%
%   The scientific question it answers: did the nested-CV scheme produce an
%   HONEST uncertainty estimate? An interval calibrated purely on OOF residuals
%   should cover the independent hold-out at its nominal rate (e.g. 95%). A
%   leaky CV produces optimistically small OOF residuals -> too-narrow
%   intervals -> hold-out UNDER-coverage. An overly pessimistic scheme
%   over-covers. So coverage near nominal is direct evidence of a structurally
%   independent (non-leaking) validation design.
%
%   Method (locally-adaptive):
%     1. abs OOF residuals e_i = |ytrue_i - yhat_i| over the calibration set.
%     2. depth-local error scale sigma(z): residuals sorted by depth, robust
%        moving-median over a window = max(100, 5% of N), plus a beta penalty
%        (10th-percentile residual) to avoid division by ~0.
%     3. normalized non-conformity scores s_i = e_i / sigma(depth_i).
%     4. conformal quantile q at level (n+1)(1-alpha)/n (finite-sample corrected).
%     5. interval half-width at a query depth = q * sigma(depth); coverage =
%        fraction of hold-out points whose |error| <= that half-width.
%
%   Inputs:
%     oofYhat,oofYtrue,oofDepth  OOF prediction / truth / depth over the dev set
%                                (clamped predictions; raw units, µmol kg⁻¹).
%     hoYhat,hoYtrue,hoDepth     Final-model prediction / truth / depth on the
%                                hold-out (raw units).
%     alpha                      (Optional) miscoverage level; default 0.05 (95%).
%
%   Output struct cp:
%     .Alpha, .TargetCoverage              nominal level
%     .nCal                                number of valid OOF calibration points
%     .NormalizedQuantile                  the conformal quantile q
%     .Coverage_Holdout, .AvgWidth_Holdout global hold-out coverage (%) and mean width
%     .DepthBands (nStrata×2), .DepthLabels
%     .Coverage_byDepth, .AvgWidth_byDepth per-stratum coverage (%) and width
%
%   References:
%     Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
%       Random World. Springer. (conformal-prediction framework)
%     Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.
%       (2018). Distribution-Free Predictive Inference for Regression. JASA,
%       113(523), 1094-1111. (split conformal + locally-weighted residuals)
%
%   No randomness, no model training — a deterministic function of its inputs.

if nargin < 7 || isempty(alpha), alpha = 0.05; end

oofYhat=oofYhat(:); oofYtrue=oofYtrue(:); oofDepth=oofDepth(:);
hoYhat=hoYhat(:);   hoYtrue=hoYtrue(:);   hoDepth=hoDepth(:);

%% 1-4. Depth-local LACP calibration (shared single source of truth — conformalCalibrate.m)
cal  = conformalCalibrate(oofYhat, oofYtrue, oofDepth, alpha);
if cal.nCal < 10
    error('conformalCoverage:TooFewOOF', 'Only %d valid OOF points; cannot calibrate.', cal.nCal);
end
nCal = cal.nCal;
q    = cal.NormalizedQuantile;
sigmaInterp = griddedInterpolant(cal.SigmaDepthGrid, cal.SigmaValues, 'linear', 'nearest');

%% 5. Apply on the hold-out: half-width = q * sigma(depth)
hw = q * sigmaInterp(hoDepth);
hoErr = abs(hoYtrue - hoYhat);
covered = hoErr <= hw;

cp = struct();
cp.Alpha             = alpha;
cp.TargetCoverage    = (1 - alpha) * 100;
cp.nCal              = nCal;
cp.NormalizedQuantile= q;
cp.Coverage_Holdout  = mean(covered) * 100;
cp.AvgWidth_Holdout  = mean(2 * hw);

%% Per-depth stratum (canonical source: getDepthStrata.m)
strata = getDepthStrata();
bands  = cell2mat(strata(:,2));
labels = strata(:,1)';
nS = size(bands,1);
covD = nan(nS,1); widD = nan(nS,1);
for s = 1:nS
    m = (hoDepth >= bands(s,1) & hoDepth <= bands(s,2));
    if any(m)
        covD(s) = mean(covered(m)) * 100;
        widD(s) = mean(2 * hw(m));
    end
end
cp.DepthBands       = bands;
cp.DepthLabels      = labels;
cp.Coverage_byDepth = covD;
cp.AvgWidth_byDepth = widD;

% Per-point covered flags + their depths (hold-out order), for a downstream
% profile-level bootstrap CI on the coverage estimate. Purely additive —
% mean(cp.Covered)*100 == cp.Coverage_Holdout by construction.
cp.Covered      = covered;
cp.CoveredDepth = hoDepth;

end
