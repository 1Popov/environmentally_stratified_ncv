function ci = computeNadeauBengioCI(perFoldRMSE, alpha)
% computeNadeauBengioCI - 95% CI via Nadeau-Bengio (2003) variance correction.
%
% Equal-split shortcut: n_test/n_train = 1/(K-1), so
%   Var(Ebar) = (1/K + 1/(K-1)) * s^2
% Matches the equal-split assumption that underlies the original
% Nadeau-Bengio (2003) variance correction.
%
% Reference: Nadeau, C. & Bengio, Y. (2003). Inference for the generalization
%            error. Machine Learning, 52, 239-281.
%
% Inputs:
%   perFoldRMSE - Vector of per-fold RMSE values (K folds).
%   alpha       - (Optional) significance level. Default 0.05 (95% CI).
%
% Output:
%   ci - [lower, upper] confidence interval bounds.

if nargin < 2, alpha = 0.05; end

K = numel(perFoldRMSE);
if K < 2
    ci = [NaN, NaN];
    return;
end

ebar   = mean(perFoldRMSE);
s2     = var(perFoldRMSE);
varE   = (1/K + 1/(K-1)) * s2;
tcrit  = tinv(1 - alpha/2, K - 1);
margin = tcrit * sqrt(varE);
ci     = [ebar - margin, ebar + margin];

end
