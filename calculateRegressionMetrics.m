function metrics = calculateRegressionMetrics(Ytrue_raw, Yhat_raw, Ytrue_norm, Yhat_norm, W, huber_delta)
%calculateRegressionMetrics Computes a standard set of weighted regression metrics.
%
%   Inputs:
%       Ytrue_raw   - Vector of true target values in their original units.
%       Yhat_raw    - Vector of predicted target values in their original
%                     units. Internally clamped at zero before metric
%                     computation (oxygen target is non-negative).
%       Ytrue_norm  - Vector of true target values, normalized.
%       Yhat_norm   - Vector of predicted target values, normalized.
%       W           - Vector of sample weights. Floored at eps internally;
%                     zero-weight rows are NOT dropped — pre-filter at the
%                     caller if exclusion is intended.
%       huber_delta - Scalar Huber-loss threshold used by training. Applied
%                     to scaled-unit residuals only (raw residuals would
%                     mismatch the threshold's scale by orders of magnitude).
%
%   Output:
%       metrics     - A structure containing the calculated metrics:
%                     RMSE_raw, MAE_raw, Bias, R2 (raw units, weighted);
%                     RMSE_scaled, MAE_scaled, Huber_scaled (normalized
%                     units, weighted).

% Ensure raw predictions are non-negative (oxygen target is non-negative)
Yhat_raw = max(Yhat_raw, 0);
% Floor weights at eps to avoid 0/0 in weighted means below. Zero-weight rows
% are NOT dropped — to exclude rows, the caller must pre-filter them out of
% Ytrue_raw / Yhat_raw / W before invoking this helper.
W = max(W, eps);
w_sum = sum(W);

% Raw-unit metrics
metrics.RMSE_raw = sqrt(sum(W .* (Ytrue_raw - Yhat_raw).^2) / w_sum);
metrics.MAE_raw  = sum(W .* abs(Ytrue_raw - Yhat_raw)) / w_sum;
metrics.Bias     = sum(W .* (Yhat_raw - Ytrue_raw)) / w_sum;

% Weighted R²
YbarW = sum(W .* Ytrue_raw) / w_sum;
SSTw  = sum(W .* (Ytrue_raw - YbarW).^2);
if SSTw > 1e-12
    SSEw = metrics.RMSE_raw^2 * w_sum;
    metrics.R2 = 1 - SSEw / SSTw;
else
    metrics.R2 = 0;
end

% Scaled-unit metrics
metrics.RMSE_scaled = rmse(Yhat_norm, Ytrue_norm, 'Weights', W);
metrics.MAE_scaled  = mae(Yhat_norm, Ytrue_norm, 'Weights', W);

% --- Huber Loss Calculation ---
% This calculation uses the specific huber_delta from the model's training
% to ensure consistency between the training objective and the final metric.

% Calculate scaled Huber loss
residual_scaled = abs(Ytrue_norm - Yhat_norm);
loss_vec_scaled = zeros(size(residual_scaled));

is_small_err_scaled = (residual_scaled <= huber_delta);
loss_vec_scaled(is_small_err_scaled) = 0.5 * residual_scaled(is_small_err_scaled).^2;
loss_vec_scaled(~is_small_err_scaled) = huber_delta * (residual_scaled(~is_small_err_scaled) - 0.5 * huber_delta);

metrics.Huber_scaled = sum(W .* loss_vec_scaled) / sum(W);

end