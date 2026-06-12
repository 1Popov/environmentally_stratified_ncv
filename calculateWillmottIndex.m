function d = calculateWillmottIndex(predictions, observations)
%calculateWillmottIndex Calculates Willmott's Index of Agreement (d).
%
%   d = calculateWillmottIndex(predictions, observations) computes the index
%   of agreement, a measure of model prediction accuracy. It varies from
%   1 (perfect match) to 0 (no agreement).
%
%   The formula is:
%   d = 1 - [ sum((P_i - O_i)^2) / sum((|P_i - O_bar| + |O_i - O_bar|)^2) ]
%
%   Inputs:
%       predictions  - A numeric vector of the model's predicted values.
%       observations - A numeric vector of the true, observed values.
%
%   Output:
%       d            - A scalar double representing the Index of Agreement.
%
%   Reference:
%   Willmott, C. J. (1981). On the validation of models. Physical Geography,
%   2(2), 184-194.

%% 1. Input Validation
assert(isequal(size(predictions), size(observations)), ...
    'Input vectors for predictions and observations must be the same size.');

%% 2. Numerator: sum of squared errors
numerator = sum((predictions - observations).^2, 'all', 'omitnan');

%% 3. Denominator: potential error
% Scalar mean across all elements ('all' guards against column-wise means
% when observations is a matrix rather than a vector).
mean_obs = mean(observations, 'all', 'omitnan');

denominator = sum((abs(predictions - mean_obs) + abs(observations - mean_obs)).^2, 'all', 'omitnan');

%% 4. Final index (denominator == 0 => identical observations, perfect fit => d = 1)
if denominator == 0
    d = 1.0;
else
    d = 1 - (numerator / denominator);
end

end