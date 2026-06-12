function [T1_norm, T2_norm, C_out, S_out] = rangeNormalizePair(T1, T2, exclude, C_in, S_in)
%% rangeNormalizePair - Learns or applies min-max normalization
%   This function normalizes specified variables in a primary table (T1) and
%   optionally applies the same transformation to a secondary table (T2).
%
%   It operates in two modes:
%   1. LEARNING MODE: If C_in/S_in are not provided, it learns normalization
%      parameters from T1 ('range' method) and returns them as structures.
%   2. APPLYING MODE: If C_in/S_in are provided as structures, it applies
%      the given parameters to T1 and T2.
%
%   The outputs C_out and S_out are always structures with field names
%   matching the variable names, ensuring robust, name-based access and
%   preventing indexing errors in calling scripts.
%
%   INPUTS:
%       T1       - Primary MATLAB table for learning or applying normalization.
%       T2       - (Optional) Secondary table for applying normalization.
%       exclude  - String array of variable names to exclude from scaling.
%       C_in     - (Optional) Struct of centering parameters to apply.
%       S_in     - (Optional) Struct of scaling parameters to apply.
%
%   OUTPUTS:
%       T1_norm  - Normalized version of T1.
%       T2_norm  - Normalized version of T2.
%       C_out    - Struct of centering parameters (min values).
%       S_out    - Struct of scaling parameters (range values).

%% 1. Identify Variables to Scale
allVars  = T1.Properties.VariableNames;
scaleVars = setdiff(allVars, ["BuoyName", "DEPTH", exclude], 'stable');

%% 2. Learn or Apply Normalization Parameters
% This block determines whether to learn new parameters or apply existing ones
% and prepares calculation-ready vectors `C_vec` and `S_vec`.

if nargin < 4 || isempty(C_in) || isempty(S_in)
    % --- LEARNING MODE ---
    % Learn parameters from the primary table T1
    [T1_scaled, C_vec, S_vec] = normalize(T1{:,scaleVars}, 'range');

    % Convert the learned vectors into structures for robust output
    C_out = struct();
    S_out = struct();
    for i = 1:length(scaleVars)
        C_out.(scaleVars{i}) = C_vec(i);
        S_out.(scaleVars{i}) = S_vec(i);
    end

else
    % --- APPLYING MODE ---
    % Use the provided structures. Pass them through as the function output.
    C_out = C_in;
    S_out = S_in;

    % Convert the input structs to correctly ordered vectors for calculation.
    % This is critical to ensure the right parameters are applied to the right columns.
    C_vec = zeros(1, length(scaleVars));
    S_vec = zeros(1, length(scaleVars));
    for i = 1:length(scaleVars)
        varName = scaleVars{i};
        % Check if the variable exists in the provided struct to avoid errors
        if isfield(C_in, varName) && isfield(S_in, varName)
            C_vec(i) = C_in.(varName);
            S_vec(i) = S_in.(varName);
        else
            error('rangeNormalizePair:MissingField', ...
                'Variable "%s" not found in provided C_in/S_in structs.', varName);
        end
    end

    % Apply normalization using the derived vectors
    T1_scaled = (T1{:,scaleVars} - C_vec) ./ S_vec;
end

%% 3. Populate Output Tables
% Assign the scaled data back to the primary output table
T1_norm = T1;
T1_norm{:,scaleVars} = T1_scaled;

% If a second table is provided, apply the same vector-based transformation
T2_norm = [];
if ~isempty(T2)
    TtestScaled = (T2{:,scaleVars} - C_vec) ./ S_vec;
    T2_norm = T2;
    T2_norm{:,scaleVars} = TtestScaled;
end

end