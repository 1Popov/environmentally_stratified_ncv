function runStrategy(strategy, excludeCoords)
% runStrategy — per-strategy orchestration: pyenv reset, Discovery → Tuning → Production.
%
%   runStrategy("RandomCV")                       % canonical 35-feature run
%   runStrategy("RandomCV_Weighted", true)        % 33-feature coord-removal run
%
% Resets the Python interpreter on entry to avoid cross-strategy CUDA /
% LightGBM session state leaking between runs, and drops any stale parallel
% pool so Discovery starts with a fresh one sized by config.ParProc.
%
% Inputs:
%   strategy      Strategy name passed through to Discovery / Tuning /
%                 Production. See Discovery.m for the valid set.
%   excludeCoords (Optional, default false) Coord-removal toggle; when
%                 true the entire pipeline runs on the 33-feature variant
%                 (LAT/LON excluded) and the flag is recorded into
%                 meta.excludeCoords at every stage. Production rejects a
%                 final-model load whose feature count does not match the
%                 flag.
%
% Output:
%   None. Side effects: resets the Python interpreter and parallel pool, then
%   runs Discovery -> Tuning -> Production, each of which writes its stage
%   .mat under Results/ (see Docs/SAVE_SCHEMA.md).
%
% The TEST/SMOKE-TEST toggle is in isTestMode.m (single file edit).

    if nargin < 2, excludeCoords = false; end

    % Explicit repo-root allowlist (NOT addpath(genpath(pwd))) so archived or
    % local helpers cannot shadow live runtime helpers of the same name.
    addpath(pwd);
    try terminate(pyenv); catch, end
    getPythonEnvironment();
    delete(gcp('nocreate'));

    discoveryFile = Discovery(strategy, excludeCoords);
    tuningFile    = Tuning(discoveryFile);
    Production(tuningFile);
end
