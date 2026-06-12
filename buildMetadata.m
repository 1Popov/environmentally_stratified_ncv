function meta = buildMetadata(strategy, stage, TestMode, upstreamFile, excludeCoords)
% buildMetadata - Standardized reproducibility metadata for every .mat artifact.
%
% Called by every Discovery / Tuning / Production driver at save time to
% give downstream consumers complete provenance: which experiment, which
% stage, what config, when it ran, where it ran, what upstream artifact it
% consumed, what code version produced it.
%
% Inputs:
%   strategy     - String name of the CV strategy. One of:
%                    "RandomCV", "RandomCV_Weighted", "GeoClusterCV",
%                    "OFCV", "EOFCV", "HybridCV".
%   stage        - String: "Discovery", "Tuning", or "Production".
%   TestMode - Logical. true = smoke-test budget, false = full-budget.
%   upstreamFile - String path to the .mat file this stage consumed.
%                  Pass '' for Discovery (entry point — no upstream).
%   excludeCoords - (Optional, default false) Whether the run used the
%                  33-feature coord-removal variant (LAT/LON excluded).
%                  Default is the canonical 35-feature set.
%
% Output:
%   meta - Struct with the standardized provenance fields:
%       .timestamp           When this artifact was written (ISO 8601 local).
%       .strategy            Which CV strategy this run is.
%       .stage               Which pipeline stage produced this .mat.
%       .TestMode            Was this a smoke-test or full-budget run?
%       .upstreamFile        Path to the .mat that fed this stage (empty for Discovery).
%       .excludeCoords       true = 33-feature coord-removal variant (LAT/LON excluded).
%       .hostName            COMPUTERNAME of the machine that ran it.
%       .userName            USERNAME at run time.
%       .matlab_version      MATLAB version string.
%       .python_version      Python interpreter version (best effort).
%       .python_executable   Full path to the Python interpreter used.
%       .lightgbm_version    LightGBM version (best effort via pyrun).
%       .optuna_version      Optuna version (best effort via pyrun).
%       .code_git_commit     Repo's HEAD git SHA at save time (best effort).
%       .code_git_dirty      Logical. true if working tree had uncommitted changes.
%
% Best-effort fields fall back to "(unknown)" strings if the underlying
% query fails (e.g. pyenv not configured, not a git repo, system() blocked).
%
% Downstream consumers (where these fields are read back):
%   .strategy       Production.m re-derives getExperimentConfig(meta.strategy)
%                   to recover the run's RandomSeed and WeightFcn.
%   .excludeCoords  Production.m enforces the coord-mode contract (35 canonical
%                   vs 33 coord-removal predictors) before training.
%   .TestMode, .code_git_commit/.code_git_dirty, and the library versions are
%                   provenance only -- surfaced in the saved schema (Docs/SAVE_SCHEMA.md).

if nargin < 5, excludeCoords = false; end

meta = struct();

% --- Run identity ---
meta.timestamp     = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
meta.strategy      = string(strategy);
meta.stage         = string(stage);
meta.TestMode      = logical(TestMode);
meta.upstreamFile  = string(upstreamFile);
meta.excludeCoords = logical(excludeCoords);

% --- Host + user ---
meta.hostName = string(getenv('COMPUTERNAME'));
meta.userName = string(getenv('USERNAME'));

% --- MATLAB version ---
meta.matlab_version = string(version());

% --- Python + ML library versions (best effort) ---
try
    pe = pyenv;
    meta.python_version    = string(pe.Version);
    meta.python_executable = string(pe.Executable);
catch
    meta.python_version    = "(unknown - pyenv inactive)";
    meta.python_executable = "(unknown)";
end

try
    meta.lightgbm_version = string(pyrun("import lightgbm; r = lightgbm.__version__", "r"));
catch
    meta.lightgbm_version = "(unknown)";
end

try
    meta.optuna_version = string(pyrun("import optuna; r = optuna.__version__", "r"));
catch
    meta.optuna_version = "(unknown)";
end

% --- Code version (git HEAD + dirty flag, best effort) ---
try
    [status, gitHash] = system('git rev-parse HEAD');
    if status == 0
        meta.code_git_commit = string(strtrim(gitHash));
    else
        meta.code_git_commit = "(unknown - not a git repo)";
    end
catch
    meta.code_git_commit = "(unknown)";
end

try
    [status, gitStatus] = system('git status --porcelain');
    if status == 0
        meta.code_git_dirty = ~isempty(strtrim(gitStatus));
    else
        meta.code_git_dirty = false;
    end
catch
    meta.code_git_dirty = false;
end

end
