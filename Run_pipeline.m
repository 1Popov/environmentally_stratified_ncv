%% Run_pipeline.m — orchestrator for the Black Sea DOXY experiments
%
% Six fold-construction strategies (cells 1–6) and two coordinate-removal
% controls (cells 7–8). Each cell calls runStrategy(<name>)
% which resets the Python interpreter and runs Discovery → Tuning → Production.
%
% SMOKE-TEST toggle: edit isTestMode.m (single file). true = smoke (15
% Optuna trials, narrow LR grid [0.04,0.05], 500-round cap, SubsetSize 0.20);
% false = full budget (per-strategy trials/cap from getExperimentConfig.m,
% SubsetSize 1.0). Running the whole script executes all eight cells in order.
%
% First-run setup on an UNREGISTERED machine — uncomment and point at your
% lgbm_env interpreter once before running (env vars survive the per-cell
% `clear`). Registered office/home hosts are auto-detected; see
% getPythonEnvironment.m for the resolution order.
%   setenv('DOXY_PYTHON', 'C:\path\to\envs\lgbm_env\python.exe');

%% 1. Random CV (unweighted) — random-baseline experiment (no depth weighting).
clear; close all;
runStrategy("RandomCV");

%% 2. Random CV (depth-weighted) — random-baseline experiment, oxycline-emphasis weighting.
clear; close all;
runStrategy("RandomCV_Weighted");

%% 3. GeoCluster-CV — geographic-blocking baseline (raw lat/lon k-means).
clear; close all;
runStrategy("GeoClusterCV");

%% 4. OF-CV — scalar physics-informed stratification (five domain-knowledge indices).
clear; close all;
runStrategy("OFCV");

%% 5. EOF-CV — PCA-on-vertical-profiles stratification (data-driven structural modes).
clear; close all;
runStrategy("EOFCV");

%% 6. Hybrid-CV — combined PCA + scalar-indices stratification.
clear; close all;
runStrategy("HybridCV");


%% ============== COORD-REMOVAL CONTROLS (paper Table 3) ==============
% Cells 7–8 rerun the two Random CV strategies with LATITUDE + LONGITUDE
% excluded. The excludeCoords flag (second argument to runStrategy) propagates
% through Discovery / Tuning / Production and switches getWorkingFeatures to the
% 33-feature variant for the entire pipeline. No manual edits required.
% Cells 7 (RandomCV unweighted) and 8 (RandomCV depth-weighted) are the
% supported coordinate-removal pair — both ship as frozen models under Models/.

%% 7. Random CV (unweighted), no Lat/Lon — coord-removal
clear; close all; runStrategy("RandomCV", true);

%% 8. Random CV (depth-weighted), no Lat/Lon — coord-removal
clear; close all; runStrategy("RandomCV_Weighted", true);


