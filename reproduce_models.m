function reproduce_models()
% reproduce_models — single ground-truth source for every documented metric.
%
% Recomputes all hold-out metrics live from the shipped models and reports the
% stored nested-CV (NCV) artefacts, so that every RESULT number in the paper
% (except the spatial-leakage / adversarial-validation diagnostics, not part of
% this release) can be read from one place, computed from real stored .mat data — never from
% a text log, and never re-training.
%
% For each frozen Models/Model_*.mat:
%   HOLD-OUT (recomputed live, both weightings) via calculateRegressionMetrics
%   / calculateWillmottIndex from FinalModelString on the shared hold-out:
%     RMSE, MAE, Bias, R^2 (unweighted + depth-weighted) and Index of
%     Agreement by depth stratum.
%   NESTED-CV (stored original-run ground truth; a final model cannot regenerate
%   its own outer folds) from PerformanceSummary_NCV: per-fold and mean+-SD of
%   RMSE / MAE / Bias / R^2. The 95% confidence interval is derived LIVE via
%   the Nadeau-Bengio variance correction (computeNadeauBengioCI) — the simple
%   t/sqrt(N) interval is not used.
%   HYPERPARAMETERS from NCV_FoldHyperparams (per-fold tuned values) +
%   FinalHyperparameters (final model): per-parameter Mean and CV% across the
%   K outer folds, and the final-model configuration.
%   FEATURE IMPORTANCE from NCV_FoldFeatureImportance (per-fold raw gain) +
%   FinalFeatureImportance: per-feature NCV mean% (normalised) and CV% across
%   folds, plus the final-model normalised importance.
%
% Generalization gaps and inflation pair NCV with the hold-out at the SAME
% per-strategy training weight (depth for the stratified strategies and the
% two *_Weighted variants, unweighted for plain RandomCV / RandomCV_NoLatLon),
% so the comparison is like-with-like.
%
% Tables print with rows in pipeline/paper order; the two coordinate-removal
% control experiments appear in a separate block. No file writes.

% Explicit subdirectory allowlist (NOT addpath(genpath(pwd))) so archived or
% local helpers cannot shadow live runtime helpers. Matches runStrategy.m.
% Resolve everything from the script location, not the caller's pwd, so the
% verifier runs from any working directory.
repoRoot = fileparts(mfilename('fullpath'));
addpath(repoRoot);
% First-run setup on an unregistered machine — uncomment and set to your lgbm_env
% interpreter (generic template; keep your real path out of the commit). Registered
% office/home hosts are auto-detected; see getPythonEnvironment.m.
%   setenv('DOXY_PYTHON', 'C:\path\to\envs\lgbm_env\python.exe');
try terminate(pyenv); catch, end
getPythonEnvironment();
% Make lgbm_wrapper.py (at the repo root) importable by py.* regardless of pwd.
if count(py.sys.path, repoRoot) == 0
    insert(py.sys.path, int32(0), repoRoot);
end

% Capture the full report with evalc so it can be echoed to the console AND
% written verbatim to a ground-truth text file. (diary is unreliable when this
% function is itself run under output capture; evalc is not.)
reportText = evalc('runReproReport(repoRoot)');
fprintf('%s', reportText);
groundTruthFile = fullfile(repoRoot, 'Ground_Truth_Results.txt');
fid = fopen(groundTruthFile, 'w', 'n', 'UTF-8');
if fid == -1, error('reproduce_models:Export', 'Cannot open %s for writing.', groundTruthFile); end
fprintf(fid, '%s', reportText);
fclose(fid);
fprintf('All results exported to %s\n', groundTruthFile);
end

function runReproReport(repoRoot)
% Builds the full ground-truth report. Prints to the console; the caller
% captures this output via evalc and also writes it to Ground_Truth_Results.txt.
% repoRoot anchors Models/ and DataFiles/ so the report is pwd-independent.

modelsDir  = fullfile(repoRoot, 'Models');
holdoutMat = fullfile(modelsDir, 'holdoutSet.mat');
if ~isfile(holdoutMat)
    error('Missing %s. Expected vendored with the repo under Models/.', holdoutMat);
end
H = load(holdoutMat);
holdoutSet = H.holdoutSet;

modelFiles = dir(fullfile(modelsDir, 'Model_*.mat'));
if isempty(modelFiles)
    error('No Model_*.mat artefacts found in %s', modelsDir);
end

strata     = getDepthStrata();              % canonical depth strata (getDepthStrata.m)
depthBands = cell2mat(strata(:,2));
nStrata    = size(depthBands, 1);

N = numel(modelFiles);
strat    = strings(N, 1);
isDepth  = false(N, 1);
RMSE_unw = zeros(N,1); RMSE_dw = zeros(N,1);
MAE_unw  = zeros(N,1); MAE_dw  = zeros(N,1);
Bias_unw = zeros(N,1); Bias_dw = zeros(N,1);
R2_unw   = zeros(N,1); R2_dw   = zeros(N,1);
IoA      = zeros(N,nStrata);
RMSE_byDepth = nan(N,nStrata); MAE_byDepth = nan(N,nStrata); Bias_byDepth = nan(N,nStrata);
pct10 = nan(N,1); pct20 = nan(N,1);
ncvRMSE_m=nan(N,1); ncvRMSE_s=nan(N,1);
ncvMAE_m =nan(N,1); ncvMAE_s =nan(N,1);
ncvBias_m=nan(N,1); ncvR2_m  =nan(N,1);
nbLo=nan(N,1); nbHi=nan(N,1);
foldRMSE = nan(N,9);
foldBias = nan(N,9); biasMean=nan(N,1); biasSD=nan(N,1); biasCV=nan(N,1);
foldLR = nan(N,9); featCount=nan(N,1);
maxdelta = zeros(N,1); verdict = strings(N,1);
% per-model cells for variable-shape artefacts
hpNamesC={}; foldHPc=cell(N,1); finalHPc=cell(N,1);
featNamesC=cell(N,1); foldFIc=cell(N,1); finalFIc=cell(N,1);
cpC=cell(N,1);
tol=1e-3; nPass=0; nFail=0;

for k = 1:N
    M = load(fullfile(modelFiles(k).folder, modelFiles(k).name));
    targetField = char(M.targetName);
    featCount(k) = numel(M.recommendedPredictorNames);

    [TholdoutN, ~, ~, ~] = rangeNormalizePair(holdoutSet, [], M.cyclicNames, M.C_final, M.S_final);
    XholdoutN = TholdoutN{:, M.recommendedPredictorNames};
    YtrueN    = TholdoutN{:, targetField};

    Yhat_N_py = py.lgbm_wrapper.predict_from_model(M.FinalModelString, XholdoutN);
    Yhat_N    = double(Yhat_N_py); Yhat_N = Yhat_N(:);
    Yhat_raw  = Yhat_N .* M.S_final.(targetField) + M.C_final.(targetField);  % unclamped; helper clamps
    Ytrue_raw = holdoutSet.(targetField);
    if istable(M.FinalHyperparameters) && any(strcmp('huber_delta', M.FinalHyperparameters.Properties.VariableNames))
        huber_delta = M.FinalHyperparameters.huber_delta;
    else
        huber_delta = 1.0;
    end

    m_unw = calculateRegressionMetrics(Ytrue_raw, Yhat_raw, YtrueN, Yhat_N, DepthWeightsUniform(holdoutSet.DEPTH), huber_delta);
    m_dw  = calculateRegressionMetrics(Ytrue_raw, Yhat_raw, YtrueN, Yhat_N, DepthWeights(holdoutSet.DEPTH),        huber_delta);

    strat(k)=string(M.strategy);
    isDepth(k) = ~ismember(strat(k), ["RandomCV","RandomCV_NoLatLon"]);
    if isfield(M,'weightFn')
        storedDepth = strcmpi(char(M.weightFn),'DepthWeights');
        if storedDepth ~= isDepth(k)
            warning('reproduce_models:WeightFnProvenance', ...
                'Model %s stored weightFn="%s" contradicts strategy-derived weight; using strategy weight.', ...
                strat(k), char(M.weightFn));
        end
    end
    RMSE_unw(k)=m_unw.RMSE_raw; RMSE_dw(k)=m_dw.RMSE_raw;
    MAE_unw(k)=m_unw.MAE_raw;   MAE_dw(k)=m_dw.MAE_raw;
    Bias_unw(k)=m_unw.Bias;     Bias_dw(k)=m_dw.Bias;
    R2_unw(k)=m_unw.R2;         R2_dw(k)=m_dw.R2;

    for s=1:nStrata
        mask=(holdoutSet.DEPTH>=depthBands(s,1) & holdoutSet.DEPTH<=depthBands(s,2));
        IoA(k,s)=calculateWillmottIndex(Yhat_raw(mask), Ytrue_raw(mask));
        % Depth-resolved hold-out error (unweighted) — restrict the same metric
        % helper to each depth band. Reports RMSE/MAE/Bias by layer, complementing
        % the depth-resolved Index of Agreement the paper already has.
        md = calculateRegressionMetrics(Ytrue_raw(mask), Yhat_raw(mask), YtrueN(mask), Yhat_N(mask), ...
                                        DepthWeightsUniform(holdoutSet.DEPTH(mask)), huber_delta);
        RMSE_byDepth(k,s)=md.RMSE_raw; MAE_byDepth(k,s)=md.MAE_raw; Bias_byDepth(k,s)=md.Bias;
    end

    % Within-tolerance accuracy on the hold-out (clamped predictions, raw units).
    absErr = abs(Ytrue_raw - max(Yhat_raw,0));
    pct10(k) = mean(absErr <= 10) * 100;
    pct20(k) = mean(absErr <= 20) * 100;

    if isfield(M,'PerformanceSummary_NCV')
        ncv=M.PerformanceSummary_NCV;
        ncvRMSE_m(k)=ncv.RMSE_raw.mean; ncvRMSE_s(k)=ncv.RMSE_raw.std;
        ncvMAE_m(k)=ncv.MAE_raw.mean;   ncvMAE_s(k)=ncv.MAE_raw.std;
        ncvBias_m(k)=ncv.Bias.mean;
        if isfield(ncv,'R2'), ncvR2_m(k)=ncv.R2.mean; end
        v=ncv.RMSE_raw.values(:)'; foldRMSE(k,1:numel(v))=v;
        ci=computeNadeauBengioCI(ncv.RMSE_raw.values); nbLo(k)=ci(1); nbHi(k)=ci(2);
        if isfield(ncv,'Bias') && isfield(ncv.Bias,'values')
            bv=ncv.Bias.values(:)'; foldBias(k,1:numel(bv))=bv;
            biasMean(k)=mean(bv,'omitnan'); biasSD(k)=std(bv,0,'omitnan');
            if abs(biasMean(k))>0, biasCV(k)=100*biasSD(k)/abs(biasMean(k)); end
        end
    end

    if isfield(M,'NCV_FoldHyperparams')
        foldHPc{k}=M.NCV_FoldHyperparams; hpNamesC=M.NCV_HyperparamNames;
        lrc=find(string(M.NCV_HyperparamNames)=="learning_rate",1);
        if ~isempty(lrc), lv=M.NCV_FoldHyperparams(:,lrc)'; foldLR(k,1:numel(lv))=lv; end
    end
    if isfield(M,'FinalHyperparameters'), finalHPc{k}=M.FinalHyperparameters; end
    if isfield(M,'NCV_FoldFeatureImportance')
        foldFIc{k}=M.NCV_FoldFeatureImportance; featNamesC{k}=M.FeatureNames;
    end
    if isfield(M,'FinalFeatureImportance'), finalFIc{k}=M.FinalFeatureImportance; end

    % Conformal coverage (LACP) — computed only where the OOF calibration set is
    % stored on the model file (OOF_Yhat/Ytrue/Depth). The coordinate-removal
    % controls carry no OOF and are reported as "—".
    if isfield(M,'OOF_Yhat') && isfield(M,'OOF_Ytrue') && isfield(M,'OOF_Depth')
        cpC{k} = conformalCoverage(M.OOF_Yhat, M.OOF_Ytrue, M.OOF_Depth, ...
                                   Yhat_raw, Ytrue_raw, holdoutSet.DEPTH);
    end

    HP=M.HoldoutPerformance;
    if ~isfield(HP,'RMSE_raw_unweighted')||~isfield(HP,'RMSE_raw_depthweighted')
        error('Model %s lacks the dual-weighting hold-out fields (RMSE_raw_unweighted / RMSE_raw_depthweighted) expected on the shipped models.', char(M.strategy));
    end
    maxdelta(k)=max(abs(RMSE_unw(k)-HP.RMSE_raw_unweighted), abs(RMSE_dw(k)-HP.RMSE_raw_depthweighted));
    if maxdelta(k)<tol, verdict(k)="PASS"; nPass=nPass+1; else, verdict(k)="FAIL"; nFail=nFail+1; end
end

% Hold-out metrics at the per-strategy TRAINING weight (basis for the gaps).
RMSE_tr=RMSE_unw; RMSE_tr(isDepth)=RMSE_dw(isDepth);
MAE_tr =MAE_unw;  MAE_tr(isDepth)=MAE_dw(isDepth);
R2_tr  =R2_unw;   R2_tr(isDepth)=R2_dw(isDepth);
relGap=(RMSE_tr-ncvRMSE_m)./ncvRMSE_m*100;
infl  =RMSE_tr./ncvRMSE_m;
inCI  =RMSE_tr>=nbLo & RMSE_tr<=nbHi;
maeGap=(MAE_tr-ncvMAE_m)./ncvMAE_m*100;
r2Gap =(R2_tr-ncvR2_m)./ncvR2_m*100;

mainOrder=["RandomCV","RandomCV_Weighted","GeoClusterCV","OFCV","EOFCV","HybridCV"];
ctrlOrder=["RandomCV_NoLatLon","RandomCV_Weighted_NoLatLon"];
mainIdx=orderIndices(strat,mainOrder); ctrlIdx=orderIndices(strat,ctrlOrder);
allIdx=[mainIdx;ctrlIdx];
unl=setdiff((1:N)',allIdx);
if ~isempty(unl)
    warning('reproduce_models:UnlistedStrategy','Models outside canonical order omitted: %s', strjoin(strat(unl),', '));
end

%% 1. Reproduction check — recomputed vs stored hold-out RMSE
b1='+------------------------------+-----------------+---------------+----------+---------+';
fprintf('\n=== 1. REPRODUCTION CHECK — recomputed vs stored hold-out RMSE (µmol kg⁻¹) ===\n');
fprintf('%s\n',b1);
fprintf('| %-28s | %-15s | %-13s | %-8s | %-7s |\n','Strategy','RMSE unweighted','RMSE weighted','max|d|','Verdict');
fprintf('%s\n',b1);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s | %15.4f | %13.4f | %8.1e | %-7s |\n',strat(k),RMSE_unw(k),RMSE_dw(k),maxdelta(k),verdict(k)); end
fprintf('%s\n',b1);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s | %15.4f | %13.4f | %8.1e | %-7s |\n',strat(k),RMSE_unw(k),RMSE_dw(k),maxdelta(k),verdict(k)); end
fprintf('%s\n',b1);

%% 2. NCV performance — stored; NB 95% CI live
b2='+------------------------------+-----------------+-----------------+---------+--------+--------------------+';
fprintf('\n=== 2. NESTED-CV PERFORMANCE — stored mean±SD; NB 95%% CI live (µmol kg⁻¹) ===\n');
fprintf('%s\n',b2);
fprintf('| %-28s | %-15s | %-15s | %-7s | %-6s | %-18s |\n','Strategy','RMSE (mean±SD)','MAE (mean±SD)','Bias','R²','NB 95% CI (RMSE)');
fprintf('%s\n',b2);
printNCV(mainIdx,strat,ncvRMSE_m,ncvRMSE_s,ncvMAE_m,ncvMAE_s,ncvBias_m,ncvR2_m,nbLo,nbHi);
fprintf('%s\n',b2);
printNCV(ctrlIdx,strat,ncvRMSE_m,ncvRMSE_s,ncvMAE_m,ncvMAE_s,ncvBias_m,ncvR2_m,nbLo,nbHi);
fprintf('%s\n',b2);
fprintf('  (Bias and R² are fold means; NB 95%% CI is on the NCV mean RMSE.)\n');

%% 3-4. Hold-out metrics
b3='+------------------------------+----------+---------+----------+--------+';
fprintf('\n=== 3. HOLD-OUT METRICS — UNWEIGHTED (recomputed, µmol kg⁻¹) ===\n');
printHoldout(b3,mainIdx,ctrlIdx,strat,RMSE_unw,MAE_unw,Bias_unw,R2_unw);
fprintf('\n=== 4. HOLD-OUT METRICS — DEPTH-WEIGHTED (recomputed, µmol kg⁻¹) ===\n');
printHoldout(b3,mainIdx,ctrlIdx,strat,RMSE_dw,MAE_dw,Bias_dw,R2_dw);

%% 5. Reliability diagnostics (RMSE) at training weight
b5='+------------------------------+------------+----------+-----------+---------+--------+-------+';
fprintf('\n=== 5. RELIABILITY DIAGNOSTICS — NCV vs hold-out RMSE at the per-strategy training weight ===\n');
fprintf('%s\n',b5);
fprintf('| %-28s | %-10s | %-8s | %-9s | %-7s | %-6s | %-5s |\n','Strategy','Weight','NCV RMSE','Hold RMSE','Gap','Infl','inCI');
fprintf('%s\n',b5);
printRel(mainIdx,strat,isDepth,ncvRMSE_m,RMSE_tr,relGap,infl,inCI);
fprintf('%s\n',b5);
printRel(ctrlIdx,strat,isDepth,ncvRMSE_m,RMSE_tr,relGap,infl,inCI);
fprintf('%s\n',b5);
fprintf('  Weight=per-strategy training weight; Gap=relative generalization gap (%%); Infl=hold/NCV; inCI=hold inside NB 95%% CI.\n');

%% 6. Generalization gap by metric (RMSE / MAE / R²) at training weight
b6='+------------------------------+----------+----------+----------+';
fprintf('\n=== 6. GENERALIZATION GAP BY METRIC — at the per-strategy training weight (%%) ===\n');
fprintf('%s\n',b6);
fprintf('| %-28s | %-8s | %-8s | %-8s |\n','Strategy','RMSE gap','MAE gap','R² gap');
fprintf('%s\n',b6);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s | %+7.1f%% | %+7.1f%% | %+7.2f%% |\n',strat(k),relGap(k),maeGap(k),r2Gap(k)); end
fprintf('%s\n',b6);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s | %+7.1f%% | %+7.1f%% | %+7.2f%% |\n',strat(k),relGap(k),maeGap(k),r2Gap(k)); end
fprintf('%s\n',b6);

%% 7. R² generalization detail (paper Table r2_generalization)
b7='+------------------------------+----------+---------------+-------------+----------+';
fprintf('\n=== 7. R² GENERALIZATION — NCV vs hold-out (both weightings) ===\n');
fprintf('%s\n',b7);
fprintf('| %-28s | %-8s | %-13s | %-11s | %-8s |\n','Strategy','NCV R²','HO unweighted','HO weighted','gap(tr)%');
fprintf('%s\n',b7);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s | %8.4f | %13.4f | %11.4f | %+7.2f%% |\n',strat(k),ncvR2_m(k),R2_unw(k),R2_dw(k),r2Gap(k)); end
fprintf('%s\n',b7);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s | %8.4f | %13.4f | %11.4f | %+7.2f%% |\n',strat(k),ncvR2_m(k),R2_unw(k),R2_dw(k),r2Gap(k)); end
fprintf('%s\n',b7);
fprintf('  gap(tr) pairs NCV R² with the hold-out R² at the training weight.\n');

%% 8. Per-fold NCV RMSE (K=9 outer folds)
b8='+------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+';
fprintf('\n=== 8. PER-FOLD NCV RMSE — K=9 outer folds (stored, µmol kg⁻¹) ===\n');
fprintf('%s\n',b8);
fprintf('| %-28s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s |\n','Strategy','F1','F2','F3','F4','F5','F6','F7','F8','F9');
fprintf('%s\n',b8);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s |%s\n',strat(k),sprintf(' %7.4f |',foldRMSE(k,:))); end
fprintf('%s\n',b8);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s |%s\n',strat(k),sprintf(' %7.4f |',foldRMSE(k,:))); end
fprintf('%s\n',b8);

%% 9. Index of Agreement by depth
b9='+------------------------------+---------+---------+---------+---------+---------+';
fprintf('\n=== 9. INDEX OF AGREEMENT (d) BY DEPTH — HOLD-OUT ===\n');
fprintf('%s\n',b9);
fprintf('| %-28s | %-7s | %-7s | %-7s | %-7s | %-7s |\n','Strategy','All','0-50','51-100','101-150','151-200');
fprintf('%s\n',b9);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f |\n',strat(k),IoA(k,1),IoA(k,2),IoA(k,3),IoA(k,4),IoA(k,5)); end
fprintf('%s\n',b9);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f |\n',strat(k),IoA(k,1),IoA(k,2),IoA(k,3),IoA(k,4),IoA(k,5)); end
fprintf('%s\n',b9);

%% 9b. Conformal-prediction coverage (LACP) — honesty of the uncertainty estimate
fprintf('\n=== 9b. CONFORMAL COVERAGE — 95%% LACP interval, OOF-calibrated, on the hold-out ===\n');
fprintf('  An interval calibrated purely on out-of-fold residuals should cover the independent\n');
fprintf('  hold-out at its 95%% nominal rate. Coverage near 95%% = structurally honest CV (no\n');
fprintf('  leakage); under-coverage flags optimism / leakage. Per strategy, below: global +\n');
fprintf('  depth-resolved coverage and mean interval width (calibration N = valid OOF dev points).\n');
for t=1:numel(mainIdx)
    k=mainIdx(t);
    printConformalBlock(strat(k),cpC{k});
end
% Coordinate-removal controls carry no stored OOF calibration set — no interval to report.
fprintf('\n  Controls without OOF calibration (no conformal interval): %s.\n', ...
    strjoin(cellstr(strat(ctrlIdx)),', '));

%% 9c. Depth-resolved hold-out error (RMSE / MAE by depth stratum)
bD='+------------------------------+---------+---------+---------+---------+---------+';
fprintf('\n=== 9c. HOLD-OUT RMSE BY DEPTH (unweighted, µmol kg⁻¹) ===\n');
fprintf('%s\n',bD);
fprintf('| %-28s | %-7s | %-7s | %-7s | %-7s | %-7s |\n','Strategy','All','0-50','51-100','101-150','151-200');
fprintf('%s\n',bD);
printByDepth(mainIdx,strat,RMSE_byDepth);
fprintf('%s\n',bD);
printByDepth(ctrlIdx,strat,RMSE_byDepth);
fprintf('%s\n',bD);
fprintf('\n=== 9d. HOLD-OUT MAE BY DEPTH (unweighted, µmol kg⁻¹) ===\n');
fprintf('%s\n',bD);
fprintf('| %-28s | %-7s | %-7s | %-7s | %-7s | %-7s |\n','Strategy','All','0-50','51-100','101-150','151-200');
fprintf('%s\n',bD);
printByDepth(mainIdx,strat,MAE_byDepth);
fprintf('%s\n',bD);
printByDepth(ctrlIdx,strat,MAE_byDepth);
fprintf('%s\n',bD);

%% 9e. Within-tolerance accuracy + cross-fold dispersion
bE='+------------------------------+-----------+-----------+-----------+------------+-----------+';
fprintf('\n=== 9e. WITHIN-TOLERANCE ACCURACY (hold-out) + CROSS-FOLD NCV DISPERSION ===\n');
fprintf('%s\n',bE);
fprintf('| %-28s | %-9s | %-9s | %-9s | %-10s | %-9s |\n', ...
    'Strategy','within±10','within±20','NCV mean','worst fold','fold CV%');
fprintf('%s\n',bE);
printAccDisp(mainIdx,strat,pct10,pct20,foldRMSE);
fprintf('%s\n',bE);
printAccDisp(ctrlIdx,strat,pct10,pct20,foldRMSE);
fprintf('%s\n',bE);
fprintf('  within±10/±20 = %% of hold-out points within that absolute error. worst fold = max per-fold NCV\n');
fprintf('  RMSE; fold CV%% = 100·SD/mean across the 9 outer folds.\n');

%% 10-11. Hyperparameters (params x strategies)
abbr = containers.Map(...
  {'RandomCV','RandomCV_Weighted','GeoClusterCV','OFCV','EOFCV','HybridCV','RandomCV_NoLatLon','RandomCV_Weighted_NoLatLon'}, ...
  {'RCV','RCVw','Geo','OF','EOF','Hyb','RCVn','RCVwn'});
if ~isempty(hpNamesC)
    cols=arrayfun(@(k) abbr(char(strat(k))), allIdx, 'UniformOutput',false);
    % stability: mean (cv%) across folds
    fprintf('\n=== 10. HYPERPARAMETER STABILITY — Mean across K=9 folds (CV%% in parentheses) ===\n');
    printHyperMatrix(hpNamesC, allIdx, foldHPc, cols, true);
    fprintf('\n=== 11. FINAL-MODEL HYPERPARAMETERS — full development-set models ===\n');
    printFinalHP(hpNamesC, allIdx, finalHPc, cols);
end

%% 12. Feature importance (per strategy: NCV mean% + CV% + final%)
fprintf('\n=== 12. FEATURE IMPORTANCE — NCV mean%% / CV%% across folds + final-model %% (gain, normalised) ===\n');
for t=1:numel(allIdx)
    k=allIdx(t);
    if isempty(foldFIc{k}) || isempty(finalFIc{k}), continue; end
    printFeatureImportance(strat(k), featNamesC{k}, foldFIc{k}, finalFIc{k});
end

%% 8b. Per-fold NCV bias (K=9 outer folds)
b8b='+------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+';
fprintf('\n=== 8b. PER-FOLD NCV BIAS — K=9 outer folds (stored, µmol kg⁻¹) ===\n');
fprintf('%s\n',b8b);
fprintf('| %-28s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s |\n','Strategy','F1','F2','F3','F4','F5','F6','F7','F8','F9');
fprintf('%s\n',b8b);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s |%s\n',strat(k),sprintf(' %+7.3f |',foldBias(k,:))); end
fprintf('%s\n',b8b);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s |%s\n',strat(k),sprintf(' %+7.3f |',foldBias(k,:))); end
fprintf('%s\n',b8b);

%% 8c. Cross-fold NCV bias dispersion (mean / SD / CV%)
bBD='+------------------------------+-----------+-----------+--------------+';
fprintf('\n=== 8c. CROSS-FOLD NCV BIAS DISPERSION ===\n');
fprintf('%s\n',bBD);
fprintf('| %-28s | %-9s | %-9s | %-12s |\n','Strategy','mean','SD','CV%');
fprintf('%s\n',bBD);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s | %+9.4f | %9.4f | %11.1f%% |\n',strat(k),biasMean(k),biasSD(k),biasCV(k)); end
fprintf('%s\n',bBD);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s | %+9.4f | %9.4f | %11.1f%% |\n',strat(k),biasMean(k),biasSD(k),biasCV(k)); end
fprintf('%s\n',bBD);
fprintf('  CV%% = 100·SD/|mean|.\n');

%% 10b. Per-fold learning rate (K=9 outer folds)
b10b='+------------------------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+';
fprintf('\n=== 10b. PER-FOLD LEARNING RATE — K=9 outer folds (stored) ===\n');
fprintf('%s\n',b10b);
fprintf('| %-28s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s |\n','Strategy','F1','F2','F3','F4','F5','F6','F7','F8','F9');
fprintf('%s\n',b10b);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s |%s\n',strat(k),sprintf(' %7.4f |',foldLR(k,:))); end
fprintf('%s\n',b10b);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s |%s\n',strat(k),sprintf(' %7.4f |',foldLR(k,:))); end
fprintf('%s\n',b10b);

%% Feature counts (predictor-set size per strategy)
fprintf('\n=== FEATURE COUNT — numel(recommendedPredictorNames) per strategy ===\n');
for t=1:numel(allIdx), k=allIdx(t); fprintf('  %-28s %d features\n',strat(k),featCount(k)); end
fprintf('  Full set = 35; coordinate-removal controls = 33 (LATITUDE/LONGITUDE excluded).\n');

%% 13. Dataset & design descriptors (reproduced from the shipped featureTable)
fprintf('\n=== 13. DATASET & DESIGN DESCRIPTORS (from DataFiles/featureTable.mat) ===\n');
dataFile = fullfile(repoRoot,'DataFiles','featureTable.mat');
if ~isfile(dataFile)
    fprintf('  DataFiles/featureTable.mat not found — dataset descriptors skipped.\n');
else
    DT = load(dataFile,'featureTable'); FT = DT.featureTable;
    nObs      = height(FT);
    nProfiles = size(unique([FT.LATITUDE, FT.LONGITUDE],'rows'),1);
    nFloats   = numel(unique(FT.BuoyName));
    % Consecutive along-track profile separation per float (haversine); 10th/25th pct.
    buoys = unique(FT.BuoyName); allKm = [];
    for b = 1:numel(buoys)
        bt = FT(strcmp(FT.BuoyName, buoys{b}), :);
        [~,ia] = unique(bt{:,{'LATITUDE','LONGITUDE'}},'rows','first');
        pd = sortrows(bt(ia,{'LATITUDE','LONGITUDE','TIME'}),'TIME');
        if height(pd) < 2, continue; end
        la = pd.LATITUDE; lo = pd.LONGITUDE;
        dLa = deg2rad(diff(la)); dLo = deg2rad(diff(lo));
        aa = sin(dLa/2).^2 + cos(deg2rad(la(1:end-1))).*cos(deg2rad(la(2:end))).*sin(dLo/2).^2;
        allKm = [allKm; 6371*2*atan2(sqrt(aa),sqrt(1-aa))]; %#ok<AGROW>
    end
    pcts = prctile(allKm,[10 25]);
    cfg  = getExperimentConfig('EOFCV'); Mblocks = cfg.nAtomicBlocks;
    eofVar = NaN;
    try
        capt = evalc('[~,~,eofVar] = CV_EOF(FT, cfg.nFolds, Mblocks);'); %#ok<NASGU>
    catch ME
        fprintf('  (EOF variance unavailable: %s)\n', ME.message);
    end
    fprintf('  Profiles (unique lat/lon)         : %d  (90/10 dev/hold-out split = 1458/169 profiles, set at Discovery on the raw pre-cleaning split; see paper Section 2)\n', nProfiles);
    fprintf('  Depth-resolved observations       : %d\n', nObs);
    fprintf('  BGC-Argo floats (unique BuoyName) : %d\n', nFloats);
    fprintf('  Consecutive-profile separation    : 10th pct %.2f km, 25th pct %.2f km\n', pcts(1), pcts(2));
    fprintf('  Atomic blocks (M)                 : %d\n', Mblocks);
    fprintf('  EOF first-5 cumulative variance    : %.2f%% (PCA on full featureTable)\n', eofVar);
end

%% Summary
fprintf('\nSummary: %d PASS, %d FAIL (hold-out RMSE tolerance %.0e µmol kg⁻¹).\n',nPass,nFail,tol);
if nFail==0, fprintf('All hold-out RMSEs reproduced exactly from the supplied models.\n'); end
fprintf('Hold-out metrics recomputed live; NCV / hyperparameters / feature importance read from stored .mat arrays.\n');
fprintf('NB 95%% CI, gaps, inflation, normalised importances all computed live — no text source, no re-training.\n\n');
end


% ===================== helpers =====================

function idx = orderIndices(strat, names)
idx=zeros(0,1);
for i=1:numel(names)
    j=find(strat==names(i),1);
    if ~isempty(j), idx(end+1,1)=j; end %#ok<AGROW>
end
end

function printNCV(idx,strat,Rm,Rs,Mm,Ms,Bm,R2,lo,hi)
for t=1:numel(idx)
    k=idx(t);
    rcell=sprintf('%7.4f±%-6.4f', Rm(k), Rs(k));
    mcell=sprintf('%7.4f±%-6.4f', Mm(k), Ms(k));
    cicell=sprintf('[%7.4f,%7.4f]', lo(k), hi(k));
    fprintf('| %-28s | %-15s | %-15s | %+7.4f | %6.4f | %18s |\n', ...
        strat(k),rcell,mcell,Bm(k),R2(k),cicell);
end
end

function printHoldout(b,mainIdx,ctrlIdx,strat,RMSE,MAE,Bias,R2)
fprintf('%s\n',b);
fprintf('| %-28s | %-8s | %-7s | %-8s | %-6s |\n','Strategy','RMSE','MAE','Bias','R²');
fprintf('%s\n',b);
for t=1:numel(mainIdx), k=mainIdx(t); fprintf('| %-28s | %8.4f | %7.4f | %+8.4f | %6.4f |\n',strat(k),RMSE(k),MAE(k),Bias(k),R2(k)); end
fprintf('%s\n',b);
for t=1:numel(ctrlIdx), k=ctrlIdx(t); fprintf('| %-28s | %8.4f | %7.4f | %+8.4f | %6.4f |\n',strat(k),RMSE(k),MAE(k),Bias(k),R2(k)); end
fprintf('%s\n',b);
end

function printByDepth(idx,strat,V)
for t=1:numel(idx)
    k=idx(t);
    fprintf('| %-28s | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f |\n', ...
        strat(k), V(k,1), V(k,2), V(k,3), V(k,4), V(k,5));
end
end

function printAccDisp(idx,strat,pct10,pct20,foldRMSE)
for t=1:numel(idx)
    k=idx(t);
    v=foldRMSE(k,:); m=mean(v,'omitnan'); worst=max(v); cv=100*std(v,0,'omitnan')/m;
    fprintf('| %-28s | %8.2f%% | %8.2f%% | %9.4f | %10.4f | %8.1f%% |\n', ...
        strat(k), pct10(k), pct20(k), m, worst, cv);
end
end

function printConformalBlock(name,cp)
% Production-style single-strategy conformal table: depth-range rows, coverage
% + mean interval width columns, footer with global coverage/gap/N and verdict.
fprintf('\n  ── %s ──\n', name);
if isempty(cp)
    fprintf('    No stored OOF calibration set — conformal interval unavailable.\n');
    return;
end
bC='  +-----------------------+-------------------+------------------------------+';
fprintf('%s\n',bC);
fprintf('  | %-21s | %-17s | %-28s |\n','Depth Range','Hold-Out Coverage','Avg Interval Width (µmol/kg)');
fprintf('%s\n',bC);
lab=cp.DepthLabels; covD=cp.Coverage_byDepth; widD=cp.AvgWidth_byDepth;
for s=1:numel(lab)
    if isnan(covD(s))
        fprintf('  | %-21s | %17s | %28s |\n',lab{s},'N/A','N/A');
    else
        fprintf('  | %-21s | %16.2f%% | %28.4f |\n',lab{s},covD(s),widD(s));
    end
end
fprintf('%s\n',bC);
gap=cp.Coverage_Holdout-cp.TargetCoverage;
fprintf('    Global coverage %.2f%% (target %.0f%%, gap %+.2f%%); calibration N=%d OOF.\n', ...
    cp.Coverage_Holdout,cp.TargetCoverage,gap,cp.nCal);
if cp.Coverage_Holdout < 90.0
    fprintf('    [WARN] Under-coverage: OOF errors optimistically small — sign of leakage / optimism.\n');
elseif cp.Coverage_Holdout >= 93.0 && cp.Coverage_Holdout <= 97.0
    fprintf('    [OK] Valid coverage: the OOF-calibrated interval transfers to the independent hold-out.\n');
else
    fprintf('    [INFO] Coverage off-target — check covariate shift or fold variance.\n');
end
end

function printRel(idx,strat,isDepth,ncv,ho,gap,infl,inCI)
for t=1:numel(idx)
    k=idx(t);
    if isDepth(k), wt='weighted'; else, wt='unweighted'; end
    if inCI(k), ci='yes'; else, ci='no'; end
    fprintf('| %-28s | %-10s | %8.4f | %9.4f | %+6.1f%% | %5.2fx | %-5s |\n',strat(k),wt,ncv(k),ho(k),gap(k),infl(k),ci);
end
end

function printHyperMatrix(names, idx, foldHPc, cols, withCV)
ncol=numel(idx);
hdr=sprintf('%-24s','Parameter');
for c=1:ncol, hdr=[hdr sprintf(' %16s', cols{c})]; end %#ok<AGROW>
fprintf('%s\n',hdr);
fprintf('%s\n',repmat('-',1,numel(hdr)));
for p=1:numel(names)
    line=sprintf('%-24s', names(p));
    for c=1:ncol
        k=idx(c); HP=foldHPc{k};
        if isempty(HP), cell12='     -      '; else
            v=HP(:,p); m=mean(v,'omitnan'); cv=100*std(v,0,'omitnan')/abs(m);
            if withCV, cell12=sprintf('%.4g(%2.0f%%)', m, cv); else, cell12=sprintf('%12.4g', m); end
        end
        line=[line sprintf(' %16s', cell12)]; %#ok<AGROW>
    end
    fprintf('%s\n',line);
end
end

function printFinalHP(names, idx, finalHPc, cols)
ncol=numel(idx);
hdr=sprintf('%-24s','Parameter');
for c=1:ncol, hdr=[hdr sprintf(' %10s', cols{c})]; end %#ok<AGROW>
fprintf('%s\n',hdr);
fprintf('%s\n',repmat('-',1,numel(hdr)));
for p=1:numel(names)
    line=sprintf('%-24s', names(p));
    for c=1:ncol
        k=idx(c); T=finalHPc{k};
        if isempty(T)||~ismember(names(p),string(T.Properties.VariableNames)), val=NaN;
        else, val=T.(char(names(p))); end
        line=[line sprintf(' %10.4g', val)]; %#ok<AGROW>
    end
    fprintf('%s\n',line);
end
end

function printFeatureImportance(stratName, featNames, foldFI, finalFI)
% NCV% / CV% use the per-fold-first normalisation convention: normalise EACH
% fold's gain to % of that fold's total FIRST, then take the mean and the
% coefficient of variation across folds (CV% = 100*Std/|Mean| on the per-fold
% percentages). Normalising-then-averaging (the reverse) would give a different
% CV and would not reproduce the paper's stability table.
rowSum  = sum(foldFI, 2, 'omitnan');          % per-fold total gain (K x 1)
foldPct = foldFI ./ rowSum * 100;             % each fold row sums to 100%
ncvPct  = mean(foldPct, 1, 'omitnan');        % cross-fold mean of the per-fold %
cvPct   = 100 * std(foldPct, 0, 1, 'omitnan') ./ abs(ncvPct);
finPct  = 100 * finalFI / sum(finalFI);       % final-model normalised gain
[~,ord]=sort(finPct,'descend');
fprintf('\n  --- %s ---\n', stratName);
fprintf('  %-20s | %8s | %7s | %8s\n','Feature','NCV %','CV %','Final %');
fprintf('  %s\n',repmat('-',1,52));
for i=1:numel(ord)
    j=ord(i);
    fprintf('  %-20s | %8.3f | %6.1f%% | %8.3f\n', featNames(j), ncvPct(j), cvPct(j), finPct(j));
end
end
