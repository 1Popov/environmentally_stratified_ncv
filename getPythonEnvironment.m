function targetPython = getPythonEnvironment()
% getPythonEnvironment  Detects the host machine and configures the Python
% environment for MATLAB-Python interop. Returns the resolved path so workers
% (or callers that want it explicitly) can use it; the function also calls
% `pyenv(Version = targetPython)` itself if the active interpreter differs.
%
% Add new machines by extending the switch below. The mapping is intentionally
% explicit so the failure mode on an unknown host is a clear error rather than
% a silent fall-back to the system Python interpreter (which would not have
% lightgbm or optuna installed).

    % Silent on parfor workers (avoid N duplicate banners); main session prints.
    isInParfor = ~isempty(getCurrentTask());

    hostName = getenv('COMPUTERNAME');

    switch hostName
        case {'IVAN-DESKTOP', 'KATANA'}     % Office machines
            targetPython = "C:\Users\ivan\anaconda3\envs\lgbm_env\python.exe";
            if ~isInParfor
                fprintf('Running on OFFICE machine: %s\n', hostName);
            end

        case 'DEEP'                          % Home machine
            targetPython = "C:\Users\ivan\miniconda3\envs\lgbm_env\python.exe";
            if ~isInParfor
                fprintf('Running on HOME machine: %s\n', hostName);
            end

        otherwise
            envPython = getenv('DOXY_PYTHON');
            if ~isempty(envPython)
                targetPython = string(envPython);
                if ~isInParfor
                    fprintf('Unknown host [%s]; using DOXY_PYTHON override: %s\n', ...
                            hostName, targetPython);
                end
            else
                error('getPythonEnvironment:UnknownHost', ...
                      ['Unknown host [%s] and DOXY_PYTHON not set. Set the DOXY_PYTHON ' ...
                       'environment variable to your lgbm_env python.exe, or add a case ' ...
                       'for this machine in getPythonEnvironment.m.\n' ...
                       'Currently registered: IVAN-DESKTOP, KATANA (office), DEEP (home).'], ...
                      hostName);
            end
    end

    currentPyEnv = pyenv;
    if ~strcmp(currentPyEnv.Version, targetPython)
        pyenv(Version = targetPython);
    end

    if ~isInParfor
        fprintf('Python environment configured. Python %s at %s\n\n', ...
                pyenv().Version, pyenv().Executable);
    end
end
