"""LightGBM training + Optuna search, called from MATLAB via the ``py.*`` bridge.

This module is the MATLAB<->Python boundary of the Black Sea DOXY pipeline. MATLAB
owns the data and the cross-validation folds; this module runs the parts MATLAB
cannot do natively -- Optuna's TPE hyperparameter search and early-stopping-aware
``lgb.cv`` -- and keeps LightGBM on the GPU throughout.

Data crosses the bridge as NumPy arrays. MATLAB column vectors arrive shaped (N, 1),
so labels and weights are ``.ravel()``-ed to 1-D before use; inner-fold index lists
arrive 1-based (MATLAB) and are converted to 0-based here.

The search/training entry points never raise across the bridge: each returns a dict
whose ``status`` key is ``'ok'`` or an error string, so a failed trial cannot crash
the MATLAB worker. (``predict_from_model`` is the deliberate exception -- see its docstring.)
"""
import math
import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fixed LightGBM params shared by every training call -- not tunable. They pin the
# backend and the numerical-reproducibility contract for the paper's results:
#   objective='huber'        Huber loss (robust to deep-water outliers)
#   metric='rmse'            CV / eval metric reported back to Optuna
#   verbosity=-1, n_jobs=-1  silence LightGBM; use all CPU threads for data prep
#   device='gpu'             GPU training (LightGBM OpenCL backend)
#   max_bin=255              histogram bins; pinned so split thresholds don't drift
#   gpu_use_dp=True          float64 GPU reductions -- required for bit-reproducible
#                            objective values (float32 atomic-adds are non-deterministic)
#   deterministic=True       deterministic histogram construction (pairs with gpu_use_dp)
#   feature_pre_filter=False keep zero-gain features reportable in feature_importance
_FIXED_GPU_PARAMS = {
    'objective': 'huber', 'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
    'device': 'gpu', 'max_bin': 255,
    'gpu_use_dp': True, 'deterministic': True,
    'feature_pre_filter': False,
}


def run_bayes_opt_in_python(X_train, y_train, W_train, cv_folds_list,
                            opt_vars_dict, max_evals, fold_index,
                            random_seed, num_boost_rounds):
    """Optuna TPE hyperparameter search for one outer CV fold.

    Parameters
    ----------
    X_train, y_train, W_train : numpy.ndarray
        Features, target, and sample weights for the fold's training rows
        (y and W arrive (N, 1) from MATLAB and are ravelled to 1-D).
    cv_folds_list : list of (array-like, array-like)
        Inner-CV (train, test) row-index pairs, 1-based from MATLAB; converted
        to 0-based here.
    opt_vars_dict : dict
        Maps each tunable parameter name to its ``[min, max]`` search range.
    max_evals, fold_index, random_seed, num_boost_rounds : int
        Trial budget, fold id (logging only), RNG seed, and the boosting-round cap.

    Returns
    -------
    dict
        ``status`` is ``'ok'`` or an error string. On success it also carries
        ``best_params``, ``best_score`` (minimum CV RMSE), ``best_iteration``
        (argmin of the validation-RMSE curve, 1-based), ``train_rmse_history``,
        ``valid_rmse_history``, and ``trials_history`` (every trial as a record).

    Notes
    -----
    Early stopping is learning-rate-adaptive: ``patience ~ 1/lr`` (one "learning
    unit"), floored at 25 and capped at ``num_boost_rounds // 2`` so it can fire
    before the ceiling; ``min_delta=1e-5`` ignores sub-noise gains. The function
    never raises -- data-prep and per-trial failures are reported via ``status``.
    """
    try:
        cv_folds = [
            (np.asarray(tr, dtype=np.int32).ravel() - 1,
             np.asarray(te, dtype=np.int32).ravel() - 1)
            for tr, te in cv_folds_list
        ]
        # MATLAB column vectors arrive as Nx1; LightGBM wants 1-D for label/weight.
        lgb_train = lgb.Dataset(np.asarray(X_train),
                                label=np.asarray(y_train).ravel(),
                                weight=np.asarray(W_train).ravel())
        num_boost_rounds = int(num_boost_rounds)
    except Exception as e:
        return {'status': f'Error in Python data preparation: {str(e)}'}

    def log_trial(study, trial):
        try:
            best_str = f"{study.best_value:.5f}"
        except ValueError:
            best_str = "N/A"
        trial_str = f"{trial.number + 1}/{study.user_attrs['n_trials']}"
        val_str = f"{trial.value:.5f}" if trial.value != float('inf') else "FAILED"
        p = trial.params
        print(
            f"   [FOLD {fold_index} | {trial_str:>7}] "
            f"RMSE: {val_str:>7} (Best: {best_str:>7}) | "
            f"D:{p.get('max_depth', -1):<2} L:{p.get('num_leaves', 0):<3} "
            f"H:{p.get('huber_delta', 0):.3f} LR:{p.get('learning_rate', 0):.4f} "
            f"C:{p.get('min_child_samples', 0):<2} "
            f"a:{p.get('reg_alpha', 0):.3f} l:{p.get('reg_lambda', 0):.3f} "
            f"Hs:{p.get('min_sum_hessian_in_leaf', 0):.4f} "
            f"Sp:{p.get('min_split_gain', 0):.4f} "
            f"Ff:{p.get('feature_fraction', 0):.2f} "
            f"Bf:{p.get('bagging_fraction', 0):.2f} "
            f"Fq:{p.get('bagging_freq', 0)}",
            flush=True,
        )

    def objective(trial):
        max_depth = trial.suggest_int(
            'max_depth',
            int(opt_vars_dict['max_depth'][0]),
            int(opt_vars_dict['max_depth'][1]),
        )
        nl_lo = int(opt_vars_dict['num_leaves'][0])
        nl_hi = int(opt_vars_dict['num_leaves'][1])
        nl_hi = min(nl_hi, 2 ** max_depth)
        nl_hi = max(nl_hi, 2)
        if nl_lo >= nl_hi:
            nl_lo = max(2, nl_hi - 1)

        params = {
            **_FIXED_GPU_PARAMS,
            'seed': int(random_seed),
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', nl_lo, nl_hi, log=True),
            'huber_delta': trial.suggest_float('huber_delta', *opt_vars_dict['huber_delta']),
            'learning_rate': trial.suggest_float('learning_rate', *opt_vars_dict['learning_rate'], log=True),
            'min_child_samples': trial.suggest_int(
                'min_child_samples',
                int(opt_vars_dict['min_child_samples'][0]),
                int(opt_vars_dict['min_child_samples'][1]),
            ),
            'reg_alpha': trial.suggest_float('reg_alpha', *opt_vars_dict['reg_alpha'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *opt_vars_dict['reg_lambda'], log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float(
                'min_sum_hessian_in_leaf',
                *opt_vars_dict['min_sum_hessian_in_leaf'], log=True,
            ),
            'min_split_gain': trial.suggest_float('min_split_gain', *opt_vars_dict['min_split_gain'], log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', *opt_vars_dict['feature_fraction']),
            'bagging_fraction': trial.suggest_float('bagging_fraction', *opt_vars_dict['bagging_fraction']),
            'bagging_freq': trial.suggest_int(
                'bagging_freq',
                int(opt_vars_dict['bagging_freq'][0]),
                int(opt_vars_dict['bagging_freq'][1]),
            ),
        }

        try:
            # LR-adaptive early stopping: patience ~ 1/lr (one "learning unit"),
            # floored at 25, capped at num_boost_rounds//2 so it can fire before
            # the ceiling. min_delta drops sub-noise gains (<1e-5 scaled).
            lr = float(params['learning_rate'])
            patience = int(min(max(round(1.0 / max(lr, 1e-6)), 25),
                               max(25, num_boost_rounds // 2)))
            cv_results = lgb.cv(
                params=params, train_set=lgb_train,
                num_boost_round=num_boost_rounds,
                folds=cv_folds,
                callbacks=[lgb.early_stopping(stopping_rounds=patience,
                                              min_delta=1e-5, verbose=False)],
                eval_train_metric=True,
            )
            valid = cv_results['valid rmse-mean']
            trial.set_user_attr('train_rmse_history', cv_results['train rmse-mean'])
            trial.set_user_attr('valid_rmse_history', valid)
            trial.set_user_attr('best_iteration', int(np.argmin(valid)) + 1)
            return float(min(valid))
        except Exception as e:
            print(f"   [FOLD {fold_index}] lgb.cv trial failed: {str(e)}", flush=True)
            return float('inf')

    sampler = TPESampler(seed=int(random_seed))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.set_user_attr('n_trials', int(max_evals))
    study.optimize(objective, n_trials=int(max_evals), callbacks=[log_trial])

    # All trials inf (GPU OOM, data pathology): refuse to train a 0-round model.
    if math.isinf(study.best_value) or 'best_iteration' not in study.best_trial.user_attrs:
        return {
            'status': f'All {int(max_evals)} trials failed inside lgb.cv for fold {int(fold_index)}; refusing to train an empty final model.',
            'best_params': {}, 'best_score': float('inf'),
            'best_iteration': 0, 'train_rmse_history': [], 'valid_rmse_history': [],
            'trials_history': study.trials_dataframe().to_dict('records'),
        }

    best_params = study.best_params
    best_attrs = study.best_trial.user_attrs

    return {
        'status': 'ok',
        'best_params': best_params,
        'best_score': float(study.best_value),
        'best_iteration': int(best_attrs.get('best_iteration', 0)),
        'train_rmse_history': best_attrs.get('train_rmse_history', []),
        'valid_rmse_history': best_attrs.get('valid_rmse_history', []),
        'trials_history': study.trials_dataframe().to_dict('records'),
    }


def train_final_model(params, X_train, y_train, W_train, num_boost_round, random_seed):
    """Train the final LightGBM model on the full dev set and serialize it.

    Parameters
    ----------
    params : dict
        Best hyperparameters from the Optuna search; merged with
        ``_FIXED_GPU_PARAMS`` and the seed before training.
    X_train, y_train, W_train : numpy.ndarray
        Full training features, target, and weights (y and W ravelled to 1-D).
    num_boost_round : int
        Boosting rounds = the search's best_iteration, used as a fixed count
        (not an early-stopping ceiling).
    random_seed : int
        Seed written into the LightGBM params.

    Returns
    -------
    dict
        ``status`` (``'ok'`` or an error string), ``model_string`` (the booster via
        ``model_to_string()``; reload with ``lgb.Booster(model_str=...)``), and
        ``feature_importance`` (gain-based, so features with many shallow splits
        are not over-counted).
    """
    try:
        # MATLAB column vectors arrive as Nx1; LightGBM wants 1-D for label/weight.
        lgb_train = lgb.Dataset(np.asarray(X_train),
                                label=np.asarray(y_train).ravel(),
                                weight=np.asarray(W_train).ravel())
        params.update({**_FIXED_GPU_PARAMS, 'seed': int(random_seed)})
        for k in ('num_leaves', 'max_depth', 'min_child_samples', 'bagging_freq'):
            if k in params:
                params[k] = int(params[k])

        model = lgb.train(params=params, train_set=lgb_train,
                          num_boost_round=int(num_boost_round))
        return {
            'status': 'ok',
            'model_string': model.model_to_string(),
            'feature_importance': model.feature_importance(importance_type='gain').tolist(),
        }
    except Exception as e:
        return {'status': f'Error during final training: {str(e)}',
                'model_string': '', 'feature_importance': []}


def predict_from_model(model_string, X_test):
    """Predict from a serialized LightGBM model string.

    Parameters
    ----------
    model_string : str
        A booster serialized via ``model_to_string()`` -- the string itself,
        not a file path.
    X_test : numpy.ndarray
        Feature matrix to predict on.

    Returns
    -------
    numpy.ndarray
        1-D prediction vector. MATLAB receives a Python array -- wrap with
        ``double(...)`` and reshape as needed.

    Notes
    -----
    Unlike the search/training entry points, errors here propagate to the MATLAB
    caller rather than being caught and returned via ``status`` (no silent
    empty-array return).
    """
    bst = lgb.Booster(model_str=model_string)
    return bst.predict(np.array(X_test))
