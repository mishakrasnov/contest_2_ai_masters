import xgboost as xgb
import numpy as np
from functools import partial
import optuna
from own_objectives import *
from time import time as tm


def start_optimization(lib, n_trials, param_info=None, param_info_fork=None, cat_features=None, **objective_kwargs):
    
    
    if lib in ('lgb', 'xgb', 'cb', 'cb_gpu'):
        sampler = optuna.samplers.TPESampler(n_startup_trials=43, n_ei_candidates=13, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
    elif lib in ('lgb_rs', 'cb_rs'): # random sampler - для исследований
        sampler = optuna.samplers.TPESampler(n_startup_trials=n_trials - 20, n_ei_candidates=33, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
    elif lib in ('lgb_msf'):
        study = optuna.create_study(directions=['maximize', 'maximize'])
    elif lib in ('lgb_mtf'):
        study = optuna.create_study(directions=['minimize', 'minimize'])
        
    if lib in ('lgb', 'lgb_rs'):
        func = partial(lgb_objective, param_info=param_info, **objective_kwargs)
    elif lib == 'xgb':
        func = partial(xgb_objective, param_info=param_info, **objective_kwargs)
    elif lib == 'lgb_msf':
        func = partial(lgb_multiscore_fork_objective, param_info=param_info, param_info_fork=param_info_fork,
                       **objective_kwargs)
    elif lib == 'lgb_mtf':
        func = partial(lgb_multitime_fork_objective, param_info=param_info, param_info_fork=param_info_fork,
                       **objective_kwargs)
    elif lib in ('cb', 'cb_gpu', 'cb_rs'):
        func = partial(сb_objective, param_info=param_info, cat_features=cat_features, **objective_kwargs)
        
        
    if lib == 'cb_gpu':    
        study.optimize(func, n_trials=n_trials, n_jobs=1)
    else:
        study.optimize(func, n_trials=n_trials, n_jobs=8)
        
    return study
    

def start_optimization_xgb(n_trials, param_info, n_startup_trials, **objective_kwargs):
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, multivariate=True)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    func = partial(xgb_objective, param_info=param_info, **objective_kwargs)
    study.optimize(func, n_trials=n_trials, n_jobs=-1)
    return study

    
    
def train(lib, study, tr_data, val_data, param_info=None, n_trees=100, no_suggest=False):
    evals_result = dict()
    if lib == 'lgb':
        times = []
        params = lgb_suggest_params(study.best_trial, param_info, no_suggest)
        callbacks = [
            lgb.record_evaluation(evals_result),
            lgb_timer(times)
        ]
        model = lgb.train(params, tr_data, valid_sets=[val_data], valid_names=['val'], num_boost_round=n_trees,
                         callbacks=callbacks)
        eval_res = evals_result['val']['auc']
        
    elif lib == 'xgb':
        params = xgb_suggest_params(study.best_trial, param_info)
        callback_timer = XGBTimer()
        model = xgb.train(params, tr_data, num_boost_round=n_trees, evals=[(val_data, 'val')],
        verbose_eval=False, evals_result=evals_result, callbacks=[callback_timer])
        times = callback_timer.res_list
        eval_res = evals_result['val']['auc']
        
    elif lib == 'lgbSklearn':
        times = []
        params = lgb_suggest_params(study.best_trial, param_info, no_suggest)
        callbacks = [
            lgb.record_evaluation(evals_result),
            lgb_timer(times)
        ]
        model = lgb.LGBMClassifier(**params, n_jobs=32, n_estimators=n_trees)
        model.fit(*tr_data, eval_set=[val_data], eval_names=['val'], callbacks=callbacks)
        eval_res = evals_result['val']['auc']
    
    if lib == 'cb': # returns only total fit time with metrics(
        params = cb_suggest_params(study.best_trial, param_info, no_suggest)
        params['iterations'] = 200
        model = cb.CatBoost(params)
        stm = tm()
        model.fit(tr_data, eval_set=val_data, verbose_eval=False)
        times = tm() - stm
        evals_result = model.get_evals_result()
        eval_res = evals_result['validation']['AUC']
        
    return times, eval_res
    
    
    
    