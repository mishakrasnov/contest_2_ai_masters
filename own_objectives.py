import xgboost as xgb
from sklearn.metrics import roc_auc_score
from time import time as tm
import pandas as pd
import catboost as cb

def _update_params(trial, params, param_info):
    for key, info in param_info.items():
        if isinstance(info, list):
            record = trial.suggest_categorical(key, info)
        elif isinstance(info, tuple): 
            if info[-1] == 'int':
                record = trial.suggest_int(key, info[0], info[1], log=True)
            else:
                record = trial.suggest_float(key, info[0], info[1], log=True)
        else: # number or None
            record = info
        params[key] = record
        if pd.isna(record):
            del params[key]
        
        
        
def xgb_suggest_params(trial = None, param_info = None):

    params_default = {
        'objective': 'binary:logistic',
        'verbosity': 0,
        'disable_default_eval_metric': True,
        'nthread': 1,
        
        'gamma': 0,
        'max_depth': 6,
        'subsample': 1.,
        'colsample_bytree': 0.75,
        'alpha': 0,
        
        'tree_method': 'hist', # GPU - gpu_hist
        'grow_policy': 'lossguide',
        
        'max_cat_threshold': 2,
        
        'eval_metric': 'auc',
        'seed': 911
    }
    if trial is None:
        params = params_default
    else: 
        params = {
        'objective': 'binary:logistic',
        'verbosity': 0,
        'disable_default_eval_metric': True,
        'nthread': 1,
        
        'eta': trial.suggest_float('eta', 1e-2, 1, log=True),
        'gamma': 0,
        'max_depth': 6,
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10000, log=True),
        'subsample': 1.,
        'colsample_bytree': 0.75,
        'lambda': trial.suggest_float('lambda', 1e-8, 100, log=True),
        'alpha': 0,
        
        'tree_method': 'hist', # GPU - gpu_hist
        'grow_policy': 'lossguide',
        'max_leaves': trial.suggest_int('max_leaves', 10, 60),
        
        
        'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 2, 6),
        'max_cat_threshold': 2,
        
        'eval_metric': 'auc',
        'seed': 911
        }
    
    # updating params via param_info
    if param_info is not None:
        _update_params(trial, params, param_info)
    
    return params



def lgb_suggest_params(trial, param_info, no_suggest=False):
    params = {
        'subsample': 1.,
        'subsample_freq': 0, # нужно указать этот параметр, чтобы subsample начал работать!
        # 'pos_subsample', 'neg_subsample'
        
        'max_depth': -1, # lossguide -> max_leaves
        
        'colsample_bytree': 0.75,
        'min_gain_to_split': 0, # вообще невозможно заранее сказать какой порядок оптимальный
        'reg_alpha': 0,
        'path_smooth': 0,
        
        'max_bin': 254,
        
        'max_cat_threshold': 32, # макс количество категориальных бинов
        
        'boosting': 'gbdt',
        'objective': 'binary',

        'metric': 'auc',
        'metric_freq': 1,
        
        'tree_learner': 'serial', # Distributed learning
        'nthread': 1, # интересно, как совмещается с optuna
        'device_type': 'cpu',
        'is_unbalance': True, # 'scale_pos_weight'
        'enable_bundle': True, # EFB (???)
        
        'verbosity': -1,
        'seed': 911
        
        # 'cegb_tradeoff', 'cegb_penalty_split', 'cegb_penalty_feature_lazy', 'cegb_penalty_feature_coupled'
        # CEGB - статья https://proceedings.neurips.cc/paper/2017/file/4fac9ba115140ac4f1c22da82aa0bc7f-Paper.pdf
        
        # 'boosting': 'dart' и серия параметров 'used only in dart' - статья https://arxiv.org/abs/1505.01866
        
        # 'linear_tree', 'linear_lambda' - статья https://arxiv.org/pdf/1802.05640.pdf
        
        # 'boosting': 'rf' - random forest mode
        
        # 'extra_trees', 'extra_seed' - extra_trees mode
    }
    if not no_suggest:
        params.update({
            'eta': trial.suggest_float('eta', 1e-2, 1, log=True), # распределение обычно в пределах 1е-3, 3
            'max_leaves': trial.suggest_int('max_leaves', 10, 60),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10000, log=True),
            # 'min_data_in_leaf' - аппроксимация того же самого в количестве объектов, через гессианы

            'lambda': trial.suggest_float('lambda', 1e-8, 100, log=True), # просто перебираем порядки - важный параметр!
            
            'min_data_in_bin': trial.suggest_int('min_data_in_bin', 3, 5000, log=True),

            'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 10000, log=True),
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 2, 6),
            'cat_l2': trial.suggest_float('cat_l2', 1e-3, 1000, log=True),
            'cat_smooth': trial.suggest_float('cat_smooth', 1e-4, 1000, log=True),

            'zero_as_missing': trial.suggest_categorical('zero_as_missing', [True, False]),
        })
    
    # updating params via param_info
    if param_info is not None:
        _update_params(trial, params, param_info)
    
    return params


def cb_suggest_params(trial, param_info, no_suggest=False):
    params = {
        'task_type': 'CPU', # вместе с 'devices' если несколько
        
        'loss_function': 'Logloss:hints=skip_train~true', # classes in [0, 1]
        'boosting_type': 'Plain',
        'eval_metric': 'Logloss:hints=skip_train~true',
        'iterations': 100,
        'has_time': False, # можно использовать колонку с Timestamp вместо рандомныых перестановок, задается в Pool

        'bootstrap_type': 'Bernoulli', # https://catboost.ai/en/docs/concepts/algorithm-main-stages_bootstrap-options
        'subsample': 0.75,
        
        'grow_policy': 'SymmetricTree',
        # 'min_data_in_leaf' - только для lossguide, depthwise деревьев

        'random_strength': 1., # борьба с переобучением - добавляем случайный шум ~ N(0, rs^2 * sigma^2) к скорам сплитов-кандидатов
        
        'fold_permutation_block': 1.,
        'max_bin': 254,
        
        # 'leaf_estimation_method', 'leaf_estimation_iterations', 'leaf_estimation_backtracking' - прикольно...
        # 'langevin' - статья https://arxiv.org/abs/2001.07248
        # 'first_feature_use_penalties', ... - стоит взглянуть
        # 'model_shrink_rate', 'model_shrink_mode' - что-то про убывание lr
        # 'model_size_reg' - штраф на размер модели с кучей кат. фичей
        
        
        # CTR settings - настройка кодиррования категорий. https://catboost.ai/en/docs/references/training-parameters/ctr
        # Quantization settings - настройка предварительного биннинга.
        #                                           https://catboost.ai/en/docs/references/training-parameters/quantization
        # Overfitting detection settings - https://catboost.ai/en/docs/references/training-parameters/overfitting-detection

        'auto_class_weights': 'Balanced', # про веса классов
        # 'class_weights', 'scale_pos_weights'
        
        'thread_count': 1,
        'random_seed': 911
    }
    
    if not no_suggest:
        params.update({
            'score_function': trial.suggest_categorical('score_function', ['Cosine', 'L2']), # ниже
            # https://catboost.ai/en/docs/concepts/algorithm-score-functions
        
            'eta': trial.suggest_float('eta', 1e-2, 1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1000, log=True),
            'depth': trial.suggest_int('depth', 3, 9), # не max_depth в ODT, а фактическая глубина
            
            'sampling_frequency': trial.suggest_categorical('sampling_frequency', ['PerTree', 'PerTreeLevel']),
            'rsm': trial.suggest_float('rsm', 0.3, 1.),
            
            'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 6),
        })
    
    # updating params via param_info
    if param_info is not None:
        _update_params(trial, params, param_info)
    
    return params



def xgb_objective(trial, param_info, X_tr, y_tr, X_val, y_val):
    params = xgb_suggest_params(trial, param_info)
    
    tr_data = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    xgb_booster = xgb.train(
        params, tr_data,
        num_boost_round=200, evals=[(val_data, 'val')], early_stopping_rounds=30,
        verbose_eval=False
    )
    
    y_prob = xgb_booster.predict(val_data)
    score = roc_auc_score(val_data.get_label(), y_prob)
    
    return score

def xgb_predict(param_info, X_tr, y_tr, X_val, trial = None):
    params = xgb_suggest_params(trial, param_info)
    
    tr_data = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    
    val_data = xgb.DMatrix(X_val, enable_categorical=True)
    
    xgb_booster = xgb.train(
        params, tr_data,
        num_boost_round=200, evals=[(val_data, 'val')], early_stopping_rounds=30,
        verbose_eval=False
    )

    return  xgb_booster.predict(val_data)





def lgb_objective(trial, param_info, X_tr, y_tr, X_val, y_val, no_suggest=False):
    params = lgb_suggest_params(trial, param_info, no_suggest)
    
    callback_early_stopping = lgb.early_stopping(stopping_rounds=30, first_metric_only=False, verbose=False)

    tr_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    lgb_booster = lgb.train(
        params, tr_data,
        num_boost_round=100, valid_sets=[val_data], valid_names=['val'],
        feval=None, # сюда можно впихнуть кастомную метрику
        # чтобы не считался objective_metric (напр logloss) на valid_sets, нужно в params прописать 'metric': 'None' (строка None!)
        callbacks=[callback_early_stopping] # early_stopping теперь тут
    )
    
    y_prob = lgb_booster.predict( # в lgb по умолчанию predict = predict_proba, и он не умеет предсказывать по lgb.Dataset
        X_val,
        # обратите внимание, что в predict есть еще аргументы, которые могут быть полезны
        start_iteration=0, num_iteration=None,
        raw_score=False, pred_leaf=False, pred_contrib=False
    )
    score = roc_auc_score(y_val, y_prob)
    
    return score



def lgb_multiscore_fork_objective(trial, param_info, param_info_fork, X_tr, y_tr, X_val, y_val):
    params = lgb_suggest_params(trial, param_info)
    
    callback_early_stopping = lgb.early_stopping(stopping_rounds=30, first_metric_only=False, verbose=False)

    tr_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    lgb_booster = lgb.train(
        params, tr_data,
        num_boost_round=100, valid_sets=[val_data], valid_names=['val'],
        callbacks=[callback_early_stopping]
    )
    
    y_prob = lgb_booster.predict(X_val)
    score_1 = roc_auc_score(y_val, y_prob)
    
    _update_params(trial, params, param_info_fork)
    lgb_booster = lgb.train(
        params, tr_data,
        num_boost_round=100, valid_sets=[val_data], valid_names=['val'],
        callbacks=[callback_early_stopping]
    )
    y_prob = lgb_booster.predict(X_val)
    score_2 = roc_auc_score(y_val, y_prob)
    
    return score_1, score_2



def lgb_multitime_fork_objective(trial, param_info, param_info_fork, X_tr, y_tr, X_val, y_val):
    params = lgb_suggest_params(trial, param_info)

    tr_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    
    stm = tm()
    lgb_booster = lgb.train(
        params, tr_data,
        num_boost_round=50
    )
    
    y_prob = lgb_booster.predict(X_val)
    time_1 = tm() - stm
    
    _update_params(trial, params, param_info_fork)
    
    stm = tm()
    lgb_booster = lgb.train(
        params, tr_data,
        num_boost_round=50
    )
    y_prob = lgb_booster.predict(X_val)
    time_2 = tm() - stm
    
    return time_1, time_2



def сb_objective(trial, param_info, X_tr, y_tr, X_val, y_val, no_suggest=False, cat_features=None):
    params = cb_suggest_params(trial, param_info, no_suggest)

    tr_data = cb.Pool(X_tr, label=y_tr, cat_features=cat_features)
    val_data = cb.Pool(X_val, label=y_val, cat_features=cat_features)
    
    cb_model = cb.CatBoost(params)
    
    cb_model.fit(tr_data, early_stopping_rounds=10, eval_set=val_data, verbose_eval=False)
    
    y_prob = cb_model.predict_proba(val_data)
    print(y_prob)
    score = roc_auc_score(y_val, y_prob)
    
    return score