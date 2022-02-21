import gc
import pickle
import warnings
import glob

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error


OUTPUT_DIR = ""

###########
# LightGBM#
###########
import lightgbm as lgb


def fit_lightgbm(X, y, params, folds, add_suffix=""):
    """
    lgbm_params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
    }
    """
    oof_pred = np.zeros(len(y), dtype=np.float64)

    fold_unique = sorted(folds.unique())
    for fold in fold_unique:
        idx_train = folds != fold
        idx_valid = folds == fold
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        lgbm_train = lgb.Dataset(x_train, y_train)
        lgbm_valid = lgb.Dataset(x_valid, y_valid, reference=lgbm_train)
        model = lgb.train(
            params=params,
            train_set=lgbm_train,
            valid_sets=[lgbm_train, lgbm_valid],
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=100,
            # callbacks=[wandb_callback()]
        )
        # log_summary(model, save_model_checkpoint=True)
        pickle.dump(model, open(f"{OUTPUT_DIR}/lgbm_fold{fold}{add_suffix}.pkl", "wb"))
        pred_i = model.predict(x_valid, num_iteration=model.best_iteration)
        oof_pred[x_valid.index] = pred_i
        score = round(mean_squared_error(y_valid, pred_i), 5)
        print(f"Performance of the prediction: {score}")

    score = round(mean_squared_error(y, oof_pred), 5)
    print(f"All Performance of the prediction: {score}")
    del model
    gc.collect()
    return oof_pred


def pred_lightgbm(X, add_suffix=""):
    models = glob(f"{OUTPUT_DIR}/lgbm*{add_suffix}.pkl")
    models = [pickle.load(open(model, "rb")) for model in models]
    preds = np.array([model.predict(X) for model in models])
    preds = np.mean(preds, axis=0)
    return preds


############
# cat-boost#
############
import catboost as cat


def fit_catboost(X, y, params, folds, categorycal_list=[], add_suffix=""):
    """
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'num_boost_round': 10000,
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'depth': 6,
    }
    """
    oof_pred = np.zeros(len(y), dtype=np.float32)

    fold_unique = sorted(folds.unique())
    for fold in fold_unique:
        idx_train = folds != fold
        idx_valid = folds == fold
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        cat_train = cat.Pool(
            x_train,
            label=y_train,
            cat_features=categorycal_list,
        )
        cat_valid = cat.Pool(
            x_valid,
            label=y_valid,
            cat_features=categorycal_list,
        )
        # model = cat.CatBoostRegressor(**params)
        model = cat.CatBoostClassifier(**params)

        model.fit(
            cat_train,
            early_stopping_rounds=100,
            plot=False,
            use_best_model=True,
            eval_set=[cat_valid],
            verbose=100,
        )
        pickle.dump(model, open(f"{OUTPUT_DIR}/cat_fold{fold}{add_suffix}.pkl", "wb"))
        # pred_i = model.predict(x_valid)
        pred_i = model.predict_proba(x_valid)[:, 1]  #
        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f"Performance of the prediction: {score}\n")

    score = round(roc_auc_score(y, oof_pred), 5)
    print(f"All Performance of the prediction: {score}")
    del model
    gc.collect()
    return oof_pred


def pred_catboost(X, add_suffix=""):
    models = glob(f"{OUTPUT_DIR}/cat*{add_suffix}.pkl")
    models = [pickle.load(open(model, "rb")) for model in models]
    preds = np.array([model.predict_proba(X)[:, 1] for model in models])
    preds = np.mean(preds, axis=0)
    return preds


##########
# xgboost#
##########
import xgboost as xgb


def fit_xgboost(X, y, params, folds, add_suffix=""):
    """
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.01,
        'tree_method':'gpu_hist'
    }
    """
    oof_pred = np.zeros(len(y), dtype=np.float32)

    fold_unique = sorted(folds.unique())
    for fold in fold_unique:
        idx_train = folds != fold
        idx_valid = folds == fold
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
        evals = [(xgb_train, "train"), (xgb_valid, "eval")]

        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=100000,
            early_stopping_rounds=100,
            evals=evals,
            verbose_eval=100,
        )
        pickle.dump(model, open(f"{OUTPUT_DIR}/xgb_fold{fold}{add_suffix}.pkl", "wb"))
        pred_i = model.predict(xgb.DMatrix(x_valid), ntree_limit=model.best_ntree_limit)
        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f"Performance of the prediction: {score}\n")

    score = round(roc_auc_score(y, oof_pred), 5)
    print(f"All Performance of the prediction: {score}")
    del model
    gc.collect()
    return oof_pred


def pred_xgboost(X, add_suffix=""):
    models = glob(f"{OUTPUT_DIR}/xgb*{add_suffix}.pkl")
    models = [pickle.load(open(model, "rb")) for model in models]
    preds = np.array(
        [
            model.predict(xgb.DMatrix(X), ntree_limit=model.best_ntree_limit)
            for model in models
        ]
    )
    preds = np.mean(preds, axis=0)
    return preds
