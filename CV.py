import pandas as pd
import numpy as np

##################
# CROSS-VALIDATION#
##################
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
)


def get_kfold(train, n_splits, seed):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(kf.split(train)):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


def get_stratifiedkfold(train, target_col, n_splits, seed):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(kf.split(train, train[target_col])):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


def get_groupkfold(train, target_col, group_col, n_splits):
    kf = GroupKFold(n_splits=n_splits)
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(
        kf.split(train, train[target_col], train[group_col])
    ):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


def get_stratifiedgroupkfold(train, target_col, group_col, n_splits, seed):
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(
        kf.split(train, train[target_col], train[group_col])
    ):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series
