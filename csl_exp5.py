# coding: utf-8
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import glob
import os
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from wKit.ML.sk_ml import grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls

from wKit.utility.file_sys import mkdirs_if_not_exist
from src.experiment_based_function import *


def load_data():
    y = pd.read_csv('data/y_csl_all.csv', index_col=0).csl
    Xs = {'RoadNet': pd.read_csv('data/x_RoadNet.csv', index_col=0)}

    for ftr_type in ['Segment', 'RoadNet+Segment']:
        for total_or_not in ['NO_TOTAL', 'TOTAL']:
            feature_combo_name = '%s_%s_%s' % (ftr_type, total_or_not, '~2014')
            Xs[feature_combo_name] = pd.read_csv('data/x_%s.csv' % feature_combo_name, index_col=0)

    return Xs, y


def init_model_params(name):
    params = grid_cv_default_params()
    if name == 'XGBreg': 
        model = xgboost.XGBRegressor()
        param = params['reg']['XGBreg']
    elif name == 'XGBcls': 
        model = xgboost.XGBClassifier()
        param = params['cls']['XGBcls']
    elif name == 'GDBcls': 
        model = GradientBoostingClassifier()
        param = params['cls']['GDBcls']
    else: raise('no model')
        
    return model, param


def exp5_one_run(exp_path, Xs, y, train_idx, test_idx):
    def upsampling():
        upsampler = SMOTE(kind=up_name, random_state=SMOTE_SEED)
        up_train_x, up_train_y = upsampler.fit_sample(train_x, train_y.round())
        up_y_dist = pd.Series(up_train_y).value_counts().to_dict()
        return up_train_x, up_train_y, up_y_dist

    def grid(model):
        grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind=model_name[-3:], name=model_name,
                                   path=cv_path)
        grid_res['ftr_combo_name'] = ftr_combo_name
        grid_res['feature_selection'] = fselect_type
        grid_res['upsample'] = up_name
        model = grid_res.pop('best_model')
        grid_res_list.append(grid_res)
        return model

    def eval_on_test():
        eval_res = evaluator_scalable_cls(model, dset['train_x'], dset['train_y'], dset['test_x'], dset['test_y'])
        eval_res['ftr_combo_name'] = ftr_combo_name
        eval_res['model_name'] = model_name
        eval_res['feature_selection'] = fselect_type
        eval_res['upsample'] = up_name
        eval_res['train_n_classes'] = train_n_classes
        eval_res['test_n_classes'] = test_n_classes
        eval_res['#ftr_all'] = len(feature_names)
        eval_res['#ftr_keep'] = dset['selected_ftr'].sum()
        eval_res['#train_sample'] = up_train_y.shape[0] if up_name != 'None' else train_y.shape[0]
        eval_res['#test_sample'] = test_y.shape[0]
        eval_res['y_dist_up'] = up_y_dist if up_name != 'None' else None
        eval_res['y_dist'] = y_dist
        eval_res_list.append(eval_res)

    # get train_y and test_y
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]
    train_n_classes = train_y.round().nunique()
    test_n_classes = test_y.round().nunique()
    print('====n classes, train: %d, test: %d' % (train_n_classes, test_n_classes))
    y_dist = train_y.round().value_counts().to_dict()
    print('====train_y: distr=%s' % (y_dist))

    # store result
    grid_res_list, eval_res_list = [], []

    # iterate combos
    for ftr_combo_name, X in Xs.items():
        print('========ftr_combo_name=%s' % ftr_combo_name)
        # get train x and test_x
        train_x, test_x = X.loc[train_idx], X.loc[test_idx]
        feature_names = train_x.columns

        for model_name, up_name, fselect_type in MODEL_UP_FSEL_NAMES:
            print('============%s,%s,%s' % (model_name, up_name, fselect_type))
            if up_name != 'None':
                try:
                    up_train_x, up_train_y, up_y_dist = upsampling()
                except ValueError as e:
                    print('------------')
                    print('path=%s, ftr_combo_name=%s, smote=%s' % (exp_path, ftr_combo_name, up_name))
                    print('SMOTE ValueError: %s' % e)
                    print('------------')
                    continue
                dset = scale_and_selection(up_train_x, up_train_y, test_x, test_y, fselect_type)

            dset = scale_and_selection(train_x, train_y, test_x, test_y, fselect_type)
            print('feature selection: %d -> %d' % (len(feature_names), dset['selected_ftr'].sum()))
            cv_path = '%s/%s#%s#%s' % (exp_path, ftr_combo_name, up_name, fselect_type)
            mkdirs_if_not_exist(cv_path)
            print('fitting models for cv_path=%s' % cv_path)

            # init model
            model, param = init_model_params(model_name)
            # grid a model, save grid_res to grid_res_list
            model = grid(model)
            # evaluate on original test set, save eval_res to eval_res_list
            eval_on_test()

        # df_res is for a whole run of up_exp, but save to disk with overwrite once a combo is done
        df_grid_res = pd.DataFrame(grid_res_list)
        df_grid_res.to_csv('%s/grid_res.csv' % exp_path)
        df_eval_res = pd.DataFrame(eval_res_list)
        df_eval_res.to_csv('%s/eval_res.csv' % exp_path)
        # break    # run one combo


def exp5_one_frac(Xs, y, frac, sample_seed):
    for seed in SEEDS:
        # set up experiment path
        exp_path = 'data/exp5/frac_%d/sample_%d/seed_%d' % (frac, sample_seed, seed)
        mkdirs_if_not_exist(exp_path)
        # get train/test index
        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(y.index, idx_fn, seed)
        print('\n====begin one run exp, in exp_path=%s' % exp_path)
        exp5_one_run(exp_path, Xs, y, train_idx, test_idx)
        # break  # run one seed


def main():
    Xs, y = load_data()
    for frac in range(100, 0, -10):
        for sample_seed in SAMPLING_SEEDS:
            sampled_y = y.sample(frac=frac/100.0, random_state=sample_seed) if frac != 100 else y
            exp5_one_frac(Xs, sampled_y, frac, sample_seed)
            # run one sample is enough for frac==100
            if frac==100:
                break

            # break  # run one sample


if __name__ == '__main__':
    SAMPLING_SEEDS = [502, 7081, 5128, 5561, 6711]
    MODEL_UP_FSEL_NAMES = [
        ('XGBcls', 'None', 'None'),
        ('XGBcls', 'svm', 'None'),
        ('XGBreg', 'None', 'None'),
        ('XGBreg', 'svm', 'rfecv_linsvc'),
        ('GDBcls', 'None', 'None'),
        ('GDBcls', 'svm', 'mrmr'),
    ]
    SMOTE_SEED = 10
    main()

