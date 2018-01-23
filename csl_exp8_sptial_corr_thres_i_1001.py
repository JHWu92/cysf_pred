# coding: utf-8
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from datetime import datetime as dtm
import glob
import os
import pandas as pd
import xgboost
from sklearn.ensemble import BaggingClassifier
from wKit.ML.sk_ml import grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls

from wKit.utility.file_sys import mkdirs_if_not_exist
from src.experiment_based_function import *


def load_data(i_tail):
    Xs = {'RoadNet': pd.read_csv('data/x_RoadNet+Spatial%s.csv' % i_tail, index_col=0)}

    for ftr_type in ['Segment', 'RoadNet+Segment']:
        for total_or_not in ['NO_TOTAL', 'TOTAL']:
            feature_combo_name = '%s_%s_%s+Spatial%s' % (ftr_type, total_or_not, '~2014', i_tail)
            Xs[feature_combo_name] = pd.read_csv('data/x_%s.csv' % feature_combo_name, index_col=0)

    y = pd.read_csv('data/y_csl_all-2017-10-01.csv', index_col=0).csl
    return Xs, y


def init_model_params(name):
    params = grid_cv_default_params()
    if name == 'XGBcls':
        model = xgboost.XGBClassifier()
        param = params['cls']['XGBcls']
    elif name == 'BAGcls':
        model = BaggingClassifier()
        param = params['cls']['BAGcls']
    else:
        raise ('no model')
    return model, param


def exp8_one_run(exp_path, Xs, y, train_idx, test_idx):
    def grid(model):
        grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind=model_name[-3:],
                                   name=model_name,
                                   path=cv_path)
        grid_res['ftr_combo_name'] = ftr_combo_name
        grid_res['feature_selection'] = fselect_type
        model = grid_res.pop('best_model')
        grid_res_list.append(grid_res)
        return model

    def eval_on_test():
        eval_res = evaluator_scalable_cls(model, dset['train_x'], dset['train_y'], dset['test_x'], dset['test_y'])
        eval_res['ftr_combo_name'] = ftr_combo_name
        eval_res['model_name'] = model_name
        eval_res['feature_selection'] = fselect_type
        eval_res['train_n_classes'] = train_n_classes
        eval_res['test_n_classes'] = test_n_classes
        eval_res['#ftr_all'] = len(feature_names)
        eval_res['#ftr_keep'] = dset['selected_ftr'].sum()
        eval_res['#train_sample'] = train_y.shape[0]
        eval_res['#test_sample'] = test_y.shape[0]
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

        for model_name, fselect_type in MODEL_NAMES:
            print('============%s' % (model_name))

            dset = scale_and_selection(train_x, train_y, test_x, test_y, fselect_type)
            print('feature selection: %d -> %d' % (len(feature_names), dset['selected_ftr'].sum()))
            cv_path = '%s/%s#%s' % (exp_path, ftr_combo_name, fselect_type)
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


def exp8(Xs, y, i):
    for seed in SEEDS:
        # set up experiment path
        exp_path = 'experiment_1001/exp8-spatial-thres-i/%s/seed_%d' % (i, seed)
        mkdirs_if_not_exist(exp_path)
        # get train/test index
        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(y.index, idx_fn, seed)
        print('====begin one run exp, in exp_path=%s' % exp_path)
        exp8_one_run(exp_path, Xs, y, train_idx, test_idx)
        # break  # run one seed


def main():
    start_time = dtm.now()
    for i in [0, 0.1, 0.3, 0.5, 0.7, 0.8]:
        print('i =', i)
        i_tail = '' if i == 0 else '-i-%0.1f' % i
        i_str = '%0.1f' % i
        Xs, y = load_data(i_tail)
        exp8(Xs, y, i_str)
        break

    # i_tail = '-Only'
    # i_str = 'Only'
    # Xs, y = load_data(i_tail)
    # exp8(Xs, y, i_str)

    end_time = dtm.now()
    print('start at:', start_time, 'end at:', end_time)


if __name__ == '__main__':
    MODEL_NAMES = [
        ('XGBcls', 'None'),
        # ('XGBcls', 'rfecv_linsvc'),
        # ('XGBcls', 'mrmr'),
        ('BAGcls', 'None'),
        # ('BAGcls', 'rfecv_linsvc'),
        # ('BAGcls', 'mrmr'),
    ]
    main()
