# coding: utf-8
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import glob
import os
from datetime import datetime as dtm
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestRegressor
from wKit.ML.sk_ml import grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls

from wKit.utility.file_sys import mkdirs_if_not_exist
from src.experiment_based_function import *


def load_xs():
    Xs = {'RoadNet': pd.read_csv('data/x_RoadNet.csv', index_col=0)}

    for ftr_type in ['Segment', 'RoadNet+Segment']:
        for total_or_not in ['NO_TOTAL', 'TOTAL']:
            feature_combo_name = '%s_%s_%s' % (ftr_type, total_or_not, '~2014')
            Xs[feature_combo_name] = pd.read_csv('data/x_%s.csv' % feature_combo_name, index_col=0)

    return Xs


def init_model_params(name):
    params = grid_cv_default_params()
    if name == 'XGBcls':
        model = xgboost.XGBClassifier()
        param = params['cls']['XGBcls']
    elif name == 'BAGcls':
        model = BaggingClassifier()
        param = params['cls']['BAGcls']
    elif name == 'GDBcls':
        model = GradientBoostingClassifier()
        param = params['cls']['GDBcls']
    elif name == 'RFreg':
        model = RandomForestRegressor()
        param = params['reg']['RFreg']
    else:
        raise ('no model')
    return model, param


def exp5_recon(train_y, test_y, Xs, exp_path):
    """
    train_y: segment level computed by frac% of ratings
    test_y: segment level computed by 100% of ratings
    """

    def upsampling():
        upsampler = SMOTE(kind=up_name, random_state=SMOTE_SEED)
        up_train_x, up_train_y = upsampler.fit_sample(train_x, train_y.round())
        up_y_dist = pd.Series(up_train_y).value_counts().to_dict()
        return up_train_x, up_train_y, up_y_dist

    def grid(model):
        grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind=model_name[-3:],
                                   name=model_name,
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

    train_n_classes = train_y.round().nunique()
    test_n_classes = test_y.round().nunique()
    print('====n classes, train: %d, test: %d' % (train_n_classes, test_n_classes))
    y_dist = train_y.round().value_counts().to_dict()
    print('====train_y: distr=%s' % (y_dist))

    train_idx = train_y.index
    test_idx = test_y.index

    # store result
    grid_res_list, eval_res_list = [], []

    # iterate combos
    for ftr_combo_name, X in Xs.items():
        print('========ftr_combo_name=%s' % ftr_combo_name)
        train_x, test_x = X.loc[train_idx], X.loc[test_idx]
        feature_names = train_x.columns

        for model_name, up_name, fselect_type in MODEL_UP_FSEL_NAMES:
            print('============%s,%s,%s' % (model_name, up_name, fselect_type))
            if up_name != 'None':
                try:
                    up_train_x, up_train_y, up_y_dist = upsampling()
                except ValueError as e:
                    print('path=%s, ftr_combo_name=%s, smote=%s' % (exp_path, ftr_combo_name, up_name))
                    print('catch a ValueError: %s' % e)
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
    return


def main():
    start_time = dtm.now()
    redo = False
    test_y = pd.read_csv('data/y_csl_all-2017-10-01.csv', index_col=0).csl
    Xs = load_xs()

    for fn in glob.glob('data/exp5_ys/1001/*'):
        _, frac, sample_seed = fn[:-4].split('#')
        exp_path = 'experiment_1001/exp5_recon/%s#%s' % (frac, sample_seed)
        if os.path.exists(exp_path) and not redo:
            print('exists:', exp_path)
            continue
        print('exp on:', exp_path)
        mkdirs_if_not_exist(exp_path)
        train_y = pd.read_csv(fn, index_col=0).csl
        exp5_recon(train_y, test_y, Xs, exp_path)
        # break

    end_time = dtm.now()
    print('start at:', start_time, 'end at:', end_time)


if __name__ == '__main__':
    MODEL_UP_FSEL_NAMES = [
        ('XGBcls', 'None', 'None'),
        ('BAGcls', 'None', 'None'),
        ('GDBcls', 'None', 'None'),
        ('RFreg', 'None', 'None'),
    ]
    SMOTE_SEED = 10
    main()
