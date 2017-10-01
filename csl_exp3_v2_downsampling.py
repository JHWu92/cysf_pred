# coding: utf-8

import pandas as pd
import xgboost
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from wKit.ML.sk_ml import (grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls,
                           show_important_features, confusion_matrix_as_df)
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
    else:
        raise ('no model')

    return model, param


def down_exp_one_run(downsample_path, model_names, y, Xs, train_idx, test_idx, seed):
    def grid(model):
        grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind=name[-3:], name=name,
                                   path=cv_path)
        grid_res['ftr_combo_name'] = ftr_combo_name
        grid_res['feature_selection'] = fselect_type
        model = grid_res.pop('best_model')
        grid_res_list.append(grid_res)
        return model

    def eval_on_test():
        eval_res = evaluator_scalable_cls(model, dset['train_x'], dset['train_y'], dset['test_x'], dset['test_y'])
        eval_res['ftr_combo_name'] = ftr_combo_name
        eval_res['model_name'] = name
        eval_res['feature_selection'] = fselect_type
        eval_res['#all'] = len(feature_names)
        eval_res['#keep'] = dset['selected_ftr'].sum()
        eval_res['down_y_dist'] = down_y_dist
        eval_res['y_dist'] = y_dist
        eval_res['down_seed'] = down_seed
        eval_res_list.append(eval_res)

    # get train_y and test_y
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]
    y_dist = train_y.round().value_counts().to_dict()
    print('train_y: %s' % (str(y_dist)))

    # store result
    grid_res_list, eval_res_list = [], []

    # iterate combos
    for ftr_combo_name, X in Xs.items():
        print('ftr_combo_name=%s' % ftr_combo_name)
        # get train x and test_x
        train_x, test_x = X.loc[train_idx], X.loc[test_idx]
        feature_names = train_x.columns
        # undersample train_x and train_y for 20 times
        for down_seed in DOWN_SEEDS:
            downsampler = RandomUnderSampler(random_state=down_seed, return_indices=True)
            down_train_x, down_train_y, down_indices = downsampler.fit_sample(train_x, train_y.round())
            down_y_dist = pd.Series(down_train_y).value_counts().to_dict()
            print('downsample seed=%d, trainy: %s, down_train_y: %s' % (down_seed, str(y_dist), str(down_y_dist)))
            down_train_y = train_y.iloc[down_indices]  # previous down_train_y is rounded, now restore it as float

            # for each combo, do a feature selection experiment
            for fselect_type in FSELECT_TYPE:
                # create folder for each fselect experiment
                cv_path = '%s/%s#%s#%d' % (downsample_path, ftr_combo_name, fselect_type, down_seed)
                mkdirs_if_not_exist(cv_path)
                # min-max scale and select
                dset = scale_and_selection(down_train_x, down_train_y, test_x, test_y, fselect_type)
                print('feature selection: %d -> %d' % (len(feature_names), dset['selected_ftr'].sum()))
                print('fitting models for cv_path=%s' % cv_path)
                # choose a ftr combo->1 out of 20 downsample->min-max scaling+feature selection->run three models
                for name in model_names:
                    # init model
                    model, param = init_model_params(name)
                    # grid a model, save grid_res to grid_res_list
                    model = grid(model)
                    # save feature importances
                    imp = show_important_features(
                        model, labels=feature_names, set_std=False, show_plt=False).drop('std', axis=1)
                    imp.columns = ['label', 'importance_%d' % seed]
                    imp.to_csv('%s/imp-%s.csv' % (cv_path, name))
                    # save confusion matrix
                    cfsn = confusion_matrix_as_df(model, dset['test_x'], dset['test_y'], labels=[1, 2, 3, 4, 5])
                    cfsn.to_csv('%s/cfsn-%s.csv' % (cv_path, name))
                    # evaluate on original test set, save eval_res to eval_res_list
                    eval_on_test()

            # df_res is for a whole run of up_exp, but save to disk with overwrite once a combo is done
            df_grid_res = pd.DataFrame(grid_res_list)
            df_grid_res.to_csv('%s/grid_res.csv' % downsample_path)
            df_eval_res = pd.DataFrame(eval_res_list)
            df_eval_res.to_csv('%s/eval_res.csv' % downsample_path)
        # break    # run one combo


def down_exp(model_names, y, Xs):
    for seed in SEEDS:
        # set up experiment path
        exp_path = 'data/up_down_experiment_v2/seed_%d' % seed
        downsample_path = '%s/downsample' % (exp_path)
        mkdirs_if_not_exist(exp_path)
        mkdirs_if_not_exist(downsample_path)

        # get train/test index
        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(y.index, idx_fn, seed)
        print('begin one run exp, in downsample_path=%s' % downsample_path)
        down_exp_one_run(downsample_path, model_names, y, Xs, train_idx, test_idx, seed)

        print('finished seed %d' % seed)
        # break  # run one seed


def main():
    Xs, y = load_data()
    model_names = ['XGBcls', 'XGBreg', 'GDBcls']
    down_exp(model_names=model_names, y=y, Xs=Xs)


DOWN_SEEDS = [1749556, 775107, 9410117, 7254933, 9096779, 266976, 3095841,
              4282790, 7964599, 6116962, 2807671, 3038865, 5435116, 7262334,
              3302627, 8296745, 1549265, 3226101, 5886901, 1986252]
if __name__ == '__main__':
    main()
