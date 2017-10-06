# coding: utf-8
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import pandas as pd
from datetime import datetime as dtm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
import xgboost

from wKit.ML.sk_ml import (grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls,
                           show_important_features, confusion_matrix_as_df)
from wKit.utility.file_sys import mkdirs_if_not_exist
from src.experiment_based_function import *


def load_data():
    y = pd.read_csv('data/y_csl_all-2017-10-01.csv', index_col=0).csl
    Xs = {'RoadNet': pd.read_csv('data/x_RoadNet.csv', index_col=0)}

    for ftr_type in ['Segment', 'RoadNet+Segment']:
        for total_or_not in ['NO_TOTAL', 'TOTAL']:
            feature_combo_name = '%s_%s_%s' % (ftr_type, total_or_not, '~2014')
            Xs[feature_combo_name] = pd.read_csv('data/x_%s.csv' % feature_combo_name, index_col=0)

    return Xs, y


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
    else: raise('no model')
    return model, param


def up_exp_one_run(upsample_path, smote_kind, model_names, y, Xs, train_idx, test_idx, seed):

    def grid(model):
        grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind=name[-3:], name=name, path=cv_path)
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
        eval_res['up_y_dist'] = up_y_dist
        eval_res['y_dist'] = y_dist
        eval_res_list.append(eval_res)

    # get train_y and test_y
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]
    y_dist = train_y.value_counts().to_dict()
    print('train_y: %s' % (str(y_dist)))

    # store result
    grid_res_list, eval_res_list = [], []

    # iterate combos
    for ftr_combo_name, X in Xs.items():
        print('ftr_combo_name=%s' % ftr_combo_name)
        # get train x and test_x
        train_x, test_x = X.loc[train_idx], X.loc[test_idx]
        feature_names = train_x.columns
        # oversample train_x and train_y
        try:
            upsampler = SMOTE(kind=smote_kind, random_state=SMOTE_SEED)
            up_train_x, up_train_y = upsampler.fit_sample(train_x, train_y)
            up_y_dist = pd.Series(up_train_y).value_counts().to_dict()
        except ValueError as e:
            print('path=%s, ftr_combo_name=%s, smote=%s' %(upsample_path, ftr_combo_name, smote_kind))
            print('catch a ValueError: %s' % e)
            continue
        print('upsampled smote=%s, trainy: %s, up_train_y: %s' % (smote_kind, str(y_dist), str(up_y_dist)))

        # for each combo, do a feature slection experiment
        for fselect_type in FSELECT_TYPE:
            # create folder for each fselect experiment
            cv_path = '%s/%s#%s' % (upsample_path, ftr_combo_name, fselect_type)
            mkdirs_if_not_exist(cv_path)
            # min-max scale and select
            dset = scale_and_selection(up_train_x, up_train_y, test_x, test_y, fselect_type)
            print('feature selection: %d -> %d' %(len(feature_names), dset['selected_ftr'].sum()))
            print('fitting models for cv_path=%s' % cv_path)
            # choose a ftr combo->upsampled->min-max scaling+feature selection->run three models
            for name in model_names:
                # init model
                model, param = init_model_params(name)
                # grid a model, save grid_res to grid_res_list
                model = grid(model)
                # evaluate on original test set, save eval_res to eval_res_list
                eval_on_test()
                # save feature importances
                try:
                        imp = show_important_features(model, labels=feature_names, set_std=False, show_plt=False).drop('std', axis=1)
                        imp.columns = ['label', 'importance_%d' % seed]
                        imp.to_csv('%s/imp-%s.csv' % (cv_path, name))
                except AttributeError as e:
                    print(name, 'no import')
                # save confusion matrix
                cfsn = confusion_matrix_as_df(model, dset['test_x'], dset['test_y'], labels=[1, 2, 3, 4, 5])
                cfsn.to_csv('%s/cfsn-%s.csv' % (cv_path, name))

        # df_res is for a whole run of up_exp, but save to disk with overwrite once a combo is done
        df_grid_res = pd.DataFrame(grid_res_list)
        df_grid_res.to_csv('%s/grid_res.csv' % upsample_path)
        df_eval_res = pd.DataFrame(eval_res_list)
        df_eval_res.to_csv('%s/eval_res.csv' % upsample_path)
        # break    # run one combo


def up_exp(smote_kind, model_names, y, Xs):

    for seed in SEEDS:
        # set up experiment path
        exp_path = 'experiment_1001/exp3/seed_%d' % seed
        upsample_path = '%s/upsample_smote_%s' % (exp_path, smote_kind)
        mkdirs_if_not_exist(exp_path)
        mkdirs_if_not_exist(upsample_path)

        # get train/test index
        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(y.index, idx_fn, seed)
        print('begin one run exp, in upsample_path=%s' % upsample_path)
        up_exp_one_run(upsample_path, smote_kind, model_names, y, Xs, train_idx, test_idx, seed)

        print('finished seed %d' % seed)
        # break  # run one seed


def main():
    start_time = dtm.now()
    Xs, y = load_data()
    y = y.round().astype(int)
    model_names = ['XGBcls', 'BAGcls', 'GDBcls']
    smote_kinds = ['regular', 'svm']
    for smote_kind in smote_kinds:
        print('running', smote_kind)
        up_exp(smote_kind=smote_kind, model_names=model_names, y=y, Xs=Xs)
    end_time = dtm.now()
    print('start at:', start_time, 'end at:', end_time)


if __name__ == '__main__':
    SMOTE_SEED = 10
    main()

