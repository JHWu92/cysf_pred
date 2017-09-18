# coding: utf-8

import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from wKit.ML.sk_ml import (grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls,
                           show_important_features, confusion_matrix_as_df)
from wKit.utility.file_sys import mkdirs_if_not_exist
from src.experiment_based_function import *


def load_data():
    y = pd.read_csv('data/y_csl_all.csv', index_col=0).csl
    X_total = pd.read_csv('data/x_TOTAL_~2014.csv', index_col=0)
    X_type = pd.read_csv('data/x_NO_TOTAL_~2014.csv', index_col=0)
    Xs = {'NO_TOTAL': X_type, 'TOTAL': X_total}
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


def upsample_one_class(y_one_class, target_num):
    num = len(y_one_class)
    factor = int(round(target_num/num))
    if factor == 1:  # don't do anything
        return y_one_class
    return pd.concat([y_one_class]*factor)  # dulicating by factor times

def upsample(train_y):
    max_ = train_y.round().value_counts().max()
    labels = train_y.round().unique()
    uped = []
    for label in labels:
        y_one_class = train_y[train_y.round()==label]
        up = upsample_one_class(y_one_class, max_)
        uped.append(up)
    return pd.concat(uped)


def get_total_or_type(total_or_not):
    return {'TOTAL': 'total', 'NO_TOTAL': 'type'}[total_or_not]


def up_exp(smote_kind, combos, y, Xs):
    for seed in SEEDS:  # the last seed raise 'Expected n_neighbors <= n_samples,  but n_samples = 5, n_neighbors = 6' in smote.fit_sample
        # set up experiment path
        exp_path = 'data/up_down_experiment/seed_%d' % seed
        upsample_path = '%s/upsample_smote_%s' % (exp_path, smote_kind)
        mkdirs_if_not_exist(exp_path)
        mkdirs_if_not_exist(upsample_path)
        print('upsample_path', upsample_path)

        # get train/test index
        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(y.index, idx_fn, seed)

        # get train_y and test_y
        train_y, test_y = y.loc[train_idx], y.loc[test_idx]
        y_dist = train_y.value_counts().to_dict()
        print('train_y: %s' % (str(y_dist)))
        #     uped_train_y = upsample(train_y)

        # store result
        df_grid_res, df_eval_res = [], []

        # iterate combos
        for total_or_not, name in combos:
            total_or_type = get_total_or_type(total_or_not)
            # get train x and test_x
            X = Xs[total_or_not]
            train_x, test_x = X.loc[train_idx], X.loc[test_idx]
            feature_names = train_x.columns
            # oversample train_x and train_y
            upsampler = SMOTE(kind=smote_kind, random_state=10)
            up_train_x, up_train_y = upsampler.fit_sample(train_x, train_y)
            up_y_dist = pd.Series(up_train_y).value_counts().to_dict()
            print('trainy: %s, up_train_y: %s' % (str(y_dist), str(up_y_dist)))

            # for each combo, do a feature slection experiment
            for fselect_type in FSELECT_TYPE:
                # create folder for each fselect experiment
                cv_path = '%s/%s' % (upsample_path, fselect_type)
                mkdirs_if_not_exist(cv_path)
                # min-max scale and select
                dset = scale_and_selection(up_train_x, up_train_y, test_x, test_y, fselect_type)
                # grid search best fit model
                model, param = init_model_params(name)
                grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind=name[-3:], name=name, path=cv_path)
                grid_res['total_or_type'] = total_or_type
                grid_res['feature_selection'] = fselect_type
                model = grid_res.pop('best_model')
                df_grid_res.append(grid_res)
                # evaluate on original test set
                eval_res = evaluator_scalable_cls(model, dset['train_x'], dset['train_y'], dset['test_x'], dset['test_y'])
                eval_res['total_or_type'] = total_or_type
                eval_res['model_name'] = name
                eval_res['feature_selection'] = fselect_type
                eval_res['#all'] = len(feature_names)
                eval_res['#keep'] = dset['selected_ftr'].sum()
                eval_res['up_y_dist'] = up_y_dist
                eval_res['y_dist'] = y_dist
                print('feature selection: %d -> %d' %(len(feature_names), dset['selected_ftr'].sum()))
                df_eval_res.append(eval_res)
                # save feature importances
                imp = show_important_features(model, labels=feature_names, set_std=False, show_plt=False).drop('std', axis=1)
                imp.columns = ['label', 'importance_%d' % seed]
                imp.to_csv('%s/imp-%s-%s.csv' % (cv_path, name, total_or_not))
                # save confusion matrix
                cfsn_norm = confusion_matrix_as_df(model, dset['test_x'], dset['test_y'], labels=[1, 2, 3, 4, 5], normalize=True)
                cfsn_norm.to_csv('%s/cfsn_norm-%s-%s.csv' % (cv_path, name, total_or_not))
                cfsn = confusion_matrix_as_df(model, dset['test_x'], dset['test_y'], labels=[1, 2, 3, 4, 5])
                cfsn.to_csv('%s/cfsn-%s-%s.csv' % (cv_path, name, total_or_not))

    #         break    # run one combos
        # save result
        df_grid_res = pd.DataFrame(df_grid_res)
        df_grid_res.to_csv('%s/grid_res.csv' % upsample_path)
        df_eval_res = pd.DataFrame(df_eval_res)
        df_eval_res.to_csv('%s/eval_res.csv' % upsample_path)
        # break  # run one seed
        print('finished seed %d' % seed)


def main():
    Xs, y = load_data()
    y = y.round().astype(int)
    combos = [('TOTAL', 'XGBcls'), ('NO_TOTAL', 'XGBreg'), ('TOTAL', 'GDBcls')]
    smote_kinds = ['regular', 'svm']
    for smote_kind in smote_kinds[1:]:
        print('running', smote_kind)
        up_exp(smote_kind=smote_kind, combos=combos, y=y, Xs=Xs)


if __name__ == '__main__':
    main()

