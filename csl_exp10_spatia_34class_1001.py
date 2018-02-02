# coding: utf-8
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import itertools
from datetime import datetime as dtm
import pickle
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from wKit.ML.sk_ml import grid_cv_a_model, grid_cv_default_params, evaluator_scalable_cls

from wKit.utility.file_sys import mkdirs_if_not_exist
from src.experiment_based_function import *

ROAD_NET_FTR = ['seg_attr', 'net_SaN', 'net_SaE', 'bk_osm', 'bk_opendc', 'elevation']
SEGMENT_FTR = ['poi', 'crash', '311', 'v0', 'crime', 'moving', 'parking']
IS_TOTAL = True
IS_TYPE = False


def load_data(i_thres, lvl):
    def neighbor_ftr(total_or_type, i):
        total_or_type = '-total' if total_or_type else ''
        df_db_binary = pd.read_csv('spatial-corr/moran-i-distanceband-binary%s.csv' % total_or_type, index_col=0)
        db_150 = df_db_binary.db_b == 150
        sig_rand = df_db_binary.p_rand < 0.05
        pass_i_thres = df_db_binary.I.abs() >= i
        keep_cols = df_db_binary[db_150 & sig_rand & pass_i_thres].column
        keep_cols = (keep_cols + '_neighbor').tolist()
        return keep_cols

    def features(keep_categories, total_or_type):
        fn = 'data/x_TOTAL_~2014_cols_by_type.pkl' if total_or_type else 'data/x_NO_TOTAL_~2014_cols_by_type.pkl'
        cols_dict = pickle.load(open(fn, 'rb'))
        keep_ftr = []
        for ftr_type, ftr_cols in cols_dict.items():
            if ftr_type in keep_categories:
                keep_ftr += ftr_cols
        return keep_ftr

    # different type of columns
    rcol_type = features(ROAD_NET_FTR, IS_TYPE)
    scol_type = features(SEGMENT_FTR, IS_TYPE)
    nb_col_type = neighbor_ftr(IS_TYPE, i_thres)
    nb_rcol_type = [c + '_neighbor' for c in rcol_type if c + '_neighbor' in nb_col_type]
    nb_scol_type = [c + '_neighbor' for c in scol_type if c + '_neighbor' in nb_col_type]

    rcol_total = features(ROAD_NET_FTR, IS_TOTAL)
    scol_total = features(SEGMENT_FTR, IS_TOTAL)
    nb_col_total = neighbor_ftr(IS_TOTAL, i_thres)
    nb_rcol_total = [c + '_neighbor' for c in rcol_total if c + '_neighbor' in nb_col_total]
    nb_scol_total = [c + '_neighbor' for c in scol_total if c + '_neighbor' in nb_col_total]

    # self and neighbor features
    x_nb_type = pd.read_csv('spatial-corr/x-neighbor-db-150-binary.csv', index_col=0).fillna(0)
    x_nb_total = pd.read_csv('spatial-corr/x_total-neighbor-db-150-binary.csv', index_col=0).fillna(0)
    x_type = pd.read_csv('data/x_NO_TOTAL_~2014.csv', index_col=0)
    x_total = pd.read_csv('data/x_TOTAL_~2014.csv', index_col=0)

    # final feature
    Xs = {}
    # Xs['RoadNet+Spatial'] = x_type[rcol_type].join(x_nb_type[nb_rcol_type])
    # Xs['Social type+Spatial'] = x_type[scol_type].join(x_nb_type[nb_scol_type])
    Xs['RoadNet+Social type+Spatial'] = x_type[rcol_type + scol_type].join(x_nb_type[nb_rcol_type + nb_scol_type])

    # Xs['Social total+Spatial'] = x_total[scol_total].join(x_nb_total[nb_scol_total])
    # Xs['RoadNet+Social total+Spatial'] = x_total[rcol_total+scol_total].join(x_nb_total[nb_rcol_total+nb_scol_total])

    y = pd.read_csv('data/y_csl_all_%s-2017-10-01.csv' % lvl, index_col=0).csl
    return Xs, y


def init_model_params(name):
    params = grid_cv_default_params()
    if name == 'XGBcls':
        model = xgboost.XGBClassifier()
        param = params['cls']['XGBcls']
    elif name == 'BAGcls':
        model = BaggingClassifier()
        param = params['cls']['BAGcls']
    elif name == 'RFcls':
        model = RandomForestClassifier()
        param = params['cls']['RFcls']
    elif name == 'SVM':
        model = SVC()
        param = params['cls']['SVM']
    elif name == 'GDBcls':
        model = GradientBoostingClassifier()
        param = params['cls']['GDBcls']
    else:
        raise ('no model',)
    return model, param


def exp8_one_run(exp_path, Xs, y, train_idx, test_idx, no_fsel=False, no_upsample=False):
    def upsampling():
        upsampler = SMOTE(kind=up_name, random_state=SMOTE_SEED)
        up_train_x, up_train_y = upsampler.fit_sample(train_x, train_y.round())
        up_y_dist = pd.Series(up_train_y).value_counts().to_dict()
        return up_train_x, up_train_y, up_y_dist

    def grid(model):
        grid_res = grid_cv_a_model(dset['train_x'], dset['train_y'], model, param, kind='cls',
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

    # get train_y and test_y
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]
    train_n_classes = train_y.round().nunique()
    test_n_classes = test_y.round().nunique()
    print('====n classes, train: %d, test: %d' % (train_n_classes, test_n_classes))
    y_dist = train_y.round().value_counts().to_dict()
    print('====train_y: distr=%s' % (y_dist))

    # store result
    eval_res_path = '%s/eval_res.csv' % exp_path
    grid_res_path = '%s/grid_res.csv' % exp_path
    grid_res_list = [] if not os.path.exists(grid_res_path) else pd.read_csv(grid_res_path, index_col=0).to_dict(
        'records')
    eval_res_list = [] if not os.path.exists(eval_res_path) else pd.read_csv(eval_res_path, index_col=0).to_dict(
        'records')

    # iterate combos
    for ftr_combo_name, X in Xs.items():
        print('========ftr_combo_name=%s' % ftr_combo_name)
        # get train x and test_x
        train_x, test_x = X.loc[train_idx], X.loc[test_idx]
        feature_names = train_x.columns

        # for model_name, up_name, fselect_type in MODEL_UP_FSEL_NAMES:
        for up_name, fselect_type in itertools.product(UP_NAMES, FSEL_NAMES):
            if no_fsel and fselect_type != 'None':
                print('skip feature selection = %s' % fselect_type)
                continue
            if no_upsample and up_name != 'None':
                print('skip upsample = %s' % up_name)
                continue

            print('============up=%s,fsel=%s' % (up_name, fselect_type))
            if up_name != 'None':
                try:
                    up_train_x, up_train_y, up_y_dist = upsampling()
                except ValueError as e:
                    print('path=%s, ftr_combo_name=%s, smote=%s' % (exp_path, ftr_combo_name, up_name))
                    print('catch a ValueError: %s' % e)
                    continue
                dset = scale_and_selection(up_train_x, up_train_y, test_x, test_y, fselect_type)
            else:
                dset = scale_and_selection(train_x, train_y, test_x, test_y, fselect_type)

            if fselect_type != 'None':
                print('feature selection: %d -> %d' % (len(feature_names), dset['selected_ftr'].sum()))

                if len(feature_names) == dset['selected_ftr'].sum():
                    cv_path = '%s/%s#%s#%s-fselnothing' % (exp_path, ftr_combo_name, up_name, fselect_type)
                    mkdirs_if_not_exist(cv_path)
                    print('feature selection %s cut down nothing' % fselect_type)
                    continue
            else:
                print('No Feature selection: %d' % len(feature_names))

            cv_path = '%s/%s#%s#%s' % (exp_path, ftr_combo_name, up_name, fselect_type)
            mkdirs_if_not_exist(cv_path)
            print('fitting models for cv_path=%s' % cv_path)

            for model_name in MODEL_NAMES:
                # init model
                model, param = init_model_params(model_name)
                # grid a model, save grid_res to grid_res_list
                model = grid(model)
                # evaluate on original test set, save eval_res to eval_res_list
                eval_on_test()

        # df_res is for a whole run of up_exp, but save to disk with overwrite once a combo is done
        df_grid_res = pd.DataFrame(grid_res_list)
        df_grid_res.to_csv(grid_res_path)
        df_eval_res = pd.DataFrame(eval_res_list)
        df_eval_res.to_csv(eval_res_path)
        # break    # run one combo


def exp8(Xs, y, i, lvl, no_fsel=False, no_upsample=False):
    for seed in SEEDS:
        # set up experiment path
        exp_path = 'experiment_1001/exp10-34class/%s/%s/seed_%d' % (lvl, i, seed)
        mkdirs_if_not_exist(exp_path)
        # get train/test index
        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(y.index, idx_fn, seed)
        print('====begin one run exp, in exp_path=%s' % exp_path)
        exp8_one_run(exp_path, Xs, y, train_idx, test_idx, no_fsel=no_fsel, no_upsample=no_upsample)
        # break  # run one seed


def main():
    start_time = dtm.now()
    for i in [0, 0.1, 0.3, 0.5, 0.7, 0.8]:
        no_fsel = False if i == 0 else True
        no_upsample = False if i == 0 else True
        for lvl in ['3lvl', '4lvl']:
            print('i =', i, 'lvl =', lvl)
            i_str = '%0.2f' % i
            Xs, y = load_data(i, lvl)
            exp8(Xs, y, i_str, lvl, no_fsel=no_fsel, no_upsample=no_upsample)

    end_time = dtm.now()
    print('start at:', start_time, 'end at:', end_time)


if __name__ == '__main__':
    # MODEL_NAMES = ['SVM', 'RFcls', 'XGBcls', 'BAGcls', 'GDBcls']
    MODEL_NAMES = ['GDBcls']
    UP_NAMES = ['None', 'svm', 'regular']

    FSEL_NAMES = ['None', 'rfecv_linsvc', 'mrmr']

    MODEL_UP_FSEL_NAMES = [
        ('SVM', 'None', 'None'),
        ('SVM', 'None', 'rfecv_linsvc'),
        ('SVM', 'svm', 'rfecv_linsvc'),
        ('SVM', 'regular', 'rfecv_linsvc'),
        ('SVM', 'None', 'mrmr'),
        ('SVM', 'svm', 'mrmr'),
        ('SVM', 'regular', 'mrmr'),

        ('RFcls', 'None', 'None'),
        ('RFcls', 'None', 'rfecv_linsvc'),
        ('RFcls', 'svm', 'rfecv_linsvc'),
        ('RFcls', 'regular', 'rfecv_linsvc'),
        ('RFcls', 'None', 'mrmr'),
        ('RFcls', 'svm', 'mrmr'),
        ('RFcls', 'regular', 'mrmr'),

        ('XGBcls', 'None', 'None'),
        ('XGBcls', 'None', 'rfecv_linsvc'),
        ('XGBcls', 'svm', 'rfecv_linsvc'),
        ('XGBcls', 'regular', 'rfecv_linsvc'),
        ('XGBcls', 'None', 'mrmr'),
        ('XGBcls', 'svm', 'mrmr'),
        ('XGBcls', 'regular', 'mrmr'),

        ('BAGcls', 'None', 'None'),
        ('BAGcls', 'None', 'rfecv_linsvc'),
        ('BAGcls', 'svm', 'rfecv_linsvc'),
        ('BAGcls', 'regular', 'rfecv_linsvc'),
        ('BAGcls', 'None', 'mrmr'),
        ('BAGcls', 'svm', 'mrmr'),
        ('BAGcls', 'regular', 'mrmr'),
    ]
    SMOTE_SEED = 10
    main()
