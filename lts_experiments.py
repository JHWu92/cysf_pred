# coding=utf-8

from sklearn.model_selection import train_test_split
from wKit.utility.file_sys import mkdirs_if_not_exist
from wKit.ML.scaler import minmax, max_cutoff
from wKit.ML.feature_selection import fselect
import os
import pandas as pd
import numpy as np
from src.constants import dir_data, fn_target_lts_dc, fn_features_dc
from src.ftr_aggregate import load_features
from wKit.ML.sk_ml import sk_models, grid_cv_default_params, grid_cv_models, evaluate_grid_cv, evaluator_scalable_cls, model_order_by_speed
from datetime import datetime as dtm

def get_max_cut_cols(cols_by_type):
    max_cutoff_candidates = ['crash', '311', 'poi', 'crime', 'v0', 'moving', 'parking']
    max_cut_cols = []
    for c in max_cutoff_candidates:
        max_cut_cols += cols_by_type[c]
    return max_cut_cols

def get_idx(lts, idx_fn, seed):
    if not os.path.exists(idx_fn):
        train_idx, test_idx = train_test_split(lts.index, test_size=0.2, random_state=seed)
        with open(idx_fn, 'w') as f:
            f.write('train\t%s\n' % ','.join(train_idx.astype(str).tolist()))
            f.write('test\t%s\n' % ','.join(test_idx.astype(str).tolist()))
    else:
        with open(idx_fn) as f:
            lines = f.readlines()
            train_idx = lines[0].strip().split('\t')[1].split(',')
            train_idx = [int(x) for x in train_idx]
            test_idx = lines[1].strip().split('\t')[1].split(',')
            test_idx = [int(x) for x in test_idx]
    return train_idx, test_idx


def scale_ftr(train_x, test_x, max_cut_cols=None):
    if max_cut_cols is not None:
        print('for', max_cut_cols[:5], '...', len(
            max_cut_cols), 'cols, do a max cut off with max=1000, alpha=0.75 first, then min max scale to [0,1]')
        for col in max_cut_cols:
            train_x[col] = max_cutoff(train_x[col], max_=1000)
            test_x[col] = max_cutoff(test_x[col], max_=1000)
    else:
        print('min max only to [0,1]')
    scaler = minmax()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    return train_x, test_x

def scale_and_selection(train_x, train_y, test_x, test_y, selection_type, max_cut=True, **kwargs):

    print('scale features')
    train_x, test_x = scale_ftr(train_x, test_x, max_cut_cols) if max_cut else scale_ftr(train_x, test_x)

    print('feature selection, choice:', selection_type)
    selected_ftr = None
    selected_ftr = fselect(train_x, train_y, selection_type, **kwargs) if selection_type != 'None' else np.array([True] * train_x.shape[1])

    if selected_ftr is None:
        print('!!!!! =============== selected feature is None =============== !!!!! ')

    train_x = train_x[:, selected_ftr]
    test_x = test_x[:, selected_ftr]
    return {'train_x' : train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'selected_ftr': selected_ftr}


def write_ftr_names(cv_dir, ftr_name, selected):

    keeps = np.array(ftr_name)[selected]
    removes = np.array(ftr_name)[~selected]
    with open(os.path.join(cv_dir, 'feature_names.txt'), 'wb') as f:
        f.write('all\t%d' % len(ftr_name) + '\t' + ', '.join(ftr_name) + '\n')
        f.write('keeps\t%d' % len(keeps) + '\t' + ', '.join(keeps) + '\n')
        f.write('removes\t%d' % len(removes) + '\t' + ', '.join(removes) + '\n')

def grid_eval(ds, cv_dir, ftr_name):
    train_x, train_y, test_x, test_y, selected_ftr = ds['train_x'], ds['train_y'], ds['test_x'], ds['test_y'], ds['selected_ftr']
    write_ftr_names(cv_dir, ftr_name, selected_ftr)

    print('get models and grid_cv tuning parameters')
    models = sk_models(stoplist=('RFcls', 'BAGcls', 'GDBcls'))
    # order = [['cls', ['RFcls', 'BAGcls', 'GDBcls']]]
    order = model_order_by_speed(4)
    params = grid_cv_default_params()

    print('running grid cv')
    df_cv_res = grid_cv_models(train_x, train_y, models, params, order=order, path=cv_dir, verbose=True)
    print('saved grid cv result for each model')

    print('evaluating best model of each kind')
    df_eval = evaluate_grid_cv(df_cv_res, train_x, train_y, test_x, test_y, evaluator_scalable_cls, path=cv_dir)
    print()


if __name__ == "__main__":
    lts = pd.read_csv(dir_data + fn_target_lts_dc, index_col=0)
    totals = ['NO_TOTAL', 'TOTAL']
    years_choices = [('~2014', (2014, 2015, 2016, 2017)), ('~2016', (2016, 2017)), ]
    features = {}
    for total_or_not in totals:
        for year_type, years in years_choices:
            ftrs, cols_by_type = load_features(lts, how=total_or_not, years=years)
            features[(total_or_not, year_type)] = (ftrs, cols_by_type)

    ys = lts.LTS

    for seed in [0, 100, 972, 5258, 7821, 40918, 57852, 168352, 291592, 789729423]:

        exp_path = 'data/experiment/seed_%d' % seed
        mkdirs_if_not_exist(exp_path)
        print(dtm.now(), 'experiment top dir =', exp_path)

        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(ys, idx_fn, seed)

        print('split train and test')

        for total_or_not in ['NO_TOTAL', 'TOTAL']:
            for year_type, years in [('~2014', (2014, 2015, 2016, 2017)), ('~2016', (2016, 2017)), ]:

                print(dtm.now(), 'loading features', total_or_not, year_type)
                ftrs, cols_by_type = features[(total_or_not, year_type)]
                train_x, train_y = ftrs.loc[train_idx], ys.loc[train_idx]
                test_x, test_y = ftrs.loc[test_idx], ys.loc[test_idx]
                ftr_name = train_x.columns

                max_cut_cols = get_max_cut_cols(cols_by_type)

                for max_cut in (True, False,):
                    for selection_type in ['None', 'rfecv_linsvc', 'mrmr']:
                        exp_param = '#'.join(
                            [total_or_not, year_type, 'max_cut' if max_cut else 'minmax_only', selection_type])
                        exp_param_path = '%s/%s' % (exp_path, exp_param)
                        mkdirs_if_not_exist(exp_param_path)
                        print(dtm.now(), 'experiment with', exp_param)
                        gridcv_ready = scale_and_selection(train_x, train_y, test_x, test_y, selection_type, max_cut)
                        grid_eval(gridcv_ready, exp_param_path, ftr_name)
