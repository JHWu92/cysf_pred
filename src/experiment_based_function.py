# coding=utf-8
import os
import numpy as np
from sklearn.model_selection import train_test_split

from wKit.ML.feature_selection import fselect
from wKit.ML.scaler import minmax, max_cutoff
from wKit.ML.sk_ml import (sk_models, grid_cv_default_params, grid_cv_models, evaluate_grid_cv, evaluator_scalable_cls,
                           model_order_by_speed)

SEEDS = [0, 100, 972, 5258, 7821, 40918, 57852, 168352, 291592, 789729423]
TEST_SIZE = 0.2
MAX_CUTOFF_CANDIDATES = ['crash', '311', 'poi', 'crime', 'v0', 'moving', 'parking']
FSELECT_TYPE = ['None', 'rfecv_linsvc', 'mrmr']

def get_max_cut_cols(cols_by_type):
    max_cut_cols = []
    for c in MAX_CUTOFF_CANDIDATES:
        max_cut_cols += cols_by_type[c]
    return max_cut_cols


def write_ftr_names(cv_dir, ftr_name, selected):
    keeps = np.array(ftr_name)[selected]
    removes = np.array(ftr_name)[~selected]
    with open(os.path.join(cv_dir, 'feature_names.txt'), 'w') as f:
        f.write('all\t%d' % len(ftr_name) + '\t' + ', '.join(ftr_name) + '\n')
        f.write('keeps\t%d' % len(keeps) + '\t' + ', '.join(keeps) + '\n')
        f.write('removes\t%d' % len(removes) + '\t' + ', '.join(removes) + '\n')


def get_idx(indices, idx_fn, seed):
    if not os.path.exists(idx_fn):
        train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=seed)
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


def scale_and_selection(train_x, train_y, test_x, test_y, selection_type, max_cut_cols=None, **kwargs):
    print('scale features')
    train_x, test_x = scale_ftr(train_x, test_x, max_cut_cols) if max_cut_cols is not None else scale_ftr(train_x,
                                                                                                          test_x)

    print('feature selection, choice:', selection_type)
    selected_ftr = None
    selected_ftr = fselect(train_x, train_y, selection_type, **kwargs) if selection_type != 'None' else np.array(
        [True] * train_x.shape[1])

    if selected_ftr is None:
        print('!!!!! =============== selected feature is None =============== !!!!! ')

    train_x = train_x[:, selected_ftr]
    test_x = test_x[:, selected_ftr]
    return {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'selected_ftr': selected_ftr}


def grid_eval(ds, cv_dir, ftr_name):
    train_x, train_y, test_x, test_y, selected_ftr = ds['train_x'], ds['train_y'], ds['test_x'], ds['test_y'], ds[
        'selected_ftr']
    write_ftr_names(cv_dir, ftr_name, selected_ftr)

    print('get models and grid_cv tuning parameters')
    models = sk_models(stoplist=())
    order = model_order_by_speed(speed=3)
    params = grid_cv_default_params()

    print('running grid cv')
    df_cv_res = grid_cv_models(train_x, train_y, models, params, order=order, path=cv_dir, verbose=True, redo=True)
    print('saved grid cv result for each model')

    print('evaluating best model of each kind')
    df_eval = evaluate_grid_cv(df_cv_res, train_x, train_y, test_x, test_y, evaluator_scalable_cls, path=cv_dir, range=(1,5))
    print()
    return df_eval