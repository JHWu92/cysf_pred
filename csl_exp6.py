# coding=utf-8
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import os
from datetime import datetime as dtm
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from wKit.ML.feature_selection import fselect
from wKit.ML.scaler import minmax, max_cutoff
from wKit.utility.file_sys import mkdirs_if_not_exist
from wKit.ML.sk_ml import (sk_models, grid_cv_default_params, grid_cv_models, evaluate_grid_cv, evaluator_scalable_cls,
                           model_order_by_speed, show_important_features, confusion_matrix_as_df)

TEST_SIZE = 0.2
MAX_CUTOFF_CANDIDATES = ['crash', '311', 'poi', 'crime', 'v0', 'moving', 'parking']


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


def cfsn_imp(df_cv_res, cv_dir, feature_names, test_x, test_y):

    for (kind, name), model in df_cv_res.best_model.iteritems():
        try:
            imp = show_important_features(model, labels=feature_names, set_std=False, show_plt=False).drop('std', axis=1)
            imp.columns = ['label', 'importance']
            imp.to_csv('%s/imp_%s.csv' %(cv_dir, name), encoding='utf8')
        except AttributeError as e:
            print(name, 'no feature importance')

        cfsn = confusion_matrix_as_df(model, test_x, test_y, labels=[1, 2, 3, 4, 5])
        cfsn.to_csv('%s/cfsn_%s.csv' %(cv_dir, name), encoding='utf8')


def grid_eval(ds, cv_dir, ftr_name):
    train_x, train_y, test_x, test_y, selected_ftr = ds['train_x'], ds['train_y'], ds['test_x'], ds['test_y'], ds[
        'selected_ftr']
    write_ftr_names(cv_dir, ftr_name, selected_ftr)

    print('get models and grid_cv tuning parameters')
    models = sk_models(stoplist=())
    order = model_order_by_speed(speed=3)
    params = grid_cv_default_params()

    print('running grid cv')
    df_cv_res = grid_cv_models(train_x, train_y, models, params, order=order, path=cv_dir, verbose=True, redo=True, n_jobs=6)
    print('saved grid cv result for each model')

    print('evaluating best model of each kind')
    df_eval = evaluate_grid_cv(df_cv_res, train_x, train_y, test_x, test_y, evaluator_scalable_cls, path=cv_dir)
    print()

    print('get confusion matrix and feature importance')
    selected_ftr_name = np.array(ftr_name)[selected_ftr]
    # cfsn_imp(df_cv_res, cv_dir, selected_ftr_name, test_x, test_y)

    return df_eval


# ======
# experiment functions
# ======


def baseline_majority_vote(single_pred, test_y, exp_path):
    print('majority vote')
    pred = [single_pred] * test_y.shape[0]

    res = {'model'           : 'majority_vote',
           'test_f1_weighted': f1_score(test_y.round(), pred, average='weighted'),
           'test_f1_micro'   : f1_score(test_y.round(), pred, average='micro'),
           'test_f1_macro'   : f1_score(test_y.round(), pred, average='macro')}
    df = pd.DataFrame.from_dict([res]).set_index('model')
    df.to_csv('%s/majority_vote.csv' % exp_path)
    return df


def exp_roadnet(y, train_idx, test_idx, exp_path):
    feature_group_name = 'RoadNet'
    feature = pd.read_csv('data/x_RoadNet.csv', index_col=0)
    train_x, test_x = feature.loc[train_idx], feature.loc[test_idx]
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]
    ftr_name = train_x.columns

    for selection_type in ['None',]:
        exp_param = feature_group_name + '#' + selection_type
        exp_param_path = '%s/%s' % (exp_path, exp_param)
        mkdirs_if_not_exist(exp_param_path)
        print(dtm.now(), 'experiment with', exp_param)
        scaled_selected_data = scale_and_selection(train_x, train_y, test_x, test_y, selection_type, max_cut_cols=None)
        grid_eval(scaled_selected_data, exp_param_path, ftr_name)


def exp_else(y, train_idx, test_idx, exp_path):
    totals = ['NO_TOTAL', 'TOTAL']
    years_choices = ['~2014']
    feature_types = ['Segment', 'RoadNet+Segment']
    train_y, test_y = y.loc[train_idx], y.loc[test_idx]

    for total_or_not in totals:
        for year in years_choices:
            cols_by_type=pickle.load(open('data/x_%s_%s_cols_by_type.pkl' % (total_or_not, year), 'rb'))
            max_cut_cols = get_max_cut_cols(cols_by_type)

            for ftr_type in feature_types:
                fn = 'data/x_%s_%s_%s.csv' % (ftr_type, total_or_not, year)
                feature = pd.read_csv(fn, index_col=0)
                train_x, test_x = feature.loc[train_idx], feature.loc[test_idx]
                ftr_name = train_x.columns

                for max_cut in (False,):
                    for selection_type in ['None',]:
                        exp_param = '#'.join(
                            [ftr_type, total_or_not, year, 'max-cutoff' if max_cut else 'min-max', selection_type])
                        exp_param_path = '%s/%s' % (exp_path, exp_param)
                        mkdirs_if_not_exist(exp_param_path)
                        print(dtm.now(), 'experiment with', exp_param)
                        gridcv_ready = scale_and_selection(train_x, train_y, test_x, test_y, selection_type,
                                                           max_cut_cols=max_cut_cols if max_cut else None)
                        grid_eval(gridcv_ready, exp_param_path, ftr_name)
    return


def exp1_one_y(y, weight_name):
    target_index_seg = y.index
    mode_of_round_y = y.round().mode().values[0]

    for seed in [0, 100, 972, 5258, 7821, 40918, 57852, 168352, 291592, 789729423][5:]:
        exp_path = 'experiment_1001/exp6/%s/seed_%d' % (weight_name, seed)
        mkdirs_if_not_exist(exp_path)
        print(dtm.now(), 'experiment top dir =', exp_path)

        idx_fn = '%s/%s' % (exp_path, 'indices.txt')
        train_idx, test_idx = get_idx(target_index_seg, idx_fn, seed)

        baseline_majority_vote(mode_of_round_y, y.loc[test_idx], exp_path)
        exp_roadnet(y, train_idx, test_idx, exp_path)
        exp_else(y, train_idx, test_idx, exp_path)


def main():
    start_time = dtm.now()
    for weight_name in [
        # '4lvl',
        # 'amplify_fcir', 'amplify_fr',
        # 'lvl_fearless_1st', 'ext_lvl_fearless_1st',
        # 'lvl_reluctant_1st', 'ext_lvl_reluctant_1st',
        # 'fam_include_noinfo', 'ext_fam_include_noinfo',
        # 'fam_exclude_noinfo', 'ext_fam_exclude_noinfo',
        'amplify_fcir2',
        # 'amplify_fcir3',
        # 'amplify_fcir3_fam_include_noinfo',
    ]:
        if weight_name != '4lvl':
            weight_name = '4lvl_' + weight_name
        print('weight_name', weight_name)
        fn = 'data/y_csl_all_%s-2017-10-01.csv' % weight_name
        csl = pd.read_csv(fn, index_col=0)
        y = csl.csl
        exp1_one_y(y, weight_name)

    end_time = dtm.now()
    print('start at:', start_time, 'end at:', end_time)

if __name__ == '__main__':
    main()
