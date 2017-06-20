# coding: utf-8

import os
from sklearn import ensemble
from wKit.ML.sk_ml import scaler_by_name, sk_models, grid_cv_default_params, \
    grid_cv_models, evaluate_grid_cv, evaluator_scalable_cls
from src.pred_lts import load_lts_dataset, prepro_lts_dataset
import numpy as np


def write_ftr_names(cv_dir, dataset):
    ftr_name = dataset['ftr_name']
    sup = dataset['selected_ftr']
    keeps = np.array(ftr_name)[sup]
    removes = np.array(ftr_name)[~sup]
    with open(os.path.join(cv_dir, 'feature_names.txt'), 'wb') as f:
        f.write('all\t%d' % len(ftr_name) + '\t' + ', '.join(ftr_name) + '\n')
        f.write('keeps\t%d' % len(keeps) + '\t' + ', '.join(keeps) + '\n')
        f.write('removes\t%d' % len(removes) + '\t' + ', '.join(removes) + '\n')


if __name__ == '__main__':
    years = (2014, 2015, 2016, 2017)
    total_or_not = 'TOTAL'
    scaler_type = 'MinMaxScaler'

    dir0 = 'data/cross_validation/'
    dir1 = 'year_{}-total_{}-norm_{}'.format(years, total_or_not, scaler_type)
    print dir1
    if not os.path.exists(dir0+dir1):
        os.mkdir(dir0+dir1)

    dataset = load_lts_dataset(None, years, total_or_not)

    selections = ['has_value_thres', 'var_thres', 'rfecv_linsvc', 'mrmr']
    for random_state in [0, 100, 291592]:
        for name in selections:
            dir2 = 'rand_{}-ftr_{}'.format(random_state, name)
            cv_dir = dir0+dir1+'/'+dir2
            if not os.path.exists(cv_dir):
                os.mkdir(cv_dir)
            print cv_dir

            scaler = scaler_by_name(scaler_type)
            ds = prepro_lts_dataset(dataset, scaler, 0.0, selection=name, random_state=random_state)
            train_x, train_y, test_x, test_y = ds['train_x'], ds['train_y'], ds['test_x'], ds['test_y']
            write_ftr_names(cv_dir, ds)

            print 'get models and grid_cv tuning parameters'
            models = {'cls': {'RFcls': ensemble.RandomForestClassifier()}}
            params = grid_cv_default_params()

            print 'running grid cv'
            df_cv_res = grid_cv_models(train_x, train_y, models, params, path=cv_dir, verbose=True)
            print 'saved grid cv result for each model'

            print 'evaluating best model of each kind'
            df_eval = evaluate_grid_cv(df_cv_res, train_x, train_y, test_x, test_y, evaluator_scalable_cls, path=cv_dir)
            print

