# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split

from wKit.ML.sk_ml import fillna
from wKit.ML.feature_selection import fselect
from constants import dir_data, fn_target_lts_dc
from ftr_aggregate import load_joint_features
import datetime

def prepare_lts_dataset(scaler, fna=0.0, years=(2014, 2015, 2016, 2017), total_or_not='total', return_type='list'):
    assert return_type in ('list', 'dict'), 'allowed return type: "list" or "dict"'

    print 'loading feature and fill NAN'
    ftr, col2code = load_joint_features(years=years, how=total_or_not)
    ftr_name = list(ftr.columns)
    ftr = fillna(ftr, how=fna)

    print 'loading LTS and remove 10'
    lts = pd.read_csv(dir_data + fn_target_lts_dc, index_col=0)
    lts = lts[lts.LTS != 10].dropna()

    print 'creating train and test set'
    dataset = lts.merge(ftr, left_index=True, right_index=True)
    train, test = train_test_split(dataset, test_size=0.2, random_state=0)
    train_y = train.LTS
    train_x = train.drop('LTS', axis=1)
    test_y = test.LTS
    test_x = test.drop('LTS', axis=1)

    print 'normalizing X'
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    if return_type == 'list':
        return train_x, train_y, test_x, test_y, ftr_name, col2code
    elif return_type == 'dict':
        return {'train_x' : train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y,
                'ftr_name': ftr_name, 'col2code': col2code}
    else:
        return None


def load_lts_dataset(fna=None, years=(2014, 2015, 2016, 2017), total_or_not='total'):
    print 'loading feature and fillna with', fna
    ftr, col2code = load_joint_features(years=years, how=total_or_not, na=fna)
    ftr_name = list(ftr.columns)
    ftr = fillna(ftr, how=fna)

    print 'loading LTS and remove 10'
    lts = pd.read_csv(dir_data + fn_target_lts_dc, index_col=0)
    lts = lts[lts.LTS != 10].dropna()

    dataset = lts.merge(ftr, left_index=True, right_index=True)

    return {'data': dataset, 'ftr_name': ftr_name, 'col2code': col2code}


def prepro_lts_dataset(dataset, scaler, fna=0.0, selection='has_value_thres', return_type='dict', **kwargs):
    """

    Parameters
    ----------
    :param dataset: dataset loaded from prepare_lts_dataset
    :param scaler: sklearn normalization/standardization
    :param fna: fillna method. default 0.0
    :param selection: feature selection name. See wKit.ML.feature_selection.help() for detail
    :param return_type: list or dict.
    :param kwargs:
        random_state: for train_test_split, default 0
        cv: k-fold cv, for rfecv_* feature selection
        n_jobs: num of threads for rfecv_* feature selection
        param: param for some feature selection model. rfecv_*, stability_*
        thres: for thres based selection: has_value_thres, var_thres

    :return: dict or list
    """
    assert return_type in ('list', 'dict'), 'allowed return type: "list" or "dict"'

    data, ftr_name, col2code = dataset['data'], dataset['ftr_name'], dataset['col2code']
    selected_ftr = None

    random_state = kwargs.get('random_state', 0)
    print datetime.datetime.now(), 'creating train and test set, with random_state =', random_state
    train, test = train_test_split(data, test_size=0.2, random_state=random_state)
    train_y = train.LTS
    train_x = train.drop('LTS', axis=1)
    test_y = test.LTS
    test_x = test.drop('LTS', axis=1)

    if selection == 'has_value_thres':  # perform this selection before fillna
        print datetime.datetime.now(), 'feature selection with', selection
        selected_ftr = fselect(train_x, train_y, selection, **kwargs)

    train_x = fillna(train_x, fna)
    test_x = fillna(test_x, fna)  # TODO: fillna with train_x's mean

    print datetime.datetime.now(), 'normalizing X'
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    if selection != 'has_value_thres':
        print datetime.datetime.now(), 'feature selection with', selection
        selected_ftr = fselect(train_x, train_y, selection, **kwargs)

    if selected_ftr is None:
        print '!!!!! =============== selected feature is None =============== !!!!! '

    train_x = train_x[:, selected_ftr]
    test_x = test_x[:, selected_ftr]

    if return_type == 'list':
        return train_x, train_y, test_x, test_y, ftr_name, col2code
    elif return_type == 'dict':
        return {'train_x' : train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y,
                'ftr_name': ftr_name, 'col2code': col2code, 'selected_ftr': selected_ftr}
    else:
        return None
