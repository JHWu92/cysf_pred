import pandas as pd
from constants import fn_features_dc, dir_data, index_seg, features_for_total


def load_joint_features(years=(2014,2015,2016,2017), how='NO_TOTAL', na=None, verbose=False):
    """
    for each feature files:
        1. get dummies, dummy_na=False;
        2. fillna with na if na is not None, default None;
        3. encode each feature names with ftr_name_No
    return joint features and code book
    """
    features = load_separate_features(verbose=verbose, na=na)
    joint_features = []
    ftr_col2code = {}
    for name, df in features.items():
        ftr = df.copy()
        ftr = filter_year(ftr, years=years)
        if name in features_for_total:
            ftr = filter_total(ftr, name, how=how)
        col2code = encode_col(ftr, name)
        ftr = ftr.groupby(level=0).sum()

        joint_features.append(ftr)
        ftr_col2code.update(col2code)

    joint_features = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), joint_features)
    
    return joint_features, ftr_col2code


def load_separate_features(verbose=True, na=0):
    """
    get dummies, dummy_na=False
    fillna with na if na is not None
    """
    features = {}
    for name, fn in fn_features_dc.items():
        if verbose:  print 'loading feature:', fn
        df = pd.read_csv(dir_data + fn, index_col=0)
        if index_seg in df.columns:
            df.set_index('index_seg', inplace=True)
        df = pd.get_dummies(df, dummy_na=False)
        if na is not None:
            df.fillna(na, inplace=True)
        features[name] = df
    return features


def filter_year(ftr, years=(2014, 2015, 2016, 2017), keep_year=False):
    if 'YEAR' in ftr.columns:
        ftr = ftr[ftr.YEAR.isin(years)]
        if not keep_year:
            ftr = ftr[ftr.columns[~ftr.columns.isin(['YEAR', 'MONTH'])]]
    return ftr


def filter_total(ftr, name, how='NO_TOTAL'):
    assert how in ('NO_TOTAL', 'TOTAL'), 'only allow two options: NO_TOTAL and TOTAL'
    if how=='NO_TOTAL':
        new_columns = [c for c in ftr.columns if 'total' not in c]
    else:
        new_columns = [c for c in ftr.columns if 'total' in c]
        if not new_columns:
            ftr[name + '_total'] = ftr.sum(axis=1)
            new_columns = [name + '_total']
    if 'YEAR' in ftr.columns and 'YEAR' not in new_columns:
        new_columns = ['YEAR','MONTH']+new_columns
    return ftr[new_columns]


def encode_col(ftr, name):
    num_col = ftr.shape[1]
    new_col = ['{}_{:03d}'.format(name, i) for i in range(num_col)]
    col2code = dict(zip(ftr.columns, new_col))
    return col2code




