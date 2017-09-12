import pandas as pd
from wKit.ML.feature_selection import has_value_thres

from .constants import fn_features_dc, dir_data, features_for_total
from wKit.ML.dprep import fillna_group_mean
from functools import reduce

import os
import pickle

ROAD_NET_FTR = ['seg_attr', 'net_SaN', 'net_SaE', 'bk_osm', 'bk_opendc', 'elevation']
SEGMENT_FTR = ['poi', 'crash', '311', 'v0', 'crime', 'moving', 'parking']
SEG_TYPE_PATH = 'data/seg_street_type.csv'
FILLNA_BY_GROUP_NAMES = ['moving', 'parking', 'crash']


def load_features_v2(total_or_not='TOTAL', year_type='~2014', feature_type='RoadNet'):

    years = {'~2014': (2014, 2015, 2016, 2017), '~2016': (2016, 2017)}[year_type]
    Xs_fn = 'data/x_%s_%s.csv' %(total_or_not, year_type)
    cols_by_type_fn = 'data/x_%s_%s_cols_by_type.pkl'%(total_or_not, year_type)

    if os.path.exists(Xs_fn):
        ftr = pd.read_csv(Xs_fn, index_col=0)
        cols_by_type = pickle.load(open(cols_by_type_fn , 'rb'))
    else:
        ftr, cols_by_type = load_features(None,how=total_or_not, years=years, pair_with_y=False)

    keep_ftr = []
    keep_ftr_type = {'RoadNet': ROAD_NET_FTR, 'Segment': SEGMENT_FTR,
                     'RoadNet+Segment': ROAD_NET_FTR + SEGMENT_FTR}[feature_type]
    for ftr_type, ftr_cols in cols_by_type.items():
        if ftr_type in keep_ftr_type:
            keep_ftr += ftr_cols

    return ftr[keep_ftr]


def load_features(ys, drop_na_thres=0.1, how='TOTAL', years=(2014, 2015, 2016, 2017), verbose=False,
                  y_column_name='LTS', pair_with_y=True):
    """
    filter data with year month by year
    filter data to either total count only or divided by different types
    filter data(columns) if has values< thres (whole population is decided by ys.index)
    fill na with group means for ['moving', 'parking', 'crash']
    """
    seg_type = pd.read_csv(SEG_TYPE_PATH)
    if pair_with_y:
        seg_type = seg_type[seg_type['index'].isin(ys.index)]

    num_org_cols = 0
    num_filtered_cols = 0
    joint_features = []
    cols_by_type = {}
    for name, fn in fn_features_dc.items():
        if verbose: print('loading', name, fn)
        ftr = pd.read_csv(dir_data + fn, index_col=0)
        ftr = filter_year(ftr, years=years)

        if name in features_for_total:
            ftr = filter_total(ftr, name, how=how)

        # ftr aggregated to one item matches one segment
        ftr = ftr.groupby(level=0).sum()

        # pair rows by y, if not, add missing rows from seg_type
        if pair_with_y:
            ftr = ys.merge(ftr, left_index=True, right_index=True, how='left').drop(y_column_name, axis=1)
        else:
            ftr = seg_type.set_index('index').merge(ftr, left_index=True, right_index=True, how='left').drop('name', axis=1)
            ftr.index.name = 'index_seg'

        # filter columns with too many NA
        keep_col = has_value_thres(ftr, thres=drop_na_thres)
        keep_col = keep_col[keep_col].index.tolist()
        if verbose: print ('all columns #:', ftr.shape[1], 'columns pass NA thres:', len(keep_col), '\n')
        num_org_cols += ftr.shape[1]
        num_filtered_cols += len(keep_col)
        ftr = ftr[keep_col]
        cols_by_type[name] = keep_col

        # fillna by means of segment types, if applicable
        if name in FILLNA_BY_GROUP_NAMES:
            ftr = fillna_group_mean(ftr, seg_type)

        joint_features.append(ftr)

    joint_features = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), joint_features)

    print('filter columns with > 90% NAs', num_org_cols, '->', num_filtered_cols)
    print(FILLNA_BY_GROUP_NAMES, 'replace NA by seg type')
    print('fill the rest NA with 0')
    joint_features.fillna(0, inplace=True)
    return joint_features, cols_by_type


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




