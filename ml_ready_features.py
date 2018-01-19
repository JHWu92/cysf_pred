# coding=utf-8
# DEPRECATED
from src.ftr_aggregate import load_features
import pandas as pd
from src.constants import dir_data
import pickle

totals = ['NO_TOTAL', 'TOTAL']
years_choices = [('~2014', (2014, 2015, 2016, 2017)), ('~2016', (2016, 2017)), ]

features = {}
for total_or_not in totals:
    for year_type, years in years_choices:
        ftrs, cols_by_type = load_features(None, how=total_or_not, years=years, pair_with_y=False)
        features[(total_or_not, year_type)] = (ftrs, cols_by_type)
        ftrs.to_csv('data/x_%s_%s.csv' %(total_or_not, year_type))
        with open('data/x_%s_%s_cols_by_type.pkl' %(total_or_not, year_type), 'wb') as f:
            pickle.dump(cols_by_type, f)
