# coding=utf-8

from src.ftr_aggregate import load_features_v2

totals = ['NO_TOTAL', 'TOTAL']
years_choices = ['~2014', '~2016']
feature_types = ['RoadNet', 'Segment', 'RoadNet+Segment']

# RoadNet is indifference across totals and years_choices
ftr_type, total_or_not, year = feature_types[0], totals[0], years_choices[0]
fn = 'data/x_%s.csv' % (ftr_type)
ftr = load_features_v2(total_or_not=total_or_not, year_type=year, feature_type=ftr_type)
ftr.to_csv(fn)

for ftr_type in feature_types[1:]:
    for total_or_not in totals:
        for year in years_choices:
            fn = 'data/x_%s_%s_%s.csv' % (ftr_type, total_or_not, year)
            ftr = load_features_v2(total_or_not=total_or_not, year_type=year, feature_type=ftr_type)
            ftr.to_csv(fn)