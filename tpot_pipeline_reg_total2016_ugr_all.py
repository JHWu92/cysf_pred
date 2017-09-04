import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=1.0, loss="huber", max_depth=7, max_features=0.6000000000000001, min_samples_leaf=1, min_samples_split=7, subsample=0.35000000000000003)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Best pipeline: ExtraTreesRegressor(GradientBoostingRegressor(input_matrix, GradientBoostingRegressor__alpha=0.95, GradientBoostingRegressor__learning_rate=1.0, GradientBoostingRegressor__loss=huber, GradientBoostingRegressor__max_depth=7, GradientBoostingRegressor__max_features=0.6, GradientBoostingRegressor__min_samples_leaf=1, GradientBoostingRegressor__min_samples_split=7, GradientBoostingRegressor__n_estimators=DEFAULT, GradientBoostingRegressor__subsample=0.35), ExtraTreesRegressor__bootstrap=False, ExtraTreesRegressor__max_features=0.1, ExtraTreesRegressor__min_samples_leaf=1, ExtraTreesRegressor__min_samples_split=3, ExtraTreesRegressor__n_estimators=100)
# 0.422652753563
#/home/jhwu92/miniconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
#  'precision', 'predicted', average, warn_for)
# mse= 0.539682539683 acc= 0.603174603175 f1_weighted = 0.564979826046 f1_macro = 0.309020791415