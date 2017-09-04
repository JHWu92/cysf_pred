import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.2, min_samples_leaf=5, min_samples_split=5, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=5, min_samples_split=7, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Best pipeline: ExtraTreesRegressor(ExtraTreesRegressor(input_matrix, ExtraTreesRegressor__bootstrap=True, ExtraTreesRegressor__max_features=0.2, ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=5, ExtraTreesRegressor__n_estimators=100), ExtraTreesRegressor__bootstrap=False, ExtraTreesRegressor__max_features=0.65, ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=7, ExtraTreesRegressor__n_estimators=100)
#0.375465437348
#/home/jhwu92/miniconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
#  'precision', 'predicted', average, warn_for)
#mse= 0.507936507937 acc= 0.68253968254 f1_weighted = 0.653781315204 f1_macro = 0.395684616341