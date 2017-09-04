import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    Normalizer(),
    RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
#Best pipeline: RandomForestRegressor(Normalizer(input_matrix, Normalizer__norm=DEFAULT), RandomForestRegressor__bootstrap=False, RandomForestRegressor__max_features=0.25, RandomForestRegressor__min_samples_leaf=1, RandomForestRegressor__min_samples_split=5, RandomForestRegressor__n_estimators=100)
# 0.427382854791
#/home/jhwu92/miniconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.  'precision', 'predicted', average, warn_for)
# mse= 0.603174603175 acc= 0.539682539683 f1_weighted = 0.5088994003 f1_macro = 0.268173956017