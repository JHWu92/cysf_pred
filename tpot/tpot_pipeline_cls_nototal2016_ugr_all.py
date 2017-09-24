import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GradientBoostingClassifier(learning_rate=0.01, max_depth=6, max_features=0.6500000000000001, min_samples_leaf=5, min_samples_split=5, n_estimators=100, subsample=1.0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Best pipeline: GradientBoostingClassifier(input_matrix, GradientBoostingClassifier__learning_rate=0.01, GradientBoostingClassifier__max_depth=6, GradientBoostingClassifier__max_features=0.65, GradientBoostingClassifier__min_samples_leaf=5, GradientBoostingClassifier__min_samples_split=5, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=1.0)
# 0.666666666667
#/home/jhwu92/miniconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
#  'precision', 'predicted', average, warn_for)
#  mse= 0.52380952381 acc= 0.666666666667 f1_weighted = 0.637657230914 f1_macro = 0.3326159732'')
