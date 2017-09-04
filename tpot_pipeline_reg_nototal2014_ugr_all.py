import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, max_depth=4, max_features=0.3, min_samples_leaf=19, min_samples_split=7, n_estimators=100, subsample=0.2)),
    Binarizer(threshold=0.8),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.45, min_samples_leaf=5, min_samples_split=17)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

#Best pipeline: ExtraTreesRegressor(LassoLarsCV(Binarizer(GradientBoostingRegressor(input_matrix, GradientBoostingRegressor__alpha=0.75, GradientBoostingRegressor__learning_rate=0.1, GradientBoostingRegressor__loss=DEFAULT, GradientBoostingRegressor__max_depth=4, GradientBoostingRegressor__max_features=0.3, GradientBoostingRegressor__min_samples_leaf=19, GradientBoostingRegressor__min_samples_split=7, GradientBoostingRegressor__n_estimators=100, GradientBoostingRegressor__subsample=0.2), Binarizer__threshold=0.8), LassoLarsCV__normalize=True), ExtraTreesRegressor__bootstrap=False, ExtraTreesRegressor__max_features=0.45, ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=17, ExtraTreesRegressor__n_estimators=DEFAULT)
#0.5193420325