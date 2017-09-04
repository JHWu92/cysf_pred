import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = XGBClassifier(max_depth=1, min_child_weight=12, nthread=1, subsample=0.7500000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Best pipeline: XGBClassifier(input_matrix, XGBClassifier__learning_rate=DEFAULT, XGBClassifier__max_depth=1, XGBClassifier__min_child_weight=12, XGBClassifier__n_estimators=DEFAULT, XGBClassifier__nthread=1, XGBClassifier__subsample=0.75)
# 0.539682539683