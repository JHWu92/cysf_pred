import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = ExtraTreesClassifier(criterion="gini", max_features=0.5, min_samples_leaf=7, min_samples_split=19)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

#Best pipeline: ExtraTreesClassifier(input_matrix, ExtraTreesClassifier__bootstrap=DEFAULT, ExtraTreesClassifier__criterion=gini, ExtraTreesClassifier__max_features=0.5, ExtraTreesClassifier__min_samples_leaf=7, ExtraTreesClassifier__min_samples_split=19, ExtraTreesClassifier__n_estimators=DEFAULT)
#0.539682539683
#/home/jhwu92/miniconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
#  'precision', 'predicted', average, warn_for)
#mse= 0.603174603175 acc= 0.539682539683 f1_weighted = 0.505854761208 f1_macro = 0.275705179587