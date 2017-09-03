from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

y_name = 'ugr_all'
X = pd.read_csv('data/%s_x_nototal_2014.csv' % y_name, index_col=0)
y = pd.read_csv('data/y_%s.csv' % y_name, index_col=0)
y=y.round()
# X = X.loc[y.index]
X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float64), y.ugr.astype(np.float64), test_size=0.2, random_state=0)

tpot = TPOTClassifier(verbosity=2, n_jobs=4, random_state=0)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline_cls_%s.py' % y_name)