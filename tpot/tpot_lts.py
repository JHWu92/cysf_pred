from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X = pd.read_csv('data/x_nototal_2014.csv', index_col=0)
y = pd.read_csv('data/y.csv', index_col=0)

X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float64), y.LTS.astype(np.float64), test_size=0.2)

tpot = TPOTClassifier(verbosity=2, n_jobs=3, random_state=0)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_lts_pipeline.py')