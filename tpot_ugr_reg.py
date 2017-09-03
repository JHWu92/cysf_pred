from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

y_name = 'ugr_all'
X = pd.read_csv('data/%s_x_total_2014.csv' % y_name, index_col=0)
y = pd.read_csv('data/y_%s.csv' % y_name, index_col=0)
# X = X.loc[y.index]
X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float64), y.ugr.astype(np.float64), test_size=0.2, random_state=0)

tpot = TPOTRegressor(verbosity=2, n_jobs=4, random_state=0, generations=100)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline_reg_total2014_%s.py' % y_name)

pred = tpot.predict(X_test).round()
y_test = y_test.round()

mse_train = mean_squared_error(y_test, pred)
acc_train = accuracy_score(y_test, pred)
f1_weighted = f1_score(y_test, pred, average='weighted')
f1_macro = f1_score(y_test, pred, average='macro')

print('mse_train =', mse_train, 'acc_train =', acc_train,  'f1_weighted =', f1_weighted, 'f1_macro =', f1_macro)