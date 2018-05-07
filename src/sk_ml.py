# coding=utf-8
import os
import warnings
from datetime import datetime as dtm

import numpy as np
import pandas as pd
from sklearn import linear_model, svm, tree, ensemble, neural_network, naive_bayes
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, confusion_matrix, precision_score, \
    recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from wKit.stat.tests import significant_level, krutest, f_oneway
from wKit.utility.check_dtype import all_int_able, check_type, all_float

try:
    import xgboost
except ImportError:
    xgboost = None


def sk_models(reg=True, cls=True, stoplist=()):
    """
    return sk models with names by regression and/or classification.
    default stoplist is ('SVM', 'SVR', 'GDBreg', 'GDBcls') because they are too slow
    """
    reg_models = {
        'ols'      : linear_model.LinearRegression(),
        'ridge'    : linear_model.Ridge(),
        'lasso'    : linear_model.Lasso(),
        'DTreg'    : tree.DecisionTreeRegressor(),
        'RFreg'    : ensemble.RandomForestRegressor(),
        'ADAreg'   : ensemble.AdaBoostRegressor(),
        'BAGreg'   : ensemble.BaggingRegressor(),
        'GDBreg'   : ensemble.GradientBoostingRegressor(),
        'SVR'      : svm.SVR(),
        'linearSVR': svm.LinearSVR(),
        'MLPreg'   : neural_network.MLPRegressor(),
    }

    cls_models = {
        'logistics': linear_model.LogisticRegression(),
        'DTcls'    : tree.DecisionTreeClassifier(),
        'RFcls'    : ensemble.RandomForestClassifier(),
        'ADAcls'   : ensemble.AdaBoostClassifier(),
        'BAGcls'   : ensemble.BaggingClassifier(),
        'GDBcls'   : ensemble.GradientBoostingClassifier(),
        'SVM'      : svm.SVC(),
        'linearSVM': svm.LinearSVC(),
        'MLPcls'   : neural_network.MLPClassifier(),
        # 'GNBcls'   : naive_bayes.GaussianNB(),  # doesn't accept sparse matrix
    }

    if xgboost is not None:
        reg_models['XGBreg'] = xgboost.sklearn.XGBRegressor()
        cls_models['XGBcls'] = xgboost.sklearn.XGBClassifier()

    models = {}
    if reg:
        for name in stoplist: reg_models.pop(name, None)
        models['reg'] = reg_models
    if cls:
        for name in stoplist: cls_models.pop(name, None)
        models['cls'] = cls_models
    return models


def model_order_by_speed(speed=2):
    """ 4 level of speed:
    0: fast
    1: fast+medium
    2: fast+medium+slow(default)
    else: fast+medium+slow + way_slow
    """
    # grid cv with default parameters option on: 10k * 800
    fast = [['reg', ['ols', 'lasso', 'ridge', 'DTreg', 'linearSVR']],
            ['cls', ['logistics', 'DTcls', 'linearSVM']]]  # less than 5 min
    medium = [['reg', ['ADAreg', 'RFreg', 'MLPreg']],
              ['cls', ['ADAcls', 'RFcls', 'MLPcls']]]  # 5 min ~ 40 min
    slow = [['reg', ['BAGreg']], ['cls', ['BAGcls']]]  # ~ 1h
    way_slow = [['reg', ['SVR', 'GDBreg']], ['cls', ['SVM', 'GDBcls']]]  # 2h~8h
    if xgboost is not None:
        way_slow += [['reg', ['XGBreg']], ['cls', ['XGBcls']]]

    if speed == 0:
        return fast
    if speed == 1:
        return fast + medium
    if speed == 2:
        return fast + medium + slow
    return fast + medium + slow + way_slow


# ################################################
# helper
# ################################################

def bounded_round(arr, mini, maxi):
    _, bins = np.histogram(arr, int(maxi-mini)+1, range=[mini, maxi])
    arr_round = np.digitize(arr, bins)
    # arr_round = arr.round()
    # arr_round[arr_round < mini] = mini
    # arr_round[arr_round > maxi] = maxi
    return arr_round


# ################################################
# pre-processing
# ################################################

def fillna(df, how='mean'):
    """df is the dataset
    """
    if how is None or how == 'None':
        return df
    if how == 'mean':
        return df.fillna(df.mean())
    return df.fillna(how)


def scaler_by_name(name):
    """return sklearn scaler by name (MinMaxScaler,)"""
    norm_choices = {'MinMaxScaler': MinMaxScaler()}
    return norm_choices[name]


# ################################################
# Grid search cross validation
# ################################################

def grid_cv_default_params():
    # TODO: read about Hyperopt https://github.com/hyperopt/hyperopt/wiki/FMin
    # GDBreg's parameters are deliberately cut down.
    params_gdb = {'n_estimators' : [10, 50, 100], 'max_features': ['sqrt', 'log2'],
                  'learning_rate': np.logspace(-4, 1, 3),
                  'max_depth'    : [3, 10, 50],
                  'random_state' : [0]}
    params_rf = {'n_estimators': [10, 100, 500], 'max_features': ['sqrt', 'log2'], 'min_samples_leaf': [1, 2],
                 'random_state': [0]}
    params_ada = {'n_estimators': [10, 30, 50, 100, 256, 500], 'learning_rate': np.logspace(-4, 1, 5),
                  'random_state': [0]}
    params_bag = {'n_estimators': [10, 30, 50, 100, 256, 500], 'max_features': [0.4, 0.7, 1.0], 'random_state': [0]}

    params_mlp = {'hidden_layer_sizes': [(100,), (5, 2), (20, 5), (100, 20), (100, 20, 5)],
                  'learning_rate'     : ['constant', 'adaptive'], 'max_iter': [10000], 'random_state': [0]}

    # SVM/SVR is way too slow
    c_s = np.logspace(-4, 2, 3)
    gamma_s = [1e-5, 1e-3, 1e-1]

    params_svm = [
        {'kernel': ['rbf'], 'C': c_s, 'gamma': gamma_s, 'random_state': [0]},
        {'kernel': ['sigmoid'], 'C': c_s, 'gamma': gamma_s, 'random_state': [0]},
        {'kernel': ['poly'], 'C': c_s, 'gamma': gamma_s, 'degree': [3], 'random_state': [0]},
    ]

    params_svr = [
        {'kernel': ['rbf'], 'C': c_s, 'gamma': gamma_s},
        {'kernel': ['sigmoid'], 'C': c_s, 'gamma': gamma_s},
        {'kernel': ['poly'], 'C': c_s, 'gamma': gamma_s, 'degree': [3]},
    ]
    params_xgb = {
        'min_child_weight': [1, 5], 'gamma': [0, 0.1],
        'learning_rate'   : [0.01, 0.05, 0.1, 0.3], 'colsample_bytree': [0.7, 1],
        'subsample'       : [0.7, 1],
        'silent'          : [1],
        'nthread'         : [1],
        # without it, XGBClassifier will hang, or XGBReg will train really slow. Bad interaction with n_jobs in GridSearchCV
        'n_estimators'    : [500], 'random_state': [0]
    }
    # params_xgb={}
    params_reg = {
        # regression
        'ols'      : {},
        'ridge'    : {'alpha': np.logspace(0, 2, 10), 'random_state': [0]},
        'lasso'    : {'alpha': np.logspace(0, 2, 10), 'random_state': [0]},
        'DTreg'    : {'max_depth': [3, 5, 10, 30, 50], 'max_features': [0.1, 0.3, 0.5, 1.], 'random_state': [0]},
        'RFreg'    : params_rf,
        'ADAreg'   : params_ada,
        'BAGreg'   : params_bag,
        'GDBreg'   : params_gdb,
        'MLPreg'   : params_mlp,
        'SVR'      : params_svr,
        'linearSVR': {'C'           : c_s, 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'epsilon'     : [0, 0.1, 1], 'random_state': [0]},
        'XGBreg'   : params_xgb,
    }

    params_cls = {
        'GNBcls'   : {},
        'logistics': {'C': np.logspace(-4, 2, 4), 'penalty': ['l1', 'l2']},
        'DTcls'    : {'max_depth': [3, 5, 10, 30, 50], 'max_features': [0.1, 0.3, 0.5, 1.],
                      'criterion': ['gini', 'entropy'], 'random_state': [0]},
        'RFcls'    : params_rf,
        'ADAcls'   : params_ada,
        'BAGcls'   : params_bag,
        'GDBcls'   : params_gdb,
        'SVM'      : params_svm,
        'MLPcls'   : params_mlp,
        'linearSVM': {'C': c_s, 'loss': ['hinge', 'squared_hinge'], 'random_state': [0]},
        'XGBcls'   : params_xgb,
    }

    return {'cls': params_cls, 'reg': params_reg}


def grid_cv_a_model(x, y, model, param, kind, name, path='', n_jobs=4, cv=5, verbose=False, redo=False, save_res=True,
                    fit_when_load=True, range=None):
    """
    cv a model, return cv result of the best model
    if model result exists, load the existing best parameters setting, but without grid_cv_time
    when loading model, run empty model.fit(x,y) if fit_when_load=True
    """
    scoring = 'neg_mean_squared_error' if kind == 'reg' else 'f1_weighted'
    path_model_res = os.path.join(path, 'cv_%d_model_%s.csv' % (cv, name))

    if range is None:
        min_y, max_y = y.min(), y.max()
    else:
        min_y, max_y = range

    if kind == 'cls' and all_float(y):
        y = bounded_round(y,min_y,max_y)
        if len(set(y)) > 100:
            print("grid_cv_a_model: rounded y has more than 100 labels")

    if os.path.exists(path_model_res) and not redo:
        print('loading existing model', kind, name)
        model_res = pd.read_csv(path_model_res, index_col=0)
        best = model_res.iloc[0].to_dict()
        param = eval(best['params'])
        if 'n_jobs' in model.get_params().keys():
            model.set_params(n_jobs=n_jobs)
        model.set_params(**param)
        if fit_when_load:
            if verbose:
                print('fitting model', kind, name)
            model.fit(x, y)
        result = {'grid_cv_time' : None, 'score': scoring, 'model_name': name, 'kind': kind,
                  'mean_test'    : best['mean_test_score'], 'mean_train': best['mean_train_score'],
                  'mean_fit_time': best['mean_fit_time'], 'best_params': param, 'best_model': model,
                  }
        print('loaded existing result for model:', name)
        return result

    sub_start = dtm.now()
    print(sub_start, 'CVing: kind = {}, model = {}'.format(kind, name))

    if name in ('BAGcls', 'BAGreg') and x.shape[1] < 4:
        param = {'n_estimators': [10, 30, 50, 100, 256, 500]}
    if name in ('XGBreg', 'XGBcls') and x.shape[1] < 4:
        import copy
        param = copy.deepcopy(param)
        param['colsample_bytree'] = [1]

    clf = GridSearchCV(model, param, n_jobs=n_jobs, cv=cv, scoring=scoring, verbose=1)
    clf.fit(x, y)
    sub_end = dtm.now()

    df = pd.DataFrame(clf.cv_results_).sort_values(by='mean_test_score', ascending=False)
    if save_res:
        df.to_csv(path_model_res)

    test_score, train_score, fit_time = df[['mean_test_score', 'mean_train_score', 'mean_fit_time']].values[0]

    if verbose:
        print('score: %s, best test = %.3f, train = %.3f, mean_fit_time = %f' % (
            scoring, test_score, train_score, fit_time))
        print('best params', clf.best_params_)
        print(sub_end, sub_end - sub_start)
        print()

    result = {
        'grid_cv_time' : sub_end - sub_start,
        'score'        : scoring,
        'model_name'   : name,
        'kind'         : kind,
        'mean_test'    : test_score,
        'mean_train'   : train_score,
        'mean_fit_time': fit_time,
        'best_params'  : clf.best_params_,
        'best_model'   : clf.best_estimator_,
    }
    return result


def grid_cv_models(x, y, models, params, order=None, path='', n_jobs=4, cv=5, save_res=True,
                   redo=False, redo_individual=False, verbose=False, fit_when_load=True, range=None):
    """
    regression model is evaluated by neg_mean_squared_error
    classification model is evaluated by f1_weighted
    if order is None, iterate over models' keys, otherwise iterate by order
    get tuning parameters based on key, if no matched paramters, that model will be skipped
    if not redo and the result exists, optimum parameters will be loaded using model.set_params(**loaded)
    if not redo_individual and result for individual classifier exists, the best params for the model will be loaded.
    when loading model, run empty model.fit(x,y) if fit_when_load=True
    :return:
        index: (kind, name);
        each line is the best parameters for that model;
        type of column "best_model" is sklearn models.
    """

    if range is None:
        min_y, max_y = y.min(), y.max()
    else:
        min_y, max_y = range

    if len(y.shape) != 1:
        print('grid_cv_models, y.shape should be (R,)', y.shape)

    path_cv_best = os.path.join(path, 'cv_%d_best_models.csv' % cv)

    # if cv best result exists and not redeo, load existing parameters
    if os.path.exists(path_cv_best) and not redo:
        loaded_df_cv_res = pd.read_csv(path_cv_best, index_col=[0, 1])
        best_models = []
        for (kind, name), row in loaded_df_cv_res.iterrows():
            param = eval(row.best_params)
            model = models[kind][name]
            model.set_params(**param)
            if fit_when_load:
                if verbose: print('fitting model', kind, name)

                if kind == 'cls' and all_float(y):
                    y = bounded_round(y, min_y, max_y)
                    if len(set(y)) > 100:
                        print("grid_cv_a_model: rounded y has more than 100 labels")

                model.fit(x, y)
            best_models.append(model)

        loaded_df_cv_res.best_model = best_models
        print('loaded existing cv-ed best parameters')
        return loaded_df_cv_res

    # redo or result not exists
    cv_results = []
    start = dtm.now()
    if order is None:
        order = [[kind, kind_models.keys()] for kind, kind_models in models.items()]

    for kind, kind_model_names in order:
        if kind not in models:
            print(kind, 'not in models')
            continue
        for name in kind_model_names:
            if name not in models[kind]:
                print('kind={} mode={} not in models, skipped'.format(kind, name))
                continue
            if name not in params[kind]:
                print('model', name, 'doesnt have tuning params, use {} instead')
                param = {}
            else:
                param = params[kind][name]
            model = models[kind][name]
            result = grid_cv_a_model(x, y, model, param, kind, name,
                                     path=path, n_jobs=n_jobs, cv=cv, verbose=verbose,
                                     redo=redo_individual, save_res=save_res, fit_when_load=fit_when_load, range=range)
            cv_results.append(result)

    end = dtm.now()
    print('finished CV', end, end - start)

    df_cv = pd.DataFrame(cv_results).set_index(['kind', 'model_name'])
    if save_res:
        df_cv.to_csv(path_cv_best)

    return df_cv


# ################################################
# Evaluation by metrics
# Evaluators for different prediction tasks
# Visualization cross validation result
# ################################################

def evaluate_grid_cv(df_cv, train_x, train_y, test_x, test_y, evaluator, path='', cv=5, save_res=True, range=None):
    """
    This function is major for evaluate result from grid_cv_models. But it can be also used to evaluate df containing
    different models. In this case, set save_res=False.

    Parameters
        df_cv: results from :func:: grid_cv_models(). index=(kind, name), 'best_model' is in columns.
        evaluator: such as :func:: evaluator_scalable_cls()
        save_res: True->save to path/cv_%d_best_models_evaluation.csv % cv
        range: the min and max label for bounded round
    Return
        evaluation result as pd.DF, columns are defined by evaluator
    """

    # TODO: fit the best params again, the f1 score is different than this output

    print('evaluating grid cv')
    results = {}
    for (kind, name), model in df_cv.best_model.iteritems():
        results[kind, name] = evaluator(model, train_x, train_y, test_x, test_y, range)

    df_results = pd.DataFrame(results).T
    if 'test_f1_weighted' in df_results.columns:
        df_results.sort_values(by='test_f1_weighted', ascending=False, inplace=True)

    if save_res:
        df_results.to_csv(os.path.join(path, 'cv_%d_best_models_evaluation.csv' % cv))

    return df_results


def evaluator_scalable_cls(model, train_x, train_y, test_x, test_y, range=None):
    """
    Evaluator for scalable classification. E.g. Ys are 1, 2, 3, 4
    Both regression and classification will be used.
    prediction by regression will be round up (bounded by max and min of Ys) as a class label
    Model will be refit to train_x and train_y.
    range: the min and max label for bounded round
    :return: metrics: mse, accuracy and f1, for both train and test
    """

    if range is None:
        min_y, max_y = train_y.min(), train_y.max()
    else:
        min_y, max_y = range

    # round real number into int, in order to get f1 score.
    if all_float(train_y):
        train_y = bounded_round(train_y, min_y, max_y)
        if len(set(train_y)) > 100:
            print("grid_cv_a_model: rounded y has more than 100 labels")

    if all_float(test_y):
        test_y = bounded_round(test_y, min_y, max_y)

    model.fit(train_x, train_y)

    train_pred = model.predict(train_x)
    train_pred_round = bounded_round(train_pred, min_y, max_y)

    train_mse = mean_squared_error(train_y, train_pred)
    train_acc = accuracy_score(train_y, train_pred_round)
    train_f1_weighted = f1_score(train_y, train_pred_round, average='weighted')
    train_f1_macro = f1_score(train_y, train_pred_round, average='macro')
    train_f1_micro = f1_score(train_y, train_pred_round, average='micro')

    train_precision_weighted = precision_score(train_y, train_pred_round, average='weighted')
    train_precision_macro = precision_score(train_y, train_pred_round, average='macro')
    train_precision_micro = precision_score(train_y, train_pred_round, average='micro')

    train_recall_weighted = recall_score(train_y, train_pred_round, average='weighted')
    train_recall_macro = recall_score(train_y, train_pred_round, average='macro')
    train_recall_micro = recall_score(train_y, train_pred_round, average='micro')

    test_pred = model.predict(test_x)
    test_pred_round = bounded_round(test_pred, min_y, max_y)

    test_mse = mean_squared_error(test_y, test_pred)
    test_acc = accuracy_score(test_y, test_pred_round)
    test_f1_weighted = f1_score(test_y, test_pred_round, average='weighted')
    test_f1_macro = f1_score(test_y, test_pred_round, average='macro')
    test_f1_micro = f1_score(test_y, test_pred_round, average='micro')

    test_precision_weighted = precision_score(test_y, test_pred_round, average='weighted')
    test_precision_macro = precision_score(test_y, test_pred_round, average='macro')
    test_precision_micro = precision_score(test_y, test_pred_round, average='micro')

    test_recall_weighted = recall_score(test_y, test_pred_round, average='weighted')
    test_recall_macro = recall_score(test_y, test_pred_round, average='macro')
    test_recall_micro = recall_score(test_y, test_pred_round, average='micro')

    result = {
        'train_f1_weighted'       : train_f1_weighted,
        'train_f1_macro'          : train_f1_macro,
        'train_f1_micro'          : train_f1_micro,
        'train_precision_weighted': train_precision_weighted,
        'train_precision_macro'   : train_precision_macro,
        'train_precision_micro'   : train_precision_micro,
        'train_recall_weighted'   : train_recall_weighted,
        'train_recall_macro'      : train_recall_macro,
        'train_recall_micro'      : train_recall_micro,
        'train_acc'               : train_acc,
        'train_mse'               : train_mse,
        'test_f1_weighted'        : test_f1_weighted,
        'test_f1_macro'           : test_f1_macro,
        'test_f1_micro'           : test_f1_micro,
        'test_precision_weighted' : test_precision_weighted,
        'test_precision_macro'    : test_precision_macro,
        'test_precision_micro'    : test_precision_micro,
        'test_recall_weighted'    : test_recall_weighted,
        'test_recall_macro'       : test_recall_macro,
        'test_recall_micro'       : test_recall_micro,
        'test_acc'                : test_acc,
        'test_mse'                : test_mse,
    }
    return result


def vis_evaluation(path, cv):
    df_eval = pd.read_csv(os.path.join(path, 'cv_%d_best_models_evaluation.csv' % cv))
    return df_eval.plot()


def vis_grid_cv_one_model(fn):
    df = pd.read_csv(fn, index_col=0)
    return df[['mean_test_score', 'mean_train_score']].boxplot()


# ################################################
# Analysis of predictor
# model inspection
# ################################################

def show_important_features(fitted_tree_model, name="", top=None, labels=None, show_plt=True, set_std=True):
    """
    Format tree_models feature importance as pd.df, along with a bar plot with error bar.

    TODO:
        the shape of estimators of gradient boosting is [n_estimator, n_classes(binary=1)].
        not sure the difference per row. Right now just flatten all estimators to calculate std.

    :param fitted_tree_model: tree like model, mostly models in sklearn.ensemble.
    :param name: name to be shown in plot title
    :param top: show top most important features. Default None, showing all features
    :param labels: labels for features. Default None, range(len(tree_model.feature_importances_))
    :param show_plt: Default True, visualize importance and std as error bar with maplotlib.pyplot.
    :return: pd.df, columns = importance, label, std
    """
    importances = fitted_tree_model.feature_importances_
    feature_size = len(importances)
    if hasattr(fitted_tree_model, 'estimators_') and set_std:
        # TODO: is it reasonable to flatten gradient boosting's estimators?
        estimators = fitted_tree_model.estimators_
        if hasattr(estimators, 'flatten'):
            estimators = estimators.flatten()
        std = np.std([tree.feature_importances_ for tree in estimators], axis=0)
    else:
        std = np.zeros(feature_size)
    if labels is None:
        labels = [i for i in range(len(importances))]
    top = min(feature_size, top) if top is not None else feature_size

    imp = pd.DataFrame(list(zip(labels, importances, std)), columns=['label', 'importance', 'std'])
    imp.sort_values('importance', ascending=False, inplace=True)
    imp = imp[:top]

    if show_plt:
        title = "%s Feature importances - top %d / %d %s" % (
            name, top, len(importances), 'no errbar' if not set_std else 'with errbar')
        imp_plt = imp.set_index('label')
        imp_plt.importance.plot(kind='barh', xerr=imp_plt['std'], title=title, figsize=(10, 7))

    return imp


def confusion_matrix_as_df(fitted_model, x, y, labels=None, normalize=False, show_plt=False, range=None):
    """
    build a confusion matrix between y and fitted_model.predict(x)

    parameters:
        fitted_model: model from sklearn. It should already perform fit(train_x, train_y)
        x,y: features and true labels
        labels: index and column names of the return pd.df. If None, pd.unique(y) will be used

    return：
        confusion matrix in the form of pd.Dataframe
    """

    if range is None:
        min_y, max_y = y.min(), y.max()
    else:
        min_y, max_y = range

    pred_y = fitted_model.predict(x)

    if all_float(pred_y):
        pred_y = bounded_round(pred_y, min_y, max_y)
    if all_float(y):
        y = bounded_round(y, min_y, max_y)
    if len(set(y)) > 100:
        print('confusion matrix as df: nunique rounded y > 100')

    if labels is None:
        labels = pd.unique(y)
    cfsn = confusion_matrix(y, pred_y, labels=labels)
    if normalize:
        cfsn = cfsn.astype('float') / cfsn.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(cfsn)
    df.columns = pd.MultiIndex.from_product([['pred'], labels])
    df.index = pd.MultiIndex.from_product([['true'], labels])
    if show_plt:
        import seaborn
        seaborn.heatmap(df)
    return df


def get_idx_by_true_pred(fitted_model, test_x, test_y, target_tp, range=None):
    """
    get indices by pairs of true and predicted label. Works for classification.
    :param fitted_model: model from sklearn. It should already perform fit(train_x, train_y)
    :param test_x: features of test set
    :param test_y: true labels of test set
    :param target_tp: true vs pred pairs to compare. [(t,p), (t,p), ...]
    :return: {(t,p): indices, }
    """

    pred_y = fitted_model.predict(test_x)

    if range is None:
        min_y, max_y = test_y.min(), test_y.max()
    else:
        min_y, max_y = range

    if check_type(test_y, int):
        pred_y = bounded_round(pred_y, min_y, max_y)
    elif all_int_able(test_y):
        warnings.warn('test y is intable, but type of elements are not all int')

    df = pd.DataFrame(zip(test_y, pred_y), columns=['true', 'pred'])

    target_xs = {}
    for t, p in target_tp:
        target_xs[(t, p)] = df[(df.true == t) & (df.pred == p)].index.tolist()

    return target_xs


def ftr_diff_stat_test(fitted_model, test_x, test_y, target_tp, verbose=False):
    """
    group test set by (true, pred) pairs, perform statistical test on each feature to find feature differences
    tests include: one way ANOVA, Kruskal-Wallis H-test

    :param fitted_model: model from sklearn. It should already perform fit(train_x, train_y)
    :param test_x: features of test set
    :param test_y: true labels of test set
    :param target_tp: true vs pred pairs to compare. [(t,p), (t,p), ...]
    :param verbose: verbose or not
    :return: pd.Dataframe, sort by kruskal p value
    """
    idx = get_idx_by_true_pred(fitted_model, test_x, test_y, target_tp)
    groups = []
    num_ftrs = test_x.shape[1]
    for tp in target_tp:
        groups.append(test_x.values[idx[tp]])
        if verbose:
            print('True = {}, Pred = {}: len = {}'.format(tp[0], tp[1], len(idx[tp])))

    test_res = []
    for i in range(num_ftrs):
        var = test_x.columns[i]
        test = [g[:, i] for g in groups]
        anova_stat, anova_p = f_oneway(test)
        kru_stat, kru_p = krutest(test)
        res = {'var_name'   : var,
               'anova1way_p': anova_p, 'anova1way_stat': anova_stat,
               'kruskal_p'  : kru_p, 'kruskal_stat': kru_stat,
               }
        if len(groups) == 2:
            res['mean_dff'] = np.abs(np.mean(test[0]) - np.mean(test[1]))
        test_res.append(res)
    df = pd.DataFrame(test_res).set_index("var_name")
    df['annova1way_siglv'] = df.anova1way_p.apply(significant_level)
    df['kruskal_signlv'] = df.kruskal_p.apply(significant_level)
    if len(groups) == 2:
        df.sort_values('mean_dff', inplace=True, ascending=False)
    else:
        df.sort_values('kruskal_p', inplace=True)
    return df
