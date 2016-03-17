"""
Xgboost fine-tuning automation script.
"""
import sys
import marktime
from datetime import timedelta
import argparse
import logging as log

import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV


def setup_logging(logfile, level=log.INFO):
    fmt = "%(asctime)s %(levelname)s %(message)s"
    if logfile is not None:
        log.basicConfig(filename=logfile, level=level, format=fmt)
    else:
        log.basicConfig(level=level, format=fmt)


def task_done(task_name):
    """
    Stop task in marktime and return timedelta object for duration
    :param task_name:
    :return:
    """
    return timedelta(seconds=marktime.stop(task_name).value)


def show_params(params):
    """
    Display state of parameters
    :param params:
    :return:
    """
    for kind in ['default', 'tuned']:
        if len(params[kind]) == 0:
            continue
        log.info("%s parameters", kind.capitalize())
        for key in sorted(params[kind].keys()):
            log.info("  %-20s %s", key+":", str(params[kind][key]))


def commit_param(params, name, value):
    """
    Assign param value and move it from default to tuned
    """
    del params['default'][name]
    params['tuned'][name] = value


def find_n_estimators(cls, data, cv_folds=5, early_stopping_rounds=50):
    marktime.start("find_n_estimators")
    xgb_params = cls.get_xgb_params()
    xgtrain = xgb.DMatrix(data['features'], label=data['labels'])
    cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round=cls.get_params()['n_estimators'],
                      nfold=cv_folds, metrics=args.metric, early_stopping_rounds=early_stopping_rounds,
                      show_progress=False)
    log.info("N_estimators search done in %s, result=%d", task_done("find_n_estimators"), cvresult.shape[0])
    return cvresult.shape[0]


def make_xgb(params, extra=None):
    opts = dict(params['default'])
    opts.update(params['tuned'])
    opts.update(params['fixed'])
    if extra is not None:
        opts.update(extra)
    return XGBClassifier(silent=True, **opts)


def find_init_learning_rate(data, params):
    """
    Find learning rate which converges in ~50 iterations. This gives more or less accurate results in
    reasonable time.
    :return: learning rate, number of estimators
    """
    for lr in np.linspace(1.0, 0.01, num=10):
        log.info("Trying LR=%f", lr)
        cls = make_xgb(params, extra={'n_estimators': 100000, 'learning_rate': lr})
        n_estimators = find_n_estimators(cls, data, early_stopping_rounds=20)
        if n_estimators > 50:
            return lr, n_estimators

    log.warn("We failed to find initial learning rate, fall back to defaults. You should report this to author!")
    return 0.3, 50


def is_on_right_bound(value, range):
    return value == range[-1]


def find_maxdepth_minchildweight(data, params):
    param_test = {
        'max_depth': range(0, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    centers = {}

    iteration = 0
    while True:
        marktime.start('first_step')
        cls = make_xgb(params)
        log.info("Fist step, iteration=%d, search in %s", iteration, param_test)
        gsearch = GridSearchCV(estimator=cls, param_grid=param_test,
                               scoring='roc_auc', n_jobs=1, iid=False, cv=5)

        gsearch.fit(data['features'], data['labels'])
        best = gsearch.best_params_
        score = gsearch.best_score_
        log.info("Frist step found params %s with score %s in %s", best, score, task_done("first_step"))

        # handle boundary value
        boundary = False
        for key in param_test.keys():
            if is_on_right_bound(best[key], param_test[key]):
                end = param_test[key][-1]
                param_test[key] = range(end, end+len(param_test[key])+2, 2)
                log.info("Optimal value for %s is on boundary (%s), shift range and iterate again",
                         key, best[key])
                boundary = True

        if not boundary:
            for key in param_test.keys():
                centers[key] = best[key]
                del param_test[key]
                log.info("Found optimal value for %s=%s", key, centers[key])
            break
        iteration += 1

    # do fine-tuning
    param_test = {
        key: [val-1, val, val+1] for key, val in centers.iteritems()
    }

    marktime.start("second_step")
    cls = make_xgb(params)
    log.info("Second step, search in %s", param_test)
    gsearch = GridSearchCV(estimator=cls, param_grid=param_test,
                           scoring='roc_auc', n_jobs=1, iid=False, cv=5)

    gsearch.fit(data['features'], data['labels'])
    best = gsearch.best_params_
    score = gsearch.best_score_
    log.info("Second step found %s with score %s in %s", best, score, task_done('second_step'))

    return best['max_depth'], best['min_child_weight']


def find_gamma(data, params):
    param_test = {
        'gamma': [i/10.0 for i in range(0, 8)]
    }

    iteration = 0
    best = 0.0
    while True:
        marktime.start('first_step')
        cls = make_xgb(params)
        log.info("Fist step, iteration=%d, search in %s", iteration, param_test)
        gsearch = GridSearchCV(estimator=cls, param_grid=param_test,
                               scoring='roc_auc', n_jobs=1, iid=False, cv=5)

        gsearch.fit(data['features'], data['labels'])
        best = gsearch.best_params_['gamma']
        score = gsearch.best_score_
        log.info("Frist step found params %s with score %s in %s", gsearch.best_params_,
                 score, task_done("first_step"))

        # handle boundary value
        if is_on_right_bound(best, param_test['gamma']):
            end = int(param_test['gamma'][-1]*10)
            param_test['gamma'] = [i/10.0 for i in range(end, end+len(param_test['gamma']))]
            log.info("Optimal value is on boundary (%s), shift range and iterate again", best)
        else:
            log.info("Found optimal value for gamma=%s", best)
            break
        iteration += 1

    return best


def calibrate_n_estimators(data, params):
    marktime.start('calibrate')
    log.info("Calibrate n_estimators to new options")
    cls = make_xgb(params, extra={'n_estimators': 100000})
    n_estimators = find_n_estimators(cls, data, early_stopping_rounds=20)
    log.info("N_estimators calibrated to %d in %s", n_estimators, task_done("calibrate"))
    return n_estimators


if __name__ == "__main__":
    DEBUG = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Input features file to use in numpy binary format")
    parser.add_argument("--labels", required=True, help="File with labels in numpy binary format")
    parser.add_argument("--log", required=False, help="Send log file to file instead of stdout")

    parser.add_argument("--seed", type=int, default=42, help="Random seed value to use, default=42")
    parser.add_argument("--cores", type=int, default=None, help="Limit amount of cores to use, default=None")

    parser.add_argument("--objective", default="binary:logistic", help="Tree objective to use, default=binary:logistic")
    parser.add_argument("--metric", default="auc", help="Metric to use, default=auc")

    args = parser.parse_args()

    marktime.start("start")
    setup_logging(args.log)
    log.info("XGB_tune started.")
    log.info("Input features: %s", args.features)
    log.info("Input labels: %s", args.labels)

    # load data
    marktime.start("data_load")
    log.info("Loading data")
    features = np.load(args.features)
    labels = np.load(args.labels)
    log.info("Data loaded in %s", task_done("data_load"))
    log.info("Features shape: %s, labels shape: %s", features.shape, labels.shape)
    if features.shape[0] != labels.shape[0]:
        log.error("Shape of features and labels don't match!")
        sys.exit(1)

    # parameters
    params_default = {
        'learning_rate': 0.5,
        'n_estimators': 100,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 0,
    }
    params_tuned = {}
    params_fixed = {
        'seed': args.seed,
        'scale_pos_weight': 1,
        'objective': args.objective,
        'nthread': args.cores if args.cores is not None else 0,
    }
    params = {
        'default': params_default,
        'tuned': params_tuned,
        'fixed': params_fixed,
    }

    # data we'll use
    data = {
        'features': features,
        'labels': labels,
    }

    show_params(params)

    # step one: find learning rate which gives reasonable amount of estimators
    if not DEBUG:
        marktime.start("learning_rate_1")
        log.info("Looking for initial learning rate")
        learning_rate_1, n_estimators_1 = find_init_learning_rate(data, params)
        log.info("Initial learning_rate %f and n_estimators %d found in %s", learning_rate_1,
                 n_estimators_1, task_done("learning_rate_1"))
    else:
        learning_rate_1, n_estimators_1 = 0.23, 81

    # move this learning rate
    commit_param(params, 'learning_rate', learning_rate_1)
    commit_param(params, 'n_estimators', n_estimators_1)

    show_params(params)

    # step two: max_depth and min_child_weight
    if not DEBUG:
        marktime.start("step_2")
        log.info("Looking for optimal max_depth and min_child_weight")
        max_depth, min_child_weight = find_maxdepth_minchildweight(data, params)
        log.info("Found max_depth=%d and min_child_weight=%d in %s", max_depth, min_child_weight,
                 task_done("step_2"))
    else:
        max_depth, min_child_weight = 4, 11

    commit_param(params, 'max_depth', max_depth)
    commit_param(params, 'min_child_weight', min_child_weight)

    show_params(params)

    # step three: gamma
    if not DEBUG:
        marktime.start("step_3")
        log.info("Looking for optimal gamma")
        gamma = find_gamma(data, params)
        log.info("Found gamma=%f in %s", gamma, task_done("step_3"))
    else:
        gamma = 0.0

    commit_param(params, 'gamma', gamma)

    show_params(params)

    # calibrate n_estimators with new params
    n_estimators_2 = calibrate_n_estimators(data, params)
    params['tuned']['n_estimators'] = n_estimators_2

    log.info("XGB_tune done in %s", task_done("start"))
    show_params(params)
