"""
Xgboost fine-tuning automation script.
"""
import sys
import os
import json
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


def read_state(state_file, params):
    """
    Read state from file into params, return last done step
    :param state_file:
    :param params:
    :return:
    """
    if state_file is None:
        return 0

    if not os.path.exists(state_file):
        log.info("State file %s not found, starting from scratch", state_file)
        return 0

    with open(state_file, "r") as fd:
        dat = json.load(fd)
        params['default'] = dat['default']
        params['tuned'] = dat['tuned']
        step_done = dat['step_done']
        log.info("State loaded from %s, last done step=%d", state_file, step_done)
        return step_done


def write_state(state_file, params, step_done):
    if state_file is None:
        return

    log.info("Save state to %s", state_file)
    with open(state_file, "w+") as fd:
        dat = {
            'default': params['default'],
            'tuned': params['tuned'],
            'step_done': step_done
        }
        json.dump(dat, fd, indent=4)


def find_n_estimators(cls, data, cv_folds=5, early_stopping_rounds=50):
    marktime.start("find_n_estimators")
    xgb_params = cls.get_xgb_params()
    xgtrain = xgb.DMatrix(data['features'], label=data['labels'])
    cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round=cls.get_params()['n_estimators'],
                      nfold=cv_folds, metrics=args.metric, early_stopping_rounds=early_stopping_rounds,
                      show_progress=False)
    n_estimators = cvresult.shape[0]-1
    log.info("N_estimators search done in %s, result=%d", task_done("find_n_estimators"), n_estimators)
    return n_estimators


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
        log.info("First step found params %s with score %s in %s", best, score, task_done("first_step"))

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


def find_subsample_colsample(data, params):
    param_test = {
        'subsample': [i/10.0 for i in range(6, 11)],
        'colsample_bytree': [i/10.0 for i in range(6, 11)],
    }

    log.info("Starting search in %s", param_test)
    cls = make_xgb(params)
    gsearch = GridSearchCV(estimator=cls, param_grid=param_test,
                           scoring='roc_auc', n_jobs=1, iid=False, cv=5)
    gsearch.fit(data['features'], data['labels'])

    best = gsearch.best_params_
    score = gsearch.best_score_
    log.info("Found params %s with score %s", best, score)
    return best['subsample'], best['colsample_bytree']


def find_alpha_lambda(data, params, reg_steps):
    param_test = {
        'reg_alpha': np.power(10.0, np.linspace(-5.0, 4.0, num=reg_steps)),
        'reg_lambda': np.power(10.0, np.linspace(-5.0, 4.0, num=reg_steps))
    }

    marktime.start("first_step")
    cls = make_xgb(params)
    log.info("First step, search in %s", param_test)
    gsearch = GridSearchCV(estimator=cls, param_grid=param_test, scoring='roc_auc',
                           n_jobs=1, iid=False, cv=5)
    gsearch.fit(data['features'], data['labels'])
    best = gsearch.best_params_
    score = gsearch.best_score_
    log.info("First step found params %s with score %s in %s",
             best, score, task_done("first_step"))

    reg_alpha = best['reg_alpha']
    reg_lambda = best['reg_lambda']

    param_test = {
        'reg_alpha': np.linspace(reg_alpha / 5.0, reg_alpha * 5.0, num=reg_steps),
        'reg_lambda': np.linspace(reg_lambda / 5.0, reg_lambda * 5.0, num=reg_steps)
    }

    marktime.start("second_step")
    log.info("Second step, search in %s", param_test)
    gsearch = GridSearchCV(estimator=cls, param_grid=param_test, scoring='roc_auc',
                           n_jobs=1, iid=False, cv=5)
    gsearch.fit(data['features'], data['labels'])
    best = gsearch.best_params_
    score = gsearch.best_score_
    log.info("Second step found params %s with score %s in %s",
             best, score, task_done("second_step"))

    return best['reg_alpha'], best['reg_lambda']


if __name__ == "__main__":
    DEBUG = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Input features file to use in numpy binary format")
    parser.add_argument("--labels", required=True, help="File with labels in numpy binary format")
    parser.add_argument("--log", required=False, help="Send log file to file instead of stdout")
    parser.add_argument("--state", help="If specified, state will be saved to or read from this file")

    parser.add_argument("--seed", type=int, default=42, help="Random seed value to use, default=42")
    parser.add_argument("--cores", type=int, default=None, help="Limit amount of cores to use, default=None")

    parser.add_argument("--objective", default="binary:logistic", help="Tree objective to use, default=binary:logistic")
    parser.add_argument("--metric", default="auc", help="Metric to use, default=auc")
    parser.add_argument("--reg-steps", default=10, type=int, help="How many steps to use in regularisation search")

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

    step_done = read_state(args.state, params)

    if step_done < 1:
        show_params(params)

        # step one: find learning rate which gives reasonable amount of estimators
        marktime.start("learning_rate_1")
        log.info("Looking for initial learning rate")
        learning_rate_1, n_estimators_1 = find_init_learning_rate(data, params)
        log.info("Initial learning_rate %f and n_estimators %d found in %s", learning_rate_1,
                 n_estimators_1, task_done("learning_rate_1"))

        # move this learning rate
        commit_param(params, 'learning_rate', learning_rate_1)
        commit_param(params, 'n_estimators', n_estimators_1)
        show_params(params)
        step_done = 1
        write_state(args.state, params, step_done)

    if step_done < 2:
        # step two: max_depth and min_child_weight
        marktime.start("step_2")
        log.info("Looking for optimal max_depth and min_child_weight")
        max_depth, min_child_weight = find_maxdepth_minchildweight(data, params)
        log.info("Found max_depth=%d and min_child_weight=%d in %s", max_depth, min_child_weight,
                 task_done("step_2"))

        commit_param(params, 'max_depth', max_depth)
        commit_param(params, 'min_child_weight', min_child_weight)
        show_params(params)
        step_done = 2
        write_state(args.state, params, step_done)

    if step_done < 3:
        # step three: gamma
        marktime.start("step_3")
        log.info("Looking for optimal gamma")
        gamma = find_gamma(data, params)
        log.info("Found gamma=%f in %s", gamma, task_done("step_3"))

        commit_param(params, 'gamma', gamma)
        show_params(params)
        step_done = 3
        write_state(args.state, params, step_done)

    if step_done < 4:
        # calibrate n_estimators with new params
        n_estimators_2 = calibrate_n_estimators(data, params)
        if n_estimators_2 > 100:
            log.info("Calibrated n_estimators is too large, ignore it")
        else:
            params['tuned']['n_estimators'] = n_estimators_2
        step_done = 4
        write_state(args.state, params, step_done)

    if step_done < 5:
        # tune subsample and colsample_bytree
        marktime.start("step_5")
        log.info("Looking for optimal subsample and colsample_bytree")
        subsample, colsample_by_tree = find_subsample_colsample(data, params)
        log.info("Found subsample=%f and colsample_bytree=%f in %s", subsample, colsample_by_tree,
                 task_done("step_5"))

        commit_param(params, "subsample", subsample)
        commit_param(params, "colsample_bytree", colsample_by_tree)
        show_params(params)
        step_done = 5
        write_state(args.state, params, step_done)

    if step_done < 6:
        # tune regularisation parameters
        marktime.start("step_6")
        log.info("Looking for optimal L1 and L2 regularisation params")
        reg_alpha, reg_lambda = find_alpha_lambda(data, params, args.reg_steps)
        log.info("Found alpha=%f and lambda=%f in %s", reg_alpha, reg_lambda,
                 task_done("step_6"))
        commit_param(params, "reg_alpha", reg_alpha)
        commit_param(params, "reg_lambda", reg_lambda)
        show_params(params)
        step_done = 6
        write_state(args.state, params, step_done)

    log.info("XGB_tune done in %s", task_done("start"))
    log.info("Now you need to manually tune learning_rate, and enjoy you kaggle score!")