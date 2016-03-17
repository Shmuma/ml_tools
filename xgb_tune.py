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
from sklearn.cross_validation import train_test_split


def setup_logging(logfile, level=log.INFO):
    format = "%(asctime)s %(levelname)s %(message)s"
    if logfile is not None:
        log.basicConfig(filename=logfile, level=level, format=format)
    else:
        log.basicConfig(level=level, format=format)


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


def find_n_estimators(cls, data, cv_folds=5, early_stopping_rounds=50):
    marktime.start("find_n_estimators")
    xgb_params = cls.get_xgb_params()
    xgtrain = xgb.DMatrix(data['features_train'], label=data['labels_train'])
    cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round=cls.get_params()['n_estimators'],
                      nfold=cv_folds, metrics=args.metric, early_stopping_rounds=early_stopping_rounds)
    log.info("N_estimators search done in %s, result=%d", task_done("find_n_estimators"), cvresult.shape[0])
    return cvresult.shape[0]


def find_init_learning_rate(data, params):
    """
    Find learning rate which converges in ~50 iterations. This gives more or less accurate results in
    reasonable time.
    :return: learning rate, number of estimators
    """
    opts = dict(params['default'])
    opts.update(params['fixed'])
    del opts['learning_rate']
    del opts['n_estimators']

    for lr in np.linspace(1.0, 0.01, num=10):
        log.info("Trying LR=%f", lr)
        cls = XGBClassifier(n_estimators=100000, learning_rate=lr, **opts)
        n_estimators = find_n_estimators(cls, data, early_stopping_rounds=20)
        if n_estimators > 50:
            return lr, n_estimators

    log.warn("We failed to find initial learning rate, fall back to defaults. You should report this to author!")
    return 0.3, 50


if __name__ == "__main__":
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

    log.info("Split data to train/test set")
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, train_size = 0.8, random_state=args.seed)

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
        'features_train': features_train,
        'features_test': features_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
    }

    show_params(params)

    # step one: find learning rate which gives reasonable amount of estimators
    marktime.start("learning_rate_1")
    log.info("Looking for initial learning rate")
    learning_rate_1, n_estimators_1 = find_init_learning_rate(data, params)
    log.info("Initial learning_rate %f and n_estimators %d found in %s", learning_rate_1,
             n_estimators_1, task_done("learning_rate_1"))

    # move this learning rate
    del params['default']['learning_rate']
    del params['default']['n_estimators']
    params['tuned']['learning_rate'] = learning_rate_1
    params['tuned']['n_estimators'] = n_estimators_1

    log.info("XGB_tune done in %s", task_done("start"))
    show_params(params)
