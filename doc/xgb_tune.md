# XGB tune: automatic xgboost tuner

## Table of contents
* [Overview](#overview)
* [Idea](#ida)
* [Steps and tuned options](#steps)
* [Installation](#installation)
* [Usage](#usage)


<a name="overview"/>
## Overview

[XGBoost](https://github.com/dmlc/xgboost) is known to be extremely powerful
tool, very popular in machine learning competitions like
[Kaggle](https://www.kaggle.com/).

But as every powerful tool, it contains tons of options to tweak and handles to
change, which can dramatically (sometimes) affect final results. To simplify
final optimisations of xgboost's parameters, I've implemented xgb_tune.

xgb_tune.py is a tool which was inspired by an article
["Complete guide to parameter tuning in XGBoost"](http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). After
couple of successfull application of this article, I decided to automate
heruistic process described in it.

<a name="idea"/>
## Idea, briefly

XGBoost has about 8 parameters which influence final result. Even with modest
grid search of 5 values per parameter we're facing with more than 390 thousands
combinations. Assuming we'll be able to check one combination per minute (not
very realistic assumption for large problems), it will take about 9 months. We
can try to use random search instead of brute force, but it adds large degree of
uncertanty to the process.

Idea proposed in article is to do sequential optimisation of sets of parameters,
step-by-step. On first steps we tune options responsible for tree structure (max
depth, child_weight) and tune booster options on subsequent steps.

To further improve speed of tuning, we do rough-tuning of paremeters first, and
perform fine-tuning using denser intervals on second step.

<a name="steps"/>
## Steps and tuned options

Xgb_tune does the following steps:
* Find rough *learning_rate* value which converges in reasonable amount of
  steps. On this step we're trying to find such learning rate with which steps count before
  overfitting is more than 50 but less than 200.
* Tune *max_depth* and *min_child_weight*
* Tune *gamma*
* Tune *subsample* and *colsample_bytree* options
* Finally, ture regularisation parameters *alpha* and *lambda*

All parameters first tuned with rough steps and after that fine-tuned using
finer intervals. For example, *max_depth* first tried with values
[2, 4, 6, 8, 10, 12, 14, 16] and after finding optimal value, we do checking of
odd values around this optimum.

<a name="installation"/>
## Installation

xgb_tune doesn't need installation and self-contained. Required modules are:
* xgboost (obviously)
* numpy
* sklearn
* [marktime](https://pypi.python.org/pypi/marktime)

<a name="usage"/>
## Usage
