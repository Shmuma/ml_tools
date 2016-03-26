# XGB tune: automatic xgboost tuner

## Table of contents
* [Overview](#overview)
* [Idea](#ida)
* [Steps and tuned options](#steps)
* [Installation](#installation)
* [Usage](#usage)


<a name="overview"/>
## Overview

[XGBoost](https://github.com/dmlc/xgboost) is known to be an extremely powerful
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

XGBoost has about 8 parameters influencing the final result. Even with a modest
grid search of 5 values per parameter we're facing with more than 390 thousands
combinations. Assuming we're be able to check one combination per minute (not
very realistic assumption for large problems), it will take about 9 months to check them all. We
can try to use random search instead of brute force, but it adds large degree of
uncertanty to the process.

The idea proposed in the article is doing sequential optimisation of sets of parameters,
step-by-step. First, we tune options responsible for structure of trees (max
depth, child_weight) and tune booster options on subsequent steps.

For further improving the speed of tuning, we split grid search of every step on 
two stages: rough and fine, as most of parameters are integer. 

<a name="steps"/>
## Steps and tuned options

Xgb_tune does the following steps:
* Finding approximate *learning_rate* value giving us convergence in reasonable amount of
  steps (between 50 and 200).
* Tuning *max_depth* and *min_child_weight*
* Tuning *gamma*
* Tuning *subsample* and *colsample_bytree* options
* Finally, ture regularisation parameters *alpha* and *lambda*

All the parameters first tuned with rough steps and after that fine-tuned using
finer intervals. For example, *max_depth* is first grid-searched with values
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
