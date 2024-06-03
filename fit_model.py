#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit ML regression model to DOC in-situ data

run this like

./fit_model.py --model rf -no 1 --mdir /tmp/models/ --pdir /tmp/plots

Models are defined below. Saves model and data to `mdir`.

Use --optimize to optimize some random forest or gradient bootsting hyperparameter
using the optuna package. Results are saved to data base and used in the next training run.

The model output can be processed with `fit_maps.py`
to generate global DOC estimates.

./fit_maps.py --model rf -no 3 --mdir /tmp/results/


@author: marko.laine@fmi.fi
"""

import sys
import os
import pickle
from joblib import dump

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna

from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

#from sklearn.tree import export_graphviz
#import pydot

from data_utils import xyline
from model_params import model_params, parser, load_params

#%%

opts = parser.parse_args(sys.argv[1:])

if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)

logging.captureWarnings(True)


# %% Load data

oldornew = 'old' if opts.old else 'new'
openorshore = 'shore' if opts.shore else 'open'
openorshore = 'all' if opts.all else openorshore

if oldornew == 'old':
    datafile = '~/DATA/ESA-BICEP/ocdocppdts2.h5'  # pandas file (uses monthly OC)
else:
    datafile = 'ocdocppdts_v9.h5'

df = pd.read_hdf(datafile)

docunit = 'µmol kg⁻¹'

# %% subset

shorelimit = opts.shorelimit
cvgroup = 'yearly'

df = df[np.abs(df['lat']) <= opts.latlimit]


if openorshore == 'shore':

    df = df[df['dts'] <= shorelimit]

    if oldornew == 'old':
        label = f'≤{shorelimit} km to the shore, Aurin (2018) data set'
    else:
        label = f'≤{shorelimit} km to the shore, Hansel (2021) data set'

    label0 = 'shore'

elif openorshore == 'open':

    df = df[(df['dts'] > shorelimit) & (df['DOC'] < opts.doclimit)]

    if oldornew == 'old':
        label = f'>{shorelimit} km from the shore, Aurin (2018) data set'
    else:
        label = f'>{shorelimit} km from the shore, Hansel (2021) data set'
    label0 = 'open'
else:

    df = df[df['DOC'] < opts.doclimit2]

    label = 'Hansel (2021) data set'
    label0 = 'all'

plotdir = opts.plotdir

os.makedirs(plotdir, exist_ok=True)
os.makedirs(opts.mdir, exist_ok=True)

# %%

if (opts.rng is None) or (opts.rng < 0):
    rng = None
else:
    rng = np.random.RandomState(opts.rng)

version = 'v1'
modelno = opts.no
modeltype = opts.model.lower()

criterion = opts.criterion

if opts.L1:
    criterion = 'absolute_error'

if modeltype == 'lr':
    criterion = 'squared_error'

if modeltype == 'nn':
    criterion = 'adam'

model_version = f'{modeltype}_{version}.{modelno}_{criterion}'

logging.info('Fitting %s', model_version)

# initial/default parameters
params = {'max_depth': 30, 'n_estimators': 400, 'max_features': 0.4}
if modeltype == 'gb':
    params = {'max_depth': 5, 'n_estimators': 200, 'max_features': 0.4}

if (modeltype != 'lr') and (modeltype != 'nn') and not opts.optimize:
    try:
        params = load_params(model_version)
    except Exception as e:
        logging.warning('%s', e)
        logging.info('using defaults')
    logging.info('Using parameters:')
    logging.info(params)

# params = model_params.get(model_version)
if params is None:
    logging.error('could not get optimized parameters')
    sys.exit(1)

# all Rrs
Rrs = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_555', 'Rrs_670']

if modelno == 1:
    # full model
    other = ['sqrtpp', 'salt', 'temp', 'depth', 'dts', 'lat']

elif modelno == 2:
    # version without geographical variables
    other = ['sqrtpp', 'salt', 'temp']

elif modelno == 3:
    # reduced linear model
#    Rrs = ['Rrs_670', 'Rrs_510']
#    other = ['temp', 'lat', 'depth', 'dts', 'sqrtpp']
    Rrs = ['Rrs_443', 'Rrs_555', 'Rrs_490', 'Rrs_510']
    other = ['temp', 'lat', 'depth', 'dts', 'sqrtpp']
#    other = ['temp']

elif modelno == 4:
    # full model with month
    #other = ['sqrtpp', 'salt', 'temp', 'depth', 'dts', 'lat', 'month']
    # test, now same as model 1
    other = ['sqrtpp', 'salt', 'temp', 'depth', 'dts', 'lat']

elif modelno == 5:
    # another reduced regression model
    #Rrs = ['Rrs_670', 'Rrs_510']
    Rrs = ['Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_555', 'Rrs_670']
    other = ['sqrtpp', 'salt', 'temp', 'depth', 'dts']
    #other = ['temp', 'lat', 'depth', 'dts']
    #other = ['salt', 'temp', 'depth', 'dts']

elif modelno == 6:
    # full model with month, and wclass
    other = ['sqrtpp', 'salt', 'temp', 'depth', 'dts', 'lat', 'month', 'wclass']

elif modelno == 7:
    # full model without month
    other = ['sqrtpp', 'salt', 'temp', 'depth', 'dts', 'lat']


else:
    logging.error('unknown model number')
    sys.exit(1)

names = Rrs + other

# %%

# max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes.

if modeltype.lower() == 'rf':

    model = RandomForestRegressor(
        criterion=criterion,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        n_jobs=-1,
        random_state=rng,
    )
    mlabel = 'random forest'
    
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
        max_features = trial.suggest_float('max_features', 0.1, 1.0)

        model = RandomForestRegressor(
            criterion=criterion,  # squared_error, friedman_mse, absolute_error
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaf_nodes=None,
            min_samples_leaf=1,
            max_features=max_features,
            n_jobs=-1,
            random_state=rng,
        )
        score = cross_val_score(model, X, y, groups=obsgroup, n_jobs=2,
                                cv=GroupShuffleSplit(n_splits=opts.n_splits,
                                                     test_size=opts.test_size,
                                                     random_state=rng)
                                )
        accuracy = score.mean()

        return accuracy


elif modeltype.lower() == 'gb':
    model = GradientBoostingRegressor(
        loss=criterion,
        criterion='friedman_mse',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        random_state=rng,
    )
    mlabel = 'gradient boosting'

    def objective(trial):

        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        n_estimators = trial.suggest_int('n_estimators', 10, 400, log=True)
        max_features = trial.suggest_float('max_features', 0.1, 1.0)

        model = GradientBoostingRegressor(
            loss=criterion,
            criterion='friedman_mse',
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=rng,
        )

        score = cross_val_score(model, X, y, groups=obsgroup, n_jobs=2,
                                cv=GroupShuffleSplit(n_splits=opts.n_splits,
                                                     test_size=opts.test_size,
                                                     random_state=rng),
                                )
        accuracy = score.mean()

        return accuracy

elif modeltype.lower() == 'lr':
    model = LinearRegression()
    opts.scale = True
    mlabel = 'linear regression'

# this is not ready yet
elif modeltype.lower() == 'nn':
    mlabel = 'neural network'
    opts.scale = True

    alpha = 0.001
    learning_rate = 0.0001
    n_layers = 3
    layer_width = 32

    n_layers = 3
    layer_width = 128


    model = MLPRegressor(
        # hidden_layer_sizes=(32, 32,),
        hidden_layer_sizes=tuple(layer_width for i in range(n_layers)),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',  # 'constant'
        learning_rate_init=learning_rate,
        alpha=alpha,
        shuffle=False,
        early_stopping=True,
        validation_fraction=0.25,
        max_iter=2000,
        random_state=rng,
        verbose=False,
    )

    def objective(trial):

        alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layer_width = trial.suggest_categorical('layer_width', [32])

        model = MLPRegressor(
            hidden_layer_sizes=tuple(layer_width for i in range(n_layers)),
            activation='relu',
            solver='adam',
            learning_rate='constant',
            learning_rate_init=learning_rate,
            alpha=alpha,
            shuffle=False,
            early_stopping=False,
            validation_fraction=0.25,
            max_iter=2000,
            random_state=rng,
            verbose=False,
        )

        score = cross_val_score(model, Xscaled, y, groups=obsgroup, n_jobs=2,
                                cv=GroupShuffleSplit(n_splits=opts.n_splits,
                                                     test_size=opts.test_size,
                                                     random_state=rng),
                                )
        accuracy = score.mean()

        return accuracy

else:
    print('unknown model')
    sys.exit(1)

# %%

Xy = df[names + ['DOC', 'time']].dropna()
X = Xy[names]
y = Xy.DOC

# scale X for model estimation, not needed for random forest
if opts.scale:
    scaler = preprocessing.StandardScaler().fit(X)
    Xscaled = scaler.transform(X)
else:
    Xscaled = X
    scaler = None

# %%
# stratified sampling
nobs = len(X)
if cvgroup == 'yearly':
    obsgroup = Xy['time'].dt.year.astype(int)
else:
    obsgroup = np.repeat(np.c_[np.zeros(50), np.ones(50)], nobs//100 + 100)[:nobs]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=opts.test_size,
#                                                    shuffle=True, stratify=st)


# %%
# Add stream handler of stdout to show the messages
if opts.optimize:
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = model_version  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name,
                                direction='maximize', load_if_exists=True)

    study.optimize(objective, timeout=60*opts.timeout)

    logging.info(study.best_trial)
    logging.info('optimization done, exiting')
    sys.exit(0)


# %%
logging.info('fitting')
# We have optimized model, so use the whole data 
#model.fit(X_train, y_train)
model.fit(Xscaled, y)

# %% cross validation
logging.info('calculating cv scores')

n_splits = opts.n_splits
score = cross_validate(model, Xscaled, y, groups=obsgroup, n_jobs=4,
                       scoring=['explained_variance',
                                'neg_mean_absolute_error',
                                'neg_root_mean_squared_error',
                                'neg_mean_absolute_percentage_error'],
                       cv=GroupShuffleSplit(n_splits=n_splits,
                                            test_size=opts.test_size,
                                            random_state=rng),
                       )
print('Q2', score['test_explained_variance'].mean())
print('MAE', -score['test_neg_mean_absolute_error'].mean())
print('RMSE', -score['test_neg_root_mean_squared_error'].mean())
print('MAPE', -score['test_neg_mean_absolute_percentage_error'].mean())


#%%
#y_pred1 = model.predict(X_train)
#y_pred2 = model.predict(X_test)
y_pred = model.predict(Xscaled)

#print(f'Test set score {model.score(X_test, y_test):0.2g}')


#%%

#rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
#print("The mean squared error (MSE) on test set: {:.4f}".format(rmse))
rmse = np.sqrt(np.var(y.values - y_pred))
resid = (y_pred - y.values) / rmse

#rmse_test = np.sqrt(mean_squared_error(y_test, y_pred2))
#me_test = mean_absolute_error(y_test, y_pred2)
#me = mean_absolute_error(y.values, y_pred)

#print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_test))
#print("The root mean squared error (RMSE) on full set: {:.4f}".format(rmse))
#print("The mean error (ME) on test set: {:.4f}".format(me_test))
#print("The mean error (ME) on full set: {:.4f}".format(me))


#%% R2

tss = np.sum((y.values - y.values.mean())**2)
rss = np.sum((y_pred - y.values)**2)
r2 = 1 - rss / tss
print(f'R²: {r2:.3g}')

#tss = np.sum((y_train - y_train.mean())**2)
#rss = np.sum((y_pred1 - y_train)**2)
#r22 = 1 - rss / tss
#print(f'R² train: {r22:.3g}')

#tss = np.sum((y_test - y_test.mean())**2)
#rss = np.sum((y_pred2 - y_test)**2)
#r23 = 1 - rss / tss
#print(f'R² test: {r23:.3g}')


cv_scores = {
    'R2': r2,
    'Q2': score['test_explained_variance'].mean(),
    'MAE': -score['test_neg_mean_absolute_error'].mean(),
    'RMSE': -score['test_neg_root_mean_squared_error'].mean(),
    'MAPE': -score['test_neg_mean_absolute_percentage_error'].mean(),
    'test_size': opts.test_size,
    'n_splits': n_splits,
    }


#%%

#n_estimators = model.n_estimators

if modeltype == 'lr':
    # Lasso 
    # Use L1 penalty
    estimator = LassoCV()
    lasso = estimator.fit(Xscaled, y)
    scores = pd.DataFrame({'name': names, 'score': np.abs(lasso.coef_)})
    scores = scores.sort_values(by='score')

elif modeltype == 'nn':
    pass

else:
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5


#%% save model


mfile = f'{opts.mdir}/{model_version}_model_{label0}.joblib'
dump(model, mfile)

logging.info('Wrote %s', mfile)

# why this???
#names = ['salinity' if n == 'salt' else n for n in names]

dfile = f'{opts.mdir}/{model_version}_data_{label0}.pkl'
pickle.dump({'X': X, 'y': y, 'names': names,
             'model_type': modeltype,
             'model_version': model_version,
             'scaler': scaler,
#             'y_train': y_train, 'y_test': y_test,
#             'X_train': X_train, 'X_test': X_test,
             'cv_scores': cv_scores,
             },
            open(dfile, 'wb'), -1)

logging.info('Wrote %s', dfile)


#%% all figures

plt.close()

#%% another version for ws slides

logging.info('plotting')


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
ax = plt.gca()
#plt.plot(y_train, y_pred1, '.', label='train data')
#plt.plot(y_test, y_pred2, '.', label='test data')
plt.plot(y, y_pred, '.', label='obs')
xyline()
#plt.legend()
plt.xlabel(f'Observed DOC [{docunit}]')
plt.ylabel(f'Predicted DOC [{docunit}]')
plt.title(f'A) Model fit - {mlabel}')
#plt.title(f'R²_{train}: {r2*100:.3g}%\nR²_{test}: {r23*100:.3g}%')
#ax.text(0.90, 0.05,
#        f"R²$_{{train}}$ = {r22*100:.3g}%\nR²$_{{test}}$ = {r23*100:.3g}%",
#        transform=ax.transAxes, fontsize=10,
#        verticalalignment='bottom', horizontalalignment='right')

ax.text(0.90, 0.05,
        f"R² = {r2*100:.3g}%",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right')


plt.subplot(1, 2, 2)
if modeltype == 'lr':
    plt.bar(scores['name'], scores['score'])
    plt.setp(plt.gca().get_xticklabels(), fontsize=12, rotation='vertical')
    plt.title(f'B) Lasso scores - {mlabel}')

elif modeltype == 'nn':
    pass

else:
    plt.bar(pos, feature_importance[sorted_idx], align='center')
    plt.xticks(pos, np.array(names)[sorted_idx])
    plt.setp(plt.gca().get_xticklabels(), fontsize=12, rotation='vertical')
    plt.title(f'B) Feature Importance - {mlabel}')

pfile = f'{plotdir}/{modeltype.lower()}_model_{label0}_{model_version}.pdf'
plt.savefig(pfile, bbox_inches='tight')

logging.info('Wrote %s', pfile)

logging.info('Done for %s', model_version)
