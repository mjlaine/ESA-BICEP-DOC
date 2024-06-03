# ML model parameters
# also some data parameters
# defines global command line parameters used in several scripts

"""
From: https://doi.org/10.1002/lno.10074
The open ocean is defined as the region having a water depth of more than 200 m as indicated by
the ETOPO1 Global Relief Model (Amante and Eakins, 2009) and having a minimum sea surface salinity of 30.
"""

import argparse

import matplotlib as mpl
import cmocean

import optuna

model_params = {
    'rf_v1.1_friedman_mse': {'max_depth': 28, 'n_estimators': 419, 'max_features': 0.8455912646361617},
    'rf_v1.1_absolute_error': {'max_depth': 18, 'n_estimators': 136, 'max_features': 0.9726092958187431},
    'gb_v1.1_absolute_error': {'max_depth': 10, 'n_estimators': 200, 'max_features': 0.622125983656194},

    'lr_v1.1_squared_error': {},
    'lr_v1.3_squared_error': {},

}


def load_params(m):
    study = optuna.load_study(study_name=m, storage=f"sqlite:///{m}.db")
    return study.best_params


def load_all_studies(models):
    out = {}
    for m in models:
        out[m] = load_params(m)
    return out


# some defaults
shorelimit = 300
#doclimit = 200
doclimit = 100
doclimit2 = 400
docmin = 40
latlimit = 70
mindepth = 100  # obs negative depth in data
maxsalt = 50
minsalt = 30  # for open water, maybe

Rrs = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_555', 'Rrs_670']

vars1 = ['sqrtpp', 'salt', 'temp', 'depth', 'dts']
othervars = ['sqrtpp', 'salt', 'temp', 'depth', 'dts', 'lat', 'month']
allvars = ['DOC'] + Rrs + othervars

units2 = {
    'DOC': 'µmol/kg',
    'pp': 'mgC/m²/d',
    'sqrtpp': 'mgC/m²/d',
    'salt': 'PSS-78',
    'temp': '°C',
    'depth': 'm',
    'dts': 'km',
    'Rrs': 'sr-1',
    'lat': 'degrees north',
    'lon': 'degrees east',
    'month': 'numeric',
    }

units = {
    'DOC': 'µmol kg⁻¹',
    'pp': 'mgC m⁻² d⁻¹',
    'sqrtpp': 'mgC m⁻² d⁻¹',
    'salt': 'PSS-78',
    'temp': '°C',
    'depth': 'm',
    'dts': 'km',
    'Rrs': 'sr⁻¹',
    'lat': 'degrees north',
    'lon': 'degrees east',
    'month': 'month number',
    }



labels = {
    'DOC': 'DOC',
    'pp': 'primary production',
    'sqrtpp': '(sqrt of) primary production',
    'salt': 'salinity',
    'temp': 'sea surface temperature',
    'depth': 'water depth',
    'dts': 'distance to shore',
    'Rrs': 'Rrs',
    'lat': 'latitude of the observation',
    'lon': 'longitude of the observation',
    'month': 'month',
    }


# older version
#wcvars = ['water_class1', 'water_class2', 'water_class3',
#          'water_class4', 'water_class5', 'water_class6',
#          'water_class7', 'water_class8', 'water_class9',
#          'water_class10', 'water_class11', 'water_class12',
#          'water_class13', 'water_class14']

wcvars = [f'owc{i:02d}' for i in range(1,15)]

parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--mdir", default='models', type=str)
parser.add_argument("--plotdir", default='plots', type=str)
parser.add_argument("--old", action='store_true')
parser.add_argument("--open", action='store_true')
parser.add_argument("--shore", action='store_true')
parser.add_argument("--L1", action='store_true')
parser.add_argument("--all", action='store_true')
parser.add_argument("--optimize", action='store_true')
parser.add_argument("--scale", action='store_true')
parser.add_argument("--shorelimit", default=shorelimit, type=float)
parser.add_argument("--doclimit", default=doclimit, type=float)
parser.add_argument("--docmin", default=docmin, type=float)
parser.add_argument("--doclimit2", default=doclimit2, type=float)
parser.add_argument("--latlimit", default=latlimit, type=float)
parser.add_argument("--mindepth", default=mindepth, type=float)
parser.add_argument("--maxsalt", default=maxsalt, type=float)
parser.add_argument("--minsalt", default=minsalt, type=float)
parser.add_argument("--model", "-m", default='rf', type=str)
parser.add_argument("--no", default=1, type=int)
parser.add_argument("--n_estimators", "-n", default=200, type=int)
parser.add_argument("--rng", default=None, type=int)
parser.add_argument("--n_splits", default=20, type=int)
parser.add_argument("--criterion", "-c", default='friedman_mse', type=str)
parser.add_argument("--test_size", default=0.25, type=float)
parser.add_argument("--timeout", "-t", default=30, type=float)  # timeout in minutes


# plotting defaults
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 13}
mpl.rc('font', **font)

# mpl.rc('figure', **{'figsize': [6, 4], 'dpi': 150})  # [6.4, 4.8] is the default

#mpl.pyplot.rcParams['figure.figsize'] = [6, 4]  # [6.4, 4.8] is the default
mpl.pyplot.rcParams['figure.dpi'] = 100
mpl.pyplot.rcParams['savefig.dpi'] = 200
mpl.pyplot.rcParams['savefig.bbox'] = 'tight'

doc_cmap = cmocean.cm.matter
doc_vmin = docmin
doc_vmax = doclimit