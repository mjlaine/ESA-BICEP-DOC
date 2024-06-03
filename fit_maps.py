#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
From fit to global prediction.

Usage ./fit_maps.py model_version ncpu

for example

./fit_maps.py lr_v1.3_squared_error 4

See `doruns.sh`, which is calling this.

@author: marko.laine@fmi.fi
"""

import sys

from genmaps import generate_maps

model_version = sys.argv[1]
njobs = int(sys.argv[2])

# model_version = 'rf_v1.1_friedman_mse'
mdir = 'models'
model_type = model_version[0:2]
outdir = f'/tmp/DOC_{model_type}'

datafile = f'{mdir}/{model_version}_data_open.pkl'
modelfile = f'{mdir}/{model_version}_model_open.joblib'

generate_maps(datafile, modelfile, outdir=outdir, njobs=njobs)
#  , years=[2011], months=[6])
