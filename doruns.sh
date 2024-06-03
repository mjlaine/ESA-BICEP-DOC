#!/bin/bash

# do the runs for BICEP-DOC paper

# python environment to be used
source /usr/local/share/venvs/sci/bin/activate

TIMEOUT=60  # timeout for each optimization run in minutes
MDIR=models  # save and load models fron here
PDIR=plots  # save plots here
NSPLIT=20  # nsplit option

# Set random state to a fixed values with --rng ${RNG}
# default and negative value is None
RNG=20240128
# RNG=-1

# select what to do
OPTIMIZE=yes
FIT=yes
MAPS=no

# optimize RF and GB, use random_state=None

if [ x${OPTIMIZE} == "xyes" ]; then
    ./fit_model.py --model gb --no 4 --L1 --optimize --timeout ${TIMEOUT} --n_splits ${NSPLIT}
    ./fit_model.py --model gb --no 4 --criterion "squared_error" --optimize --timeout ${TIMEOUT} --n_splits ${NSPLIT}
    ./fit_model.py --model rf --no 4 --optimize --timeout ${TIMEOUT} --n_splits ${NSPLIT}
    ./fit_model.py --model rf --no 4 --L1 --optimize --timeout ${TIMEOUT} --n_splits ${NSPLIT}
fi

# model fits

NSPLIT=30
NO=4

if [ x${FIT} == "xyes" ]; then

    # two first are "static" multiple linear models
    ./fit_model.py --model lr --no 1 --mdir ${MDIR} --plotdir ${PDIR} --n_splits 100 --rng ${RNG}
    ./fit_model.py --model lr --no 3 --mdir ${MDIR} --plotdir ${PDIR} --n_splits 100 --rng ${RNG}
    ./fit_model.py --model rf --no ${NO} --mdir ${MDIR} --plotdir ${PDIR} --n_splits ${NSPLIT} --rng ${RNG}
    ./fit_model.py --model rf --no ${NO} --L1 --mdir ${MDIR} --plotdir ${PDIR} --n_splits ${NSPLIT} --rng ${RNG}
    ./fit_model.py --model gb --no ${NO} --L1 --mdir ${MDIR} --plotdir ${PDIR} --n_splits ${NSPLIT} --rng ${RNG}
    ./fit_model.py --model gb --no ${NO} --criterion "squared_error" --mdir ${MDIR} --plotdir ${PDIR} --n_splits ${NSPLIT} --rng ${RNG}

fi

# generate maps
if [ x${MAPS} == "xyes" ]; then

    ./fit_maps.py lr_v1.3_squared_error 4
    ./fit_maps.py rf_v1.4_friedman_mse 2
    ./fit_maps.py rf_v1.4_absolute_error 2
    ./fit_maps.py gb_v1.4_squared_error 4
    ./fit_maps.py gb_v1.4_absolute_error 4

fi
