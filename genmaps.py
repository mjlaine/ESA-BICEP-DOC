# generate global monthly maps

#import sys
import os
import pickle
import joblib
#import argparse

import time

from itertools import product

import numpy as np
import xarray as xr

#import multiprocessing
from joblib import Parallel, delayed

from data_utils import salinity, nppdata
from data_utils import ocdata, era5sst, era5dts, pmlsst
from data_utils import bathymetrydata


def processmonth(year, month, model, Rrs, names, dts, depths,
                 outdir='/tmp/DOC',
                 scaler=None,
                 model_type='na',
                 output=False):
    print(f'started {year}-{month:02d}')
    
    if Rrs is not None:
        oc = ocdata(year, month, vars=Rrs)     # (lat: 4320, lon: 8640)
    npp = nppdata(year, month)   # (lat: 2160, lon: 4320)
    sss = salinity(year, month)  # (lat: 584, lon: 1388)
    # PML version of SST from Bror
    sst = pmlsst(year, month) - 273.15  # same resolution as PP
    # scale all to same as NPP (1/12°)
    sss2 = sss.reindex_like(npp, method='nearest')
    if Rrs is not None:
        oc2 = oc.reindex_like(npp, method='nearest')
    npp2 = npp.reindex_like(npp, method='nearest')
    sst2 = sst.reindex_like(npp, method='nearest')
    dts2 = dts.reindex_like(npp, method='nearest')
    depth2 = depths.reindex_like(npp, method='nearest')

    out = sss2.merge(npp2)
    #out = npp2.merge(sss2)
    if Rrs is not None:
        out = out.merge(oc2, compat='override')
    out = out.merge(sst2, compat='override')
    out = out.merge(dts2, compat='override')
    out = out.merge(depth2, compat='override')
    out = out.rename({'sss': 'salt'})

    out = out.rename({'sst': 'temp'})
    out = out.rename({'bathymetry': 'depth'})

    outdf = out.to_dataframe()  # to pandas
    outdf = outdf.reset_index(level=['lat', 'lon'])
    outdf['sqrtpp'] = np.sqrt(outdf['pp'])
    outdf['month'] = outdf['time'].dt.month
    # need to have global data for this
    # outdf['wclass'] = np.argmax(outdf[wcvars].values, axis=1) + 1

    X = outdf[names].copy()
    if scaler is not None:
        X[names] = scaler.transform(X[names])

    inds = np.isfinite(X).all(axis=1)

    mpred = np.zeros(X.shape[0], dtype=np.float32) * np.nan
    #p = model.predict(X.loc[inds])
    #print(p.shape)
    #print(p)
    #print(inds)
    mpred[inds] = model.predict(X.loc[inds]).ravel()
    # mpred = np.maximum(mpred, 0.0001)  # some predictions might be negative

    # add predictions and prediction std to data
    out['DOC'] = (['lat', 'lon'], mpred.reshape((len(out.lat), len(out.lon))))
    out['DOC'].attrs['long_name'] = f'Estimated DOC (using {model_type.upper()} method)'
    # out['DOCse'] = (['lat', 'lon'], mpredse.values.reshape((len(out.lat), len(out.lon))))
    out = out['DOC'].to_dataset()  # save only doc
    out.to_netcdf(f'{outdir}/DOC_{year}_{month:02d}.nc')
    print(f'done {year}-{month:02d}')
    if output:
        return out


Rrsall = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_555', 'Rrs_670']
yearsall = np.arange(2010, 2019)
monthsall = np.arange(1, 13)


def domaps(model, names, years, months, scaler, outdir,
           Rrs=Rrsall,
           model_type='na',
           njobs=-1):
    
    dts = era5dts()  # distance to shore, latitude: 1801, longitude: 3600
    depths = bathymetrydata()

    Parallel(n_jobs=njobs)(
            delayed(processmonth)(year, month, model=model, Rrs=Rrs,
                                  names=names,
                                  dts=dts, depths=depths,
                                  scaler=scaler,
                                  model_type=model_type,
                                  outdir=outdir)
            for year, month in product(years, months))


def combine_years(years, model_type, model_version, outdir='/tmp/DOC'):

    for year in years:
        print(f'Year {year}')
        files = os.path.expanduser(f'{outdir}/DOC_{year}_*.nc')
        x = xr.open_mfdataset(files, concat_dim='time',
                              combine='nested')
        x = x['DOC']
        file = f'{outdir}/global_DOC_{model_type.upper()}_{model_version}_{year}.nc'
        x.to_netcdf(file)
        print(f'Wrote {file}')


def generate_maps(datafile, modelfile, outdir, model=None,
                  Rrs=Rrsall, years=yearsall, months=monthsall,
                  njobs=-1):
        
    t0 = time.perf_counter()

    data = pickle.load(open(datafile, 'rb'))
    names = data['names']
    scaler = data['scaler']
    model_version = data['model_version']
    model_type = data['model_type']
    if model is None:
        model = joblib.load(modelfile)
    
    os.makedirs(outdir, exist_ok=True)
    
    domaps(model, names, years, months, scaler, outdir, Rrs=Rrs,
           model_type=model_type, njobs=njobs)
    if len(months) > 1:
        print('done, now combining the datasets yearly')
        combine_years(years, model_type, model_version, outdir)

    e1 = time.perf_counter() - t0
    print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(e1))}')


