#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data utilities for ESA-BICEP.

Some utilities to load Oceancolour data from PML.

Also code to fetch salinity CCI data and some plotting routines.

The NPP dataset is stored locally in ~/DATA/ESA-BICEP/NPP/month/.
It is downloaded from
https://rsg.pml.ac.uk/shared_files/gku/ESA_comms/annual/
https://rsg.pml.ac.uk/shared_files/gku/ESA_comms/month/

These are a set of netCDF files that include primary production in
units of mg C m-2 month-1 (for monthly products) or mg C m-2 y-1 (for
annual climatologies). In the folders, you can find monthly products
(this is the highest temporal resolution) and climatologies for each
year between 1998-2018 and one climatology for the whole period of
1998-2018. Spatial resolution is 9km.

Ocean colour from: https://catalogue.ceda.ac.uk/uuid/99348189bd33459cbd597a58c30d8d10
salinity from : https://catalogue.ceda.ac.uk/uuid/7813eb75a131474a8d908f69c716b031


Distance to sea is constructed from ERA5 land sea map downloaded from
https://confluence.ecmwf.int/pages/viewpage.action?pageId=140385202#ERA5Land:datadocumentation-parameterlistingParameterlistings

ERA5 land sea mask file `lsm_1279l4_0.1x0.1.grb_v4_unpack.nc` is transferred to distances with a R script:
```R
r <- raster::raster('lsm_1279l4_0.1x0.1.grb_v4_unpack.nc')
d <- raster::gridDistance(raster::rotate(raster::subs(r, data.frame(from=c(0),to=c(NA)), subsWithNA=FALSE)),1)
raster::writeRaster(d,'dts_1279l4_0.1x0.1.grb_v4_unpack.nc')
```


Created on Tue Jun  2 17:24:22 2020
@author: marko.laine@fmi.fi

"""
# %% imports

import datetime
import os
import re

import numpy as np
import xarray as xr
import dask
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cpf
import cmocean

import seaborn as sns
# sns.set_theme('paper')

# %% Save with compression
# https://stackoverflow.com/questions/40766037/specify-encoding-compression-for-many-variables-in-xarray-dataset-when-write-to

# location of BICEP data
bicep_data_dir = '~/DATA/ESA-BICEP'


def savenc(ds, file):
    """Save to netcdf with zlib compression."""
    encoding = {}
    encoding_keys = ("_FillValue", "dtype", "scale_factor",
                     "add_offset", "grid_mapping")
    for data_var in ds.data_vars:
        encoding[data_var] = {key: value for key,
                              value in ds[data_var].encoding.items()
                              if key in encoding_keys}
        encoding[data_var].update(zlib=True, complevel=5)

    ds.to_netcdf(file, encoding=encoding)


# save pandas df with attributes
def dfsave(df, file, complevel=9):
    """Save pandas data frame to h5 with attributes"""
    with pd.HDFStore(file) as store:
        store.put('df', df, format='table', complevel=complevel)
        store.get_storer('df').attrs.metadata = df.attrs


def dfload(file):
    """Load h5 pandas dataframe with attributes"""
    with pd.HDFStore(file) as store:
        df = store.get('df')
        metadata = store.get_storer('df').attrs.metadata
        df.attrs.update(metadata)
    return df


# %% NPP
def npp_file(year, month, old=False):
    """File name of one OC dataset."""
    if old:
        return f'{bicep_data_dir}/NPP/month/OC-CCI_PP_Kulk_et_al_{year}_{month:02d}.nc'
    else:
        return f'{bicep_data_dir}/NPP/PP_CEDA/PP_OC-CCI_v4-2_{year}_{month:02d}_Kulk_et_al.nc'


# %%

def nppdata(year, month, old=False):
    """NPP loader."""
    url = npp_file(year, month, old=old)
    if old:
        return xr.open_dataset(url)['pp'].to_dataset().load()
    else:
        return xr.open_dataset(url)['primary_production'].to_dataset().rename({'primary_production': 'pp'}).load()

# %%


def preprocess(ds):
    """Preprosessor for combining the files in a single xarray dataset."""
    f = ds.encoding['source']
    m = int(f[-5:-3])
    y = int(f[-10:-6])
    d = datetime.datetime(y, m, 15, 0, 0)
    print(d)
    ds['date'] = d
    return ds


def preprocess2(ds):
    """Preprosessor for combining the files in a single xarray dataset."""
    f = ds.encoding['source']
    m = int(f[-16:-14])
    y = int(f[-21:-17])
    d = datetime.datetime(y, m, 15, 0, 0)
    print(d)
    ds['date'] = d
    return ds

# %% open all!


def npp_dataset_old():
    """Open all NPP files."""
    files = os.path.expanduser(f'{bicep_data_dir}/NPP/month/*.nc')
    xa = xr.open_mfdataset(files, concat_dim='date',
                           preprocess=preprocess,
                           combine='nested')
    return xa


def npp_dataset():
    """Open all NPP files."""
    files = os.path.expanduser(f'{bicep_data_dir}/NPP/PP_CEDA/*.nc')
    xa = xr.open_mfdataset(files, concat_dim='date',
                           preprocess=preprocess2,
                           combine='nested').rename({'primary_production': 'pp'})
    return xa


# %% OC data loader

def ocdata(year, month, vars=['Rrs_555', 'Rrs_510', 'Rrs_490'], remote=False):
    """OC data loader."""
    url = 'http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v4.2-MONTHLY'
    if remote:
        time = f'{year}-{month:02d}-01'
        ds = xr.open_dataset(url)
        return ds.sel(time=time)[vars].load()
    else:  # load all variables from local file
        file = f'{bicep_data_dir}/OC_Rrs/OC_Rrs_{year}_{month:02d}.nc'
        ds = xr.open_dataset(file).load()
        return ds

# %% salinity CCI data loader


def salinity(year, month, remote=False):
    """Load CCI salinity."""
    day = 15
    if remote:  # load from opendap server
        url = f'http://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/sea_surface_salinity/data/v02.31/30days/{year}/ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-{year}{month:02d}{day:02d}-fv2.31.nc'
    else:  # Use local data
        url = os.path.expanduser(f'~/DATA/esacci/sea_surface_salinity/data/v02.31/30days/{year}/ESACCI-SEASURFACESALINITY-L4-SSS-MERGED_OI_Monthly_CENTRED_15Day_25km-{year}{month:02d}{day:02d}-fv2.31.nc')
    ds = xr.open_dataset(url)
    return ds.sss.isel(time=0).to_dataset().load()


# %% SST CCI data loader
# data from  09/1981 to 12/2016
# daily data (lat: 3600, lon: 7200)

def ccisst(year, month, day=15):
    """Load CCI sea surface temperature."""
    url = f'http://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc'
    ds = xr.open_dataset(url)
    # return ds
    return ds.analysed_sst.isel(time=0).to_dataset().load()


# Data from Bror
# 201811-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP_UPSCALE9KM-v02.0-fv02.0.nc
def pmlsst(year, month):
    """Load PML SST data."""
    file = os.path.expanduser(f'{bicep_data_dir}/SST/{year}{month:02d}-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP_UPSCALE9KM-v02.0-fv02.0.nc')
    da = xr.open_dataset(file).sst.isel(time=0)
    da['time'] = da['time'] + np.timedelta64(14, 'D')
    return da


def era5sst(year=None, month=None):
    """ERA5 monthly SST."""
    files = os.path.expanduser(f'{bicep_data_dir}/ERA5/ERA5_sst*.grib')
    ds = xr.open_mfdataset(files, engine='cfgrib')
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    sst = ds.sst  # - 273.15
    sst.coords['lon'] = (sst.coords['lon'] + 180) % 360 - 180
    sst = sst.sortby(sst.lon)
    return sst


# %% Load distance to shore data
# need to regenerate this using -180 - +180, so better at lon 0°

def era5dts():
    """Read distance to shore data set."""
    dts = xr.open_dataset(f'{bicep_data_dir}/dts_rot_1279l4_0.1x0.1.grb_v4_unpack.nc')
    dts = dts.rename({'longitude': 'lon', 'latitude': 'lat'})

# using rotated version now
#    dts.coords['lon'] = (dts.coords['lon'] + 180) % 360 - 180
#    dts = dts.sortby(dts.lon)

    dts = dts.layer.rename('dts') / 1000  # to km
    return dts


def era5lsm(d=200):
    """Load land sea mask"""
    dts = xr.open_dataset(f'{bicep_data_dir}/dts_rot_1279l4_0.1x0.1.grb_v4_unpack.nc')
    dts = dts.rename({'longitude': 'lon', 'latitude': 'lat'})
    dts = dts.layer.rename('dts') / 1000
    lsm = dts < d
    # lsm.coords['lon'] = (lsm.coords['lon'] + 180) % 360 - 180
    # lsm = lsm.sortby(lsm.lon)
    return lsm


# %% Bathymetry data

def bathymetrydata(load=True):
    b = xr.open_dataset(f'{bicep_data_dir}/bathy_9km_new_fill_val.nc')
    # b.coords['lon'] = (b.coords['lon'] + 180) % 360 - 180
    # b = b.sortby(b.lon)
    b = b['bathymetry']
    if load:
        b.load()
        b.close()
    return b


# %% some plotting utilities

def ocplot(vre, df, x='DOC', hue='lat', height=15, file=None):
    """Plot selected variables against DOC."""
    vars = list(filter(lambda x: re.search(vre, x), list(df.keys())))
    n = len(vars)
    fig = plt.figure(figsize=(10, height))
    i = 1
    for v in vars:
        ax = fig.add_subplot(n, 1, i)
        sns.scatterplot(x=x, y=v, data=df, hue=hue, ax=ax)
        i += 1
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()
# %%


def xyline(a=0, b=1, scaley=False, **kwargs):
    """Plot straight line given intercept and slope."""
    ax = plt.gca()
    x = np.array(ax.get_xlim())
    y = a + b * x
    ax.plot(x, y, '-', scaley=scaley, **kwargs)


def plotts_old(da, lon, lat, dx=10,
               df=None, decdiff=1, file=None, variable='DOC'):
    """Plot time series on a location."""
    # find location index
    xi = np.argmin(np.abs(da['lon'].values - lon))
    yi = np.argmin(np.abs(da['lat'].values - lat))
    # average over ±dx pixels
    dai = da.isel(lon=slice(xi - dx, xi + dx),
                  lat=slice(yi - dx, yi + dx))
    m = dai.mean(dim=['lon', 'lat'], skipna=True)
    m.attrs = dai.attrs
    lx = (da.lon[1]-da.lon[0]).values
    p = m.plot(linestyle='--', marker='o', markersize=3, linewidth=1)
    plt.title(f'{lon:.5g}, {lat:.5g} (±{dx} pixels of size {lx:.2})')
    if df is not None:
        # add observations
        inds = ((np.abs(df['lat'] - lat) < decdiff) &
                (np.abs(df['lon'] - lon) < decdiff))
        fdi = df[inds].set_index('time')
        if len(fdi) > 0:
            fdi[variable].plot(marker='o', linestyle='none',
                               markersize=5)
    if file is not None:
        plt.savefig(file, dpi=100, bbox_inches='tight')
        plt.close()
    return p


def plotts(da, lon, lat, dx=10,
           df=None, decdiff=1, file=None, variable='DOC', label=None,
           dflabel=None, dfmarker='o',
           ax=None):
    """Plot time series on a location."""
    # find location index
    if ax is None:
        ax = plt.gca()
    xi = np.argmin(np.abs(da['lon'].values - lon))
    yi = np.argmin(np.abs(da['lat'].values - lat))
    # average over ±dx pixels
    dai = da.isel(lon=slice(xi - dx, xi + dx),
                  lat=slice(yi - dx, yi + dx))
    m = dai.mean(dim=['lon', 'lat'], skipna=True)
    m.attrs = dai.attrs
    lx = (da.lon[1]-da.lon[0]).values
    p = m.plot(linestyle='--', marker='o', markersize=3, linewidth=1, label=label, ax=ax)
    #plt.title(f'{lon:.5g}, {lat:.5g} (±{dx} pixels of size {lx:.2})')
    ax.set_title(f'{lon:.5g}, {lat:.5g} (±{dx} pixels of size {lx:.2})')
    if df is not None:
        # add observations
        inds = ((np.abs(df['lat'] - lat) < decdiff) &
                (np.abs(df['lon'] - lon) < decdiff))
        fdi = df[inds].set_index('time')
        if len(fdi) > 0:
            fdi[variable].plot(marker=dfmarker, linestyle='none',
                               color='black',
                               markersize=8, ax=ax, label=dflabel)
    if file is not None:
        plt.savefig(file)
        plt.close()
    return p


def plotmap(z, title='', vmin=20, vmax=100, cm=cmocean.cm.algae,
            p0=ccrs.PlateCarree(), p1=ccrs.Robinson(),
            variable='DOC',
            label='Estimated DOC [µmol kg⁻¹]',
            file=None, df=None, ax=None, cbar=True):
    """Plot global DOC on a map."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': p1})

    if cbar:
        cbargs = {'shrink': 1.0, 'label': label,
                  'orientation': 'horizontal', 'pad': 0.01}
    else:
        cbargs = None

    # ax.coastlines()
    ax.add_feature(cpf.OCEAN, zorder=0, facecolor=[0.8, 0.8, 0.8],
                   edgecolor='none')
    ax.add_feature(cpf.LAND, zorder=0, facecolor=[0.6, 0.6, 0.6],
                   edgecolor='none')
    ax.set_global()
    ax.gridlines(linewidth=1, color='gray', alpha=0.7,
                 linestyle='dotted')
    z.plot(ax=ax, transform=p0, cmap=cm,
           cbar_kwargs=cbargs, add_colorbar=cbar,
           vmin=vmin, vmax=vmax)
    if df is not None:
        ax.scatter(df['lon'], df['lat'], transform=p0,
                   c=df[variable], s=10, cmap=cm, vmin=vmin, vmax=vmax,
                   linewidths=0.2,
                   edgecolors='black')
    plt.title(title, fontsize=18)
#    ax.add_title(title, fontsize=18)
    if file is not None:
        plt.savefig(file)


def plotmap_points(df, variable='DOC', vmin=40, vmax=90,
                   cmap=cmocean.cm.matter,
                   p0=ccrs.PlateCarree(),
                   p1=ccrs.Robinson(),
                   title='In-situ DOC locations',
                   label='DOC [µmol kg⁻¹]',
                   cbar=True,
                   file=None):

    if cbar:
        cbargs = {'shrink': 1.0, 'label': label,
                  'orientation': 'horizontal', 'pad': 0.01}
    else:
        cbargs = None

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=p1)
    ax.add_feature(cpf.OCEAN, zorder=0)
    #ax.add_feature(cpf.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cpf.LAND, zorder=0)
    ax.set_extent([-180, 180, -85, 85], crs=ccrs.PlateCarree())
    #ax.gridlines()
    s = ax.scatter(df['lon'], df['lat'], 5, c=df[variable],
                   cmap=cmap,
                   transform=p0,
                   vmin=vmin, vmax=vmax,
                   )
    if cbar:
        fig.colorbar(s, ax=ax, **cbargs)
    plt.title(title)

    if file is not None:
        plt.savefig(file)
    return None
