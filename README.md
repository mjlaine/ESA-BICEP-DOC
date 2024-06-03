# ESA-BICEP project WP3.4 Dissolved Organic Carbon

This repository supplements the following article:

M. Laine, G. Kulk, B. Jönsson, S. Sathyendranath: A machine learning model-based satellite data record of dissolved organic carbon concentration in surface waters of the global pelagic region, *Frontiers in Marine Science*. 2024
[doi:10.3389/fmars.2024.1305050](https://doi.org/10.3389/fmars.2024.1305050)

It contains python code to work with Oceancolour (OC), other satellite-based global data sets and build a machine leaning model for dissolved organic carbon (DOC) data using in-situ data for DOC from Hansel (2021).

There is vscode devcontainer setup file included as well as Dockerfile and requirements.txt to create a coding environment similar that was used in this work. The training of the several machine learning models should work directly with the files provided here. For generating global predictions, several global data sets need to be available, descibed below in detail.

## Code


|                                  |                         |
|----------------------------------|-------------------------|
| [`data_utils.py`](data_utils.py) | Manage data sets        |
| [`fit_run.py`](fit_run.py)       | Train or optimize model |
| [`fit_maps.py`](fit_maps.py)     | Generate global maps    |


## Example

Estimate model number 3 defined in `fit_run.py` and generate monthly global datasets of DOC in folder `/tmp/DOC`:
```
./fit_run.py --model rf --no 4 --optimize
./fit_run.py --model rf --no 4 --mdir /tmp/results
./fit_maps.py --model 3 --mdir /tmp/results --outdir /tmp/DOC
```


## Data

In-situ data from Hansel (2021) aggregated monthly and combined with ESA Ocean Colour and other global data needed in the model training is in file `ocdocppdts_v9.h5`. 
In order to produce global maps, global satellite data must be available from the followig sources. The code assumes that some of the data have been preloaded and are stored local folder `~/DATA/ESA-BICEP/`.

- In-situ data for DOC from [https://doi.org/10.25921/s4f4-ye35]().
- Primary productions data from [https://doi.org/10.3390/rs12050826]()
- Oceancolour data from
[http://www.oceancolour.org/thredds/dodsC/CCI_ALL-v4.2-DAILY]()
- Salinity data (30 day average) from 
[http://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/sea_surface_salinity/data/v02.31/30days/]()
- ERA5 monthly SST [doi:10.24381/cds.f17050d7](https://doi.org/10.24381/cds.f17050d7).
- Distance to sea is constructed from ERA5 land sea map downloaded from
[https://confluence.ecmwf.int/pages/viewpage.action?pageId=140385202#ERA5Land:datadocumentation-parameterlistingParameterlistings]()
ERA5 land sea mask file `lsm_1279l4_0.1x0.1.grb_v4_unpack.nc` is transferred to distances with a R script:
```R
r <- raster::raster('lsm_1279l4_0.1x0.1.grb_v4_unpack.nc')
d <- raster::gridDistance(raster::rotate(raster::subs(r, data.frame(from=c(0),to=c(NA)), subsWithNA=FALSE)),1)
raster::writeRaster(d,'dts_1279l4_0.1x0.1.grb_v4_unpack.nc')
```

## References

- G. Kulk et al. Primary production, an index of climate change in the ocean: Satellite-based estimates over two decades. *Remote Sensing*, 12(5), 2020. [doi:10.3390/rs12050826](https://doi.org/10.3390/rs12050826)
- D. A. Hansell, et al. Compilation of dissolved organic matter (DOM) data obtained from the global ocean surveys from 1994 to 2020 (NCEI accession 0227166). Dataset, 2021. [doi:10.25921/s4f4-ye35](https://doi.org/10.25921/s4f4-ye35).
- D. Aurin, A. Mannino, and D. J. Lary. Remote sensing of cdom, cdom spectral slope, and dissolved organic carbon in the global ocean. *Applied Sciences*, 8(12), 2018. [doi:10.3390/app8122687](https://doi.org/10.3390/app8122687)
- T. DeVries and T. Weber. The export and fate of organic matter in the ocean: New constraints from combining satellite and oceanographic tracer observations. *Global Biogeochemical Cycles*, 31(3):535–555, 2017.
[doi:10.1002/2016GB005551](https://doi.org/10.1002/2016GB005551)
- S. Sathyendranath et al. An ocean-colour time series for use in climate studies: The experience of the Ocean-Colour Climate Change Initiative (OC-CCI). *Sensors*, 19(19), 2019. [doi:10.3390/s19194285](https://doi.org/10.3390/s19194285)
- A. G. Bonelli, et al. A new method to estimate the dissolved organic carbon concentration from remote sensing in the global open ocean. *Remote Sensing of Environment*, 281:113227, 2022. [doi:10.1016/j.rse.2022.113227](https://doi.org/10.1016/j.rse.2022.113227)


---
marko.laine@fmi.fi
