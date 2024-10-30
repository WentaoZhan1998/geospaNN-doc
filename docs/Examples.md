## Running examples

### Notes
Python packages time, pandas, seaborn, geopandas, and matplotlib are required to run the following experiments.


1. [A simulation experiment with a common spatial setting is shown here.](Example_simulation.md)
2. For the linear regression case, [a performance comparison with the R package BRISC is shown here.](Example_realdata.md)
3. [A real data experiment is shown here.](Example_realdata.md)

In the real data experiment, the PM2.5 data is collected from the [U.S. Environmental Protection Agency](https://www.epa.gov/outdoor-air-quality-data/download-daily-data) datasets for each state are collected and bound together to obtain 'pm25_2022.csv'. daily PM2.5 files are subsets of 'pm25_2022.csv' produced by 'realdata_preprocess.py'. One can skip the preprocessing and use the daily files directory.

The meteorological data is collected from the [National Centers for Environmental Prediction’s (NCEP) North American Regional Reanalysis (NARR) product](https://psl.noaa.gov/data/gridded/data.narr.html). The '.nc' (netCDF) files should be downloaded from the website and saved in the root directory to run 'realdata_preprocess.py'. Otherwise, one may skip the preprocessing and use covariate files directly.