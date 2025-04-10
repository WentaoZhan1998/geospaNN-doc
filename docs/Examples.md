## Running examples

### Notes

Python packages time, pandas, seaborn, geopandas, and matplotlib are required to run the following experiments.

-   A simple pipeline illsutrating the basic features of the package is available [here](Example_utils/Example_utils.md).
-   Several simulation examples are available, to illustrate the usage of geospaNN for different statistical tasks:
    -   [Choice of neural network architectures.](Example_architecture/Example_architecture.md)
    -   [Application on spatial linear mixed model (SPLMM).](Example_linear/Example_linear.md)
    -   [Compare with the add-to-spatial-feature approaches.](Example_addcovariates_new/Example_addcovariates_new.md)
    -   [Application on time series data.](Example_time/Example_time.md)
    -   [Compare with GAM.](Example_GAM/Example_GAM.md)
    -   A real data experiment is shown [here](Example_realdata.md).

In the real data experiment, the PM2.5 data is collected from the [U.S. Environmental Protection Agency](https://www.epa.gov/outdoor-air-quality-data/download-daily-data) datasets for each state are collected and bound together to obtain 'pm25_2022.csv'. daily PM2.5 files are subsets of 'pm25_2022.csv' produced by 'realdata_preprocess.py'. One can skip the preprocessing and use the daily files directory.

The meteorological data is collected from the [National Centers for Environmental Prediction's (NCEP) North American Regional Reanalysis (NARR) product](https://psl.noaa.gov/data/gridded/data.narr.html). The '.nc' (netCDF) files should be downloaded from the website and saved in the root directory to run 'realdata_preprocess.py'. Otherwise, one may skip the preprocessing and use covariate files directly.
