## BioDT - Copernicus Weather Data 
Building block for obtaining selected weather data at given location(s) at hourly resolution, 
   from Copernicus ERA5-Land dataset (interpolated from 0.1° x 0.1° or 0.25° x 0.25° spatial resolution)
   for desired time periods. 
   Optional conversion to and calculation of specific target variables at daily resolution. 

<a href="https://doi.org/10.5281/zenodo.15730814"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15730814.svg" alt="DOI"></a>

## Installation
The current development version can be installed as:

    pip install git+https://github.com/BioDT/general-copernicus-weather-data.git@main

## Usage
Preparatory step: setup the CDS API personal access token as described [here](https://cds.climate.copernicus.eu/how-to-api).

Request data for a span of years at a few locations:

```python
from copernicus import get_weather_data

years = list(range(2022, 2025))
coordinates = [{"lat": 51.123456, "lon": 11.987654}, {"lat": 51.234, "lon": 11.876}, {"lat": 51.33, "lon": 11.66}]
get_weather_data(years, coordinates)
```

Full function signature: 

`get_weather_data(years, coordinates_list, *,
    months=list(range(1, 13)),
    data_format='grib',
    download_area=True,
    check_larger_areas=True,
    grid_resolution=0.1,
    final_time_resolution='daily',
    target_folder=None)`

Parameters:
- years (list of int): Years list.
- coordinates_list (list of dict): List of dictionaries with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).

- months (list of int): Months list (1-12, default is [1, 2, ... 12]).
- data_format (str): Data format ('grib' or 'netcdf', default is 'grib').
- download_area (bool): Download raw weather data for whole area covering all locations from the coordinates list at grid resolution (default is True). If False, data will be downloaded for each location separately (only available for 'netcdf' format).
- check_larger_areas (bool): If existing weather data for the area from previous downloads is not available on the OPeNDAP server, look for larger available areas containing the area (up to the search limit) for which (some) weather data is already available and use this area (default is True).
- grid_resolution (float): Grid resolution (0.1 or 0.25, default is 0.1).
- final_time_resolution (str): Resolution for final text file ('hourly' or 'daily', default is 'daily').
- target_folder (str or Path): Target folder for .txt files (default is 'weatherDataPrepared').

## Developers
Developed in the BioDT project (until 2025-05) by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ) 
and Tuomas Rossi (CSC).

Further developed (from 2025-06) by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ).

## Copyright
Copyright (C) 2024
- Helmholtz Centre for Environmental Research GmbH - UFZ, Germany
- CSC - IT Center for Science Ltd., Finland

Licensed under the EUPL, Version 1.2 or - as soon they will be approved
by the European Commission - subsequent versions of the EUPL (the "Licence").
You may not use this work except in compliance with the Licence.

You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

## Funding
The BioDT project has received funding from the European Union's Horizon Europe Research and Innovation
Programme under grant agreement No 101057437 (BioDT project, https://doi.org/10.3030/101057437).
The authors acknowledge the EuroHPC Joint Undertaking and CSC - IT Center for Science Ltd., Finland
for awarding this project access to the EuroHPC supercomputer LUMI, hosted by CSC - IT Center for
Science Ltd., Finland and the LUMI consortium through a EuroHPC Development Access call.

## Data source
ERA5-Land hourly data from 1950 to present:
- Muñoz Sabater, J. (2019):
ERA5-Land hourly data from 1950 to present. 
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.e2161bac
- Website: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- Online documentation: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
- Full licence to use Copernicus products: https://apps.ecmwf.int/datasets/licences/copernicus/
- Access via Climate Data Store (CDS) Application Program Interface (API):
   - URL: https://cds.climate.copernicus.eu/api
   - Documentation: https://cds.climate.copernicus.eu/how-to-api
   - Python package 'cdsapi': https://pypi.org/project/cdsapi/
   - Access to the CDS API requires:
      - An ECMWF account (https://www.ecmwf.int/)
      - Agreement to the Terms of Use of the "ERA5-Land hourly data from 1950 to present" data set
      - A CDS API personal access token (usually put in a '$HOME/.cdsapirc' file)
      - See detailed instructions at: https://cds.climate.copernicus.eu/how-to-api
