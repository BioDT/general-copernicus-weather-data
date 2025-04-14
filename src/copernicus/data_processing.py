"""
Module Name: data_processing.py
Description: Function for obtaining selected weather data at given location(s) at hourly resolution,
   from Copernicus ERA5-Land dataset (interpolated from 0.1° x 0.1° or 0.25° x 0.25° spatial resolution)
   for desired time periods.
   Optional conversion to and calculation of specific target variables at daily resolution.

Developed in the BioDT project by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ)
and Tuomas Rossi (CSC).

Copyright (C) 2024
- Helmholtz Centre for Environmental Research GmbH - UFZ, Germany
- CSC - IT Center for Science Ltd., Finland

Licensed under the EUPL, Version 1.2 or - as soon they will be approved
by the European Commission - subsequent versions of the EUPL (the "Licence").
You may not use this work except in compliance with the Licence.

You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

This project has received funding from the European Union's Horizon Europe Research and Innovation
Programme under grant agreement No 101057437 (BioDT project, https://doi.org/10.3030/101057437).
The authors acknowledge the EuroHPC Joint Undertaking and CSC - IT Center for Science Ltd., Finland
for awarding this project access to the EuroHPC supercomputer LUMI, hosted by CSC - IT Center for
Science Ltd., Finland and the LUMI consortium through a EuroHPC Development Access call.

Data source:
    ERA5-Land hourly data from 1950 to present:
    - Muñoz Sabater, J. (2019):
      ERA5-Land hourly data from 1950 to present.
      Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.e2161bac
    - Website: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
    - Documentation: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
    - Full Licence to Use Copernicus Products: https://apps.ecmwf.int/datasets/licences/copernicus/
    - Access via Climate Data Store (CDS) Application Program Interface (API):
        - URL: https://cds.climate.copernicus.eu/api
        - Documentation: https://cds.climate.copernicus.eu/how-to-api
        - Python package 'cdsapi': https://pypi.org/project/cdsapi/
        - Access to the CDS API requires:
            - An ECMWF account (https://www.ecmwf.int/)
            - Agreement to the Terms of Use of the "ERA5-Land hourly data from 1950 to present" data set
            - A CDS API personal access token (usually put in a '$HOME/.cdsapirc' file)
            - See detailed instructions at: https://cds.climate.copernicus.eu/how-to-api
"""

from types import MappingProxyType

from copernicus import get_weather_data as gwd
from copernicus.logger_config import logger

# Define data variable specifications for download variables, including:
#     long_name (str): Long name of the variable.
#     short_name (str): Abbreviation of the variable.
#     unit_conversion (str): Unit conversion from source to target data.
#     col_name_hourly (str): Column name for the variable in hourly data (source data units converted).
#     col_name_daily (str): Column name for the variable in daily data (source data units converted).
#
# Additional vars for PET FAO calculation in commits before 2024-09-25.
DATA_VAR_SPECS = MappingProxyType(
    {
        "precipitation": {
            "long_name": "total_precipitation",
            "short_name": "tp",
            "unit_conversion": "_to_Milli",
            "col_name_hourly": "Precipitation[mm] (acc.)",
            "col_name_daily": "Precipitation[mm]",
        },
        "temperature": {
            "long_name": "2m_temperature",
            "short_name": "t2m",
            "unit_conversion": "Kelvin_to_Celsius",
            "col_name_hourly": "Temperature[degC]",
            "col_name_daily": "Temperature[degC]",
        },
        "solar_radiation_down": {
            "long_name": "surface_solar_radiation_downwards",
            "short_name": "ssrd",
            "unit_conversion": 0,
            "col_name_hourly": "SSRD[Jm-2] (acc.)",
            "col_name_daily": "PAR[µmolm-2s-1]",
        },
    }
)


def data_processing(
    years,
    months,
    coordinates_list,
    *,
    data_format="grib",
    download_whole_area=True,
    grid_resolution=0.1,
    final_time_resolution="daily",
    target_folder=None,
):
    """
    Download data from CDS API. Convert to .txt files.

    Parameters:
        years (list of int): Years list.
        months (list of int): Months list.
        coordinates_list (list of dict): List of dictionaries with 'lat' and 'lon' keys
            ({'lat': float, 'lon': float}).
        data_format (str): Data format ('grib' or 'netcdf', default is 'grib').
        download_whole_area (bool): Download data for whole area covering all locations from the coordinates list
          (default is True). If False, data will be downloaded for each location separately.
        grid_resolution (float): Grid resolution (0.1 or 0.25, default is 0.1).
        final_time_resolution (str): Resolution for final text file ('hourly' or 'daily', default is 'daily').
        target_folder (str or Path): Target folder for .txt files (default is 'weatherDataPrepared').

    Returns:
        None
    """
    # Prepare requests
    months_list = gwd.construct_months_list(years, months)

    if download_whole_area:
        # Configure requests to download whole area (for each time period)
        area_coordinates = gwd.get_area_coordinates(
            coordinates_list, resolution=grid_resolution
        )
        logger.info("Requesting weather data for area ...")
        logger.info(
            f"latitude: {area_coordinates['lat_start']} - {area_coordinates['lat_end']}, "
            f"longitude: {area_coordinates['lon_start']} - {area_coordinates['lon_end']}",
        )
        coordinate_digits = 1
        data_requests = gwd.configure_data_requests(
            DATA_VAR_SPECS,
            area_coordinates,
            months_list,
            data_format=data_format,
            coordinate_digits=coordinate_digits,
        )
    else:
        # Configure requests to separately download data for each location (for each time period)
        area_coordinates = None
        logger.info("Requesting weather data for single locations ...")
        coordinate_digits = 6
        data_requests = []

        for coordinates in coordinates_list:
            logger.info(
                f"latitude: {coordinates['lat']}, longitude: {coordinates['lon']}"
            )
            location_as_area = gwd.get_area_coordinates(
                [coordinates], resolution=0, map_to_grid=False
            )
            data_requests.extend(
                gwd.configure_data_requests(
                    DATA_VAR_SPECS, location_as_area, months_list
                )
            )

    # Download raw data
    cds_api_url = gwd.download_weather_data(data_requests)

    # Process raw data to final files
    for coordinates in coordinates_list:
        gwd.weather_data_to_txt_file(
            DATA_VAR_SPECS,
            coordinates,
            months_list,
            area_coordinates=area_coordinates,
            coordinate_digits=coordinate_digits,
            final_time_resolution=final_time_resolution,
            target_folder=target_folder,
            data_format=data_format,
            data_source=cds_api_url,
        )
