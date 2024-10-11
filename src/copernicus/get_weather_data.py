"""
Module Name: get_weather_data.py
Description: Functions for downloading and processing selected weather data at given location
             (0.1° x 0.1° spatial resolution) for desired time periods, at hourly
             resolution, from Copernicus ERA5-Land dataset.

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

from pathlib import Path

import cdsapi
import netCDF4
import numpy as np
import pandas as pd
from netCDF4 import num2date

from copernicus import convert_weather_data as cwd
from copernicus import utils as ut


def construct_months_list(years, months=list(range(1, 13))):
    """
    Construct a list of year-month pairs.

    Parameters:
        years (list of int): List of years.
        months (list of int): List of months (default is 1 to 12, no other option if more than one year).

    Returns:
        list of tuples: List of (year, month_string) tuples representing all combinations
            of years and months in the input lists, plus months before first and after last month,
            with ranges of four months for each full year ('01-04', '05-08', '09-12').
    """
    months_list = []

    # Check years and months input lists for correctness and for gaps
    years = list(np.unique(years))
    months = list(np.unique(months))

    if months[0] < 1 or months[-1] > 12:
        raise ValueError(
            f"Month has invalid entries ({months}). Please provide only values between 1 and 12!"
        )

    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(
            f"Years list has gaps ({years}). Please provide consecutive years!"
        )

    if len(years) > 1:
        # Force complete months list if more than one year
        if months != list(range(1, 13)):
            print(
                f"All months required for multiple years! Replacing months list ({months}) by [1, 2, ... 12]."
            )
            months = list(range(1, 13))
    else:
        if months != list(range(months[0], months[-1] + 1)):
            raise ValueError(
                f"Month list has gaps ({months}). Please provide consecutive months!"
            )

    for year in years:
        # Add month before first month of first year (can be December of previous year)
        if year == years[0]:
            if months[0] > 1:
                months_list.append((year, f"{months[0] - 1:02}"))
            else:
                months_list.append((year - 1, "12"))

        # Add all months, combine always 4 months if full year
        if months == list(range(1, 13)):
            for month in range(1, 13, 4):
                months_list.append((year, f"{month:02}-{month+3:02}"))
        else:
            for month in months:
                months_list.append((year, f"{month:02}"))

        # Add month after last month of last year (can be January of next year)
        if year == years[-1]:
            if months[-1] < 12:
                months_list.append((year, f"{months[-1] + 1:02}"))
            else:
                months_list.append((year + 1, "01"))

    return months_list


def get_var_specs():
    """
    Retrieve dictionary of variable specifications.

    Create dictionary that provides specifications for potential download variables.
    Each variable is identified by its name and includes the following information:
        long_name: Long name of the variable.
        short_name: Abbreviation of the variable.
        unit_conversion: Unit conversion from source to target data.
        col_name_hourly: Column name for the variable in hourly data (source data units converted).
        col_name_daily: Column name for the variable in daily data (source data units converted).

    Returns:
        dict: Dictionary of variable specifications, where each key is a variable name,
              and each value is a dictionary of specifications.

    """
    # Additional vars for PET FAO calculation in commits before 2024-09-25.

    data_var_specs = {
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

    return data_var_specs


def construct_request(
    coordinates,
    year,
    month_str,
    variables,
    *,
    data_format="netcdf",
):
    """
    Construct data request.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        year (int): Year for the data request.
        month_str (str): Month(s) for the data request, can be one (e.g. '03')
            or range of months (e.g. '01-04').
        variables (list): List of variables to request.
        data_format (str): Data format (default is 'netcdf').

    Returns:
        dict: Dictionary representing the data request parameters.
    """
    # Fragments for daily requests in commits before 2023-11-08.

    month_formatted = ut.format_month_str(month_str)
    request = {
        "variable": variables,
        "product_type": "reanalysis",
        "grid": "0.1/0.1",
        "area": [  # works with exact location as lower and upper bounds of area
            coordinates["lat"],
            coordinates["lon"],
            coordinates["lat"],
            coordinates["lon"],
        ],
        "data_format": data_format,
        "day": [f"{d:02}" for d in range(1, 32)],  # works also for shorter months
        "year": str(year),
        "month": month_formatted,
        "time": [f"{i:02}:00" for i in range(24)],
        "download_format": "unarchived",
    }

    return request


def configure_data_request(
    data_var_specs,
    coordinates,
    months_list,
    *,
    data_format="netcdf",
):
    """
    Configure data requests.

    Parameters:
        data_var_specs (dict): Dictionary of variable specifications.
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        months_list (list): List of (year, month_str) pairs, 'month_str' can be one (e.g. '03')
            or range of months (e.g. '01-04').
        data_format (str): Data format (default is 'netcdf').

    Returns:
        list: List of data requests and corresponding file names.
    """
    # Fragments for daily requests in commits before 2023-11-08.

    data_requests = []
    var_names = list(data_var_specs.keys())
    long_names = [data_var_specs[key]["long_name"] for key in var_names]

    for year, month_str in months_list:
        request = construct_request(
            coordinates,
            year,
            month_str,
            long_names,  # requested variables
        )
        file_name = ut.construct_weather_data_file_name(
            coordinates,
            folder="weatherDataRaw",
            data_format=data_format,
            time_specifier=f"{year}_{month_str}",
            data_specifier="hourly",
        )
        data_requests.append((request, file_name))

    return data_requests


def download_weather_data(data_requests):
    """
    Download weather data from CDS API.

    Parameters:
        list: List of data requests and corresponding file names.
    """
    # Fragments for daily requests in commits before 2023-11-08.
    # Asynchronous option? https://docs.python.org/3/library/asyncio.html

    client = cdsapi.Client()

    for request, file_name in data_requests:
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        client.retrieve("reanalysis-era5-land", request, file_name)

    return client.url


def weather_data_to_txt_file(
    data_var_specs,
    coordinates,
    months_list,
    *,
    final_resolution="daily",
    target_folder="weatherDataPrepared",
    data_format="netcdf",
    data_source="https://cds.climate.copernicus.eu/api",
):
    """
    Process and write weather data to .txt files, considering time zone shifts (full hours only).

    Parameters:
        data_var_specs (dict): Dictionary of variable specifications.
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        months_list (list): List of (year, month_str) pairs, 'month_str' can be one (e.g. '03')
            or a range of months (e.g. '01-04')..
        final_resolution (str): Resolution for final .txt file ('hourly' or 'daily', default is 'daily').
        target_folder (str or Path): Target folder for .txt files (default is 'weatherDataPrepared').
        data_format (str): Data format (default is 'netcdf', no other option currently).
        data_source (str): URL used in data requests (default is 'https://cds.climate.copernicus.eu/api').
    """
    # Fragments for daily requests (toolbox version before CDS update) in commits before 2023-11-08.
    # Requests from dataset "reanalysis-era5-single-levels" in commits before 2024-01.
    # PET reading from CDS in commits before 2024-01.
    # PET calculation by FAO-56 Penman-Monteith equation in commits before 2024-10.

    col_names = [entries["col_name_hourly"] for entries in data_var_specs.values()]
    data_hourly = pd.DataFrame(columns=["Valid time", "Local time"] + col_names)
    var_names = list(data_var_specs.keys())
    data_query_protocol = []
    years_to_check = np.unique([item[0] for item in months_list])
    tz_offset = ut.get_time_zone(
        coordinates, return_as_offset=True, years=years_to_check
    )
    tz_label = ut.format_offset(tz_offset, add_utc=False)

    for year, month_str in months_list:
        file_name = ut.construct_weather_data_file_name(
            coordinates,
            folder="weatherDataRaw",
            data_format=data_format,
            time_specifier=f"{year}_{month_str}",
            data_specifier="hourly",
        )

        # Open netCDF4 file and extract variables
        data_raw = netCDF4.Dataset(file_name)
        history = getattr(data_raw, "history")
        time_stamp, extra_info = history.split(" ", 1)
        data_query_protocol.append(
            [year, month_str, data_source, time_stamp + "+00:00", extra_info]
        )

        # Init data frame with time data
        valid_time = data_raw.variables["valid_time"]
        valid_time = num2date(valid_time[:], valid_time.units)  # cds times are UTC
        local_time = valid_time + tz_offset  # local time, but w/o daylight saving time
        data_temp = pd.DataFrame(
            {
                "Valid time": [
                    t.isoformat(timespec="minutes") + "+00:00" for t in valid_time
                ],
                "Local time": [
                    t.isoformat(timespec="minutes") + tz_label for t in local_time
                ],
            }
        )

        # Collect and convert hourly data from data_raw
        for var_name in var_names:
            data_var = cwd.convert_units(
                data_raw.variables[data_var_specs[var_name]["short_name"]][:].flatten(),
                data_var_specs[var_name]["unit_conversion"],
            )
            data_temp[data_var_specs[var_name]["col_name_hourly"]] = data_var

        if not data_hourly.empty:
            data_hourly = pd.concat([data_hourly, data_temp], ignore_index=True)
        else:
            data_hourly = data_temp

    # Remove all data before 00:00 local time at first day (no time gaps in data was checked in construct_months_list)
    tz_offset_hours = int(tz_offset.total_seconds() / 3600)
    len_month_start = ut.get_days_in_month(months_list[0][0], int(months_list[0][1]))
    data_hourly = data_hourly.iloc[len_month_start * 24 - tz_offset_hours :]

    # Remove all data after 00:00 local time at last day + 1 (no time gaps in data was checked in construct_months_list)
    len_month_end = ut.get_days_in_month(months_list[-1][0], int(months_list[-1][1]))
    data_hourly = data_hourly.iloc[: -len_month_end * 24 + 1 - tz_offset_hours]

    # Save DataFrame to .txt file, create data directory if missing
    time_range = (
        f"{data_hourly["Local time"].values[0].split("T")[0]}"
        f"_{data_hourly["Local time"].values[-2].split("T")[0]}"
    )
    file_name = ut.construct_weather_data_file_name(
        coordinates,
        folder=target_folder,
        data_format="txt",
        time_specifier=time_range,
        data_specifier="hourly",
    )
    data_hourly.to_csv(
        file_name, sep="\t", index=False, float_format="%.6f", na_rep="nan"
    )
    print("Text file with hourly resolution prepared.")

    # Save data query protocal if existing
    if data_query_protocol:
        file_name = file_name.with_name(
            file_name.stem + "__data_query_protocol" + file_name.suffix
        )
        ut.list_to_file(
            data_query_protocol,
            ["year", "month", "data_source", "time_stamp", "info"],
            file_name,
        )

    # Convert hourly to daily data if needed (e.g. for grassland model)
    if final_resolution == "daily":
        cwd.hourly_to_daily(
            data_hourly,
            data_var_specs,
            coordinates,
            tz_offset_hours=tz_offset_hours,
            target_folder=target_folder,
        )
