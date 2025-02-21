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

import os
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import num2date
from scipy.interpolate import griddata

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
                months_list.append((year, f"{month:02}-{month + 3:02}"))
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


def get_area_coordinates(coordinates_list, *, resolution=0.1, map_to_grid=True):
    """
    Get area coordinates based on a list of coordinates.

    Parameters:
        coordinates_list (list): List of coordinates dictionaries with 'lat' and 'lon' keys.
        resolution (float): Grid resolution (default is 0.1 [degrees], e.g. ERA5 grid resolution used).
        map_to_grid (bool): Map area coordinates to the nearest grid points (default is True,
            otherwise use margin around the coordinates).

    Returns:
        dict: Dictionary with 'lat_start', 'lat_end', 'lon_start' and 'lon_end' keys.

    Raises:
        ValueError: If coordinates are not correctly defined.
        ValueError: If resolution is not 0.1 or 0.25.
    """
    if not isinstance(resolution, (int, float)) or resolution < 0:
        raise ValueError("Resolution must be a number greater than or equal to 0!")

    if map_to_grid and resolution not in [0.1, 0.25]:
        raise ValueError("Grid resolution must be 0.1 or 0.25 degrees!")

    # Check if each entry in the list has 'lat' and 'lon' keys
    if not all(
        all(key in coordinates for key in ["lat", "lon"])
        for coordinates in coordinates_list
    ):
        raise ValueError(
            "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
        )

    lat_list = [coordinates["lat"] for coordinates in coordinates_list]
    lon_list = [coordinates["lon"] for coordinates in coordinates_list]

    def get_max_decimal_digits(list_of_floats):
        """Helper function to get maximum number of decimal digits in a list of floats."""
        max_decimal_digits = 0

        for f in list_of_floats:
            str_f = str(f)
            decimal_digits = len(str_f.split(".")[1]) if "." in str_f else 0
            max_decimal_digits = max(max_decimal_digits, decimal_digits)

        return max_decimal_digits

    digits_required = get_max_decimal_digits(lat_list + lon_list + [resolution])

    if map_to_grid:
        # Area around the coordinates, extended to the nearest grid points
        round_factor = round(1 / resolution)
        area_coordinates = {
            "lat_start": round(
                np.floor(min(lat_list) * round_factor) / round_factor, digits_required
            ),
            "lat_end": round(
                np.ceil(max(lat_list) * round_factor) / round_factor, digits_required
            ),
            "lon_start": round(
                np.floor(min(lon_list) * round_factor) / round_factor, digits_required
            ),
            "lon_end": round(
                np.ceil(max(lon_list) * round_factor) / round_factor, digits_required
            ),
        }
    else:
        # Area with margin around the min and max coordinate values, according to the resolution
        area_coordinates = {
            "lat_start": round((min(lat_list)) - resolution, digits_required),
            "lat_end": round(max(lat_list) + resolution, digits_required),
            "lon_start": round(min(lon_list) - resolution, digits_required),
            "lon_end": round(max(lon_list) + resolution, digits_required),
        }

    return area_coordinates


def construct_request(
    area_coordinates,
    year,
    month_str,
    variables,
    *,
    data_format="grib",
):
    """
    Construct data request.

    Parameters:
        area_coordinates (dict): Dictionary with 'lat_start', 'lat_end', 'lon_start', and 'lon_end' keys
          ({'lat_start': float, 'lat_end': float, 'lon_start': float, 'lon_end': float}).
        year (int): Year for the data request.
        month_str (str): Month(s) for the data request, can be one (e.g. '03')
            or range of months (e.g. '01-04').
        variables (list): List of variables to request.
        data_format (str): Data format ('grib' (default), 'netcdf' (netCDF4), or 'netcdf_legacy' to get
            output in the format of the CDS API before their update in 24-09).

    Returns:
        dict: Dictionary representing the data request parameters.
    """
    # Fragments for daily requests in commits before 2023-11-08.

    # Notes on GRIB versus NetCDF data format:
    # - download/queuing times seem similar
    # - max number of items per request is the same
    # - raw data files are larger for GRIB
    # - GRIB is recommended option by CDS,
    #   cf. https://confluence.ecmwf.int/display/CKB/GRIB+to+netCDF+conversion+on+new+CDS+and+ADS+systems
    # - GRIB data use a reduced Gaussian grid for quasi-uniform spacing around the globe,
    #   cf. https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference
    # - NetCDF area data already come mapped to a regular grid by interpolation,
    #   this means we then interpolate again for single points (double interpolation)
    # - NetCDF single point data get interpolated directly from the reduced Gaussian grid,
    #   but this needs separate requests for each point (not efficient for many clustered points)

    month_formatted = ut.format_month_str(month_str)
    request = {
        "variable": variables,
        "product_type": "reanalysis",
        "grid": "0.1/0.1",
        "area": [  # works also with exact location as lower and upper bounds of area
            area_coordinates["lat_start"],
            area_coordinates["lon_start"],
            area_coordinates["lat_end"],
            area_coordinates["lon_end"],
        ],
        "data_format": data_format,
        "day": [f"{d:02}" for d in range(1, 32)],  # works also for shorter months
        "year": str(year),
        "month": month_formatted,
        "time": [f"{index:02}:00" for index in range(24)],
        "download_format": "unarchived",
    }

    return request


def configure_data_requests(
    data_var_specs,
    area_coordinates,
    months_list,
    *,
    data_format="grib",
    coordinate_digits=6,
):
    """
    Configure data requests.

    Parameters:
        data_var_specs (dict): Dictionary of variable specifications.
        area_coordinates (dict): Dictionary with 'lat_start', 'lat_end', 'lon_start', and 'lon_end' keys
          ({'lat_start': float, 'lat_end': float, 'lon_start': float, 'lon_end': float}).
        months_list (list): List of (year, month_str) pairs, 'month_str' can be one (e.g. '03')
            or range of months (e.g. '01-04').
        data_format (str): Data format ('grib' or 'netcdf', default is 'grib').
        coordinate_digits (int): Number of digits for coordinates in file name (default is 6).

    Returns:
        list: List of data requests and corresponding file names.
    """
    # Fragments for daily requests in commits before 2023-11-08.

    data_requests = []
    var_names = list(data_var_specs.keys())
    long_names = [data_var_specs[key]["long_name"] for key in var_names]

    for year, month_str in months_list:
        request = construct_request(
            area_coordinates,
            year,
            month_str,
            long_names,  # requested variables
            data_format=data_format,
        )
        file_name = ut.construct_weather_data_file_name(
            area_coordinates,
            folder="weatherDataRaw",
            data_format=data_format,
            time_specifier=f"{year}_{month_str}",
            data_specifier="hourly",
            precision=coordinate_digits,
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
    area_coordinates=None,
    coordinate_digits=6,
    final_time_resolution="daily",
    target_folder="weatherDataPrepared",
    data_format="grib",
    data_source="https://cds.climate.copernicus.eu/api",
):
    """
    Process and write weather data to .txt files, considering time zone shifts (full hours only).

    Parameters:
        data_var_specs (dict): Dictionary of variable specifications.
        coordinates (dict): Location as dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        months_list (list): List of (year, month_str) pairs, 'month_str' can be one (e.g. '03')
            or a range of months (e.g. '01-04').
        area_coordinates (dict): Area of raw weather data with 'lat_start', 'lat_end', 'lon_start', and
            'lon_end' keys ({'lat_start': float, 'lat_end': float, 'lon_start': float, 'lon_end': float}).
            If None, the area is defined by the location coordinates. (default is None).
        grid_resolution (float): Spatial grid resolution for area data (default is 0.1).
        final_time_resolution (str): Temporal resolution for final .txt file ('hourly' or 'daily', default is 'daily').
        target_folder (str or Path): Target folder for .txt files (default is 'weatherDataPrepared').
        data_format (str): Data format ('grib' or 'netcdf', default is 'grib').
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
    single_point_data = False

    if area_coordinates is None:
        # Single point data (no area coordinates provided)
        single_point_data = True
        area_coordinates = get_area_coordinates(
            [coordinates], resolution=0, map_to_grid=False
        )

    if data_format == "netcdf":
        for year, month_str in months_list:
            file_name = ut.construct_weather_data_file_name(
                area_coordinates,
                folder="weatherDataRaw",
                data_format=data_format,
                time_specifier=f"{year}_{month_str}",
                data_specifier="hourly",
                precision=coordinate_digits,
            )

            # Open netCDF4 file and extract variables
            with netCDF4.Dataset(file_name) as data_raw:
                history = getattr(data_raw, "history")
                time_stamp, extra_info = history.split(" ", 1)
                data_query_protocol.append(
                    [year, month_str, data_source, time_stamp + "+00:00", extra_info]
                )

                # Init data frame with time data
                valid_time = data_raw.variables["valid_time"]
                valid_time = num2date(
                    valid_time[:], valid_time.units
                )  # CDS times are UTC
                local_time = (
                    valid_time + tz_offset
                )  # local time, but w/o daylight saving time
                data_temp = pd.DataFrame(
                    {
                        "Valid time": [
                            t.isoformat(timespec="minutes") + "+00:00"
                            for t in valid_time
                        ],
                        "Local time": [
                            t.isoformat(timespec="minutes") + tz_label
                            for t in local_time
                        ],
                    }
                )

                if not single_point_data:
                    # Meshgrid of latitude and longitude
                    # (works for increasing and decreasing latitudes and longitudes,
                    # but should be obtained from data file to get correct order)
                    lon_values = data_raw.variables["longitude"][:]
                    lat_values = data_raw.variables["latitude"][:]
                    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
                    grid_points = (lat_grid.flatten(), lon_grid.flatten())

                # Collect and convert hourly data from data_raw
                for var_name in var_names:
                    # Extract the data for the variable of interest
                    data_var = data_raw.variables[
                        data_var_specs[var_name]["short_name"]
                    ][:]

                    if single_point_data:
                        data_values = data_var.flatten()
                    else:
                        # Perform bilinear interpolation for each time step
                        data_values = []

                        for time_step in range(data_var.shape[0]):
                            data_slice = data_var[time_step, :, :]
                            interpolated_value = griddata(
                                grid_points,  # points
                                data_slice.flatten(),  # values
                                (
                                    coordinates["lat"],
                                    coordinates["lon"],
                                ),  # point of interest
                                method="linear",
                            )
                            data_values.append(interpolated_value)

                    # Convert values to numpy array, and to target units
                    data_values = np.array(data_values)
                    converted_values = cwd.convert_units(
                        data_values, data_var_specs[var_name]["unit_conversion"]
                    )
                    data_temp[data_var_specs[var_name]["col_name_hourly"]] = (
                        converted_values
                    )

            if not data_hourly.empty:
                data_hourly = pd.concat([data_hourly, data_temp], ignore_index=True)
            else:
                data_hourly = data_temp

    elif data_format == "grib":
        # Note: GRIB data come with last day of previous month as first entry,
        #       but contain NANs for:
        #       - all hourly values of first day except last value
        #       - last value of last day
        #       --> omitting these values for consistent concatenation of time series
        # Option: consider xr.merge or xr.open_mfdataset to combine several files to one time series?
        start_hours_skipped = 23
        end_hours_skipped = 1

        for year, month_str in months_list:
            file_name = ut.construct_weather_data_file_name(
                area_coordinates,
                folder="weatherDataRaw",
                data_format=data_format,
                time_specifier=f"{year}_{month_str}",
                data_specifier="hourly",
                precision=coordinate_digits,
            )

            # Get file creation time (or modification time if creation time is not available)
            time_stamp = (
                datetime.fromtimestamp(os.path.getctime(file_name))
                .astimezone(timezone.utc)  # convert to UTC
                .isoformat(timespec="seconds")
            )

            # Open GRIB file and extract variables
            with xr.load_dataset(file_name, engine="cfgrib") as data_raw:
                # Get extra info from history attribute
                extra_info = getattr(data_raw, "history").split(" ", 1)[1]
                data_query_protocol.append(
                    [year, month_str, data_source, time_stamp, extra_info]
                )

                # Init data frame with time data flattened to one-dimensional array
                valid_time = data_raw.variables["valid_time"].values.flatten()[
                    start_hours_skipped:-end_hours_skipped
                ]
                data_temp = pd.DataFrame(
                    {
                        "Valid time": [
                            pd.to_datetime(t).isoformat(timespec="minutes") + "+00:00"
                            for t in valid_time  # CDS times are UTC
                        ],
                        "Local time": [
                            (pd.to_datetime(t) + tz_offset).isoformat(
                                timespec="minutes"
                            )
                            + tz_label  # local time, but w/o daylight saving time
                            for t in valid_time
                        ],
                    }
                )

                # Extract data for the variable of interest, and the coordinates
                for var_name in var_names:
                    data_var = getattr(data_raw, data_var_specs[var_name]["short_name"])
                    data_values = data_var.interp(
                        latitude=coordinates["lat"], longitude=coordinates["lon"]
                    ).values.flatten()  # default linear interpolation, enough points for cubic is not guaranteed

                    # Use only non-NaN values, must be equal to range used for valid time
                    nan_indexes = np.isnan(data_values)

                    if any(nan_indexes[start_hours_skipped:-end_hours_skipped]):
                        raise ValueError(
                            "Data values contain NaNs at time points for which values were expected!"
                        )

                    if not (
                        all(nan_indexes[:start_hours_skipped])
                        and all(nan_indexes[-end_hours_skipped:])
                    ):
                        raise ValueError(
                            "Data values contain values at time points for which NaNs were expected!"
                        )

                    # Convert values to numpy array, and to target units
                    data_values = np.array(data_values[~nan_indexes])
                    converted_values = cwd.convert_units(
                        data_values, data_var_specs[var_name]["unit_conversion"]
                    )
                    data_temp[data_var_specs[var_name]["col_name_hourly"]] = (
                        converted_values
                    )

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
        f"{data_hourly['Local time'].values[0].split('T')[0]}"
        f"_{data_hourly['Local time'].values[-2].split('T')[0]}"
    )
    file_name = ut.construct_weather_data_file_name(
        coordinates,
        folder=target_folder,
        data_format="txt",
        time_specifier=time_range,
        data_specifier="hourly__" + data_format,
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
            file_name,
            column_names=["year", "month", "data_source", "time_stamp", "info"],
        )

    # Convert hourly to daily data if needed (e.g. for grassland model)
    if final_time_resolution == "daily":
        cwd.hourly_to_daily(
            data_hourly,
            data_var_specs,
            coordinates,
            tz_offset_hours=tz_offset_hours,
            target_folder=target_folder,
        )
