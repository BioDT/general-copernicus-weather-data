"""
Module Name: get_weather_data.py
Description: Functions for downloading and processing selected weather data.

Copyright (C) 2024
- Thomas Banitz, Franziska Taubert, Helmholtz Centre for Environmental Research GmbH - UFZ, Leipzig, Germany
- Tuomas Rossi, CSC – IT Center for Science Ltd., Espoo, Finland

Licensed under the EUPL, Version 1.2 or - as soon they will be approved
by the European Commission - subsequent versions of the EUPL (the "Licence").
You may not use this work except in compliance with the Licence.

You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

This project has received funding from the European Union's Horizon Europe Research and Innovation
Programme under grant agreement No 101057437 (BioDT project, https://doi.org/10.3030/101057437).
The authors acknowledge the EuroHPC Joint Undertaking and CSC – IT Center for Science Ltd., Finland
for awarding this project access to the EuroHPC supercomputer LUMI, hosted by CSC – IT Center for
Science Ltd., Finland and the LUMI consortium through a EuroHPC Development Access call.
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
        months (list of int): List of months (default is 1 to 12).

    Returns:
        list of tuples: A list of (year, month) tuples representing all combinations
        of years and months in the input lists.
    """
    months_list = []

    # Check if months list is correct and has no gaps
    months = list(np.unique(months))

    if months[0] < 1 or months[-1] > 12:
        raise ValueError(
            f"Month has invalid entries ({months}). Please provide only values between 1 and 12!"
        )

    if months != list(range(months[0], months[-1] + 1)):
        raise ValueError(
            f"Month list has gaps ({months}). Please provide consecutive months!"
        )

    for year in years:
        # Add all months
        for month in months:
            months_list.append((year, month))

        # Add month before first month (can be December of previous year, can create duplicates)
        if months[0] > 1:
            months_list.append((year, months[0] - 1))
        else:
            months_list.append((year - 1, 12))

        # Add month after last month (can be January of next year, can create duplicates)
        if months[-1] < 12:
            months_list.append((year, months[-1] + 1))
        else:
            months_list.append((year + 1, 1))

    # Remove duplicates and sort
    months_list = list(set([y_m for y_m in months_list]))
    months_list.sort(key=lambda x: (x[0], x[1]))

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
        daily_stat: Statistic for downloading daily data (option currently not available).

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
            "daily_stat": "daily_maximum",
        },
        "temperature": {
            "long_name": "2m_temperature",
            "short_name": "t2m",
            "unit_conversion": "Kelvin_to_Celsius",
            "col_name_hourly": "Temperature[degC]",
            "col_name_daily": "Temperature[degC]",
            "daily_stat": "daily_mean",
        },
        "solar_radiation_down": {
            "long_name": "surface_solar_radiation_downwards",
            "short_name": "ssrd",
            "unit_conversion": 0,
            "col_name_hourly": "SSRD[Jm-2] (acc.)",
            "col_name_daily": "PAR[µmolm-2s-1]",  # was: PPFD_down
            "daily_stat": "daily_maximum",
        },
    }

    return data_var_specs


def construct_weather_data_file_name(
    folder,
    data_format,
    coordinates,
    year,
    month,
    data_specifier,
):
    """
    Construct data file name.

    Parameters:
        folder (str or Path): Folder where the data file will be stored.
        data_format (str): Data format ('netcdf' or 'grib' or 'txt').
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        year (int): Year for the data file.
        month (int): Month for the data file.
        data_specifier (str): Data specifier (e.g. 'hourly', 'weather').

    Returns:
        Path: Constructed data file name as a Path object.
    """
    # Get folder with path appropriate for different operating systems
    folder = Path(folder)

    if "lat" in coordinates and "lon" in coordinates:
        formatted_lat = f"lat{coordinates['lat']:.6f}"
        formatted_lon = f"lon{coordinates['lon']:.6f}"
        file_suffix = ut.get_file_suffix(data_format)
        formatted_year = str(year)
        formatted_month = str(month).zfill(2)
        file_name = (
            folder
            / f"{formatted_lat}_{formatted_lon}_{formatted_year}_{formatted_month}__{data_specifier}{file_suffix}"
        )
    else:
        raise ValueError(
            "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
        )

    return file_name


def construct_request(
    coordinates,
    year,
    month,
    variables,
    *,
    data_format="netcdf",
):
    """
    Construct data request.

    Parameters:
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon'.
        year (int): Year for the data request.
        month (int): Month for the data request.
        variables (list): List of variables to request.
        data_format (str): Data format (default if 'netcdf').

    Returns:
        dict: Dictionary representing the data request parameters.
    """
    # Fragments for daily requests in commits before 2023-11-08.

    request = {
        "variable": variables,
        "product_type": "reanalysis",
        "grid": "0.1/0.1",
        "area": [  # Works with exact location as lower and upper bounds of area
            coordinates["lat"],
            coordinates["lon"],
            coordinates["lat"],
            coordinates["lon"],
        ],
        "data_format": data_format,
        "day": ut.generate_day_values(year, month),
        "year": str(year),
        "month": str(month),
        "time": [f"{i:02}:00" for i in range(24)],
        "download_format": "unarchived",
    }  # not working: "time_zone": "UTC+01:00",

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
        months_list (list): List of (year, month) pairs.
        data_format (str): Data format (default is 'netcdf').

    Returns:
        list: List of data requests and corresponding file names.
    """
    # Fragments for daily requests in commits before 2023-11-08.

    data_requests = []
    var_names = list(data_var_specs.keys())
    long_names = [data_var_specs[key]["long_name"] for key in var_names]

    for year, month in months_list:
        request = construct_request(
            coordinates,
            year,
            month,
            long_names,  # requested variables
        )
        file_name = construct_weather_data_file_name(
            "weatherDataRaw",
            data_format,
            coordinates,
            year,
            month,
            "hourly",
        )
        data_requests.append((request, file_name))

    return data_requests


def download_weather_data(data_requests):
    """
    Download weather data from CDS API.

    Parameters:
        data_requests (list): List of data requests and corresponding file names.
    """
    # Fragments for daily requests in commits before 2023-11-08.
    # Asynchronous option? https://docs.python.org/3/library/asyncio.html

    client = cdsapi.Client()

    for request, file_name in data_requests:
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        # client.retrieve("reanalysis-era5-land", request, file_name)


# TODO: split into several functions?
def weather_data_to_txt_file(
    data_var_specs,
    coordinates,
    months_list,
    *,
    final_resolution="daily",
    data_format="netcdf",
    data_source="https://cds-beta.climate.copernicus.eu/api",
):
    """
    Process and write weather data to .txt files.

    Parameters:
        data_var_specs (dict): Dictionary of variable specifications.
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon' keys.
        months_list (list): List of (year, month) pairs.
        data_format (str): Data format (default is 'netcdf', no other option currently).
        final_resolution (str): Resolution for final text file ('hourly' or 'daily', default is 'daily').
    """
    # Fragments for daily requests in commits before 2023-11-08.
    # Code for requests from dataset "reanalysis-era5-single-levels" in commits before 2024-01.
    # Code for PET reading from CDS in commits before 2024-01.

    col_names = [entries["col_name_hourly"] for entries in data_var_specs.values()]
    df_collect = pd.DataFrame(columns=["valid_time"] + col_names)
    var_names = list(data_var_specs.keys())
    data_query_protocol = []

    for year, month in months_list:
        file_name = construct_weather_data_file_name(
            "weatherDataRaw",
            data_format,
            coordinates,
            year,
            month,
            "hourly",
        )

        # Open netCDF4 file and extract variables
        ds = netCDF4.Dataset(file_name)
        history = getattr(ds, "history")
        time_stamp, extra_info = history.split(" ", 1)
        data_query_protocol.append(
            [year, month, data_source, time_stamp + "+00:00", extra_info]
        )

        # Init data frame with time data
        valid_time = ds.variables["valid_time"]
        valid_time = num2date(valid_time[:], valid_time.units)
        df_temp = pd.DataFrame({"valid_time": [t.isoformat() for t in valid_time]})

        # Collect and convert hourly data from ds
        for var_name in var_names:
            data_temp = cwd.convert_units(
                ds.variables[data_var_specs[var_name]["short_name"]][:].flatten(),
                data_var_specs[var_name]["unit_conversion"],
            )
            df_temp[data_var_specs[var_name]["col_name_hourly"]] = data_temp

        df_collect = pd.concat([df_collect, df_temp], ignore_index=True)

    # Remove all data except very first entry for last month
    # This will not work correctly in case of multiple years, but not all months!
    count_years = len(np.unique([item[0] for item in months_list]))
    count_months = len(np.unique([item[1] for item in months_list]))

    if (count_years < 2) or (count_months == 12):
        len_last_month = ut.get_days_in_month(months_list[-1][0], months_list[-1][1])
        df_collect = df_collect.iloc[: (-len_last_month * 24 + 1)]
    else:
        raise ValueError(
            "Discontinuous time periods cannot be combined properly from raw data! Please select only one year or all 12 months!"
        )

    # Save DataFrame to .txt file
    file_name = construct_weather_data_file_name(
        "weatherDataPrepared",
        "txt",
        coordinates,
        f"{months_list[1][0]:04d}-{months_list[1][1]:02d}",
        f"{months_list[-2][0]:04d}-{months_list[-2][1]:02d}",
        "hourly",
    )

    # Create data directory if missing
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    # Write file to directory
    df_collect.to_csv(
        file_name, sep="\t", index=False, float_format="%.6f", na_rep="nan"
    )
    print("Text file with hourly resolution prepared.")

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
        # Dates (omit last entry from last Day + 1 at 00:00)
        valid_date = df_collect["valid_time"][:-24:24].str.split("T").str[0].values

        # Day lengths
        day_length = ut.get_day_length(coordinates, valid_date)

        # Precipitation (omit first entry from first Day at 00:00)
        precipitation = df_collect[data_var_specs["precipitation"]["col_name_hourly"]][
            24::24
        ].values

        # Temperature (hourly still needed for PET calculation, get daily means with average of 00:00 and 24:00 as one of 24 values)
        temperature_hourly = df_collect[
            data_var_specs["temperature"]["col_name_hourly"]
        ].values
        temperature = cwd.daily_mean_00_24(temperature_hourly)
        temperature_daylight = cwd.daily_mean_daylight(
            temperature_hourly, valid_date, coordinates
        )

        # SSRD & SSR (omit first entry from first Day at 00:00)
        ssrd = df_collect[data_var_specs["solar_radiation_down"]["col_name_hourly"]][
            24::24
        ].values
        # Convert net radiation to PAR
        ssrd_par = cwd.par_from_net_radiation(ssrd)
        # ssr = df_collect[data_var_specs["solar_radiation_net"]["col_name_hourly"]][
        #     24::24
        # ].values

        # # CO2 default value 400
        # co2 = [400] * len(valid_date)

        # # Remaining data needed for PET calculations
        # # SLHF (omit first entry from first Day at 00:00)
        # slhf = df_collect[
        #     data_var_specs["surface_latent_heat_flux"]["col_name_hourly"]
        # ][24::24].values

        # # Wind speed data (10 m, convert to 2m, get daily means with average of 00:00 and 24:00 as one of 24 values)
        # wind_speed_10m = cwd.wind_speed_from_u_v(
        #     df_collect[data_var_specs["eastward_wind"]["col_name_hourly"]].values,
        #     df_collect[data_var_specs["northward_wind"]["col_name_hourly"]].values,
        # )
        # wind_speed_2m = cwd.wind_speed_height_change(
        #     wind_speed_10m, height1=10, height2=2, z0=0.03
        # )
        # wind_speed_2m = cwd.daily_mean_00_24(wind_speed_2m)

        # # Dewpoint temperature (get daily means with average of 00:00 and 24:00 as one of 24 values)
        # dewpoint_temperature_hourly = df_collect[
        #     data_var_specs["dewpoint_temperature"]["col_name_hourly"]
        # ].values
        # # dewpoint_temperature = cwd.daily_mean_00_24(dewpoint_temperature_hourly)

        # # Surface pressure (get daily means with average of 00:00 and 24:00 as one of 24 values)
        # surface_pressure = cwd.daily_mean_00_24(
        #     df_collect[data_var_specs["surface_pressure"]["col_name_hourly"]].values
        # )

        # # PET from FAO-56 Penman-Monteith equation
        # pet_fao = cwd.get_pet_fao(
        #     ssr,
        #     slhf,
        #     temperature,
        #     temperature_hourly,
        #     wind_speed_2m,
        #     dewpoint_temperature_hourly,
        #     surface_pressure,
        # )

        # PET from Thornthwaite equation
        pet_thornthwaite = cwd.get_pet_thornthwaite(
            temperature,
            temperature_hourly,
            day_length,
            valid_date,
            use_effective_temperature=False,
        )

        # Overwrite hourly dataframe with daily values
        # Include only PET Thornthwaite version
        df_collect = pd.DataFrame(
            {
                "Date": valid_date,
                data_var_specs["precipitation"]["col_name_daily"]: precipitation,
                data_var_specs["temperature"]["col_name_daily"]: temperature,
                "Temperature_Daylight[degC]": temperature_daylight,
                data_var_specs["solar_radiation_down"]["col_name_daily"]: ssrd_par,
                "Daylength[h]": day_length,
                "PET[mm]": pet_thornthwaite,  # was: PET_thornthwaite
            }
        )
        # # Include both PET versions and SSRD, SSRD, SLHF for analysis
        # df_collect = pd.DataFrame(
        #     {
        #         "Date": valid_date,
        #         data_var_specs["precipitation"]["col_name_daily"]: precipitation,
        #         data_var_specs["temperature"]["col_name_daily"]: temperature,
        #         data_var_specs["solar_radiation_down"]["col_name_daily"]: ssrd_par,
        #         "Daylength[h]": day_length,
        #         "PET_fao[mm]": pet_fao,
        #         "PET_thornthwaite[mm]": pet_thornthwaite,
        #         "CO2[ppm]": co2,
        #         "SSRD[Jm-2]": ssrd,
        #         data_var_specs["solar_radiation_net"]["col_name_daily"]: ssr,
        #         data_var_specs["surface_latent_heat_flux"]["col_name_daily"]: slhf,
        #     }
        # )

        # Save DataFrame to .txt file
        file_name = construct_weather_data_file_name(
            "weatherDataPrepared",
            "txt",
            coordinates,
            f"{months_list[0][0]:04d}-{months_list[0][1]:02d}",
            f"{months_list[-2][0]:04d}-{months_list[-2][1]:02d}",
            "weather",
        )
        df_collect.to_csv(
            file_name, sep="\t", index=False, float_format="%.6f", na_rep="nan"
        )
        print(f"Text file with {final_resolution} resolution prepared.")
