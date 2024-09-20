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
"""

from pathlib import Path

import cdsapi
import netCDF4
import numpy as np
import pandas as pd
from netCDF4 import num2date

from copernicus import convert_weather_data as cwd
from copernicus import utils as ut


def construct_months_list(years, months):
    """
    Construct a list of year-month pairs.

    Parameters:
        years (list of int): List of years.
        months (list of int): List of months (1 to 12).

    Returns:
        list of tuples: A list of (year, month) tuples representing all combinations
        of years and months in the input lists.
    """
    months_list = []

    for year in years:
        for month in months:
            months_list.append((year, month))

        # Always add next month if months don't go until the end of the year, or else the January after the final year
        if months[-1] < 12:
            months_list.append((year, months[-1] + 1))
        elif year == years[-1]:
            months_list.append((year + 1, 1))

    return months_list


def get_var_specs():
    """
    Retrieve a dictionary of variable specifications.

    Create a dictionary that provides specifications for potential download variables.
    Each variable is identified by its name and includes the following information:
        long_name: Long name of the variable.
        short_name: Abbreviation of the variable.
        data_set_hourly: Dataset name for hourly data.
        data_set_daily: Dataset name for daily data (currently not available).
        daily_stat: Statistic for downloading daily data (currently not available).
        unit_conversion: Unit conversion from source to target data.
        col_name_hourly: Column name for the variable in hourly data (source data units converted).
        col_name_daily: Column name for the variable in daily data (source data units converted).

    Returns:
        dict: A dictionary of variable specifications, where each key is a variable name,
              and each value is a dictionary of specifications.

    """
    data_var_specs = {
        "precipitation": {
            "long_name": "total_precipitation",
            "short_name": "tp",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_maximum",
            "unit_conversion": "_to_Milli",
            "col_name_hourly": "Precipitation[mm] (acc.)",
            "col_name_daily": "Precipitation[mm]",
        },
        "temperature": {
            "long_name": "2m_temperature",
            "short_name": "t2m",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_mean",
            "unit_conversion": "Kelvin_to_Celsius",
            "col_name_hourly": "Temperature[degC]",
            "col_name_daily": "Temperature[degC]",
        },
        "solar_radiation_down": {
            "long_name": "surface_solar_radiation_downwards",
            "short_name": "ssrd",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_maximum",
            "unit_conversion": 0,
            "col_name_hourly": "SSRD[Jm-2] (acc.)",
            "col_name_daily": "PPFD_down[µmolm-2s-1]",  # as converted to PAR
        },
        "solar_radiation_net": {
            "long_name": "surface_net_solar_radiation",
            "short_name": "ssr",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_maximum",
            "unit_conversion": 0,
            "col_name_hourly": "SSR[Jm-2] (acc.)",
            "col_name_daily": "SSR[Jm-2]",
        },
        "surface_latent_heat_flux": {
            "long_name": "surface_latent_heat_flux",
            "short_name": "slhf",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_sum",
            "unit_conversion": 0,
            "col_name_hourly": "SLHF[Jm-2] (acc.)",
            "col_name_daily": "SLHF[Jm-2]",
        },
        "eastward_wind": {
            "long_name": "10m_u_component_of_wind",
            "short_name": "u10",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_mean",
            "unit_conversion": 0,
            "col_name_hourly": "U10[ms-1]",
            "col_name_daily": "",
        },
        "northward_wind": {
            "long_name": "10m_v_component_of_wind",
            "short_name": "v10",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_mean",
            "unit_conversion": 0,
            "col_name_hourly": "V10[ms-1]",
            "col_name_daily": "",
        },
        "dewpoint_temperature": {
            "long_name": "2m_dewpoint_temperature",
            "short_name": "d2m",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_mean",
            "unit_conversion": "Kelvin_to_Celsius",
            "col_name_hourly": "DewpointTemperature[degC]",
            "col_name_daily": "",
        },
        "surface_pressure": {
            "long_name": "surface_pressure",
            "short_name": "sp",
            "data_set_hourly": "reanalysis-era5-land",
            "daily_stat": "daily_mean",
            "unit_conversion": "_to_Kilo",
            "col_name_hourly": "SurfacePressure[kPa]",
            "col_name_daily": "",
        },
    }

    return data_var_specs


def check_missing_entries(entry_name, data_var_specs):
    """
    Check for missing 'entry_name' in variable specifications.
    Prints a warning for each variable without 'entry_name'.

    Parameters:
        entry_name (str): Name of the entry to check for.
        data_var_specs (dict): Variable specifications dictionary.
    """
    for key, entries in data_var_specs.items():
        if entry_name not in entries:
            print(f"Warning: '{entry_name}' entry missing for key '{key}'")


def construct_weather_data_file_name(
    folder,
    data_set,
    data_resolution,
    data_suffix,
    location,
    year,
    month,
    var_short,
):
    """
    Construct data file name.

    Parameters:
        folder (str or Path): Folder where the data file will be stored.
        data_set (str): Name of the data set.
        data_resolution (str): Data resolution ('hourly', 'daily').
        data_suffix (str): File suffix ('.nc').
        location (str or dict): Location information ('DEIMS.iD' or {'lat': float, 'lon': float}).
        year (int): Year for the data file.
        month (int): Month for the data file.
        var_short (str or list of str): Variable short name(s).

    Returns:
        Path: Constructed data file name as a Path object.
    """
    # Get folder with path appropriate for different operating systems
    folder = Path(folder)

    # Set default var name if multiple vars, or get var name from list if one var
    if len(var_short) > 1:
        var_short = "multVars"
    else:
        var_short = var_short[0]

    formatted_year = str(year)
    formatted_month = str(month).zfill(2)

    if ("lat" in location) and ("lon" in location):
        formatted_lat = f"lat{location['lat']:.6f}".replace(".", "-")
        formatted_lon = f"lon{location['lon']:.6f}".replace(".", "-")
        file_start = f"{data_set}_{data_resolution}_{formatted_lat}_{formatted_lon}"
    elif "deims_id" in location:  # DEIMS.iD
        file_start = location["deims_id"]
    elif isinstance(location, str):  # location as string (e.g. DEIMS.iD)
        file_start = location
    else:
        raise ValueError("Unsupported location format.")

    file_name = (
        folder
        / f"{file_start}_{formatted_year}_{formatted_month}_{var_short}{data_suffix}"
    )

    return file_name


def construct_request(
    data_format, data_resolution, coordinates, year, month, variables
):
    """
    Construct data request.

    Parameters:
        data_format (str): Data format ('netcdf').
        data_resolution (str): Data resolution ('hourly').
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon'.
        year (int): Year for the data request.
        month (int): Month for the data request.
        variables (list): List of variables to request.

    Returns:
        dict: Dictionary representing the data request parameters.
    """
    if data_resolution == "hourly":
        request = {
            "variable": variables,
            "product_type": "reanalysis",  # ['reanalysis']
            "grid": "0.1/0.1",
            "area": [  # Works with exact location as lower and upper bounds of area
                coordinates["lat"],
                coordinates["lon"],
                coordinates["lat"],
                coordinates["lon"],
            ],
            "data_format": data_format,  # 'data_format': data_format
            "day": ut.generate_day_values(year, month),
            "year": str(year),
            "month": str(month),
            "time": [f"{i:02}:00" for i in range(24)],
            "download_format": "unarchived",
        }  # 'download_format': 'unarchived'

    # # Option for daily requests not fully developed!
    # # Fragments in commits before 2023-11-08
    else:
        raise ValueError("Unsupported data resolution.")

    return request


def configure_data_request(
    data_set, data_var_specs, data_format, data_resolution, coordinates, months_list
):
    """
    Configure data requests.

    Parameters:
        data_set (str): Name of the data set.
        data_var_specs (dict): Dictionary of variable specifications.
        data_format (str): Data format ('netcdf').
        data_resolution (str): Data resolution ('hourly').
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon'.
        months_list (list): List of (year, month) pairs.

    Returns:
        list: List of data requests and corresponding file names.
    """
    data_requests = []
    data_suffix = ut.get_data_suffix(data_format)

    data_set_var_name = "data_set_" + data_resolution
    check_missing_entries(data_set_var_name, data_var_specs)

    var_names = [
        key
        for key, entries in data_var_specs.items()
        if entries[data_set_var_name] == data_set
    ]
    long_names = [data_var_specs[key]["long_name"] for key in var_names]
    short_names = [data_var_specs[key]["short_name"] for key in var_names]

    if data_resolution == "hourly":
        for year, month in months_list:
            request = construct_request(
                data_format,
                data_resolution,
                coordinates,
                year,
                month,
                long_names,  # Requested variables
            )

            file_name = construct_weather_data_file_name(
                "weatherDataRaw",
                data_set,
                data_resolution,
                data_suffix,
                coordinates,
                year,
                month,
                short_names,  # Requested variables' short names
            )

            data_requests.append((request, file_name))

    # # Option for daily requests not fully developed!
    # # Fragments in commits before 2023-11-08
    else:
        raise ValueError("Unsupported data resolution.")

    return data_requests


def download_weather_data(data_set, data_requests, data_resolution):
    """
    Download weather data from CDS API.

    Parameters:
        data_set (str): Name of the data set to download.
        data_requests (list): List of data requests and corresponding file names.
        data_resolution (str): Data resolution ('hourly').
    """
    client = cdsapi.Client()

    if data_resolution == "hourly":
        for request, file_name in data_requests:
            # Create data directory if missing
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
            # Retrieve data
            client.retrieve(data_set, request, file_name)

    # asynchronous https://docs.python.org/3/library/asyncio.html

    # # Option for daily requests not fully developed!
    # # Fragments in commits before 2023-11-08
    else:
        raise ValueError("Unsupported data resolution.")


# TODO: split into several functions?
def weather_data_to_txt_file(
    data_sets,
    data_var_specs,
    data_format,
    data_resolution,
    final_resolution,
    deims_id,
    coordinates,
    months_list,
):
    """
    Process and write weather data to .txt files.

    Parameters:
        data_sets (list): List of data set names.
        data_var_specs (dict): Dictionary of variable specifications.
        data_format (str): Data format ('netcdf').
        data_resolution (str): Data resolution ('hourly').
        final_resolution (str): Final data resolution ('hourly' or 'daily').
        deims_id (str): DEIMS.iD if available, or None.
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon'.
        months_list (list): List of (year, month) pairs.
    """
    data_suffix = ut.get_data_suffix(data_format)
    check_missing_entries("col_name_hourly", data_var_specs)
    col_names = [
        entries["col_name_hourly"]
        for entries in data_var_specs.values()
        if "col_name_hourly" in entries
    ]

    df_collect = pd.DataFrame(columns=["time"] + col_names)

    if data_resolution == "hourly":
        # Create list of data that need to be read, as they may sit in multiple files for the same month
        data_read_out = []
        data_set_var_name = "data_set_" + data_resolution
        check_missing_entries(data_set_var_name, data_var_specs)
        check_missing_entries("short_name", data_var_specs)

        for data_set in data_sets:
            # can be simplified when only ERA5-Land dataset is used
            var_names = [
                key
                for key, entries in data_var_specs.items()
                if entries[data_set_var_name] == data_set
            ]

            if len(var_names) > 1:
                var_short = ["multVars"]
            elif len(var_names) == 1:
                var_short = [data_var_specs[var_names[0]]["short_name"]]
            else:
                print(
                    f"Warning: No entries found that use the '{data_set}' data set for {data_resolution} resolution."
                )

            data_read_out.append((data_set, var_short))

    # # Option for daily requests not fully developed!
    # # Fragments in commits before 2023-11-08
    else:
        raise ValueError("Unsupported data resolution.")

    for year, month in months_list:
        df_temp = None

        for data_set, var_short in data_read_out:
            file_name = construct_weather_data_file_name(
                "weatherDataRaw",
                data_set,
                data_resolution,
                data_suffix,
                coordinates,
                year,
                month,
                var_short,
            )

            # Open netCDF4 file and extract variables
            ds = netCDF4.Dataset(file_name)
            # time = ds.variables["time"]
            time = ds.variables["valid_time"]
            time = num2date(time[:], time.units)

            if data_resolution == "hourly":
                # Init data frame with time data
                df_temp = pd.DataFrame({"time": [t.isoformat() for t in time]})

                # Collect and convert hourly data from ds
                for var_name in var_names:
                    data_temp = cwd.convert_units(
                        ds.variables[data_var_specs[var_name]["short_name"]][
                            :
                        ].flatten(),
                        data_var_specs[var_name]["unit_conversion"],
                    )
                    df_temp[data_var_specs[var_name]["col_name_hourly"]] = data_temp

                # Option for requests from dataset "reanalysis-era5-single-levels" removed
                # Code in commits before 2024-01

            # # Option for daily requests not fully developed!
            # # Fragments in commits before 2023-11-08
            else:
                raise ValueError("Unsupported data resolution.")

        df_collect = pd.concat([df_collect, df_temp], ignore_index=True)

    # Remove all data except very first entry for last month
    # This will not work correctly in case of multiple years, but not all months!
    count_years = len(np.unique([item[0] for item in months_list]))
    count_months = len(np.unique([item[1] for item in months_list]))

    if (count_years < 2) or (count_months == 12):
        len_last_month = ut.get_days_in_month(months_list[-1][0], months_list[-1][1])

        if data_resolution == "hourly":
            df_collect = df_collect.iloc[: (-len_last_month * 24 + 1)]
        elif data_resolution == "daily":
            df_collect = df_collect.iloc[: (-len_last_month + 1)]
    else:
        raise ValueError(
            "Discontinuous time periods cannot be combined properly from raw data! Please select only one year or all 12 months!"
        )

    # Save DataFrame to .txt file
    file_name = construct_weather_data_file_name(
        "weatherDataPrepared",
        data_set,
        data_resolution,
        ".txt",
        coordinates,
        f"{months_list[0][0]:04d}-{months_list[0][1]:02d}",
        f"{months_list[-2][0]:04d}-{months_list[-2][1]:02d}",
        ["multVars"],
    )

    # Create data directory if missing
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    # Write file to directory
    df_collect.to_csv(
        file_name, sep="\t", index=False, float_format="%.6f", na_rep="nan"
    )
    print(f"Text file with {data_resolution} resolution prepared.")

    # ####
    # Convert hourly to daily data if needed (e.g. for grassland model)
    if data_resolution == "hourly" and final_resolution == "daily":
        # Dates (omit last entry from last Day + 1 at 00:00)
        time = df_collect["time"][:-24:24].str.split("T").str[0].values

        # Day lengths
        day_length = ut.get_day_length(coordinates, time)

        # Precipitation (omit first entry from first Day at 00:00)
        precipitation = df_collect[data_var_specs["precipitation"]["col_name_hourly"]][
            24::24
        ].values

        # Temperature (hourly still needed for PET calculation, get daily means with average of 00:00 and 24:00 as one of 24 values)
        temperature_hourly = df_collect[
            data_var_specs["temperature"]["col_name_hourly"]
        ].values
        temperature = cwd.daily_mean_00_24(temperature_hourly)

        # SSRD & SSR (omit first entry from first Day at 00:00)
        ssrd = df_collect[data_var_specs["solar_radiation_down"]["col_name_hourly"]][
            24::24
        ].values
        # Convert net radiation to PAR
        ssrd_par = cwd.par_from_net_radiation(ssrd)
        ssr = df_collect[data_var_specs["solar_radiation_net"]["col_name_hourly"]][
            24::24
        ].values

        # CO2 default value 400
        co2 = [400] * len(time)

        # Code for PET reading from CDS in commits before 2024-01
        # Remaining data needed for PET calculations
        # SLHF (omit first entry from first Day at 00:00)
        slhf = df_collect[
            data_var_specs["surface_latent_heat_flux"]["col_name_hourly"]
        ][24::24].values

        # Wind speed data (10 m, convert to 2m, get daily means with average of 00:00 and 24:00 as one of 24 values)
        wind_speed_10m = cwd.wind_speed_from_u_v(
            df_collect[data_var_specs["eastward_wind"]["col_name_hourly"]].values,
            df_collect[data_var_specs["northward_wind"]["col_name_hourly"]].values,
        )
        wind_speed_2m = cwd.wind_speed_height_change(
            wind_speed_10m, height1=10, height2=2, z0=0.03
        )
        wind_speed_2m = cwd.daily_mean_00_24(wind_speed_2m)

        # Dewpoint temperature (get daily means with average of 00:00 and 24:00 as one of 24 values)
        dewpoint_temperature_hourly = df_collect[
            data_var_specs["dewpoint_temperature"]["col_name_hourly"]
        ].values
        # dewpoint_temperature = cwd.daily_mean_00_24(dewpoint_temperature_hourly)

        # Surface pressure (get daily means with average of 00:00 and 24:00 as one of 24 values)
        surface_pressure = cwd.daily_mean_00_24(
            df_collect[data_var_specs["surface_pressure"]["col_name_hourly"]].values
        )

        # PET from FAO-56 Penman-Monteith equation
        pet_fao = cwd.get_pet_fao(
            ssr,
            slhf,
            temperature,
            temperature_hourly,
            wind_speed_2m,
            dewpoint_temperature_hourly,
            surface_pressure,
        )

        # PET from Thornthwaite equation
        pet_thornthwaite = cwd.get_pet_thornthwaite(
            temperature,
            temperature_hourly,
            day_length,
            time,
            use_effective_temperature=False,
        )

        # Overwrite hourly dataframe with daily values
        # Include both PET versions and SSRD, SSRD, SLHF for analysis
        df_collect = pd.DataFrame(
            {
                "Date": time,
                data_var_specs["precipitation"]["col_name_daily"]: precipitation,
                data_var_specs["temperature"]["col_name_daily"]: temperature,
                data_var_specs["solar_radiation_down"]["col_name_daily"]: ssrd_par,
                "Daylength[h]": day_length,
                "PET_fao[mm]": pet_fao,
                "PET_thornthwaite[mm]": pet_thornthwaite,
                "CO2[ppm]": co2,
                "SSRD[Jm-2]": ssrd,
                data_var_specs["solar_radiation_net"]["col_name_daily"]: ssr,
                data_var_specs["surface_latent_heat_flux"]["col_name_daily"]: slhf,
            }
        )

        # Save DataFrame to .txt file, FileName with DEIMS.iD if existing
        file_name = construct_weather_data_file_name(
            "weatherDataPrepared",
            data_set,
            final_resolution,
            ".txt",
            deims_id if deims_id is not None else coordinates,
            f"{months_list[0][0]:04d}-{months_list[0][1]:02d}",
            f"{months_list[-2][0]:04d}-{months_list[-2][1]:02d}",
            ["Weather"],
        )
        df_collect.to_csv(
            file_name, sep="\t", index=False, float_format="%.6f", na_rep="nan"
        )
        print(f"Text file with {final_resolution} resolution prepared.")
