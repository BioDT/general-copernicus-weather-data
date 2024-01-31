"""
Module Name: get_weather_data.py
Author: Thomas Banitz, Tuomas Rossi, Franziska Taubert, BioDT
Date: November 8, 2023
Description: Functions for downloading and processing selected weather data. 
"""
import cdsapi
import netCDF4
from netCDF4 import num2date
import pandas as pd
from . import utils as ut
import statistics as stats
import numpy as np
from pathlib import Path


def construct_months_list(years, months):
    """
    Construct a list of year-month pairs.

    Args:
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
    - long_name: Long name of the variable.
    - short_name: Abbreviation of the variable.
    - data_set_hourly: Dataset name for hourly data.
    - data_set_daily: Dataset name for daily data.
    - daily_stat: Statistic for downloading daily data (currently not available).
    - col_name_raw: Raw column name for the variable in the data source.
    - col_name: Adjusted column name for the variable in final files.

    Returns:
        dict: A dictionary of variable specifications, where each key is a variable name,
              and each value is a dictionary of specifications.

    """
    data_var_specs = {
        "precipitation": {
            "long_name": "total_precipitation",
            "short_name": "tp",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_maximum",
            "col_name_raw": "Precipitation[mm] (acc.)",
            "col_name": "Precipitation[mm]",
        },
        "temperature": {
            "long_name": "2m_temperature",
            "short_name": "2t",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_mean",
            "col_name_raw": "Temperature[degC]",
            "col_name": "Temperature[degC]",
        },
        "solar_radiation_down": {
            "long_name": "surface_solar_radiation_downwards",
            "short_name": "ssrd",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_maximum",
            "col_name_raw": "SSRD_PPFD[mmolm-2s-1] (acc.)",
            "col_name": "PPFD_down[mmolm-2s-1]",  # can be adjusted for final files
        },
        "solar_radiation_net": {
            "long_name": "surface_net_solar_radiation",
            "short_name": "ssr",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_maximum",
            "col_name_raw": "SSR_PPFD[mmolm-2s-1] (acc.)",
            "col_name": "PPFD_net[mmolm-2s-1]",  # can be adjusted for final files
        },
        "pot_evapotranspiration_land": {
            "long_name": "potential_evaporation",
            "short_name": "pev",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_maximum",
            "col_name_raw": "PET_land[mm] (acc.)",
            "col_name": "PET_land[mm]",  # can be adjusted for final files
        },
        "pot_evapotranspiration_single": {
            "long_name": "potential_evaporation",
            "short_name": "pev",
            "data_set_hourly": "reanalysis-era5-single-levels",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_mean",  # daily sum not possible
            "col_name_raw": "PET_single[mm]",
            "col_name": "PET_single[mm]",  # can be adjusted for final files
        },
        "surface_latent_heat_flux": {
            "long_name": "surface_latent_heat_flux",
            "short_name": "slhf",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_sum",  # ??
            "col_name_raw": "SLHF[Jm-2]",  # ??
            "col_name": "SLHF[Jm-2]",  # can be adjusted for final files
        },
        "eastward_wind": {
            "long_name": "10m_u_component_of_wind",
            "short_name": "u10",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_mean",  # ??
            "col_name_raw": "U10[ms-1]",  # ??
            "col_name": "U10[ms-1]",  # can be adjusted for final files
        },
        "northward_wind": {
            "long_name": "10m_v_component_of_wind",
            "short_name": "v10",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_mean",  # ??
            "col_name_raw": "V10[ms-1]",  # ??
            "col_name": "V10[ms-1]",  # can be adjusted for final files
        },
        "dewpoint_temperature": {
            "long_name": "2m_dewpoint_temperature",
            "short_name": "2d",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_mean",
            "col_name_raw": "DewpointTemperature[degC]",  # check?
            "col_name": "DewpointTemperature[degC]",  # check?
        },
        "surface_pressure": {
            "long_name": "surface_pressure",
            "short_name": "sp",
            "data_set_hourly": "reanalysis-era5-land",
            "data_set_daily": "tool.toolbox.orchestrator.workflow",
            "daily_stat": "daily_mean",
            "col_name_raw": "SurfacePressure[Pa]",  # check?
            "col_name": "SurfacePressure[Pa]",  # check?
        },
    }

    return data_var_specs


def check_missing_entries(entry_name, data_var_specs):
    """
    Check for missing 'entry_name' in variable specifications.
    Prints a warning for each variable without 'entry_name'.

    Args:
        entry_name (str): Name of the entry to check for.
        data_var_specs (dict): Variable specifications dictionary.
    """
    for key, entries in data_var_specs.items():
        if entry_name not in entries:
            print(f"Warning: '{entry_name}' entry missing for key '{key}'")


def construct_data_file_name(
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

    Args:
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

    if ut.is_dict_of_2_floats(location) and set(location.keys()) == {
        "lat",
        "lon",
    }:  # location as dictionary with lat, lon
        formatted_lat = f"lat{location['lat']:.2f}".replace(".", "-")
        formatted_lon = f"lon{location['lon']:.2f}".replace(".", "-")
        file_name = (
            folder
            / f"{data_set}_{data_resolution}_{formatted_lat}_{formatted_lon}_{formatted_year}_{formatted_month}_{var_short}{data_suffix}"
        )
    elif isinstance(location, str):  # location as string (DEIMS.iD)
        file_name = (
            folder
            / f"{location}_{formatted_year}_{formatted_month}_{var_short}{data_suffix}"
        )
    else:
        raise ValueError("Unsupported location format.")

    return file_name


# # def find_raw_file(src_folder, extension):
# # def move_raw_file(file_name, extension):
# # only needed for daily resolution, in commits before 2023-11-08


def construct_request(
    data_format, data_resolution, coordinates, year, month, variables
):
    """
    Construct data request.

    Args:
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
            "product_type": "reanalysis",
            "grid": "0.1/0.1",
            "area": [  # Works with exact location as lower and upper bounds of area
                coordinates["lat"],
                coordinates["lon"],
                coordinates["lat"],
                coordinates["lon"],
            ],
            "format": data_format,
            "day": ut.generate_day_values(year, month),
            "year": str(year),
            "month": str(month),
            "time": [f"{i:02}:00" for i in range(24)],
        }

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

    Args:
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

    data_setVarName = "data_set_" + data_resolution
    check_missing_entries(data_setVarName, data_var_specs)

    var_names = [
        key
        for key, entries in data_var_specs.items()
        if entries[data_setVarName] == data_set
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

            file_name = construct_data_file_name(
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

    Args:
        data_set (str): Name of the data set to download.
        data_requests (list): List of data requests and corresponding file names.
        data_resolution (str): Data resolution ('hourly').
    """
    c = cdsapi.Client()

    if data_resolution == "hourly":
        for request, file_name in data_requests:
            # Create data directory if missing
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
            # Retrieve data
            c.retrieve(data_set, request, file_name)

    # # Option for daily requests not fully developed!
    # # Fragments in commits before 2023-11-08
    else:
        raise ValueError("Unsupported data resolution.")


# TODO: split into several functions?
def weather_data_2_txt_file(
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

    Args:
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
    check_missing_entries("col_name_raw", data_var_specs)
    col_names = [
        entries["col_name_raw"]
        for entries in data_var_specs.values()
        if "col_name_raw" in entries
    ]

    df_collect = pd.DataFrame(columns=["time"] + col_names)

    if data_resolution == "hourly":
        # Create list of data that need to be read, as they may sit in multiple files for the same month
        data_read_out = []
        data_setVarName = "data_set_" + data_resolution
        check_missing_entries(data_setVarName, data_var_specs)
        check_missing_entries("short_name", data_var_specs)

        for data_set in data_sets:
            # can be simplified when only ERA5-Land dataset is used
            var_names = [
                key
                for key, entries in data_var_specs.items()
                if entries[data_setVarName] == data_set
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
            file_name = construct_data_file_name(
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
            times = ds.variables["time"]
            times = num2date(times[:], times.units)

            if data_resolution == "hourly":
                if df_temp is None:
                    df_temp = pd.DataFrame(
                        {
                            "time": [t.isoformat() for t in times],
                            data_var_specs["precipitation"][
                                "col_name_raw"
                            ]: ds.variables[
                                data_var_specs["precipitation"]["short_name"]
                            ][
                                :
                            ].flatten()
                            * 1000,  # Conversion from m to mm
                            data_var_specs["temperature"]["col_name_raw"]: ds.variables[
                                data_var_specs["temperature"]["short_name"]
                            ][:].flatten()
                            - 273.15,  # Conversion from K to C
                            data_var_specs["solar_radiation_down"][
                                "col_name_raw"
                            ]: ds.variables[
                                data_var_specs["solar_radiation_down"]["short_name"]
                            ][
                                :
                            ].flatten()
                            / (24 * 60 * 60)  # Conversion from /d to /s
                            * 4.57  # Conversion J to µmol (4.57 µmol/J)
                            * 0.5,  # PAR fraction
                            data_var_specs["solar_radiation_net"][
                                "col_name_raw"
                            ]: ds.variables[
                                data_var_specs["solar_radiation_net"]["short_name"]
                            ][
                                :
                            ].flatten()
                            / (24 * 60 * 60)  # Conversion from /d to /s
                            * 4.57  # Conversion J to µmol (4.57 µmol/J)
                            * 0.5,  # PAR fraction
                            data_var_specs["pot_evapotranspiration_land"][
                                "col_name_raw"
                            ]: ds.variables[
                                data_var_specs["pot_evapotranspiration_land"][
                                    "short_name"
                                ]
                            ][
                                :
                            ].flatten()
                            * 1000,  # Conversion from m to mm
                        }
                    )
                else:
                    if len(ds.variables["time"]) == len(df_temp["time"]):
                        df_temp[
                            data_var_specs["pot_evapotranspiration_single"][
                                "col_name_raw"
                            ]
                        ] = (
                            ds.variables[
                                data_var_specs["pot_evapotranspiration_single"][
                                    "short_name"
                                ]
                            ][:].flatten()
                            * 1000
                        )  # Conversion from m to mm
                    else:
                        raise ValueError("Unequal lengths of the data to be combined!")

            # # Option for daily requests not fully developed!
            # # Fragments in commits before 2023-11-08
            else:
                raise ValueError("Unsupported data resolution.")

        df_collect = pd.concat([df_collect, df_temp], ignore_index=True)

    # Remove all data except very first entry for last month
    # This will not work correctly in case of multiple years, but not all months!
    ny = len(np.unique([item[0] for item in months_list]))
    nm = len(np.unique([item[1] for item in months_list]))

    if (ny < 2) or (nm == 12):
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
    file_name = construct_data_file_name(
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
    df_collect.to_csv(file_name, sep="\t", index=False, float_format="%.6f")
    print(f"Text file with {data_resolution} resolution prepared.")

    # ####
    # Convert hourly to daily data if needed (e.g. for Grassmind)
    if data_resolution == "hourly" and final_resolution == "daily":
        # Dates (omit last entry from last Day + 1 at 00:00)
        time_data = df_collect["time"][:-24:24].str.split("T").str[0]
        time_data = time_data.values

        # Day lengths
        day_length_data = ut.get_day_length(coordinates, time_data)

        # Precipitation (omit first entry from first Day at 00:00)
        prec_data = df_collect[data_var_specs["precipitation"]["col_name_raw"]][
            24::24
        ].values

        # Temperature
        temp_data = df_collect[data_var_specs["temperature"]["col_name_raw"]].values
        # Calculate the mean of the two temperatures at 0:00 and 24:00 (day + 1 at 00:00)
        mean_temp_00_24 = [
            stats.mean(
                [
                    temp_data[i],
                    temp_data[i + 24],
                ]
            )
            for i in range(0, len(df_collect) - 1, 24)
        ]

        # Calculate daily means
        temp_data = temp_data[:-1]  # Omit last entry from last Day + 1 at 00:00
        temp_data[
            ::24
        ] = mean_temp_00_24  # Replace 00:00 entries with calculated means of 00:00 and 24:00
        temp_data = [temp_data[i : i + 24].mean() for i in range(0, len(temp_data), 24)]

        # SSRD & SSR (omit first entry from first Day at 00:00)
        ssrd_data = df_collect[data_var_specs["solar_radiation_down"]["col_name_raw"]][
            24::24
        ].values
        # ssrData = df_collect[data_var_specs["solar_radiation_net"]["col_name_raw"]][24::24].values

        # PET (correct negative values to 0)
        # TODO: replace PET values from download by calculation using PyETo package
        pet_land_data = df_collect[
            data_var_specs["pot_evapotranspiration_land"]["col_name_raw"]
        ][
            24::24
        ].values  # entries are accumulated
        pet_land_data[pet_land_data < 0] = 0

        pet_single_data = df_collect[
            data_var_specs["pot_evapotranspiration_single"]["col_name_raw"]
        ][1:].values
        pet_single_data[pet_single_data < 0] = 0  # correction before summing up
        pet_single_data = [
            pet_single_data[i : i + 24].sum()
            for i in range(0, len(pet_single_data), 24)
        ]  # entries not accumulated, so sum seems to make most sense (calculation not entirely clear!)

        co2_data = [400] * len(time_data)  # default value 400

        # Overwrite hourly dataframe with daily values
        df_collect = pd.DataFrame(
            {
                "Date": time_data,
                data_var_specs["precipitation"]["col_name"]: prec_data,
                data_var_specs["temperature"]["col_name"]: temp_data,
                data_var_specs["solar_radiation_down"]["col_name"]: ssrd_data,
                "Daylength[h]": day_length_data,
                data_var_specs["pot_evapotranspiration_land"][
                    "col_name"
                ]: pet_land_data,
                data_var_specs["pot_evapotranspiration_single"][
                    "col_name"
                ]: pet_single_data,
                "CO2[ppm]": co2_data,
            }
        )

        # Save DataFrame to .txt file, FileName with DEIMS.iD if existing
        file_name = construct_data_file_name(
            "weatherDataPrepared",
            data_set,
            final_resolution,
            ".txt",
            deims_id if deims_id is not None else coordinates,
            f"{months_list[0][0]:04d}-{months_list[0][1]:02d}",
            f"{months_list[-2][0]:04d}-{months_list[-2][1]:02d}",
            ["Weather"],
        )
        df_collect.to_csv(file_name, sep="\t", index=False, float_format="%.6f")
        print(f"Text file with {final_resolution} resolution prepared.")
