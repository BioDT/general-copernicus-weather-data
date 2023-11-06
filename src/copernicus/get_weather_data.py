import os
import shutil
import cdsapi
import netCDF4
from netCDF4 import num2date
import pandas as pd
from . import utils as ut
import statistics as stats
import numpy as np
from pathlib import Path

# import xarray


def construct_months_list(years, months):
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
            "short_name": "t2m",
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
    }

    return data_var_specs


def check_missing_entries(entry_name, data_var_specs):
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
        file_name = folder / f"{data_set}_{data_resolution}_{formatted_lat}_{formatted_lon}_{formatted_year}_{formatted_month}_{var_short}{data_suffix}"
    elif isinstance(location, str):  # location as string (DEIMS.iD)
        file_name = folder / f"{location}_{formatted_year}_{formatted_month}_{var_short}{data_suffix}"
    else:
        raise ValueError("Unsupported location format.")

    return file_name


def find_raw_file(src_folder, extension):
    # Find youngest file with extension, as this has cryptic unknown name after downlod of daily data
    raw_file = None
    raw_file_mod_time = float("-inf")

    for file_name in os.listdir(src_folder):
        if file_name.endswith(extension):
            file_path = os.path.join(src_folder, file_name)
            file_mod_time = os.path.getmtime(file_path)

            if file_mod_time > raw_file_mod_time:
                raw_file = file_path
                raw_file_mod_time = file_mod_time

    return raw_file


def move_raw_file(file_name, extension):
    # Rename/move downloaded raw file to desired location
    src_folder = os.getcwd()  # Current folder
    raw_file = find_raw_file(src_folder, extension)

    if raw_file:
        shutil.move(raw_file, file_name)
        print(f"Moved {raw_file} to {file_name}")
    else:
        print(f"No '{extension}' files found in the current folder.")


def construct_request(
    data_set, data_format, data_resolution, coordinates, year, month, variables
):
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
    # # elif data_resolution == "daily":
    #     request = {
    #         "realm": "user-apps",
    #         "project": "app-c3s-daily-era5-statistics",
    #         "version": "master",
    #         "kwargs": {
    #             "dataset": data_set,
    #             "variable": var_name,
    #             "product_type": "reanalysis",
    #             "grid": "0.1/0.1",
    #             "area": {  # Span area around location that contains one grid point
    #                 "lat": [
    #                     coordinates['lat'] - 0.05,
    #                     coordinates['lat'] + 0.05,
    #                 ],
    #                 "lon": [
    #                     coordinates['lon'] - 0.05,
    #                     coordinates['lon'] + 0.05,
    #                 ],
    #             },
    #             "statistic": var_stat,
    #             "year": str(year),
    #             "month": str(month).zfill(2),
    #             "time_zone": ut.get_time_zone(coordinates),
    #             "frequency": "1-hourly",
    #         },
    #         "workflow_name": "application",
    #     }

    return request


def configure_data_request(
    data_set, data_var_specs, data_format, data_resolution, coordinates, months_list
):
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
                data_set,
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
    # elif data_resolution == "daily":
    #     for year, month in months_list:
    #         # for year in years:
    #         # for month in months:
    #         for var_name, var_short, var_stat in data_var_specs:
    #             # test if single value requests work for all vars
    #             # option to send as single value requests..
    #             request = {
    #                 "realm": "user-apps",
    #                 "project": "app-c3s-daily-era5-statistics",
    #                 "version": "master",
    #                 "kwargs": {
    #                     "dataset": data_set,
    #                     "variable": var_name,
    #                     "product_type": "reanalysis",
    #                     "grid": "0.1/0.1",
    #                     "area": {  # Span area around location that contains one grid point
    #                         "lat": [
    #                             coordinates['lat'] - 0.05,
    #                             coordinates['lat'] + 0.05,
    #                         ],
    #                         "lon": [
    #                             coordinates['lon'] - 0.05,
    #                             coordinates['lon'] + 0.05,
    #                         ],
    #                     },
    #                     "statistic": var_stat,
    #                     "year": str(year),
    #                     "month": str(month).zfill(2),
    #                     "time_zone": ut.get_time_zone(coordinates),
    #                     "frequency": "1-hourly",
    #                 },
    #                 "workflow_name": "application",
    #             }

    #             file_name = construct_data_file_name(
    #                 "weatherDataRaw",
    #                 data_set,
    #                 data_resolution,
    #                 data_suffix,
    #                 coordinates,
    #                 year,
    #                 month,
    #                 var_short,
    #             )
    #
    #             data_requests.append((request, file_name))
    else:
        raise ValueError("Unsupported data resolution.")

    return data_requests


def download_weather_data(data_set, data_requests, data_resolution):
    c = cdsapi.Client()
    # verify=False,  # This was a fix to avoid unresolvable certificate verification errors!
    # It will produce warnings.
    # Error messages: " ... SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]
    # certificate verify failed: unable to get local issuer certificate (_ssl.c:1002)')"
    # (It used to work fine before, but somehow changed.)
    # (Found some stuff on manually modifying 'cacert.pem', but got lost..)

    if data_resolution == "hourly":
        for request, file_name in data_requests:
            # Create data directory if missing
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
            # Retrieve data
            c.retrieve(data_set, request, file_name)
    # # Option for daily requests not fully developed!
    # elif data_resolution == "daily":
    #     for request, file_name in data_requests:
    #         # Resource app-c3s-daily-era5-statistics does not accept requests via "retrieve"
    #         result = c.service("tool.toolbox.orchestrator.workflow", request)
    #         c.download(result)
    #         # Seems impossible to specify the target file for download --> find, rename & move
    #         move_raw_file(file_name, "nc")  # no .grib files for daily resolution, could be adapted
    else:
        raise ValueError("Unsupported data resolution.")


def weather_data_2_txt_file(
    data_sets,
    data_var_specs,
    data_format,
    data_resolution,
    deims_id,
    coordinates,
    months_list,
):
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
    # elif data_resolution == "daily": ...
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

            # option for daily values not fully developed yet
            #
            # problem seems that taking maximum value per day includes the value at 0:00
            # which is the accumulated value of the previous day for prec. and ssr
            # and excludes the value at day + 1 00:00, which should be the correct accumulated
            # value for that day
            #
            # also: not fully clear, how mean temperature is calculated
            # should it include the value at day + 1 00:00 too?
            # or maybe the average of day 00:00 and day + 1 00:00 as one of 24 contributions?
            # because taking day 00:00 01:00 .. 23:00 seems a slight bias?
            #
            # implemented manually from hourly data above
            # download of daily data seems slow anyways, also needs seperate download requests for each variable...
            #
            # elif data_resolution == "daily":
            # varData[var_name] = ds.variables[var_name]
            # # temperature = ds.variables["t2m"]
            # # surfaceSolarRadiationDownwards = ds.variables["ssrd"]
            # # surfaceNetSolarRadiation = ds.variables["ssr"]

            # df = pd.DataFrame(
            #     {
            #         "time": [t.isoformat() for t in times],
            #         var_name: totalPrecipitation[:].flatten()
            #         * 24,  # Conversion from daily mean to daily sum.
            #         # "Temperature[degC]": temperature[:].flatten()
            #         # - 273.15,  # conversion from K to C
            #     }
            # )

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

    # Save DataFrame to CSV
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
    df_collect.to_csv(file_name, sep="\t", index=False, float_format="%.6f")
    print(f"Text file with {data_resolution} resolution prepared.")

    # ToDo: introduce separation here and make the part below an own function, to be used in a GM-specific BB, not general BB

    # Conversion to daily data as needed for Grassmind
    if data_resolution == "hourly":
        # Dates (omit last entry from last Day + 1 at 00:00)
        time_data = df_collect["time"][:-24:24].str.split("T").str[0]
        time_data = time_data.values

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

        # Correct negative PET values to 0
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

        # Save DataFrame to CSV, FileName with DEIMS.iD if existing
        file_name = construct_data_file_name(
            "weatherDataPrepared",
            data_set,
            "daily",
            ".txt",
            deims_id if deims_id is not None else coordinates,
            f"{months_list[0][0]:04d}-{months_list[0][1]:02d}",
            f"{months_list[-2][0]:04d}-{months_list[-2][1]:02d}",
            ["Weather"],
        )
        df_collect.to_csv(file_name, sep="\t", index=False, float_format="%.6f")
        print("Grassmind input text file with daily resolution prepared.")
