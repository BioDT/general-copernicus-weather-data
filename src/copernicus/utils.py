"""
Module Name: utils.py
Description: Utility functions for copernicus building block.

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
"""

import calendar
import csv
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from timezonefinder import TimezoneFinder as tzf

from copernicus.logger_config import logger


def get_days_in_year(year):
    """
    Get the number of days in a year, considering leap years.

    Parameters:
        year (int): The year.

    Returns:
        int: Number of days in the specified year.
    """
    return 366 if calendar.isleap(year) else 365


def get_days_in_month(year, month):
    """
    Get the number of days in a month for a given year, considering leap years.

    Parameters:
        year (int): The year.
        month (int): The month.

    Returns:
        int: Number of days in the specified month.
    """
    return calendar.monthrange(year, month)[1]


def format_month_str(month_str):
    """
    Generate month values as strings for a given 'month_str'.

    Parameters:
        month_str (str): String to specify the month(s), can be one (e.g. '03')
            or a range of months (e.g. '01-04').

    Returns:
        list of str: List of month(s) as strings.
    """
    # Check if month_str contains a comma, raise error if so
    if "," in month_str:
        try:
            raise ValueError(
                f"Comma not supported in 'month_str' ({month_str}). Use hyphen for month range."
            )
        except ValueError as e:
            logger.error(e)
            raise

    month_items = month_str.split("-")

    if len(month_items) == 1:
        month_formatted = [f"{int(month_items[0]):02}"]
    elif len(month_items) == 2:
        month_formatted = [
            f"{m:02}" for m in range(int(month_items[0]), int(month_items[1]) + 1)
        ]
    else:
        try:
            raise ValueError(f"Wrong format of 'month_str' ({month_str})!")
        except ValueError as e:
            logger.error(e)
            raise

    return month_formatted


def format_offset(offset, add_utc=True):
    """
    Format a time zone offset in hours and minutes.

    Parameters:
        offset (timezone.utcoffset): Time zone offset to UTC.
        add_utc (bool): Inlude 'UTC' in result string (default is True).

    Returns:
        str: Formatted time zone offset string (e.g. 'UTC+02:00').
    """
    hours, remainder = divmod(abs(offset.total_seconds()), 3600)
    minutes = remainder // 60
    sign = "+" if offset.total_seconds() >= 0 else "-"
    offset_str = f"{sign}{int(hours):02}:{int(minutes):02}"

    if add_utc:
        return "UTC" + offset_str

    return offset_str


def get_time_zone(coordinates, *, return_as_offset=False, years=[2021]):
    """
    Get the time zone for a given set of coordinates.

    Parameters:
        coordinates (dict): Coordinates with 'lat' and 'lon'.
        return_as_offset (bool): Return time zone as offset to UTC (default is False).
        years (iterable): Years for obtaining time zone offset to UTC (default is [2021]).

    Returns:
        zoneinfo.ZoneInfo or zoneinfo.ZoneInfo.utcoffset:
            time zone as zoneinfo.ZoneInfo object (if return_as_offset is False).
            time zone as zoneinfo.ZoneInfo.utcoffset (if return_as_offset is True).
    """
    tz_loc = tzf().timezone_at(lat=coordinates["lat"], lng=coordinates["lon"])

    if tz_loc:
        tz = ZoneInfo(tz_loc)

        if return_as_offset:
            # Offest for last year, use any winter day to avoid daylight saving time
            offset = tz.utcoffset(datetime(years[-1], 1, 1))

            # Check if offset differs for other years
            for year in years[:-1]:
                offset_check = tz.utcoffset(datetime(year, 1, 1))

                if offset != offset_check:
                    logger.warning(
                        f"Timezone offset varies among years! Using final year ({years[-1]}, offset: {format_offset(offset)})."
                    )
                    return offset

            return offset

        return tz

    try:
        raise ValueError("Time zone not found.")
    except ValueError as e:
        logger.error(e)
        raise


def get_day_length(coordinates, dates):
    """
    Get day length in hours for a given location and list of dates.

    Parameters:
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon' keys.
        dates (iterable): Iterable of date strings.

    Returns:
        list: List of day lengths in hours for the specified dates.
    """
    time_zone = get_time_zone(coordinates)
    location = LocationInfo(
        "name", "region", time_zone.key, coordinates["lat"], coordinates["lon"]
    )
    day_lengths = []

    for date_str in dates:
        try:
            year, month, day = map(int, date_str.split("-"))
            day_date = date(year, month, day)
            sun_local = sun(location.observer, date=day_date, tzinfo=time_zone)
            day_length = sun_local["sunset"] - sun_local["sunrise"]
            day_lengths.append(day_length.total_seconds() / 3600)
        except Exception as e:
            # Error handling (no sunset or sunrise on given location)
            logger.error(e)
            day_lengths.append(0)

    return np.array(day_lengths)


def get_file_suffix(data_format):
    """
    Determine data file ending based on data format.

    Parameters:
        data_format (str): Data format ('netcdf' or 'grib' or 'txt').

    Returns:
        str: Data file suffix.
    """
    if data_format == "netcdf":
        return ".nc"
    elif data_format in ["grib", "txt"]:
        return f".{data_format}"
    else:
        try:
            raise ValueError("Unsupported data format.")
        except ValueError as e:
            logger.error(e)
            raise


def list_to_file(list_to_write, file_name, *, column_names=None):
    """
    Write a list to a text file (tab-separated) or csv file (;-separated) or an Excel file.

    Parameters:
        list_to_write (list): List of strings or tuples or dictionaries to be written to the file.
        file_name (str or Path): Path of output file (suffix determines file type).
        column_names (list): List of column names (strings) to write as header line (default is None).
    """
    # Convert string entries to single item tuples
    list_to_write = [
        (entry,) if isinstance(entry, str) else entry for entry in list_to_write
    ]

    # Check if list_to_write contains dictionaries
    if any(isinstance(entry, dict) for entry in list_to_write):
        if not all(isinstance(entry, dict) for entry in list_to_write):
            try:
                raise ValueError(
                    "All entries in the list must be either dictionaries or not dictionaries. Cannot write list with mixed types."
                )
            except ValueError as e:
                logger.error(e)
                raise

        # Get column names from dictionaries (keys of first dictionary) if not provided
        if not column_names:
            logger.warning(
                "No column names provided. Using keys from first dictionary in list to obtain column names."
            )
            column_names = list(list_to_write[0].keys())

        if not column_names:
            try:
                raise ValueError(
                    "No column names provided and no keys found in dictionaries to obtain column names. Cannot write list."
                )
            except ValueError as e:
                logger.error(e)
                raise

        # Convert dictionaries to lists of values based on column_names, empty string if key not found
        list_to_write = [
            [entry.get(col, "") for col in column_names] for entry in list_to_write
        ]
    else:
        # Check if all tuples in list have the same length as the column_names list
        if column_names and not all(
            len(entry) == len(column_names) for entry in list_to_write
        ):
            try:
                raise ValueError(
                    "All entries in the list must have the same length as the column names list."
                )
            except ValueError as e:
                logger.error(e)
                raise

    file_path = Path(file_name)
    file_suffix = file_path.suffix.lower()

    # Create data directory if missing
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    if file_suffix in [".txt", ".csv"]:
        with open(
            file_path, "w", newline="", encoding="utf-8", errors="replace"
        ) as file:
            writer = (
                csv.writer(file, delimiter="\t")
                if file_suffix == ".txt"
                else csv.writer(file, delimiter=";")
            )

            if column_names:
                header = column_names
                writer.writerow(header)  # Header row

            for entry in list_to_write:
                writer.writerow(entry)
    elif file_suffix == ".xlsx":
        df = pd.DataFrame(list_to_write, columns=column_names)
        df.to_excel(file_path, index=False)
    else:
        try:
            raise ValueError(
                "Unsupported file format. Supported formats are '.txt', '.csv' and '.xlsx'."
            )
        except ValueError as e:
            logger.error(e)
            raise

    logger.info(f"List written to file '{file_name}'.")


def construct_weather_data_file_name(
    coordinates,
    *,
    folder="weatherDataFolder",
    data_format="txt",
    time_specifier="timeRange",
    data_specifier="noInfo",
    precision=6,
):
    """
    Construct weather data file name and create folder if missing.

    Parameters:
        coordinates (dict): Dictionary either with 'lat_start', 'lat_end', 'lon_start', and 'lon_end' keys
            for area ({'lat_start': float, 'lat_end': float, 'lon_start': float, 'lon_end': float})
            or with 'lat' and 'lon' keys for a single point ({'lat': float, 'lon': float}).
        folder (str or Path): Folder of weather data file (default is 'weatherDataFolder').
        data_format (str): Data format ('netcdf', 'grib' or 'txt', default is 'txt').
        time_specifier (str): Time range specifier (e.g. '1995-03', default is 'timeRange')
        data_specifier (str): Data specifier (e.g. 'hourly', 'weather', default is 'noInfo').
        precision (int): Precision for latitude and longitude values (default is 6).

    Returns:
        Path: Constructed data file name as a Path object.
    """
    # Get folder with path appropriate for different operating systems, create folder if missing
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Coordinates for area
    if all(
        key in coordinates for key in ["lat_start", "lat_end", "lon_start", "lon_end"]
    ):
        if coordinates["lat_start"] == coordinates["lat_end"]:
            precision = 6
            formatted_lat = f"lat{coordinates['lat_start']:.{precision}f}"
        elif coordinates["lat_start"] < coordinates["lat_end"]:
            formatted_lat = f"lat{coordinates['lat_start']:.{precision}f}_{coordinates['lat_end']:.{precision}f}"
        else:
            raise ValueError(
                f"Latitude range not correctly defined. Start value ({coordinates['lat_start']}) "
                f"must be not higher than end value ({coordinates['lat_end']})."
            )

        if coordinates["lon_start"] == coordinates["lon_end"]:
            formatted_lon = f"lon{coordinates['lon_start']:.{precision}f}"
        elif coordinates["lon_start"] < coordinates["lon_end"]:
            formatted_lon = f"lon{coordinates['lon_start']:.{precision}f}_{coordinates['lon_end']:.{precision}f}"
        else:
            try:
                raise ValueError(
                    f"Longitude range not correctly defined. Start value ({coordinates['lon_start']}) "
                    f"must be not higher than end value ({coordinates['lon_end']})."
                )
            except ValueError as e:
                logger.error(e)
                raise
    # Coordinates for single point
    elif all(key in coordinates for key in ["lat", "lon"]):
        formatted_lat = f"lat{coordinates['lat']:.{precision}f}"
        formatted_lon = f"lon{coordinates['lon']:.{precision}f}"
    else:
        try:
            raise ValueError(
                "Coordinates not correctly defined. Please provide as dictionary, either for area "
                "({'lat_start': float, 'lat_end': float, 'lon_start': float, 'lon_end': float}) "
                "or for a single point ({'lat': float, 'lon': float})."
            )
        except ValueError as e:
            logger.error(e)
            raise

    file_suffix = get_file_suffix(data_format)
    file_name = (
        folder
        / f"{formatted_lat}_{formatted_lon}__{time_specifier}__{data_specifier}{file_suffix}"
    )

    return file_name
