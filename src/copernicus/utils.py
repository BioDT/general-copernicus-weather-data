"""
Module Name: utils.py
Description: Utility functions for copernicus building block.

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

import csv
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from suntime import Sun, SunTimeException
from timezonefinder import TimezoneFinder as tzf


def is_dict_of_2_floats(variable):
    """
    Check if 'variable' is a dictionary of two float values.

    Parameters:
        variable (dict): Input dictionary to check.

    Returns:
        bool: True if 'variable' is a valid dictionary, False otherwise.
    """
    if isinstance(variable, dict) and len(variable) == 2:
        for key, value in variable.items():
            if not isinstance(value, float):
                return False
        return True
    return False


def is_leap_year(year):
    """
    Check if a given year is a leap year.

    Parameters:
        year (int): The year.

    Returns:
        bool: True if the year is a leap year, False otherwise.
    """
    # A year is a leap year if it is divisible by 4,
    # except for years that are divisible by 100 but not by 400
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def get_days_in_month(year, month):
    """
    Get the number of days in a month for a given year, considering leap years.

    Parameters:
        year (int): The year.
        month (int): The month.

    Returns:
        int: Number of days in the specified month.
    """
    # Define the number of days in each month, considering leap years
    days_in_month = {
        1: 31,  # January
        2: 29 if is_leap_year(year) else 28,  # February
        3: 31,  # March
        4: 30,  # April
        5: 31,  # May
        6: 30,  # June
        7: 31,  # July
        8: 31,  # August
        9: 30,  # September
        10: 31,  # October
        11: 30,  # November
        12: 31,  # December
    }

    # Check if the provided month is valid
    if month not in days_in_month:
        raise ValueError(
            "Invalid month value. Please provide a month between 1 and 12."
        )

    return days_in_month[month]


def generate_day_values(year, month):
    """
    Generate day value strings for a given year and month.

    Parameters:
        year (int): The year.
        month (int): The month.

    Returns:
        list: List of day value strings.
    """

    return [f"{i:02}" for i in range(1, get_days_in_month(year, month) + 1)]


def format_offset(offset_seconds, add_utc=True):
    """
    Format a time zone offset in hours and minutes.

    Parameters:
        offset_seconds (int): Time zone offset in seconds.
        add_utc (bool): Inlude 'UTC' in result string (default is True).

    Returns:
        str: Formatted time zone offset string (e.g. 'UTC+02:00').
    """
    hours, remainder = divmod(abs(offset_seconds), 3600)
    minutes = remainder // 60
    sign = "+" if offset_seconds >= 0 else "-"
    offset_str = f"{sign}{hours:02}:{minutes:02}"

    if add_utc:
        return "UTC" + offset_str

    return offset_str


def get_time_zone(coordinates, *, return_as_offset=False):
    """
    Get the time zone for a given set of coordinates.

    Parameters:
        coordinates (dict): Coordinates with 'lat' and 'lon'.
        return_as_string (bool): Return time zone as formatted string (e.g. 'UTC+02:00', default is False).

    Returns:
        pytz.timezone or str:
            time zone as pytz.timezone object (if return_as_string is False).
            time zone as formatted string (if return_As_string is True).
    """
    tz_loc = tzf().timezone_at(lat=coordinates["lat"], lng=coordinates["lon"])

    if tz_loc:
        tz = pytz.timezone(tz_loc)

        if return_as_offset:
            ref_date = datetime(
                2021, 1, 1
            )  # just any winter day to avoid daylight saving time
            offset = tz.utcoffset(ref_date)

            return offset

        return tz

    raise ValueError("Time zone not found.")


def get_day_length(coordinates, date_iterable):
    """
    Get day length in hours for a given location and list of dates.

    Parameters:
        coordinates (dict): Coordinates dictionary with 'lat' and 'lon' keys.
        date_iterable (iterable): Iterable of date strings.

    Returns:
        list: List of day lengths in hours for the specified dates.
    """
    sun = Sun(coordinates["lat"], coordinates["lon"])
    day_lengths = []

    for date_str in date_iterable:
        try:
            year, month, day = map(int, date_str.split("-"))
            day_date = date(year, month, day)
            sunrise = sun.get_local_sunrise_time(day_date)
            sunset = sun.get_local_sunset_time(day_date)
            day_length = sunset - sunrise
            day_lengths.append(day_length.total_seconds() / 3600)
        except SunTimeException as e:
            # Error handling (no sunset or sunrise on given location)
            print(f"Error: {e}.")
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
        file_suffix = ".nc"
    elif data_format == "grib":
        file_suffix = ".grib"
    elif data_format == "txt":
        file_suffix = ".txt"
    else:
        raise ValueError("Unsupported data format.")

    return file_suffix


def list_to_file(list_to_write, column_names, file_name):
    """
    Write a list of tuples to a text file (tab-separated) or csv file (;-separated) or an Excel file.

    Parameters:
        list_to_write (list): List of strings or tuples or dictionaries to be written to the file.
        column_names (list): List of column names (strings).
        file_name (str or Path): Path of output file (suffix determines file type).
    """
    # Convert string entries to single item tuples
    list_to_write = [
        (entry,) if isinstance(entry, str) else entry for entry in list_to_write
    ]

    # Check if list_to_write contains dictionaries
    if isinstance(list_to_write[0], dict):
        # Convert dictionaries to lists of values based on column_names
        list_to_write = [
            [entry.get(col, "") for col in column_names] for entry in list_to_write
        ]
    # Check if all tuples in list have the same length as the column_names list
    elif not all(len(entry) == len(column_names) for entry in list_to_write):
        print(
            f"Error: All tuples in the list must have {len(column_names)} entries (same as column_names)."
        )

        return

    file_path = Path(file_name)
    file_suffix = file_path.suffix.lower()

    # Create data directory if missing
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    if file_suffix in [".txt", ".csv"]:
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = (
                csv.writer(file, delimiter="\t")
                if file_suffix == ".txt"
                else csv.writer(file, delimiter=";")
            )
            header = column_names
            writer.writerow(header)  # Header row

            for entry in list_to_write:
                writer.writerow(entry)
    elif file_suffix == ".xlsx":
        df = pd.DataFrame(list_to_write, columns=column_names)
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(
            "Unsupported file format. Supported formats are '.txt', '.csv' and '.xlsx'."
        )

    print(f"List written to file '{file_name}'.")


def construct_weather_data_file_name(
    coordinates,
    *,
    folder="weatherDataFolder",
    data_format="txt",
    time_specifier="timeRange",
    data_specifier="noInfo",
):
    """
    Construct data file name and create folder if missing.

    Parameters:
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        folder (str or Path): Folder where the data file will be stored (default is 'weatherDataFolder').
        data_format (str): Data format ('netcdf', 'grib' or 'txt', default is 'txt').
        time_specifier (str): Time range specifier (e.g. '1995-03', default is 'timeRange')
        data_specifier (str): Data specifier (e.g. 'hourly', 'weather', default is 'noInfo').

    Returns:
        Path: Constructed data file name as a Path object.
    """
    # Get folder with path appropriate for different operating systems, create folder if missing
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if "lat" in coordinates and "lon" in coordinates:
        formatted_lat = f"lat{coordinates['lat']:.6f}"
        formatted_lon = f"lon{coordinates['lon']:.6f}"
        file_suffix = get_file_suffix(data_format)
        file_name = (
            folder
            / f"{formatted_lat}_{formatted_lon}__{time_specifier}__{data_specifier}{file_suffix}"
        )
    else:
        raise ValueError(
            "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
        )

    return file_name
