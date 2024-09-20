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
"""

import datetime

import deims
import numpy as np
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

    return [str(i).zfill(2) for i in range(1, get_days_in_month(year, month) + 1)]


def format_offset(offset_seconds):
    """
    Format a time zone offset in hours and minutes.

    Parameters:
        offset_seconds (int): Time zone offset in seconds.

    Returns:
        str: Formatted time zone offset string (e.g. 'UTC+02:00').
    """
    hours, remainder = divmod(abs(offset_seconds), 3600)
    minutes = remainder // 60
    sign = "+" if offset_seconds >= 0 else "-"
    offset_str = f"UTC{sign}{hours:02}:{minutes:02}"

    return offset_str


def get_time_zone(coordinates):
    """
    Get the time zone for a given set of coordinates.

    Parameters:
        coordinates (dict): Coordinates with 'lat' and 'lon'.

    Returns:
        str: Time zone as formatted string (e.g. 'UTC+02:00').
    """
    tz_loc = tzf().timezone_at(lat=coordinates["lat"], lon=coordinates["lon"])

    if tz_loc:
        tz = pytz.timezone(tz_loc)
        ref_date = datetime.datetime(
            2021, 1, 1
        )  # Just any winter day to avoid daylight saving time
        offset = tz.utcoffset(ref_date)
        offset_str = format_offset(offset.seconds)

        return offset_str

    raise ValueError("Time zone not found.")


def get_day_length(coordinates, date_iterable):
    """
    Get day length in hours for a given location and list of dates.

    Parameters:
        coordinates (dict): Coordinates with 'lat' and 'lon'.
        date_iterable (iterable): Iterable of date strings.

    Returns:
        list: List of day lengths in hours for the specified dates.
    """
    sun = Sun(coordinates["lat"], coordinates["lon"])
    day_lengths = []

    for date_str in date_iterable:
        try:
            year, month, day = map(int, date_str.split("-"))
            date = datetime.date(year, month, day)
            sunrise = sun.get_local_sunrise_time(date)
            sunset = sun.get_local_sunset_time(date)
            day_length = sunset - sunrise
            day_lengths.append(day_length.total_seconds() / 3600)
        except SunTimeException as e:
            # Error handling (no sunset or sunrise on given location)
            print("Error: {0}.".format(e))
            day_lengths.append(0)

    return np.array(day_lengths)


def get_data_suffix(data_format):
    """
    Determine data file ending based on data format.

    Parameters:
        data_format (str): Data format ('netcdf' or 'grib').

    Returns:
        str: Data file suffix.
    """
    if data_format == "netcdf":
        data_suffix = ".nc"
    elif data_format == "grib":
        data_suffix = ".grib"
    else:
        raise ValueError("Unsupported data format")

    return data_suffix


def get_deims_coordinates(deims_id):
    """
    Get coordinates for a DEIMS.iD.

    Parameters:
        deims_id (str): DEIMS.iD.

    Returns:
        dict: Coordinates as a dictionary with 'lat' and 'lon'.
    """
    try:
        deims_gdf = deims.getSiteCoordinates(deims_id, filename=None)
        # deims_gdf = deims.getSiteBoundaries(deims_id, filename=None)  # option: collect all coordinates from deims_gdf.boundary[0] ...

        lon = deims_gdf.geometry[0].x
        lat = deims_gdf.geometry[0].y
        name = deims_gdf.name[0]
        print(f"Coordinates for DEIMS.id '{deims_id}' found ({name}).")
        print(f"Latitude: {lat}, Longitude: {lon}")

        return {
            "lat": lat,
            "lon": lon,
            "deims_id": deims_id,
            "found": True,
            "name": name,
        }
    except Exception as e:
        print(f"Error: coordinates for DEIMS.id '{deims_id}' not found ({e})!")

        return {"deims_id": deims_id, "found": False}
