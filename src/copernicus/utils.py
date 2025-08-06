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
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import paramiko
import requests
from astral import LocationInfo
from astral.sun import sunrise, sunset
from dotenv import dotenv_values
from timezonefinder import TimezoneFinder as tzf

from copernicus.logger_config import logger

# # will be "https://opendap.biodt.eu/..."
OPENDAP_ROOT = "http://opendap.biodt.eu/grasslands-pdt/"


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
                        f"Timezone offset varies among years. Using final year ({years[-1]}, offset: {format_offset(offset)})."
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
            sunrise_time = sunrise(location.observer, date=day_date, tzinfo=time_zone)
            sunset_time = sunset(location.observer, date=day_date, tzinfo=time_zone)
            day_length = sunset_time - sunrise_time
            day_lengths.append(day_length.total_seconds() / 3600)
        except Exception as e:
            # Error handling (no sunset or sunrise on given location)
            logger.error(f"Error calculating day length for {date_str} ({e}).")
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
            try:
                raise ValueError(
                    f"Latitude range not correctly defined. Start value ({coordinates['lat_start']}) "
                    f"must be not higher than end value ({coordinates['lat_end']})."
                )
            except ValueError as e:
                logger.error(e)
                raise

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


def download_file_opendap(
    file_name,
    opendap_folder,
    target_folder,
    *,
    new_file_name=None,
    attempts=5,
    delay=2,
    warn_not_found=True,
):
    """
    Download a file from OPeNDAP server 'grasslands-pdt'.

    Parameters:
        file_name (str): Name of file to download.
        opendap_folder (str): Folder where file is expected on OPeNDAP server.
        target_folder (str): Local folder where file will be saved.
        new_file_name (str): New name for downloaded file (default is None, file_name will be used).
        attempts (int): Number of attempts to download the file (default is 5).
        delay (int): Number of seconds to wait between attempts (default is 2).
        warn_not_found (bool): Warn if file not found on OPeNDAP server (default is True).

    Returns:
        None
    """
    url = f"{OPENDAP_ROOT}{opendap_folder}/{file_name}"
    logger.info(f"Trying to download '{url}' ...")

    while attempts > 0:
        try:
            response = requests.get(url)

            # # Variant with authentication using OPeNDAP credentials from .env file.
            # dotenv_config = dotenv_values(".env")
            # session = requests.Session()
            # session.auth = (dotenv_config["opendap_user"], dotenv_config["opendap_pw"])
            # response = session.get(url)

            if response.status_code == 200:
                if not new_file_name:
                    new_file_name = file_name

                target_file = target_folder / new_file_name
                Path(target_file).parent.mkdir(parents=True, exist_ok=True)

                with open(target_file, "wb") as file:
                    file.write(response.content)

                logger.info(f"File downloaded successfully to '{target_file}'.")
                return
            elif response.status_code == 404:
                if warn_not_found:
                    logger.warning(f"File '{file_name}' not found on OPeNDAP server.")
                else:
                    logger.info(f"File '{file_name}' not found on OPeNDAP server.")
                return
            else:
                attempts -= 1

                if attempts > 0:
                    time.sleep(delay)
        except requests.ConnectionError:
            attempts -= 1

            if attempts > 0:
                time.sleep(delay)

    logger.warning(f"File '{file_name}' download failed repeatedly.")


def upload_file_opendap(file_name, opendap_folder, *, new_file_name=None):
    """
    Upload a file to OPeNDAP server 'grasslands-pdt' using SFTP.

    Parameters:
        file_name (Path): Path of the file to upload.
        opendap_folder (str): Folder where file will be uploaded on OPeNDAP server.
        new_file_name (str): New name for uploaded file (default is None, file_name will be used).
    """
    if file_name.is_file():
        if new_file_name is None:
            new_file_name = file_name.name

        # Get IP, credentials and port from environment variables
        dotenv_config = dotenv_values(".env")

        if (
            "FTP_SERVER_IP" in dotenv_config
            and "FTP_LOGIN_USER" in dotenv_config
            and "FTP_LOGIN_PASSWORD" in dotenv_config
        ):
            ftp_ip = dotenv_config["FTP_SERVER_IP"]
            ftp_user = dotenv_config["FTP_LOGIN_USER"]
            ftp_password = dotenv_config["FTP_LOGIN_PASSWORD"]
            ftp_port = (
                int(dotenv_config["FTP_CONNECT_PORT"])
                if "FTP_CONNECT_PORT" in dotenv_config
                else 22  # default SFTP port
            )

            try:
                # Connect to SFTP server
                transport = paramiko.Transport((ftp_ip, ftp_port))
                transport.connect(username=ftp_user, password=ftp_password)
                sftp = paramiko.SFTPClient.from_transport(transport)

                # Define remote path (use what comes after the IP in OPeNDAP_ROOT)
                opendap_root_relative = OPENDAP_ROOT.partition("://")[2].partition("/")[
                    2
                ]
                remote_path = (
                    f"/{opendap_root_relative}{opendap_folder}/{new_file_name}"
                )

                # Check if folder exists, if not create it
                try:
                    sftp.stat(f"/{opendap_root_relative}{opendap_folder}")
                except FileNotFoundError:
                    sftp.mkdir(f"/{opendap_root_relative}{opendap_folder}")
                    logger.info(
                        f"Folder '{opendap_folder}' created on OPeNDAP server '{OPENDAP_ROOT}'."
                    )

                # Upload file
                sftp.put(str(file_name), remote_path)
                logger.info(
                    f"File '{file_name}' uploaded successfully to '{remote_path}'."
                )

                # Close connections
                sftp.close()
                transport.close()

            except Exception as e:
                logger.error(f"OPeNDAP upload failed for file '{file_name}' ({e}).")

        else:
            logger.warning(
                "OPeNDAP upload skipped: Valid FTP credentials not available in .env file."
            )
    else:
        logger.warning(f"OPeNDAP upload skipped: File '{file_name}' not found.")


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
    """
    if not isinstance(resolution, (int, float)) or resolution < 0:
        try:
            raise ValueError("Resolution must be a number greater than or equal to 0!")
        except ValueError as e:
            logger.error(e)
            raise

    if map_to_grid and resolution not in [0.1, 0.25]:
        try:
            raise ValueError("Grid resolution must be 0.1 or 0.25 degrees!")
        except ValueError as e:
            logger.error(e)
            raise

    # Check if each entry in the list has 'lat' and 'lon' keys
    if not all(
        all(key in coordinates for key in ["lat", "lon"])
        for coordinates in coordinates_list
    ):
        try:
            raise ValueError(
                "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
            )
        except ValueError as e:
            logger.error(e)
            raise

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
        # Area around the coordinates, extended to the nearest grid points, ensure minimum width is resolution in both directions
        round_factor = round(1 / resolution)
        lat_start = round(
            np.floor(min(lat_list) * round_factor) / round_factor, digits_required
        )
        lat_end = round(
            max(
                lat_start + resolution,
                np.ceil(max(lat_list) * round_factor) / round_factor,
            ),
            digits_required,
        )
        lon_start = round(
            np.floor(min(lon_list) * round_factor) / round_factor, digits_required
        )
        lon_end = round(
            max(
                lon_start + resolution,
                np.ceil(max(lon_list) * round_factor) / round_factor,
            ),
            digits_required,
        )
        area_coordinates = {
            "lat_start": lat_start,
            "lat_end": lat_end,
            "lon_start": lon_start,
            "lon_end": lon_end,
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
