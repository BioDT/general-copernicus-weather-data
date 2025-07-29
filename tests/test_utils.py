"""
Module Name: test_utils.py
Description: Test utility functions for copernicus building block.

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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from dotenv import dotenv_values

from copernicus.utils import (
    OPENDAP_ROOT,
    construct_weather_data_file_name,
    download_file_opendap,
    format_month_str,
    format_offset,
    get_area_coordinates,
    get_day_length,
    get_days_in_month,
    get_days_in_year,
    get_file_suffix,
    get_time_zone,
    list_to_file,
    upload_file_opendap,
)


def test_format_month_str():
    """Test formatting of month strings."""
    assert format_month_str("03") == ["03"]
    assert format_month_str("3") == ["03"]
    assert format_month_str("003") == ["03"]
    assert format_month_str("01-04") == ["01", "02", "03", "04"]
    assert format_month_str("1 -  004") == ["01", "02", "03", "04"]

    with pytest.raises(ValueError):
        format_month_str("01-04-07")  # Invalid format
        format_month_str("2, 3, 4")  # Invalid format


def test_get_days_in_year():
    """Test the number of days in a year for both leap and non-leap years."""
    # Test non-leap year
    assert get_days_in_year(2021) == 365
    assert get_days_in_year(2100) == 365

    # Test leap year
    assert get_days_in_year(2020) == 366


def test_get_days_in_month():
    """Test the number of days in each month for both leap and non-leap years."""
    # Test valid months
    assert get_days_in_month(2021, 1) == 31  # January
    assert get_days_in_month(2021, 2) == 28  # February (non-leap year)
    assert get_days_in_month(2020, 2) == 29  # February (leap year)
    assert get_days_in_month(2021, 3) == 31  # March
    assert get_days_in_month(2021, 4) == 30  # April
    assert get_days_in_month(2021, 5) == 31  # May
    assert get_days_in_month(2021, 6) == 30  # June
    assert get_days_in_month(2021, 7) == 31  # July
    assert get_days_in_month(2021, 8) == 31  # August
    assert get_days_in_month(2021, 9) == 30  # September
    assert get_days_in_month(2021, 10) == 31  # October
    assert get_days_in_month(2021, 11) == 30  # November
    assert get_days_in_month(2021, 12) == 31  # December

    # Test invalid month
    with pytest.raises(ValueError):
        get_days_in_month(2021, 0)  # Invalid month (0)

    with pytest.raises(ValueError):
        get_days_in_month(2021, 13)  # Invalid month (13)


def test_get_time_zone():
    """Test retrieval of time zone information."""
    test_locations = [
        ({"lat": 60.1699, "lon": 24.9384}, "Europe/Helsinki", "+02:00", "UTC+02:00"),
        ({"lat": 51.5, "lon": 0}, "Europe/London", "+00:00", "UTC+00:00"),
        ({"lat": 40.7128, "lon": -74.0060}, "America/New_York", "-05:00", "UTC-05:00"),
    ]

    for (
        coordinates,
        expected_tz,
        expected_offset,
        expected_offset_utc,
    ) in test_locations:
        tz = get_time_zone(coordinates)
        assert tz.key == expected_tz

        offset = get_time_zone(coordinates, return_as_offset=True)
        assert format_offset(offset, add_utc=False) == expected_offset
        assert format_offset(offset, add_utc=True) == expected_offset_utc


def test_get_day_length():
    """Test calculation of day length."""
    coordinates = {"lat": 51.3919, "lon": 11.8787}  # GCEF Bad Lauchstädt
    dates = ["2013-01-01", "2022-04-01", "2023-07-01"]  # Example dates
    day_lengths = get_day_length(coordinates, dates)
    assert 7.9 < day_lengths[0] < 8  # Jan 1st
    assert 12.8 < day_lengths[1] < 13  # April 1st
    assert 16.5 < day_lengths[2] < 16.6  # Jul 1st


def test_get_file_suffix():
    """Test determination of file suffix based on data format."""
    assert get_file_suffix("netcdf") == ".nc"
    assert get_file_suffix("grib") == ".grib"
    assert get_file_suffix("txt") == ".txt"

    with pytest.raises(ValueError):
        get_file_suffix("unsupported_format")  # Unsupported format


def test_list_to_file(tmp_path):
    """Test writing a list to a file."""

    # Different test lists: tuples, strings, dictionaries
    lists_to_write = [
        [("a", "b", "c"), ("d", "e", "f")],
        ["abc", "def"],
        [
            {"col1": "a", "col2": "b", "col3": "c"},
            {
                "col2": "e",
                "col3": "f",
                "colx": "x",
                "col1": "d",
            },  # Order of keys is not guaranteed, extra keys may exist
        ],
    ]
    target_strings = {
        "txt": [("a\tb\tc", "d\te\tf"), ("abc", "def"), ("a\tb\tc", "d\te\tf")],
        "csv": [("a;b;c", "d;e;f"), ("abc", "def"), ("a;b;c", "d;e;f")],
    }
    column_names_list = [["col1", "col2", "col3"], ["col1"], ["col1", "col2", "col3"]]
    column_names_strings = {
        "txt": ["col1\tcol2\tcol3", "col1", "col1\tcol2\tcol3"],
        "csv": ["col1;col2;col3", "col1", "col1;col2;col3"],
    }

    def validate_file_content(content, target_strings, *, column_names_string=None):
        if column_names_string is None:
            for index, target_string in enumerate(target_strings):
                assert content[index].strip() == target_string
        else:
            assert content[0].strip() == column_names_string

            for index, target_string in enumerate(target_strings):
                assert content[index + 1].strip() == target_string

    for index, list_to_write in enumerate(lists_to_write):
        # Test with txt file and csv files
        for suffix in ["txt", "csv"]:
            file_path = tmp_path / f"test.{suffix}"
            list_to_file(list_to_write, file_path)  # Default is without column names

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.readlines()

                if isinstance(list_to_write[0], dict):
                    # Keys of first entry are used as column names for dictionaries
                    validate_file_content(
                        content,
                        target_strings[suffix][index],
                        column_names_string=column_names_strings[suffix][index],
                    )
                else:
                    validate_file_content(content, target_strings[suffix][index])

            # Add column names
            list_to_file(
                list_to_write, file_path, column_names=column_names_list[index]
            )

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.readlines()
                validate_file_content(
                    content,
                    target_strings[suffix][index],
                    column_names_string=column_names_strings[suffix][index],
                )

        # Test with xls files
        file_path = tmp_path / "test.xlsx"
        list_to_file(list_to_write, file_path)
        content = pd.read_excel(file_path)

        if isinstance(list_to_write[0], dict):
            # use first list as target also for dictionaries
            target_list = lists_to_write[0]
            assert np.all(content.columns == column_names_list[index])
        else:
            target_list = list_to_write

        for row_index, row in enumerate(target_list):
            assert np.all(content.values[row_index] == row)

        # Add column names
        list_to_file(list_to_write, file_path, column_names=column_names_list[index])
        content = pd.read_excel(file_path)
        assert np.all(content.columns == column_names_list[index])

        for row_index, row in enumerate(target_list):
            assert np.all(content.values[row_index] == row)

    # Test with invalid column names
    with pytest.raises(ValueError):
        list_to_file([("a", "b", "c")], file_path, column_names=["col1", "col2"])


def test_construct_weather_data_file_name():
    """Test construction of weather data file name."""
    file_name = construct_weather_data_file_name(
        {"lat": 12.123456789, "lon": 99.9},
        folder="test_folder",
        data_format="netcdf",
        time_specifier="2021-06",
        data_specifier="specialInfo",
        precision=3,
    )
    assert str(file_name.as_posix()).endswith(
        "test_folder/lat12.123_lon99.900__2021-06__specialInfo.nc"
    )

    file_name = construct_weather_data_file_name(
        {"lat": 12.123456789, "lon": -12},
    )
    assert str(file_name.as_posix()).endswith(
        "weatherDataFolder/lat12.123457_lon-12.000000__timeRange__noInfo.txt"
    )

    file_name = construct_weather_data_file_name(
        {"lat_start": 12.123456789, "lat_end": 13, "lon_start": -13.89, "lon_end": -12}
    )
    assert str(file_name.as_posix()).endswith(
        "weatherDataFolder/lat12.123457_13.000000_lon-13.890000_-12.000000__timeRange__noInfo.txt"
    )

    with pytest.raises(ValueError):
        construct_weather_data_file_name(
            {
                "lat_start": 12.123456789,
                "lat_end": 10,
                "lon_start": -13.89,
                "lon_end": -12,
            }  # lat_end < lat_start
        )

    with pytest.raises(ValueError):
        construct_weather_data_file_name({"lat": 12})  # Missing key

    with pytest.raises(ValueError):
        construct_weather_data_file_name(
            {"lat_start": 12, "lon_end": -12}  # Missing keys
        )

    with pytest.raises(ValueError):
        construct_weather_data_file_name(
            {"latitude": 12, "longitude": -12}
        )  # Invalid keys

    with pytest.raises(ValueError):
        construct_weather_data_file_name(
            {"lat": 12, "lon": -12},
            data_format="any_format",  # Invalid data format
        )

    # Remove test folders if they were just created (i.e. they are empty)
    folders = ["weatherDataFolder", "test_folder"]

    for folder in folders:
        folder = Path(folder)

        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def test_upload_file_opendap(tmp_path, caplog):
    """Test upload of a file to the OPeNDAP server."""
    # Create a temporary file to upload
    file_path = tmp_path / "test_file.txt"
    test_folder = "_test_folder"
    with open(file_path, "w") as f:
        # generate some unique content, like a timestamp
        test_content = f"Test file content: {pd.Timestamp.now()}"
        f.write(test_content)

    # Upload file to OPeNDAP server, needs credentials
    with caplog.at_level("WARNING"):
        upload_file_opendap(file_path, test_folder)

    # Check if credentials allowing upload are available,
    dotenv_config = dotenv_values(".env")

    if (
        "FTP_SERVER_IP" in dotenv_config
        and "FTP_LOGIN_USER" in dotenv_config
        and "FTP_LOGIN_PASSWORD" in dotenv_config
    ):
        # Test that file exists under the URL and has the expected content
        file_url = f"{OPENDAP_ROOT}{test_folder}/{file_path.name}"
        response = requests.get(file_url)
        assert response.status_code == 200, "File not found on OPeNDAP server."
        assert response.content.decode() == test_content, "File content does not match."
    else:
        # Test that correct warning is logged when not all credentials are available
        assert (
            "OPeNDAP upload skipped: Valid FTP credentials not available in .env file."
            in caplog.text
        )


def test_download_file_opendap(tmp_path, caplog):
    """Test download of a file from the OPeNDAP server."""
    # Create a temporary file to upload
    file_name = "test_file.txt"
    test_folder = "_test_folder"
    local_file_path = tmp_path / file_name

    download_file_opendap(file_name, test_folder, tmp_path)

    # Test that the downloaded file exists and has the expected content
    assert local_file_path.exists()
    assert local_file_path.is_file()

    with open(local_file_path, "r") as f:
        downloaded_content = f.read()
        assert downloaded_content.startswith("Test file content:")

    # Test download of a non-existing file
    file_name = "non_existing_file.txt"
    local_file_path = tmp_path / file_name

    with caplog.at_level("WARNING"):
        download_file_opendap(file_name, test_folder, tmp_path)

    assert f"File '{file_name}' not found on OPeNDAP server." in caplog.text
    assert local_file_path.exists() is False, "Local file should not exist."


def test_get_area_coordinates():
    """Test area coordinates calculation."""
    assert get_area_coordinates([{"lat": 1.101, "lon": 7.8}]) == {
        "lat_start": 1.1,
        "lat_end": 1.2,
        "lon_start": 7.8,
        "lon_end": 7.8,
    }

    assert get_area_coordinates([{"lat": 1.101, "lon": 7.8}], map_to_grid=False) == {
        "lat_start": 1.001,
        "lat_end": 1.201,
        "lon_start": 7.7,
        "lon_end": 7.9,
    }

    coordinates = [
        {"lat": 1.101, "lon": 7.8},
        {"lat": 1.102, "lon": 7.9},
        {"lat": 1.103, "lon": 8.25},
    ]

    assert get_area_coordinates(coordinates) == {
        "lat_start": 1.1,
        "lat_end": 1.2,
        "lon_start": 7.8,
        "lon_end": 8.3,
    }

    assert get_area_coordinates(coordinates, resolution=0.25) == {
        "lat_start": 1,
        "lat_end": 1.25,
        "lon_start": 7.75,
        "lon_end": 8.25,
    }

    assert get_area_coordinates(coordinates, map_to_grid=False) == {
        "lat_start": 1.001,
        "lat_end": 1.203,
        "lon_start": 7.7,
        "lon_end": 8.35,
    }

    assert get_area_coordinates(coordinates, resolution=0, map_to_grid=False) == {
        "lat_start": 1.101,
        "lat_end": 1.103,
        "lon_start": 7.8,
        "lon_end": 8.25,
    }

    with pytest.raises(ValueError):
        get_area_coordinates(coordinates, resolution=-0.1)

    with pytest.raises(ValueError):
        get_area_coordinates(coordinates, resolution=None)

    with pytest.raises(ValueError):
        get_area_coordinates([{"lat": 1}])  # "lon" key missing
