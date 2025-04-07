"""
Module Name: test_get_weather_data.py
Description: Test get_weather_data functions of copernicus building block.

Note: 'download_weather_data' function for download from CDS API not tested here.

Developed in the BioDT project by Thomas Banitz (UFZ) with contributions by Franziska Taubert (UFZ)
and Tuomas Rossi (CSC).

Copyright (C) 2025
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

import glob
import os
import shutil
from pathlib import Path

import pytest

from copernicus.data_processing import DATA_VAR_SPECS
from copernicus.get_weather_data import (
    configure_data_requests,
    construct_months_list,
    construct_request,
    get_area_coordinates,
    weather_data_to_txt_file,
)
from copernicus.utils import construct_weather_data_file_name


def test_construct_months_list():
    """Test months list construction."""
    target_list = [
        (2019, "12"),
        (2020, "01-04"),
        (2020, "05-08"),
        (2020, "09-12"),
        (2021, "01-04"),
        (2021, "05-08"),
        (2021, "09-12"),
        (2022, "01"),
    ]

    assert construct_months_list([2020, 2021]) == target_list

    # Months correction if more than one year
    assert construct_months_list([2020, 2021], [3, 4, 5]) == target_list

    # Single months if only one year
    assert construct_months_list(2020, [3, 4, 5]) == [
        (2020, "02"),
        (2020, "03"),
        (2020, "04"),
        (2020, "05"),
        (2020, "06"),
    ]

    with pytest.raises(ValueError):
        construct_months_list(2020, [3, 5, 7])  # months not consecutive

    with pytest.raises(ValueError):
        construct_months_list(2020, [3, 4, 5, 13])  # 13 is invalid month

    with pytest.raises(ValueError):
        construct_months_list(2020, [3, 4, 5, 0])  # 0 is invalid month

    with pytest.raises(ValueError):
        construct_months_list([2020, 2022])  # years not consecutive


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


def test_construct_request():
    """Test request construction."""
    coordinates = {"lat_start": 1.1, "lat_end": 1.2, "lon_start": 7.8, "lon_end": 7.8}
    year = 2020
    month_str = "01-04"
    variables = ["var_name1", "var_name2"]
    target_request = {
        "variable": variables,
        "product_type": "reanalysis",
        "grid": "0.1/0.1",
        "area": [1.1, 7.8, 1.2, 7.8],
        "data_format": "grib",
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "year": "2020",
        "month": ["01", "02", "03", "04"],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "download_format": "unarchived",
    }

    assert construct_request(coordinates, year, month_str, variables) == target_request

    target_request["data_format"] = "netcdf"

    assert (
        construct_request(coordinates, year, month_str, variables, data_format="netcdf")
        == target_request
    )


def test_configure_data_requests():
    """Test data request configuration."""
    test_specs = {
        "var1": {
            "long_name": "long_name1",
        },
        "var2": {
            "long_name": "long_name2",
        },
    }
    coordinates = {"lat_start": 1.1, "lat_end": 1.2, "lon_start": 7.8, "lon_end": 7.8}
    months_list = [(2020, "01-04"), (2020, "05")]

    def get_target_request(
        coordinates, months_list, *, data_format="grib", precision=6
    ):
        """Helper function to define target requests"""
        target_requests = []

        for year, month_str in months_list:
            request = construct_request(
                coordinates,
                year,
                month_str,
                ["long_name1", "long_name2"],
                data_format=data_format,
            )
            file_name = construct_weather_data_file_name(
                coordinates,
                folder="weatherDataRaw",
                data_format=data_format,
                time_specifier=f"{year}_{month_str}",
                data_specifier="hourly",
                precision=precision,
            )  # construct_weather_data_file_name tested in test_utils.py
            target_requests.append((request, file_name))

        return target_requests

    assert configure_data_requests(
        test_specs, coordinates, months_list
    ) == get_target_request(coordinates, months_list)

    # Test with different data format
    assert configure_data_requests(
        test_specs, coordinates, months_list, data_format="netcdf"
    ) == get_target_request(coordinates, months_list, data_format="netcdf")

    # Test with different precision
    assert configure_data_requests(
        test_specs, coordinates, months_list, coordinate_digits=3
    ) == get_target_request(coordinates, months_list, precision=3)

    # Test invalid data format
    with pytest.raises(ValueError):
        configure_data_requests(
            test_specs, coordinates, months_list, data_format="invalid_format"
        )


def test_weather_data_to_txt_file():
    """
    Test weather data to text file conversion.

    Notes:
      This test uses DATA_VAR_SPECS from copernicus.data_processing module.
      This test uses reference files created with the same function for result comparison.
    """
    coordinates = {"lat": 51.3919, "lon": 11.8787}
    months_list = construct_months_list([1998])

    # Copy each file to "weatherDataRaw" directory, remember if it was empty
    os.makedirs("weatherDataRaw", exist_ok=True)
    remove_folder = not any(Path("weatherDataRaw").iterdir())

    for file_path in glob.glob(os.path.join("weatherDataTestFiles", "*__hourly.nc")):
        shutil.copy(file_path, "weatherDataRaw")

    for file_path in glob.glob(os.path.join("weatherDataTestFiles", "*__hourly.grib")):
        shutil.copy(file_path, "weatherDataRaw")

    def compare_file_contents(example_type):
        """Helper function to compare contents of generated files with expected contents."""
        raw_example_string = f"example_{example_type}__"

        for file_path in glob.glob(
            os.path.join("weatherDataTestFiles", raw_example_string + "*.txt")
        ):
            with open(file_path, "r") as file:
                expected_content = file.read()  # .replace("\r\n", "\n")

            generated_file_path = file_path.replace(raw_example_string, "")
            with open(generated_file_path, "r") as file:
                generated_content = file.read()  # .replace("\r\n", "\n")

            assert generated_content == expected_content

            # Delete generated file
            os.remove(generated_file_path)

    # Convert raw data (GRIB, area location) to text files
    weather_data_to_txt_file(
        DATA_VAR_SPECS,
        coordinates,
        months_list,
        area_coordinates=get_area_coordinates([coordinates]),
        coordinate_digits=1,
        target_folder="weatherDataTestFiles",
    )
    compare_file_contents("grib_area")

    # # Convert raw data (NetCDF, area location) to text files
    # weather_data_to_txt_file(
    #     DATA_VAR_SPECS,
    #     coordinates,
    #     months_list,
    #     area_coordinates=get_area_coordinates([coordinates]),
    #     coordinate_digits=1,
    #     target_folder="weatherDataTestFiles",
    #     data_format="netcdf",
    # )
    # compare_file_contents("netcdf_area")

    # Convert raw data (NetCDF, point location) to text files
    weather_data_to_txt_file(
        DATA_VAR_SPECS,
        coordinates,
        months_list,
        target_folder="weatherDataTestFiles",
        data_format="netcdf",
    )
    compare_file_contents("netcdf_point")

    # Clean up
    if remove_folder:
        shutil.rmtree("weatherDataRaw")

    file_path = glob.glob(
        os.path.join(
            "weatherDataTestFiles", "lat*__hourly__grib__data_query_protocol.txt"
        )
    )  # GRIB protocol file created, but excluded from comparison, as time_stamps differ

    assert len(file_path) == 1  # must be exactly one file

    os.remove(file_path[0])
