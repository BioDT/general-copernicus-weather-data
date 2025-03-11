"""
Module Name: test_convert_weather_data.py
Description: Test convert_weather_data functions of copernicus building block.

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

import calendar
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astral import LocationInfo
from astral.sun import sun

from copernicus.convert_weather_data import (
    check_length_of_values_00_24,
    convert_units,
    daily_accumulated,
    daily_max_00_24,
    daily_mean_00_24,
    daily_mean_daylight,
    daily_min_00_24,
    effective_temperature,
    get_pet_thornthwaite,
    hourly_to_daily,
    monthly_mean,
    par_from_net_radiation,
)
from copernicus.data_processing import DATA_VAR_SPECS
from copernicus.utils import get_day_length, get_time_zone


def test_convert_units():
    """Test convert_units function."""
    values = np.array([0, 1000, -10])
    precision = 12  # 12 decimal places to avoid differences only due to floating point arithmetic

    assert all(
        np.round(convert_units(values, "_to_Milli"), precision)
        == np.array([0, 1000000, -10000])
    )
    assert all(
        np.round(convert_units(values, "_to_Kilo"), precision)
        == np.array([0, 1, -0.01])
    )
    assert all(
        np.round(convert_units(values, "_to_Mega"), precision)
        == np.array([0, 0.001, -0.00001])
    )

    assert all(
        np.round(convert_units(values, "Kelvin_to_Celsius"), precision)
        == np.array([-273.15, 726.85, -283.15])
    )
    assert all(
        np.round(convert_units(values, "Celsius_to_Kelvin"), precision)
        == np.array([273.15, 1273.15, 263.15])
    )

    assert all(
        np.round(convert_units(values, "d-1_to_s-1"), precision)
        == np.round(
            [
                0,
                1000 / 24 / 60 / 60,
                -10 / 24 / 60 / 60,
            ],
            precision,
        )
    )  # 1 day = 24 * 60 * 60 seconds

    with pytest.raises(ValueError):
        convert_units(values, "invalid_conversion")


def test_par_from_net_radiation():
    """Test par_from_net_radiation function."""
    precision = 12  # 12 decimal places to avoid differences only due to floating point arithmetic

    # Input unit: J/m^2/day
    radiation_input = np.array([1e6, 0, -1e6])

    # Target unit: µmol/m^2/s;
    # 1 J = 4.57 µmol, PAR fraction: 50% of net radiation, 1 day = 24 * 60 * 60 seconds
    # Negative values to be corrected to 0
    radiation_target = np.round([1e6 * 4.57 / 2 / 24 / 60 / 60, 0, 0], precision)

    assert all(
        np.round(par_from_net_radiation(radiation_input), precision) == radiation_target
    )


def test_check_length_of_values_00_24():
    """Test length check function for hourly data from 00:00 to 24:00."""
    for length in range(25, 49):
        values = np.arange(length)
        values_unchanged = values.copy()
        assert all(check_length_of_values_00_24(values) == np.arange(25))
        assert all(values == values_unchanged)

    for length in range(49, 51):
        values = np.arange(length)
        values_unchanged = values.copy()
        assert all(check_length_of_values_00_24(values) == np.arange(49))
        assert all(values == values_unchanged)

    with pytest.raises(ValueError):
        check_length_of_values_00_24(np.arange(24))


def test_daily_mean_00_24():
    """Test daily_mean function for hourly data from 00:00 to 24:00."""
    # Test values; 25 hourly values for first day,
    # 24 hourly values for other days, as first value comes from last value of previous day
    test_day_0 = [0] * 24 + [
        24
    ]  # inner values mean: 0, 1st & last values (0 & 24) mean: 12
    test_day_1 = [0] * 23 + [
        24
    ]  # inner values mean: 0, 1st & last values (24 & 24) mean: 24
    test_day_2 = [1] * 23 + [
        26
    ]  # inner values mean: 1, 1st & last values (24 & 26) mean: 25
    test_day_3 = list(range(23, 0, -1)) + [
        46
    ]  # inner values mean: 12, 1st & last values (26 & 46) mean: 36
    values = np.array(test_day_0 + test_day_1 + test_day_2 + test_day_3)

    assert all(daily_mean_00_24(values) == np.array([0.5, 1, 2, 13]))

    # Test with incomplete days, missing last value of last day
    values = values[:-1]
    values_unchanged = values.copy()

    assert all(daily_mean_00_24(values) == np.array([0.5, 1, 2]))
    assert all(values == values_unchanged)

    # Test single day
    assert daily_mean_00_24(np.array(test_day_0)) == 0.5

    with pytest.raises(ValueError):
        daily_mean_00_24(np.array(test_day_1))


def test_daily_min_00_24():
    """Test daily_min function for hourly data from 00:00 to 24:00."""
    # Test values; 25 hourly values for first day,
    # 24 hourly values for other days, as first value comes from last value of previous day
    test_day_0 = [0] * 24 + [12.345]
    test_day_1 = [0] * 24
    test_day_2 = [0] * 24
    test_day_3 = [0] + [-1] * 23
    values = np.array(test_day_0 + test_day_1 + test_day_2 + test_day_3)

    assert all(daily_min_00_24(values) == np.array([0, 0, 0, -1]))

    # Test with incomplete days, missing last value of last day
    values = values[:-1]
    values_unchanged = values.copy()

    assert all(daily_min_00_24(values) == np.array([0, 0, 0]))
    assert all(values == values_unchanged)

    # Test single day
    assert daily_min_00_24(np.array(test_day_0)) == 0

    with pytest.raises(ValueError):
        daily_min_00_24(np.array(test_day_1))


def test_daily_max_00_24():
    """Test daily_max function for hourly data from 00:00 to 24:00."""
    # Test values; 25 hourly values for first day,
    # 24 hourly values for other days, as first value comes from last value of previous day
    test_day_0 = [0] * 24 + [12.345]
    test_day_1 = [0] * 24
    test_day_2 = [0] * 24
    test_day_3 = [0] + [-1] * 23
    values = np.array(test_day_0 + test_day_1 + test_day_2 + test_day_3)

    assert all(daily_max_00_24(values) == np.array([12.345, 12.345, 0, 0]))

    # Test with incomplete days, missing last value of last day
    values = values[:-1]
    values_unchanged = values.copy()

    assert all(daily_max_00_24(values) == np.array([12.345, 12.345, 0]))
    assert all(values == values_unchanged)

    # Test single day
    assert daily_max_00_24(np.array(test_day_0)) == 12.345

    with pytest.raises(ValueError):
        daily_max_00_24(np.array(test_day_1))


def test_daily_accumulated():
    """Test daily_accumulated function."""
    test_day0 = [0] * 25
    test_day1 = list(range(1, 25))
    test_day2 = list(range(2, 49, 2))
    test_day3 = list(range(3, 73, 3))
    utc_values = np.array(test_day0 + test_day1 + test_day2 + test_day3)

    assert all(daily_accumulated(utc_values) == np.array([0, 24, 48, 72]))

    # Test all possible positive offsets, including 0
    for offset in range(15):
        values = utc_values[24 - offset : 96 - offset + 1]  # cut out 3 days with offset
        # Final accumulated values of UTC days sit in each local day's "offset" hour
        target_offset = (
            values[0 + offset] - values[0] + values[24],
            values[24 + offset] - values[24] + values[48],
            values[48 + offset] - values[48] + values[72],
        )

        assert all(daily_accumulated(values, tz_offset_hours=offset) == target_offset)

    # Test all possible negative offsets, excluding 0
    for offset in range(-12, 0):
        values = utc_values[0 - offset : 72 - offset + 1]  # cut out 3 days with offset
        # Final accumulated values of UTC days sit in each local day's "offset" hour
        target_offset = (
            values[24 + offset] - values[0] + values[24],
            values[48 + offset] - values[24] + values[48],
            values[72 + offset] - values[48] + values[72],
        )

        assert all(daily_accumulated(values, tz_offset_hours=offset) == target_offset)


def test_daily_mean_daylight():
    """Test daily_mean_daylight function."""
    # Note: unreasonable test values only, data should be continuous, not just 2 unconnected days
    #       midnight values (1000) do not get used in this example
    dates = ["2020-01-01", "2020-06-30"]
    coordinates = {"lat": 51.3919, "lon": 11.8787}
    values_hourly = np.concatenate(
        ([1000], np.arange(1, 24), [1000], np.arange(1, 24), [1000])
    )

    # Get daily mean values over period of daylight from function
    daylight_values = daily_mean_daylight(values_hourly, dates, coordinates)

    # Example target specs for the defined winter and summer day
    # Note: seconds are cut w/o rounding,
    #       day in June has daylight saving time (DST) in effect
    target_specs = {
        "2020-01-01": {
            "sunrise": "2020-01-01 08:17+0100",
            "sunset": "2020-01-01 16:13+0100",
            "start_index": 8,  # 8:00 is first hourly value to be (partly) used
            "end_index": 16,  # 16:00 is last hourly value to be (partly) used
            "start_fraction": 13 / 60,  # 13 minutes of 7:30-8:30 to be used
            "end_fraction": 43 / 60,  # 43 minutes of 15:30-16:30 to be used
        },
        "2020-06-30": {
            "sunrise": "2020-06-30 05:00+0200",
            "sunset": "2020-06-30 21:32+0200",
            "start_index": 4,  # 4:00 is first hourly value to be (partly) used, DST!
            "end_index": 21,  # 21:00 is last hourly value to be (partly) used, DST!
            "start_fraction": 30 / 60,  # 30 minutes of 3:30-4:30 to be used
            "end_fraction": 2 / 60,  # 2 minutes of 20:30-21:30 to be used
        },
    }
    precision = 12  # 12 decimal places to avoid differences only due to floating point arithmetic

    # Get test sunrise and sunset times
    time_zone = get_time_zone(coordinates)
    location = LocationInfo(
        "name", "region", time_zone.key, coordinates["lat"], coordinates["lon"]
    )

    for index, date_str in enumerate(dates):
        day_date = datetime.strptime(date_str, "%Y-%m-%d")
        sun_local = sun(location.observer, date=day_date, tzinfo=time_zone)
        specs = target_specs[date_str]

        # Test if retrieved sunrise and sunset times match target times (also obtained from astral)
        assert specs["sunrise"] == sun_local["sunrise"].strftime("%Y-%m-%d %H:%M%z")
        assert specs["sunset"] == sun_local["sunset"].strftime("%Y-%m-%d %H:%M%z")

        # Calculate expected value for daylight period
        day_values = values_hourly[24 * index : 24 * (index + 1) + 1]
        weighted_sum = (
            day_values[specs["start_index"]] * (specs["start_fraction"])
            + np.sum(values_hourly[specs["start_index"] + 1 : specs["end_index"]])
            + day_values[specs["end_index"]] * (specs["end_fraction"])
        )
        sum_of_weights = (
            specs["start_fraction"]
            + (specs["end_index"] - specs["start_index"] - 1)
            + specs["end_fraction"]
        )
        target_value = round(weighted_sum / sum_of_weights, precision)

        assert round(daylight_values[index], precision) == target_value


def test_hourly_to_daily():
    """Test hourly_to_daily function for conversion of hourly to daily data."""
    # Create data hourly dataframes, example offsets are 0, 2 and -2 hours
    dates_local = {"+00:00": [], "+02:00": [], "-02:00": []}
    dates_UTC = {"+00:00": [], "+02:00": [], "-02:00": []}
    year = 2021
    coordinates = {"lat": 50, "lon": 10}
    values_continuous = []

    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            for hour in range(24):
                values_continuous.append(round(month + day / 100 + hour / 10000, 6))

                date_str = f"2021-{month:02d}-{day:02d}T{hour:02d}:00"
                dates_UTC["+00:00"].append(f"{date_str}+00:00")
                dates_local["+02:00"].append(f"{date_str}+02:00")  # 2 hours ahead
                dates_local["-02:00"].append(f"{date_str}-02:00")  # 2 hours behind

    # First entries of next year is needed
    values_continuous.append(12.3124)

    dates_UTC["+00:00"].append("2022-01-01T00:00+00:00")
    dates_local["+00:00"] = dates_UTC["+00:00"]
    dates_local["+02:00"].append("2022-01-01T00:00+02:00")  # 2 hours ahead
    dates_local["-02:00"].append("2022-01-01T00:00-02:00")  # 2 hours behind

    # Shift UTC dates by offsets
    dates_UTC["+02:00"] = [
        "2020-12-31T22:00+00:00",
        "2020-12-31T23:00+00:00",
    ] + dates_UTC["+00:00"][:-2]
    dates_UTC["-02:00"] = dates_UTC["+00:00"][2:] + [
        "2022-01-01T01:00+00:00",
        "2022-01-01T02:00+00:00",
    ]

    values_accumulated_UTC = {
        "+00:00": np.array([24] + list(range(1, 25)) * 365),
        "+02:00": np.array(
            np.array([22, 23, 24] + list(range(1, 25)) * 364 + list(range(1, 23)))
        ),
        "-02:00": np.array(list(range(2, 25)) + list(range(1, 25)) * 364 + [1, 2]),
    }

    # Specify expected data format
    precision = 6  # 6 decimal places expected in results data
    results_file_path = Path(
        "weatherDataTestResults"
        + os.sep
        + "lat50.000000_lon10.000000__2021-01-01_2021-12-31__weather.txt"
    )
    target_data_created = False

    # Create folder for prepared daily data
    os.makedirs("weatherDataTestResults", exist_ok=True)
    remove_folder = not any(Path("weatherDataTestResults").iterdir())

    for offset_hours in [0, 2, -2]:
        offset_str = f"{offset_hours:+03d}:00"
        data_hourly = pd.DataFrame(
            {
                "Valid time": dates_UTC[offset_str],
                "Local time": dates_local[offset_str],
                "Precipitation[mm] (acc.)": values_accumulated_UTC[offset_str],
                "Temperature[degC]": np.array(values_continuous),
                "SSRD[Jm-2] (acc.)": values_accumulated_UTC[offset_str] * 100000,
            }
        )
        hourly_to_daily(
            data_hourly,
            DATA_VAR_SPECS,
            coordinates,
            tz_offset_hours=offset_hours,
            target_folder="weatherDataTestResults",
        )
        generated_content = pd.read_csv(results_file_path, delimiter="\t")

        # Specify expected data (independent of offset, because input data were adjusted)
        # Note: functions used for mean etc. are tested separately
        if not target_data_created:
            target_data = {
                "Date": data_hourly["Local time"][:-24:24].str.split("T").str[0].values,
                "Precipitation[mm]": daily_accumulated(
                    data_hourly["Precipitation[mm] (acc.)"].values,
                    tz_offset_hours=offset_hours,
                ),
                "Temperature[degC]": daily_mean_00_24(
                    data_hourly["Temperature[degC]"].values
                ),
            }
            target_data["Temperature_Daylight[degC]"] = daily_mean_daylight(
                data_hourly["Temperature[degC]"].values,
                target_data["Date"],
                coordinates,
            )
            target_data["PAR[µmolm-2s-1]"] = par_from_net_radiation(
                daily_accumulated(
                    data_hourly["SSRD[Jm-2] (acc.)"].values,
                    tz_offset_hours=offset_hours,
                )
            )
            target_data["Daylength[h]"] = get_day_length(
                coordinates, target_data["Date"]
            )
            target_data["PET[mm]"] = get_pet_thornthwaite(
                target_data["Temperature[degC]"],
                data_hourly["Temperature[degC]"].values,
                target_data["Daylength[h]"],
                target_data["Date"],
                use_effective_temperature=False,
            )
            target_data_created = True

        assert generated_content.shape == (365, len(target_data.keys()))
        assert list(generated_content.columns) == list(target_data.keys())

        for key in target_data.keys():
            if key == "Date":
                assert all(generated_content[key] == target_data[key])
            else:
                assert all(
                    generated_content[key] == np.round(target_data[key], precision)
                )

    if remove_folder:
        shutil.rmtree("weatherDataTestResults")


def test_monthly_mean():
    """Test monthly_mean function."""
    dates = []
    values = []
    target_means = np.zeros((3, 12))

    for year_index, year in enumerate(range(2020, 2023)):  # include leap year
        for month_index, month in enumerate(range(1, 13)):
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                dates.append(f"{year}-{month:02d}-{day:02d}")
                values.append((year - 2020) * 100 + month + day / 100)

            # Calculate target mean for each month
            target_means[year_index, month_index] = np.mean(
                values[-calendar.monthrange(year, month)[1] :]
            )

    assert (monthly_mean(np.array(values), dates) == target_means).all()


def test_effective_temperature():
    """Test effective temperature calculation."""
    precision = 12  # 12 decimal places to avoid differences only due to floating point arithmetic
    year = 2021
    temperature_hourly = []
    dates = []

    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            for hour in range(24):
                temperature_hourly.append(round(month + day / 100 + hour / 10000, 6))

            dates.append(f"{year}-{month:02d}-{day:02d}")

    # First entries of next year is needed
    temperature_hourly.append(12.3124)
    temperature_hourly = np.array(temperature_hourly)

    # Specify expected results
    day_length = get_day_length({"lat": 50, "lon": 10}, dates)
    t_max = daily_max_00_24(temperature_hourly)
    t_min = daily_min_00_24(temperature_hourly)
    t_avg = (t_max + t_min) / 2  # approximation as used in Pereira and Pruitt 2004

    # Eq. (6) in Pereira and Pruitt 2004, k = 0.69
    target_t_eff_default = 0.5 * 0.69 * (3 * t_max - t_min)
    target_t_eff_corrected = target_t_eff_default * (day_length / (24 - day_length))
    target_t_eff_corrected = np.maximum(
        target_t_eff_corrected, t_avg
    )  # t_eff not lower than t_avg
    target_t_eff_corrected = np.minimum(
        target_t_eff_corrected, t_max
    )  # t_eff not higher than t_max

    assert all(
        np.round(effective_temperature(temperature_hourly), precision)
        == np.round(target_t_eff_default, precision)
    )
    assert all(
        np.round(
            effective_temperature(
                temperature_hourly, correct_by_day_length=True, day_length=day_length
            ),
            precision,
        )
        == np.round(target_t_eff_corrected, precision)
    )

    # Test missing day length data
    with pytest.raises(ValueError):
        effective_temperature(temperature_hourly, correct_by_day_length=True)
