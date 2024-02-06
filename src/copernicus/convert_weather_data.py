"""
Module Name: convert_weather_data.py
Author: Thomas Banitz, Tuomas Rossi, Franziska Taubert, BioDT
Date: February, 2024
Description: Functions for conversions and subsequent calculations from Copernicus weather data. 
"""

import numpy as np
import statistics as stats


def convert_units(data, unit_conversion):
    if unit_conversion:
        if unit_conversion == "_to_Milli":
            return data * 1e3
        elif unit_conversion == "_to_Kilo":
            return data * 1e-3
        elif unit_conversion == "_to_Mega":
            return data * 1e-6
        elif unit_conversion == "Kelvin_to_Celsius":
            return data - 273.15
        elif unit_conversion == "J_m-2_d-1_to_PAR_Micromol_m-2_s-1":
            # 4.57 µmol/J, PAR fraction 0.5, /d to /s
            return data * 4.57 / (24 * 60 * 60) * 0.5
        else:
            print(
                f"Error: Unit conversion '{unit_conversion}' not found! No conversion performed."
            )
            return data
    else:
        return data


def get_daily_mean_00_24(data):
    # Calculate the mean of the two values at 0:00 and 24:00 (day + 1 at 00:00)
    mean_00_24 = [
        stats.mean(
            [
                data[i],
                data[i + 24],
            ]
        )
        for i in range(0, len(data) - 1, 24)
    ]

    # Calculate daily means
    data = data[:-1]  # Omit last entry from last Day + 1 at 00:00
    data[::24] = (
        mean_00_24  # Replace 00:00 entries with calculated means of 00:00 and 24:00
    )
    data = [data[i : i + 24].mean() for i in range(0, len(data), 24)]

    return data


def wind_speed_from_u_v(u, v):
    return np.power((np.power(u, 2) + np.power(v, 2)), 0.5)


def wind_speed_height_change(wind_speed, height1=10, height2=2, z0=0.03):
    return wind_speed * np.log(height2 / z0) / np.log(height1 / z0)


def get_pet_fao(
    ssrd,
    slhf,
    temperatures,
    temperatures_hourly,
    wind_speed_2m,
    dewpoint_temperatures,
    surface_pressure,
):
    # test before implementing formulae
    return ssrd
