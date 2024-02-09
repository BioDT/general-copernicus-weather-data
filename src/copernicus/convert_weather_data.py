"""
Module Name: convert_weather_data.py
Author: Thomas Banitz, Tuomas Rossi, Franziska Taubert, BioDT
Date: February, 2024
Description: Functions for conversions and subsequent calculations from Copernicus weather data. 
             Includes PET calculation based on FAO recommendation (Allen et al., cf. https://pyeto.readthedocs.io)
"""

import numpy as np
import statistics as stats

# import pyeto impossible, not on PyPI, copy source code suggested by author


def convert_units(values, unit_conversion):
    """
    Convert units of the input values based on the specified unit conversion.

    Args:
        values (float or numpy.ndarray): Input values to be converted.
        unit_conversion (str): Unit conversion specifier. Options include:
            - '_to_Milli': Convert to milli-units.
            - '_to_Kilo': Convert to kilo-units.
            - '_to_Mega': Convert to mega-units.
            - 'Kelvin_to_Celsius': Convert from Kelvin to Celsius.
            - 'Celsius_to_Kelvin': Convert from Celsius to Kelvin.
            - 'd-1_to_s-1': Convert from per day to per second.

    Returns:
        float or numpy.ndarray: Converted values.
    """
    if unit_conversion:
        if unit_conversion == "_to_Milli":
            return values * 1e3
        elif unit_conversion == "_to_Kilo":
            return values * 1e-3
        elif unit_conversion == "_to_Mega":
            return values * 1e-6
        elif unit_conversion == "Kelvin_to_Celsius":
            return values - 273.15
        elif unit_conversion == "Celsius_to_Kelvin":
            return values + 273.15
        elif unit_conversion == "d-1_to_s-1":
            return values / (24 * 60 * 60)
        else:
            print(
                f"Error: Unit conversion '{unit_conversion}' not found! No conversion performed."
            )
            return values
    else:
        return values


def par_from_net_radiation(values):
    """
    Calculate Photosynthetically Active Radiation (PAR) from net radiation values.

    Args:
        values (float or numpy.ndarray): Net radiation values (unit: J/m²/d).

    Returns:
        numpy.ndarray: PAR values (unit: µmol/m²/s).
    """
    # 4.57 µmol/J, PAR fraction 0.5
    par = values * 4.57 * 0.5

    # Convert from radiation per day to per second
    return convert_units(par, "d-1_to_s-1")


def daily_mean_00_24(values_hourly):
    """
    Calculate daily mean values from hourly data.

    Args:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Daily mean values.
    """
    # Calculate the mean of the two values at 0:00 and 24:00 (day + 1 at 00:00)
    mean_00_24 = [
        stats.mean(
            [
                values_hourly[i],
                values_hourly[i + 24],
            ]
        )
        for i in range(0, len(values_hourly) - 1, 24)
    ]

    # COPY to prevent changing the original 'values_hourly' variable!
    # Omit last entry from last Day + 1 at 00:00,
    values_for_means = values_hourly[:-1].copy()

    # Replace 00:00 entries with calculated means of 00:00 and 24:00
    values_for_means[::24] = mean_00_24

    # Calculate and return daily means
    means = [
        values_for_means[i : i + 24].mean() for i in range(0, len(values_for_means), 24)
    ]

    return np.array(means)


def daily_min_00_24(values_hourly):
    """
    Calculate daily minimum values from hourly data.

    Args:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Daily minimum values.
    """
    # Calculate the min of the 25 values from 0:00 to 24:00 (day + 1 at 00:00)
    mins = [
        values_hourly[i : i + 25].min() for i in range(0, len(values_hourly) - 1, 24)
    ]

    return np.array(mins)


def daily_max_00_24(values_hourly):
    """
    Calculate daily maximum values from hourly data.

    Args:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Daily maximum values.
    """
    # Calculate the min of the 25 values from 0:00 to 24:00 (day + 1 at 00:00)
    maxs = [
        values_hourly[i : i + 25].max() for i in range(0, len(values_hourly) - 1, 24)
    ]

    return np.array(maxs)


def wind_speed_from_u_v(u_component, v_component):
    """
    Calculate wind speed from u and v components.

    Args:
        u_component (float or numpy.ndarray): U-component of wind.
        v_component (float or numpy.ndarray): V-component of wind.

    Returns:
        numpy.ndarray: Wind speed.
    """
    return np.power(np.power(u_component, 2) + np.power(v_component, 2), 0.5)


def wind_speed_height_change(wind_speed, height1=10, height2=2, z0=0.03):
    """
    Calculate wind speed change due to height difference.

    Args:
        wind_speed (float or numpy.ndarray): Wind speed at height1 (unit: m/s).
        height1 (float, optional): Initial height. Default is 10 (unit: m).
        height2 (float, optional): Final height. Default is 2 (unit: m).
        z0 (float, optional): Roughness length. Defaults is 0.03 (unit: m).

    Returns:
        numpy.ndarray: Wind speed at height2 (unit: m/s).
    """
    return wind_speed * np.log(height2 / z0) / np.log(height1 / z0)


# from pyeto
def svp_from_temperature(temperature):
    """
    Calculate saturation vapor pressure (svp) from temperature.
    Eq. (11) in Allen et al.

    Args:
        temperature (float or numpy.ndarray): Temperature (unit: degC).

    Returns:
        numpy.ndarray: Saturation vapor pressure (unit: kPa).
    """
    return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))


def delta_svp(temperature):
    """
    Calculate slope of saturation vapor pressure curve at given temperature.
    Eq. (13) in Allen et al.

    Args:
        temperature (numpy.ndarray): Temperature (unit: degC).

    Returns:
        numpy.ndarray: Slope of saturation vapor pressure curve (unit: kPa/degC).
    """
    return 4098 * svp_from_temperature(temperature) / np.power(temperature + 237.3, 2)


def gamma_from_atmospheric_pressure(pressure):
    """
    Calculate psychrometric constant from atmospheric pressure.
    Eq. (8) in Allen et al.

    Args:
        pressure (numpy.ndarray): Atmospheric pressure (unit: kPa).

    Returns:
        numpy.ndarray: Psychrometric constant (unit: kPa/degC).
    """
    return 0.000665 * pressure


def get_pet(
    ssrd,
    slhf,
    temperature,
    temperature_hourly,
    wind_speed_2m,
    dewpoint_temperature_hourly,
    surface_pressure,
    method="fao",
):
    """
    Calculate Potential Evapotranspiration (PET) using the FAO-56 Penman-Monteith equation.
    Eq. (13) in Allen et al.

    Args:
        ssrd (float or numpy.ndarray): Surface solar radiation downward (unit: J/m²/d).
        slhf (float or numpy.ndarray): Surface latent heat flux (unit: J/m²/d).
        temperature (numpy.ndarray): Daily mean temperature (unit: degC).
        temperature_hourly (numpy.ndarray): Hourly temperature (unit: degC).
        wind_speed_2m (float or numpy.ndarray): Wind speed at 2 meters height (unit: m/s).
        dewpoint_temperature_hourly (numpy.ndarray): Hourly dewpoint temperature (unit: degC).
        surface_pressure (numpy.ndarray): Surface atmospheric pressure (unit: kPa).
        method (str, optional): PET calculation method. Default is "fao".

    Returns:
        numpy.ndarray: Potential Evapotranspiration (PET) values (unit: mm).
    """
    if method == "fao":
        # Prepare and convert variables as needed for PET from FAO-56 Penman-Monteith equation
        delta = delta_svp(temperature)
        ssrd_megaJ = convert_units(ssrd, "_to_Mega")
        slhf_megaJ = convert_units(slhf, "_to_Mega")
        temperature_kelvin = convert_units(temperature, "Celsius_to_Kelvin")

        # Best alternative: mean of svp from hourly temperatures
        # svp = svp_from_temperature(temperature)  # from daily temperature
        # svp = (
        #     svp_from_temperature(daily_min_00_24(temperature_hourly))
        #     + svp_from_temperature(daily_max_00_24(temperature_hourly))
        # ) / 2  # from daily min/max temperatures
        svp = daily_mean_00_24(
            svp_from_temperature(temperature_hourly)
        )  # from hourly temperatures

        # Best alternative: mean of svp from hourly dewpoint temperatures
        # avp = svp_from_temperature(dewpoint_temperature)  # from daily temperature
        # avp = (
        #     svp_from_temperature(daily_min_00_24(dewpoint_temperature_hourly))
        #     + svp_from_temperature(daily_max_00_24(dewpoint_temperature_hourly))
        # ) / 2  # from daily min/max temperatures
        avp = daily_mean_00_24(
            svp_from_temperature(dewpoint_temperature_hourly)
        )  # from hourly temperatures

        gamma = gamma_from_atmospheric_pressure(surface_pressure)

        # Eq. (13) in Allen et al.
        pet = (
            0.408 * delta * (ssrd_megaJ - slhf_megaJ)
            + gamma * 900 / temperature_kelvin * wind_speed_2m * (svp - avp)
        ) / (delta + gamma * (1 + 0.34 * wind_speed_2m))

        return pet
    else:
        raise ValueError(f"Invalid PET calculation method selected: {method}.")
