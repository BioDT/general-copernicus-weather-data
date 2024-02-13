"""
Module Name: convert_weather_data.py
Author: Thomas Banitz, Tuomas Rossi, Franziska Taubert, BioDT
Date: February, 2024
Description: Functions for conversions and subsequent calculations from Copernicus weather data. 
             Includes PET calculation based on FAO recommendation (Allen et al. 1998 FAO Irrigation and drainage paper 56)
             and Thornthwaite equation (e.g. Pereira and Pruitt 2004 Agricultural Water Management, https://doi.org/10.1016/j.agwat.2003.11.003)
             Cf. https://pyeto.readthedocs.io, but pyeto package is not on PyPI
"""

from copernicus import utils as ut
import numpy as np
import statistics as stats


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


def monthly_mean(values_daily, time):
    """
    Calculate monthly mean from daily data for each month of each year.

    Args:
        values_daily (array-like): Array of daily data.
        time (array-like): Array of daily time strings in the format yyyy-mm-dd.

    Returns:
        numpy.ndarray: 2D array containing monthly mean values for each month of each year.
    """
    # Extract year and month from time strings
    years = np.array([int(date[:4]) for date in time])
    months = np.array([int(date[5:7]) for date in time])
    unique_years = np.unique(years)

    # Check if all days from a year are present in the data (as per total number)
    for year in unique_years:
        expected_days = 366 if ut.is_leap_year(year) else 365
        if np.sum(years == year) != expected_days:
            raise ValueError(
                f"Length of daily data for the year {year} must be {expected_days}."
            )

    # Calculate monthly means for each month of each year
    monthly_means = []
    for year in unique_years:
        for month in range(1, 13):
            values_this_month = values_daily[(years == year) & (months == month)]
            monthly_means.append(np.mean(values_this_month))

    return np.array(monthly_means).reshape(len(unique_years), 12)


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


def svp_from_temperature(temperature):
    """
    Calculate saturation vapor pressure (svp) from temperature.
    Eq. (11) in Allen et al. 1998 FAO

    Args:
        temperature (float or numpy.ndarray): Temperature (unit: degC).

    Returns:
        numpy.ndarray: Saturation vapor pressure (unit: kPa).
    """
    return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))


def delta_svp(temperature):
    """
    Calculate slope of saturation vapor pressure curve at given temperature.
    Eq. (13) in Allen et al. 1998

    Args:
        temperature (numpy.ndarray): Temperature (unit: degC).

    Returns:
        numpy.ndarray: Slope of saturation vapor pressure curve (unit: kPa/degC).
    """
    return 4098 * svp_from_temperature(temperature) / np.power(temperature + 237.3, 2)


def gamma_from_atmospheric_pressure(pressure):
    """
    Calculate psychrometric constant from atmospheric pressure.
    Eq. (8) in Allen et al. 1998

    Args:
        pressure (numpy.ndarray): Atmospheric pressure (unit: kPa).

    Returns:
        numpy.ndarray: Psychrometric constant (unit: kPa/degC).
    """
    return 0.000665 * pressure


def heat_index(temperature_monthly):
    """
    Calculate the heat index for each year based on monthly temperature values.
    Eq. (2) in Pereira and Pruitt 2004

    Args:
        temperature_monthly (numpy.ndarray): Array of monthly temperature values for each year.
            Shape should be (number of years, 12).

    Returns:
        numpy.ndarray: Array of heat index values for each year.
    """
    if temperature_monthly.shape[1] != 12:
        raise ValueError("Length of monthly temperatured data must be 12.")

    return np.sum(np.power(np.maximum(0, 0.2 * temperature_monthly), 1.514), axis=1)


def exponent_a(heat_index_yearly):
    """
    Calculate the exponent 'a' for the Thornthwaite PET formula based on yearly heat index values.
    Eq. (2) in Pereira and Pruitt 2004

    Args:
        heat_index_yearly (numpy.ndarray): Array of yearly heat index values.

    Returns:
        numpy.ndarray: Array of exponent 'a' values.
    """
    return (
        6.75e-7 * np.power(heat_index_yearly, 3)
        - 7.71e-5 * np.power(heat_index_yearly, 2)
        + 1.7912e-2 * heat_index_yearly
        + 0.49239
    )


def get_pet_fao(
    ssr,
    slhf,
    temperature,
    temperature_hourly,
    wind_speed_2m,
    dewpoint_temperature_hourly,
    surface_pressure,
):
    """
    Calculate Potential Evapotranspiration (PET) using the FAO-56 Penman-Monteith equation.
    Eq. (13) in Allen et al. 1998

    Args:
        ssr (float or numpy.ndarray): Surface net solar radiation (unit: J/m²/d).
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
    # Prepare and convert variables as needed for PET from FAO-56 Penman-Monteith equation
    delta = delta_svp(temperature)
    ssr_megaJ = convert_units(ssr, "_to_Mega")
    slhf_megaJ = -convert_units(
        slhf, "_to_Mega"
    )  # "minus" because the ECMWF convention for vertical fluxes is positive downwards
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

    # Eq. (13) in Allen et al. 1998
    pet_fao = (
        0.408 * delta * (ssr_megaJ - slhf_megaJ)
        + gamma * 900 / temperature_kelvin * wind_speed_2m * (svp - avp)
    ) / (delta + gamma * (1 + 0.34 * wind_speed_2m))

    # Correct negative values to 0
    return np.maximum(pet_fao, 0)


def get_pet_thornthwaite(temperature, day_length, time):
    """
    Calculate Potential Evapotranspiration (PET) using the Thornthwaite equation.
    Daily version, Eqs. (1-5) in Pereira and Pruitt 2004

    Args:
        temperature (array-like):  Daily mean temperature (unit: degC).
        day_length (array-like): Daily day length (unit: hours).
        time (array-like): Array of daily time strings in the format yyyy-mm-dd.

    Returns:
        numpy.ndarray: Potential Evapotranspiration (PET) values (unit: mm).
    """
    temperature_monthly = monthly_mean(temperature, time)
    heat_index_yearly = heat_index(temperature_monthly)
    exponent_a_yearly = exponent_a(heat_index_yearly)
    correct_to_daily = day_length / 360  # Eq. (5) in Pereira and Pruitt 2004

    # Initialize array to store PET values for each day
    pet_thorn = np.zeros_like(temperature)
    years = np.array([int(date[:4]) for date in time])
    unique_years = np.unique(years)

    # Iterate over each entry in time and calculate PET for each day
    for i, year in enumerate(years):
        if temperature[i] <= 0:
            pet_thorn[i] = 0
        elif temperature[i] > 26:
            # Eq. (5) in Pereira and Pruitt 2004
            # Remark: Thornthwaite and Mather 1957 seem to use 26.5 °C, but newer sources usually 26 C°
            pet_thorn[i] = correct_to_daily[i] * (
                -415.85 + 32.24 * temperature[i] - 0.43 * temperature[i] ** 2
            )
            # Only for checking, Eq. (1) as below
            pet_check = (
                correct_to_daily[i]
                * 16
                * np.power(
                    10 * temperature[i] / heat_index_yearly[unique_years == year][0],
                    exponent_a_yearly[unique_years == year][0],
                )
            )

            print(
                f"{time[i]}: Temperature {temperature[i]} > 26 °C, "
                f"using PET from Eq. (4): {pet_thorn[i]:.2f}, "
                f"PET from Eq. (1) would be: {pet_check:.2f}."
            )
        else:
            # Eq. (1) in Pereira and Pruitt 2004
            pet_thorn[i] = (
                correct_to_daily[i]
                * 16
                * np.power(
                    10 * temperature[i] / heat_index_yearly[unique_years == year][0],
                    exponent_a_yearly[unique_years == year][0],
                )
            )

    return pet_thorn
