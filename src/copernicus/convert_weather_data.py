"""
Module Name: convert_weather_data.py
Description: Functions for conversions and subsequent calculations from Copernicus weather data.

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

Sources:
    PET calculation based on FAO recommendation:
    - Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
      Crop evapotranspiration-Guidelines for computing crop water requirements-FAO Irrigation and drainage paper 56.
      Fao, Rome, 300(9), D05109.

    PET calculation based on Thornthwaite equation:
    - Pereira, A.R., Pruitt, W.O. (2004).
      Adaptation of the Thornthwaite scheme for estimating daily reference evapotranspiration.
      Agricultural Water Management 66, 251-257. https://doi.org/10.1016/j.agwat.2003.11.003
    - Cf. https://pyeto.readthedocs.io, but pyeto package is not on PyPI.
"""

import statistics as stats
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sunrise, sunset

from copernicus import utils as ut
from copernicus.logger_config import logger


def convert_units(values, unit_conversion):
    """
    Convert units of the input values based on the specified unit conversion.

    Parameters:
        values (float or numpy.ndarray): Input values to be converted.
        unit_conversion (str): Unit conversion specifier. Options include:
            '_to_Milli': Convert to milli-units.
            '_to_Kilo': Convert to kilo-units.
            '_to_Mega': Convert to mega-units.
            'Kelvin_to_Celsius': Convert from Kelvin to Celsius.
            'Celsius_to_Kelvin': Convert from Celsius to Kelvin.
            'd-1_to_s-1': Convert from per day to per second.

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
            try:
                raise ValueError(f"Unit conversion '{unit_conversion}' not found!")
            except ValueError as e:
                logger.error(e)
                raise
    else:
        return values


def par_from_net_radiation(values):
    """
    Calculate Photosynthetically Active Radiation (PAR) from net radiation values.

    Parameters:
        values (float or numpy.ndarray): Net radiation values (unit: J/m²/d).

    Returns:
        numpy.ndarray: PAR values (unit: µmol/m²/s).
    """
    if np.any(values < 0):
        logger.warning("Negative values in net radiation data were corrected to 0.")

    values = np.maximum(values, 0)

    # 4.57 µmol/J, PAR fraction 0.5
    par = values * 4.57 * 0.5

    # Convert from radiation per day to per second
    par = convert_units(par, "d-1_to_s-1")

    return np.array(par)


def check_length_of_values_00_24(values_hourly):
    """
    Check length of hourly values for calculation of daily values from 00:00 to 24:00.

    Parameters:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Checked and adjusted hourly values, i.e. length reduced to full days.
    """
    length_input = len(values_hourly)

    if length_input < 25:
        try:
            raise ValueError(
                f"Length of hourly values ({length_input}) must be at least 25 for calculation of daily values."
            )
        except ValueError as e:
            logger.error(e)
            raise
    else:
        # COPY to prevent changing the original 'values_hourly' variable!
        values_checked = values_hourly.copy()

        if length_input % 24 == 0:
            values_checked = values_checked[:-23]
        else:
            values_checked = values_checked[
                : len(values_checked) - (len(values_checked) % 24 - 1)
            ]

        if len(values_checked) < length_input:
            logger.warning(
                f"Length of hourly values ({length_input}) was reduced to {len(values_checked)} for calculation of daily values."
            )

        return values_checked


def daily_mean_00_24(values_hourly):
    """
    Calculate daily mean values from hourly data.

    Parameters:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Daily mean values.
    """
    values_checked = check_length_of_values_00_24(values_hourly)

    # Calculate the mean of the two values at 0:00 and 24:00 (day + 1 at 00:00)
    mean_00_24 = [
        stats.mean(
            [
                values_checked[index],
                values_checked[index + 24],
            ]
        )
        for index in range(0, len(values_checked) - 24, 24)
    ]

    # Omit last entry from last Day + 1 at 00:00,
    values_for_means = values_checked[:-1]

    # Replace 00:00 entries with calculated means of 00:00 and 24:00
    values_for_means[::24] = mean_00_24

    # Calculate and return daily means
    means = [
        values_for_means[index : index + 24].mean()
        for index in range(0, len(values_for_means), 24)
    ]

    return np.array(means)


def daily_min_00_24(values_hourly):
    """
    Calculate daily minimum values from hourly data.

    Parameters:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Daily minimum values.
    """
    values_checked = check_length_of_values_00_24(values_hourly)

    # Calculate the min of the 25 values from 0:00 to 24:00 (day + 1 at 00:00)
    mins = [
        values_checked[index : index + 25].min()
        for index in range(0, len(values_checked) - 24, 24)
    ]

    return np.array(mins)


def daily_max_00_24(values_hourly):
    """
    Calculate daily maximum values from hourly data.

    Parameters:
        values_hourly (numpy.ndarray): Hourly data.

    Returns:
        numpy.ndarray: Daily maximum values.
    """
    values_checked = check_length_of_values_00_24(values_hourly)

    # Calculate the min of the 25 values from 0:00 to 24:00 (day + 1 at 00:00)
    maxs = [
        values_checked[index : index + 25].max()
        for index in range(0, len(values_checked) - 24, 24)
    ]

    return np.array(maxs)


def daily_accumulated(values_hourly, tz_offset_hours=0):
    """
    Calculate daily values from accumulative hourly data, considering time zone shifts (full hours only).

    Parameters:
        values_hourly (numpy.ndarray): Hourly data.
        tz_offset_hours (int): Offset of local time zone to UTC in hours (default is 0).

    Returns:
        numpy.ndarray: Daily accumulated values.
    """
    # Accumulated values at start of local day (at 00:00) since UTC day start of previous day,
    # i.e. fully belonging to previous day
    values_local_day_start = values_hourly[::24]

    if tz_offset_hours == 0:
        # Omit first entry because it belongs to day before first day
        return values_local_day_start[1:]
    else:
        # Accumulated values at the UTC day start within the current day (and since previous UTC day start),
        # i.e. only partly belonging to current local day
        if tz_offset_hours > 0:
            # Values at 00:00 + offset
            values_utc_day_start = values_hourly[tz_offset_hours::24]
        else:
            # Values at 00:00 of day+1 + negative offset
            values_utc_day_start = values_hourly[24 + tz_offset_hours :: 24]

        values_accumulated = (
            values_utc_day_start  # until UTC day start
            - values_local_day_start[:-1]  # until local day start (i.e. previous day)
            + values_local_day_start[1:]  # since UTC day start
        )

    return values_accumulated


def daily_mean_daylight(values_hourly, dates, coordinates):
    """
    Calculate daily mean values from hourly data, considering only daylight time span.

    Parameters:
        values_hourly (numpy.ndarray): Hourly data.
        dates (list or numpy.ndarray): Iterable of date strings (e.g., 'YYYY-MM-DD') corresponding to each day.
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).

    Returns:
        numpy.ndarray: Daily mean temperature values for daylight time span only.
    """
    time_zone = ut.get_time_zone(coordinates)
    location = LocationInfo(
        "name", "region", time_zone.key, coordinates["lat"], coordinates["lon"]
    )
    values_daylight = []

    for index, date_str in enumerate(dates):
        try:
            day = int(date_str.split("-")[2])
            day_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Get local sunrise and sunset times
            sunrise_time = sunrise(location.observer, date=day_date, tzinfo=time_zone)
            sunset_time = sunset(location.observer, date=day_date, tzinfo=time_zone)

            # Adjust sunrise and sunset to take into account the CDS hour representation,
            # and ignore daylight saving time DST
            sunrise_plus_30 = (
                sunrise_time + timedelta(minutes=30) - time_zone.dst(sunrise_time)
            )
            sunset_plus_30 = (
                sunset_time + timedelta(minutes=30) - time_zone.dst(sunset_time)
            )

            # Determine indices of hours not fully considered
            sunrise_index = sunrise_plus_30.hour
            sunset_index = sunset_plus_30.hour if sunset_plus_30.day == day else 24

            # Fractional weights for sunset and sunrise hour (simply cutting off seconds, no rounding),
            # weights=1 in between, weights=0 before and after
            weights = np.zeros(25)
            weights[sunrise_index] = 1 - sunrise_plus_30.minute / 60
            weights[sunset_index] = sunset_plus_30.minute / 60
            weights[sunrise_index + 1 : sunset_index] = 1

            # Get weighted mean for daylight hours
            day_values = values_hourly[index * 24 : index * 24 + 25]
            weighted_mean = np.sum(day_values * weights) / np.sum(weights)
            values_daylight.append(weighted_mean)
        except Exception as e:
            # Error handling (no sunset or sunrise on given location)
            logger.error(f"Error calculating daylight time span for {date_str} ({e}).")
            values_daylight.append(np.nan)

    return np.array(values_daylight)


def hourly_to_daily(
    data_hourly,
    data_var_specs,
    coordinates,
    *,
    tz_offset_hours=0,
    target_folder="weatherDataPrepared",
):
    """
    Convert hourly weather data to daily, considering time zone shifts (full hours only).

    Parameters:
        data_hourly (dataframe): Hourly data.
        data_var_specs (dict): Dictionary of variable specifications.
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        tz_offset_hours (int): Offset of local time zone to UTC in hours (default is 0).
        target_folder (str or Path): Target folder for .txt files (default is 'weatherDataPrepared').
    """
    # Dates (omit last entry from last Day + 1 at 00:00)
    local_date = data_hourly["Local time"][:-24:24].str.split("T").str[0].values

    # Day lengths
    day_length = ut.get_day_length(coordinates, local_date)

    # Precipitation (omit first entry from first Day at 00:00)
    precipitation = data_hourly[
        data_var_specs["precipitation"]["col_name_hourly"]
    ].values
    precipitation = daily_accumulated(precipitation, tz_offset_hours)

    # Temperature (hourly still needed for PET calculation, daily means considering 00:00 and 24:00 values)
    temperature_hourly = data_hourly[
        data_var_specs["temperature"]["col_name_hourly"]
    ].values
    temperature = daily_mean_00_24(temperature_hourly)
    temperature_daylight = daily_mean_daylight(
        temperature_hourly, local_date, coordinates
    )

    # SSRD (omit first entry from first Day at 00:00), convert net radiation to PAR
    ssrd = data_hourly[data_var_specs["solar_radiation_down"]["col_name_hourly"]].values
    ssrd = daily_accumulated(ssrd, tz_offset_hours)
    ssrd_par = par_from_net_radiation(ssrd)

    # PET from Thornthwaite equation
    pet_thornthwaite = get_pet_thornthwaite(temperature, day_length, local_date)

    # Write dataframe with daily values
    data_daily = pd.DataFrame(
        {
            "Date": local_date,
            data_var_specs["precipitation"]["col_name_daily"]: precipitation,
            data_var_specs["temperature"]["col_name_daily"]: temperature,
            "Temperature_Daylight[degC]": temperature_daylight,
            data_var_specs["solar_radiation_down"]["col_name_daily"]: ssrd_par,
            "Daylength[h]": day_length,
            "PET[mm]": pet_thornthwaite,  # was: PET_thornthwaite
        }
    )

    # Save DataFrame to .txt file
    time_range = f"{local_date[0]}_{local_date[-1]}"
    file_name = ut.construct_weather_data_file_name(
        coordinates,
        folder=target_folder,
        data_format="txt",
        time_specifier=time_range,
        data_specifier="weather",
    )
    data_daily.to_csv(
        file_name, sep="\t", index=False, float_format="%.6f", na_rep="nan"
    )
    logger.info("Text file with daily resolution prepared.")


def monthly_mean(values_daily, dates):
    """
    Calculate monthly mean from daily data for each month of each year.

    Parameters:
        values_daily (numpy.ndarray): Array of daily data.
        dates (array-like): Array of daily time strings in the format yyyy-mm-dd.

    Returns:
        numpy.ndarray: 2D array containing monthly mean values for each month of each year.
    """
    # Extract year and month from time strings
    years = np.array([int(day_date[:4]) for day_date in dates])
    months = np.array([int(day_date[5:7]) for day_date in dates])
    unique_years = np.unique(years)

    # Calculate monthly means for each month of each year
    monthly_means = []

    for year in unique_years:
        for month in range(1, 13):
            values_this_month = values_daily[(years == year) & (months == month)]
            monthly_means.append(np.mean(values_this_month))

    return np.array(monthly_means).reshape(len(unique_years), 12)


def effective_temperature(
    temperature_hourly, *, correct_by_day_length=False, day_length=None
):
    """
    Calculate daily effective temperature based on hourly temperature data and day lengths.

    Parameters:
        temperature_hourly (numpy.ndarray): Hourly temperature (unit: degC).
        correct_by_day_length(bool): Apply day length correction (default is False).
        day_length (numpy.ndarray): Daily day length (unit: hours, default is None).

    Returns:
        numpy.ndarray: Array of effective temperature values.
    """
    t_max = daily_max_00_24(temperature_hourly)
    t_min = daily_min_00_24(temperature_hourly)

    if correct_by_day_length:
        if day_length is None:
            try:
                raise ValueError(
                    "Day length data must be provided for day length correction."
                )
            except ValueError as e:
                logger.error(e)
                raise

        # Day length correction, Eqs. (6, 7) in Pereira and Pruitt 2004, k = 0.69
        day_night_ratio = day_length / (24 - day_length)
        t_eff = 0.69 / 2 * (3 * t_max - t_min) * day_night_ratio

        # Tavg ≤ Tef∗ ≤ Tmax
        t_eff = np.maximum(t_eff, (t_max + t_min) / 2)
        t_eff = np.minimum(t_eff, t_max)
    else:
        # no day length correction, Eq. (6) in Pereira and Pruitt 2004, k = 0.69 (not 0.72)
        t_eff = 0.69 / 2 * (3 * t_max - t_min)

    return t_eff


def wind_speed_from_u_v(u_component, v_component):
    """
    Calculate wind speed from u and v components.

    Parameters:
        u_component (float or numpy.ndarray): U-component of wind.
        v_component (float or numpy.ndarray): V-component of wind.

    Returns:
        numpy.ndarray: Wind speed.
    """
    return np.sqrt(u_component**2 + v_component**2)


def wind_speed_height_change(
    wind_speed, *, initial_height=10, target_height=2, roughness_length=0.03
):
    """
    Calculate wind speed change due to height difference using log wind profile.

    Parameters:
        wind_speed (float or numpy.ndarray): Wind speed at initial height (unit: m/s).
        initial_height (float, optional): Initial height. Default is 10 (unit: m).
        target_height (float, optional): Final height. Default is 2 (unit: m).
        roughness_length (float, optional): Roughness length (z0). Default is 0.03 (unit: m).

    Returns:
        numpy.ndarray: Wind speed at target height (unit: m/s).
    """
    if roughness_length <= 0:
        try:
            raise ValueError("Roughness length must be larger than 0.")
        except ValueError as e:
            logger.error(e)
            raise

    # Remark: roughness length = 0.03 used, as most established for grass and similar,
    #         Eq. (47) in Allen et al. 1998 uses slightly lower value (~0.017)
    return (
        wind_speed
        * np.log(target_height / roughness_length)
        / np.log(initial_height / roughness_length)
    )


def svp_from_temperature(temperature):
    """
    Calculate saturation vapor pressure (svp) from temperature.
    Eq. (11) in Allen et al. 1998 FAO

    Parameters:
        temperature (float or numpy.ndarray): Temperature (unit: degC).

    Returns:
        numpy.ndarray: Saturation vapor pressure (unit: kPa).
    """
    return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))


def delta_svp(temperature):
    """
    Calculate slope of saturation vapor pressure curve at given temperature.
    Eq. (13) in Allen et al. 1998

    Parameters:
        temperature (numpy.ndarray): Temperature (unit: degC).

    Returns:
        numpy.ndarray: Slope of saturation vapor pressure curve (unit: kPa/degC).
    """
    return 4098 * svp_from_temperature(temperature) / np.power(temperature + 237.3, 2)


def gamma_from_atmospheric_pressure(pressure):
    """
    Calculate psychrometric constant gamma from atmospheric pressure.
    Eq. (8) in Allen et al. 1998

    Parameters:
        pressure (numpy.ndarray): Atmospheric pressure (unit: kPa).

    Returns:
        numpy.ndarray: Psychrometric constant (unit: kPa/degC).
    """
    return 0.000665 * pressure


def heat_index(temperature_monthly):
    """
    Calculate the heat index for each year based on monthly temperature values.
    Eq. (2) in Pereira and Pruitt 2004

    Parameters:
        temperature_monthly (numpy.ndarray): Monthly temperature values for each year (unit: degC).
            Shape should be (number of years, 12).

    Returns:
        numpy.ndarray: Heat index values for each year.
    """
    if temperature_monthly.shape[1] != 12:
        try:
            raise ValueError("Length of monthly temperature data must be 12.")
        except ValueError as e:
            logger.error(e)
            raise

    return np.sum(np.power(np.maximum(0, 0.2 * temperature_monthly), 1.514), axis=1)


def exponent_a(heat_index_yearly):
    """
    Calculate the exponent 'a' for the Thornthwaite PET formula based on yearly heat index values.
    Eq. (3) in Pereira and Pruitt 2004

    Parameters:
        heat_index_yearly (numpy.ndarray): Yearly heat index values.

    Returns:
        numpy.ndarray: Exponent 'a' values for each year.
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

    Parameters:
        ssr (numpy.ndarray): Daily surface net solar radiation (unit: J/m²/d).
        slhf (numpy.ndarray): Daily surface latent heat flux (unit: J/m²/d).
        temperature (numpy.ndarray): Daily mean temperature (unit: degC).
        temperature_hourly (numpy.ndarray): Hourly temperature (unit: degC).
        wind_speed_2m (numpy.ndarray): Daily wind speed at 2 meters height (unit: m/s).
        dewpoint_temperature_hourly (numpy.ndarray): Hourly dewpoint temperature (unit: degC).
        surface_pressure (numpy.ndarray): Daily surface atmospheric pressure (unit: kPa).

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

    # Eq. (6) in Allen et al. 1998
    pet_fao = (
        0.408 * delta * (ssr_megaJ - slhf_megaJ)
        + gamma * 900 / temperature_kelvin * wind_speed_2m * (svp - avp)
    ) / (delta + gamma * (1 + 0.34 * wind_speed_2m))

    # Correct negative values to 0
    return np.maximum(pet_fao, 0)


def get_pet_thornthwaite(
    temperature,
    day_length,
    dates,
    *,
    temperature_hourly=None,
):
    """
    Calculate Potential Evapotranspiration (PET) using the Thornthwaite equation.
    Daily version, Eqs. (1-5) in Pereira and Pruitt 2004.
    Requires full years of daily data for calculation.

    Parameters:
        temperature (numpy.ndarray):  Daily mean temperature (unit: degC).
        day_length (numpy.ndarray): Daily day length (unit: hours).
        dates (array-like): Array of daily time strings in the format yyyy-mm-dd.
        temperature_hourly (numpy.ndarray): Hourly temperature (unit: degC) if effective temperature
            calculation from hourly data is desired (default is None).

    Returns:
        numpy.ndarray: Daily Potential Evapotranspiration (PET) values (unit: mm),
            if full year of daily input data were provided, otherwise 'nan' for all days
            of the respective year.
    """
    # Initialize array to store PET values for each day
    pet_thorn = np.full_like(day_length, np.nan)

    # Extract year from time strings
    years = np.array([int(day_date[:4]) for day_date in dates])
    unique_years = np.unique(years)

    # Check if all days from a year are present in the data (as per total number), otherwise return 'nan'
    for year in unique_years:
        days_found = np.sum(years == year)
        days_expected = ut.get_days_in_year(year)

        if days_found != days_expected:
            logger.warning(
                f"Length of daily data for {year} ({days_found}) differs "
                f"from days in that year ({days_expected}). "
                "PET by Thornthwaite equation not calculated as it requires data for full years!"
            )
            return pet_thorn

    # Prepare variables for PET calculation
    temperature_monthly = monthly_mean(temperature, dates)
    heat_index_yearly = heat_index(temperature_monthly)
    exponent_a_yearly = exponent_a(heat_index_yearly)
    correct_to_daily = day_length / 360  # Eq. (5) in Pereira and Pruitt 2004

    # Correct to effective daily temperature if hourly temperatures specified
    # In both cases:
    # Use daily temperature values for calculation of "standard month of 30 days,
    # each day with 12 h of photoperiod" (Eq. (1) in Pereira and Pruitt 2004)
    # i.e. assume the temperature of the given day as average for the whole reference month,
    # than correct PET to daily, which includes the actual day length
    temperature_used = (
        effective_temperature(temperature_hourly)
        if temperature_hourly is not None
        else temperature
    )

    # Iterate over each entry in dates and calculate PET for each day
    for index, year in enumerate(years):
        if temperature_used[index] <= 0:
            pet_thorn[index] = 0
        else:
            # Eq. (1) in Pereira and Pruitt 2004
            pet_eq1 = (
                correct_to_daily[index]
                * 16
                * np.power(
                    10
                    * temperature_used[index]
                    / heat_index_yearly[unique_years == year][0],
                    exponent_a_yearly[unique_years == year][0],
                )
            )

            if temperature_used[index] > 26:
                # Eq. (4) in Pereira and Pruitt 2004
                # Remark: Thornthwaite and Mather 1957 seem to use 26.5 °C, but newer sources usually 26 C°
                pet_thorn[index] = correct_to_daily[index] * (
                    -415.85
                    + 32.24 * temperature_used[index]
                    - 0.43 * temperature_used[index] ** 2
                )
                logger.info(
                    f"PET (Thornthwaite), {dates[index]}: "
                    f"temperature {temperature_used[index]:.2f} °C > 26 °C, "
                    f"using Eq. (4): {pet_thorn[index]:.2f} mm, "
                    f"Eq. (1) would give: {pet_eq1:.2f} mm."
                )
            else:
                # Use value from Eq. (1) in Pereira and Pruitt 2004
                pet_thorn[index] = pet_eq1

    return pet_thorn
