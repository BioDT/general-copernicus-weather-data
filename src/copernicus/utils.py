import os
import pytz
import datetime
from timezonefinder import TimezoneFinder as tzf
from suntime import Sun, SunTimeException


def is_dict_of_2_floats(variable):
    if isinstance(variable, dict) and len(variable) == 2:
        for key, value in variable.items():
            if not isinstance(value, float):
                return False
        return True
    return False


def is_leap_year(year):
    # A year is a leap year if it is divisible by 4,
    # except for years that are divisible by 100 but not by 400
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def get_days_in_month(year, month):
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
    # Generate day value strings based on the number of days in the month
    return [str(i).zfill(2) for i in range(1, get_days_in_month(year, month) + 1)]


def format_offset(offset_seconds):
    hours, remainder = divmod(abs(offset_seconds), 3600)
    minutes = remainder // 60
    sign = "+" if offset_seconds >= 0 else "-"
    offset_str = f"UTC{sign}{hours:02}:{minutes:02}"

    return offset_str


def get_time_zone(coordinates):
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


def get_day_length(coordinates, date_str_list):
    # Get day light durations in hours for given location and dates
    sun = Sun(coordinates["lat"], coordinates["lon"])
    day_lengths = []

    for date_str in date_str_list:
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

    return day_lengths


def get_data_suffix(data_format):
    #  Determine the data file ending based on the data format
    if data_format == "netcdf":
        data_suffix = ".nc"
    elif data_format == "grib":
        data_suffix = ".grib"
    else:
        raise ValueError("Unsupported data format")

    return data_suffix
