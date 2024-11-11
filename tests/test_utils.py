import pytest

import copernicus.utils as ut


def test_is_leap_year():
    """Test leap year logic."""
    assert ut.is_leap_year(2020)
    assert not ut.is_leap_year(2019)
    assert not ut.is_leap_year(1900)  # Century non-leap year
    assert ut.is_leap_year(2000)  # Century leap year


def test_get_days_in_month():
    """Test the number of days in each month for both leap and non-leap years."""
    # Test valid months
    assert ut.get_days_in_month(2021, 1) == 31  # January
    assert ut.get_days_in_month(2021, 2) == 28  # February (non-leap year)
    assert ut.get_days_in_month(2020, 2) == 29  # February (leap year)
    assert ut.get_days_in_month(2021, 3) == 31  # March
    assert ut.get_days_in_month(2021, 4) == 30  # April
    assert ut.get_days_in_month(2021, 5) == 31  # May
    assert ut.get_days_in_month(2021, 6) == 30  # June
    assert ut.get_days_in_month(2021, 7) == 31  # July
    assert ut.get_days_in_month(2021, 8) == 31  # August
    assert ut.get_days_in_month(2021, 9) == 30  # September
    assert ut.get_days_in_month(2021, 10) == 31  # October
    assert ut.get_days_in_month(2021, 11) == 30  # November
    assert ut.get_days_in_month(2021, 12) == 31  # December

    # Test invalid month
    with pytest.raises(ValueError):
        ut.get_days_in_month(2021, 0)  # Invalid month (0)

    with pytest.raises(ValueError):
        ut.get_days_in_month(2021, 13)  # Invalid month (13)
