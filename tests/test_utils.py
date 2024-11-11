import pytest

from copernicus.utils import get_days_in_month


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
