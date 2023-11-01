import argparse

# import sys
# sys.path.append("scripts")
from scripts import get_weather_data as gwd
from scripts import get_deims_coordinates as gdc


def main(
    data_sets,
    data_format,
    data_resolution,
    years,
    months,
    coordinates,
    deims_id,
):
    if years is None:
        years = list(range(2013, 2014))  # list(range(..., ...))

    if months is None:
        months = list(range(1, 3))  # list(range(1, 13))

    deims_id = "102ae489-04e3-481d-97df-45905837dc1a"  # GCEF site
    # deims_id = "474916b5-8734-407f-9179-109083c031d8"  # Doode Bemde site, Belgium

    if coordinates is None:
        if deims_id:
            coordinates = gdc.get_deims_coordinates(deims_id)
        else:
            raise ValueError(
                "No location defined. Please provide coordinates or DEIMS.iD!"
            )

    months_list = gwd.construct_months_list(years, months)
    data_var_specs = gwd.get_var_specs()

    for data_set in data_sets:
        data_requests = gwd.configure_data_request(
            data_set,
            data_var_specs,
            data_format,
            data_resolution,
            coordinates,
            months_list,
        )

        gwd.download_weather_data(data_set, data_requests, data_resolution)

    gwd.weather_data_2_txt_file(
        data_sets,
        data_var_specs,
        data_format,
        data_resolution,
        deims_id,
        coordinates,
        months_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your description here")

    # Define command-line arguments
    parser.add_argument(
        "--data_sets",
        nargs="*",
        default=["reanalysis-era5-land", "reanalysis-era5-single-levels"],
        help="List of data sets",
    )
    parser.add_argument(
        "--data_format",
        default="netcdf",
        choices=["netcdf", "grib"],
        help="Data format",
    )
    parser.add_argument(
        "--data_resolution",
        default="hourly",
        choices=["hourly", "daily"],
        help="Data resolution",
    )
    parser.add_argument("--years", nargs="*", type=int, help="List of years")
    parser.add_argument("--months", nargs="*", type=int, help="List of months")
    parser.add_argument(
        "--coordinates",
        type=lambda s: dict(lat=float(s.split(",")[0]), lon=float(s.split(",")[1])),
        help="Coordinates as 'lat,lon'",
    )
    parser.add_argument("--deims_id", help="DEIMS.iD")

    args = parser.parse_args()

    main(
        data_sets=args.data_sets,
        data_format=args.data_format,
        data_resolution=args.data_resolution,
        years=args.years,
        months=args.months,
        coordinates=args.coordinates,
        deims_id=args.deims_id,
    )
