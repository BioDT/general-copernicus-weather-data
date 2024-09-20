"""
Module Name: data_processing.py
Description: Building block for obtaining selected weather data at given location
             (0.1° x 0.1° spatial resolution) for desired time periods, at hourly
             resolution, e.g. from Copernicus ERA5-Land dataset.

Copyright (C) 2024
- Thomas Banitz, Franziska Taubert, Helmholtz Centre for Environmental Research GmbH - UFZ, Leipzig, Germany
- Tuomas Rossi, CSC – IT Center for Science Ltd., Espoo, Finland

Licensed under the EUPL, Version 1.2 or - as soon they will be approved
by the European Commission - subsequent versions of the EUPL (the "Licence").
You may not use this work except in compliance with the Licence.

You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl
"""

from copernicus import get_weather_data as gwd
from copernicus import utils as ut


def data_processing(
    data_sets,
    final_resolution,
    years,
    months,
    coordinates,
    deims_id,
):
    """
    Download data from CDS API. Convert to .txt files.

    Parameters:
        data_sets (list of str): Names of Copernicus datasets.
        final_resolution (str): Resolution for final text file ("hourly" or "daily").
        years (list of int): Years list.
        months (list of int): Months list.
        coordinates (list of dict): List of dictionaries with "lat" and "lon" keys.
        deims_id (str): Identifier of the eLTER site.
    """
    # hard coded because only options for now, but still passed as args to functions using them
    data_resolution = "hourly"
    data_format = "netcdf"

    if coordinates is None:
        if deims_id:
            coordinates = ut.get_deims_coordinates(deims_id)
        else:
            raise ValueError(
                "No location defined. Please provide coordinates or DEIMS.iD!"
            )

    months_list = gwd.construct_months_list(years, months)
    data_var_specs = gwd.get_var_specs()

    for data_set in data_sets:  # can be simplified when only ERA5-Land dataset is used
        data_requests = gwd.configure_data_request(
            data_set,
            data_var_specs,
            data_format,
            data_resolution,
            coordinates,
            months_list,
        )

        gwd.download_weather_data(data_set, data_requests, data_resolution)

    gwd.weather_data_to_txt_file(
        data_sets,
        data_var_specs,
        data_format,
        data_resolution,
        final_resolution,
        deims_id,
        coordinates,
        months_list,
    )
