"""
Module Name: data_processing.py
Description: Function for obtaining selected weather data at given location
             (0.1° x 0.1° spatial resolution) for desired time periods, at hourly
             resolution, from Copernicus ERA5-Land dataset.

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

Data source:
    ERA5-Land hourly data from 1950 to present:
    - Muñoz Sabater, J. (2019):
      ERA5-Land hourly data from 1950 to present.
      Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.e2161bac
    - Website: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
    - Documentation: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
    - Full Licence to Use Copernicus Products: https://apps.ecmwf.int/datasets/licences/copernicus/
    - Access via Climate Data Store (CDS) Application Program Interface (API):
        - URL: https://cds.climate.copernicus.eu/api
        - Documentation: https://cds.climate.copernicus.eu/how-to-api
        - Python package 'cdsapi': https://pypi.org/project/cdsapi/
        - Access to the CDS API requires:
            - An account
            - Agreement to the Terms of Use of the "ERA5-Land hourly data from 1950 to present" data set
            - A CDS API personal access token (usually put in a '$HOME/.cdsapirc' file)
            - See detailed instructions at: https://cds.climate.copernicus.eu/how-to-api
"""

from copernicus import get_weather_data as gwd


def data_processing(
    years,
    months,
    coordinates,
    *,
    final_resolution="daily",
    target_folder=None,
):
    """
    Download data from CDS API. Convert to .txt files.

    Parameters:
        years (list of int): Years list.
        months (list of int): Months list.
        coordinates (dict): Dictionary with 'lat' and 'lon' keys ({'lat': float, 'lon': float}).
        final_resolution (str): Resolution for final text file ('hourly' or 'daily', default is 'daily').
        target_folder (str or Path): Target folder for .txt files (default is 'weatherDataPrepared').
    """
    if "lat" in coordinates and "lon" in coordinates:
        print(
            f"Preparing weather data for latitude: {coordinates['lat']}, longitude: {coordinates['lon']} ..."
        )
    else:
        raise ValueError(
            "Coordinates not correctly defined. Please provide as dictionary ({'lat': float, 'lon': float})!"
        )

    # Prepare requests
    months_list = gwd.construct_months_list(years, months)
    data_var_specs = gwd.get_var_specs()
    data_requests = gwd.configure_data_request(data_var_specs, coordinates, months_list)

    # Download raw data
    cds_api_url = gwd.download_weather_data(data_requests)

    # Process raw data to final files
    gwd.weather_data_to_txt_file(
        data_var_specs,
        coordinates,
        months_list,
        final_resolution=final_resolution,
        target_folder=target_folder,
        data_source=cds_api_url,
    )
