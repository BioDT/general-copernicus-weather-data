from setuptools import setup
from pathlib import Path

# Project metadata
name = "copernicus"
version = "0.1.0"
author = "Thomas Banitz, Franziska Taubert, BioDT"
description = (
    "Retrieve weather data from Copernicus and prepare as GRASSMIND input files"
)
url = "https://github.com/BioDT/general-copernicus-weather-data"
license = "MIT"

# Specify project dependencies from a requirements.txt file
with open("requirements.txt", "r") as req_file:
    install_requires = req_file.readlines()

# Setup configurationpip
setup(
    name=name,
    version=version,
    author=author,
    description=description,
    url=url,
    license=license,
    entry_points={'console_scripts':
        ['copernicus_data_processing = copernicus.data_processing:main']},
    install_requires=install_requires,
)
