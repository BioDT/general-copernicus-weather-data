from setuptools import find_packages, setup

# Project metadata
name = "copernicus"
version = "0.1.0"
author = "Thomas Banitz, Franziska Taubert, Tuomas Rossi, BioDT"
description = (
    "Retrieve weather data from Copernicus and prepare as grassland model input files"
)
url = "https://github.com/BioDT/general-copernicus-weather-data"
license = "EUPL v1.2"

# Specify project dependencies from a requirements.txt file
with open("requirements.txt", "r") as req_file:
    install_requires = req_file.readlines()

# Setup configuration pip
setup(
    name=name,
    version=version,
    author=author,
    description=description,
    url=url,
    license=license,
    python_requires=">=3.10",
    install_requires=install_requires,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
