from setuptools import setup, find_packages
from pathlib import Path

# Project metadata
name = "CopernicusWeatherData"
version = "1.0.0"
author = "Thomas Banitz, Franziska Taubert, BioDT"
description = (
    "Retrieve weather data from Copernicus and prepare as GRASSMIND input files"
)
url = ""  # Replace with appropriate gitlab url
license = ""  # Replace with appropriate license

# Define the project's packages (if any)
packages = find_packages()

# # List script files in the 'scripts' folder, attempt to make modules from "scripts" folder available, didn't work.
# script_files = [f"scripts.{p.stem}" for p in Path("scripts").glob("*.py")]

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
    packages=packages,
    # py_modules=script_files,
    install_requires=install_requires,
)
