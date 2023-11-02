import requests
from bs4 import BeautifulSoup
import re


def get_deims_coordinates(deims_id):
    # Construct the DEIMS.ID URL from the ID
    deims_url = f"https://deims.org/{deims_id}"

    # Send an HTTP GET request to the DEIMS.ID URL
    response = requests.get(deims_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the link to the .kml file for download
        kml_link = None
        for a in soup.find_all("a", href=True):
            if a.get_text() == "[.kml]":
                kml_link = a["href"]
                break

        # Download the .kml file
        if kml_link:
            kml_response = requests.get(
                kml_link, timeout=(100, 10)
            )  # long connectTimeout to avoid errors

            if kml_response.status_code == 200:
                # Parse the .kml file using regular expressions, search for coordinates
                kml_content = kml_response.text
                coordinates_match = re.search(
                    r"<coordinates>(.*?)</coordinates>", kml_content, re.DOTALL
                )

                if coordinates_match:
                    coordinates = coordinates_match.group(1).strip().split(",")
                    if len(coordinates) >= 2:
                        lat = float(coordinates[1].strip())
                        lon = float(coordinates[0].strip())
                        print(f"Coordinates for DEIMS.id '{deims_id}' found.")
                        print(f"Latitude: {lat}, Longitude: {lon}")
                        return {"lat": lat, "lon": lon}
                    else:
                        return {"error": "Coordinates not found in the .kml file."}
                else:
                    return {"error": "Coordinates not found in the .kml file."}
            else:
                return {
                    "error": f"Failed to download .kml file. Status code: {kml_response.status_code}"
                }
        else:
            return {"error": "No .kml file found for download."}
    else:
        return {
            "error": f"Failed to retrieve data. Status code: {response.status_code}"
        }
