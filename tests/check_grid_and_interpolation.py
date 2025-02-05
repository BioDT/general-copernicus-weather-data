import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Define arbitrary latitude and longitude values with different lengths
latitudes = np.array([40, 30, 20, 10])  # Decreasing latitudes
longitudes = np.array([100, 110, 120, 130, 140])  # Increasing longitudes

# Create a meshgrid of latitude and longitude
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Define an arbitrary data slice (e.g., temperature values) with matching dimensions
data_slice = np.array(
    [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34],
    ]
)

# Define the coordinates of the point of interest
lat_point = 38  # Latitude between 20 and 30
lon_point = 101  # Longitude between 110 and 120

# Perform bilinear interpolation for the specific point
interpolated_value = griddata(
    (lat_grid.flatten(), lon_grid.flatten()),  # Points
    data_slice.flatten(),  # Values
    (lat_point, lon_point),  # Point of interest
    method="linear",
)

print(f"Interpolated value at ({lat_point}, {lon_point}): {interpolated_value}")

# Plot the data slice and the interpolation point for visualization
plt.figure(figsize=(8, 6))
plt.contourf(lon_grid, lat_grid, data_slice, cmap="viridis", alpha=0.7)
plt.colorbar(label="Temperature")
plt.scatter(lon_point, lat_point, color="red", label="Interpolation Point")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Data Slice and Interpolation Point")
plt.legend()
plt.show()
