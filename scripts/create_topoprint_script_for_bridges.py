import osgeo.ogr as ogr
import osgeo.osr as osr
import sqlite3
import gzip
import base64
import json
import math

import logging

# Setup logger
logger = logging.getLogger(__name__)

# Add a formatter that includes timestamp, level, and module
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Add INFO level logging to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# Define tile size in mm
tile_size_mm = 150
min_buffer_distance = 24
MAX_SCALE=2000

# Create a helper function to round coordinates
def round_coordinates(geometry, decimals=6):
    """Recursively round coordinates in GeoJSON geometry to a fixed number of decimals."""
    if geometry["type"] == "Point":
        geometry["coordinates"] = [round(coord, decimals) for coord in geometry["coordinates"]]
    elif geometry["type"] in ["LineString", "MultiPoint"]:
        geometry["coordinates"] = [[round(coord, decimals) for coord in point] for point in geometry["coordinates"]]
    elif geometry["type"] in ["Polygon", "MultiLineString"]:
        geometry["coordinates"] = [
            [[round(coord, decimals) for coord in point] for point in ring]
            for ring in geometry["coordinates"]
        ]
    elif geometry["type"] == "MultiPolygon":
        geometry["coordinates"] = [
            [[[round(coord, decimals) for coord in point] for point in ring] for ring in polygon]
            for polygon in geometry["coordinates"]
        ]
    return geometry

# for which datasets do I want to create a perimeter2stl.sh 
process_list = [
    {
        "filename" : "assets/road_bridges.gpkg.zip",
        "layer":  "tlm_strassen_strasse"
    },
    {
        "filename" : "assets/railway_bridges.gpkg.zip",
        "layer":  "tlm_oev_eisenbahn"
    }
]
    

# Open the GPKG file
gpkg_driver = ogr.GetDriverByName("GPKG")

# As the output is a bash script, make sure we run the script in a "strict" mode
print("#!/usr/bin/env bash")
print("set -euo pipefail")
print("IFS=$'\n\t'")

for item in process_list:
    gpkg = gpkg_driver.Open(item["filename"], 0)
    layer = gpkg.GetLayer(item["layer"])

    # Set up input and output spatial references
    input_srs = layer.GetSpatialRef()  # This is EPSG:2056
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(4326)  # This is WGS84 (EPSG:4326)
    output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # Set up coordinate transformation from input_srs to output_srs
    transform = osr.CoordinateTransformation(input_srs, output_srs)

    # Connect to SQLite database
    conn = sqlite3.connect("assets/bridge_parameters.db")
    cursor = conn.cursor()

    # Get valid UUIDs from bridge_parameters table
    cursor.execute("SELECT UUID FROM bridge_parameters")
    valid_uuids = {row[0] for row in cursor.fetchall()}

    # Process features and create base64 GeoJSON output
    for feature in layer:
        feature_uuid = feature.GetField("UUID")
        if feature_uuid in valid_uuids:
            # Get the geometry and create a buffer
            geom = feature.GetGeometryRef()
            # buffer should be at least 10 meters, but no more than 10% of the line length
            buffer_distance = max(min_buffer_distance, geom.Length() * 0.1) 
            buffer_geom = geom.Buffer(buffer_distance)

            # Calculate the envelope in the original coordinate system (EPSG:2056)
            env = buffer_geom.GetEnvelope()  # Returns (minX, maxX, minY, maxY)
            width_meters = env[1] - env[0]   # width in meters (since EPSG:2056 is in meters)
            height_meters = env[3] - env[2]  # height in meters

            # Get the larger dimension in meters
            logger.info(f"uuid={feature_uuid}")
            logger.debug(f"width_meters={width_meters:.2f}")
            logger.debug(f"height_meters={height_meters:.2f}")
            max_dimension_meters = max(width_meters, height_meters)

            # Calculate scale: how many meters per mm on the print
            # We want the max dimension to fit within tile_size_mm
            scale = int(max_dimension_meters / ((1/1000) * tile_size_mm)) + 10
            if scale > MAX_SCALE:
                scale = MAX_SCALE
            logger.debug(f"scale= 1:{scale}")
            logger.debug(f"tile_size_mm={tile_size_mm:.2f}")

            # Now transform the buffered geometry to WGS84 for the GeoJSON output
            buffer_geom.Transform(transform)

            # Create a GeoJSON object for the buffered geometry
            geojson_feature = {
                "type": "Feature",
                "geometry": json.loads(buffer_geom.ExportToJson()),
                "properties": {}
            }

            # Round coordinates to 6 decimal places
            geojson_feature["geometry"] = round_coordinates(geojson_feature["geometry"], decimals=6)
            
            geojson = {
                "type": "FeatureCollection",
                "features": [geojson_feature]
            }

            # Convert the GeoJSON to string
            geojson_text = json.dumps(geojson)

            # Compress the GeoJSON using gzip
            compressed = gzip.compress(geojson_text.encode('utf-8'), compresslevel=9)

            # Encode the compressed data in Base64
            base64_encoded = base64.b64encode(compressed).decode('utf-8')

            # We already calculated the scale in the original coordinate system
            
            # Print the perimeter2stl.sh command with the base64 string
            print(f'bash scripts/perimeter2stl.sh -u "bridge_{feature_uuid}" --scale {scale} --tile-size {tile_size_mm} --polygon {base64_encoded} --skip-web-view')

    # Final cleanup
    layer = None
    gpkg = None
    cursor.close()
    conn.close()
