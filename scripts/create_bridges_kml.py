import geopandas as gpd
import pandas as pd
import sqlite3
import os

# Paths to your files
railway_file = "assets/railway_bridges.gpkg.zip"
road_file = "assets/road_bridges.gpkg.zip"
params_db = "assets/bridge_parameters.db"

# Read the GeoPackage files - identifying the layer names
railway_layer = "tlm_oev_eisenbahn"  # From your ogrinfo output
railway_bridges = gpd.read_file(railway_file, layer=railway_layer)

# Check the layer name for road bridges
# You can replace "road_layer_name" with the actual layer name
road_layer = "tlm_strassen_strasse"  # Adjust this based on ogrinfo for road bridges
road_bridges = gpd.read_file(road_file, layer=road_layer)

# Read the SQLite database
conn = sqlite3.connect(params_db)
bridge_params = pd.read_sql_query("SELECT * FROM bridge_parameters WHERE deck_width > 0", conn)
conn.close()

# Add a source field to identify the bridge source
railway_bridges['source'] = 'railway'
road_bridges['source'] = 'road'

# Filter bridges that have entries in the parameters table
railway_bridges_filtered = railway_bridges[railway_bridges['uuid'].isin(bridge_params['uuid'])]
road_bridges_filtered = road_bridges[road_bridges['uuid'].isin(bridge_params['uuid'])]

# Merge the two bridge layers
all_bridges = pd.concat([railway_bridges_filtered, road_bridges_filtered])

# Convert to GeoDataFrame
all_bridges_gdf = gpd.GeoDataFrame(all_bridges, geometry='geometry')

# Join with the parameters table to include parameter data
joined_bridges = all_bridges_gdf.merge(bridge_params, on='uuid', how='inner')

# Export to KML (KML is always in WGS84)
kml_output = "assets/bridges_with_parameters.kml"
joined_bridges.to_file(kml_output, driver='KML')

# Ensure GeoJSON is in WGS84 (EPSG:4326)
joined_bridges_wgs84 = joined_bridges.to_crs(epsg=4326)

# Export to GeoJSON with WGS84
geojson_output = "assets/bridges_with_parameters.geojson"
joined_bridges_wgs84.to_file(geojson_output, driver='GeoJSON')

print(f"Created KML file with {len(joined_bridges)} bridges at {kml_output}")
print(f"Created GeoJSON file with {len(joined_bridges_wgs84)} bridges at {geojson_output} (WGS84)")
print(f"Each feature includes both geometry and parameter data")
