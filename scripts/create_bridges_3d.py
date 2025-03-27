import os
import sys
from osgeo import ogr
import geopandas as gpd
import pandas as pd
import shapely 
import numpy as np
import zlib
import re
import pyvista as pv
import numpy as np
import zlib
import io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import pyvista as pv
import numpy as np
import logging
from shapely.geometry import (
    LineString, MultiLineString, Point, MultiPoint,
    Polygon, MultiPolygon, GeometryCollection
)
import httpx
import json
from shapely.geometry import LineString
from hishel import CacheClient, FileStorage

from bridge_creation import create_bridge

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


def get_min_height_swissalti_service(linestring: LineString) -> float:
    coords = [[round(x), round(y)] for x, y, *_ in linestring.coords]
    
    # If there are more than 4094 coordinates, thin the list to take only every n-th coordinate
    if len(coords) > 4094:
        # Calculate how many points to skip to get under 4094 points
        n = len(coords) // 4094 + 1
        coords = coords[::n]
    
    geom = {"type": "LineString", "coordinates": coords}
    geom_str = json.dumps(geom)

    url = "https://api3.geo.admin.ch/rest/services/profile.json"
    params = {"geom": geom_str}

    try:
        # Create a cache storage with 30-day TTL
        storage = FileStorage(ttl=60 * 60 * 24 * 30)  # 30 days in seconds
        
        # Create a client with the cache
        with CacheClient(storage=storage) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()  # List of points

        heights = [pt["alts"]["COMB"] for pt in data if "alts" in pt and "COMB" in pt["alts"]]
        if not heights:
            raise httpx.HTTPStatusError("No valid elevation data returned.")
        return min(heights)
    except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
        if e.response.status_code >= 400:
            # If status is 400, calculate a fallback height
            # Find the smallest z-coordinate and subtract 100m
            z_coords = [coord[2] for coord in linestring.coords if len(coord) > 2]
            if z_coords:
                return min(z_coords) - 100
            else:
                # If no z-coordinates are available, use a default value
                return 0  # or some other sensible default
        else:
            # Re-raise other HTTP errors
            raise

def get_min_z(geom):
    def extract_z_coords(g):
        if isinstance(g, (LineString, Point)):
            return [c[2] for c in g.coords if len(c) > 2]
        elif isinstance(g, (Polygon)):
            zs = [c[2] for ring in [g.exterior, *g.interiors] for c in ring.coords if len(c) > 2]
            return zs
        elif isinstance(g, (MultiLineString, MultiPoint, MultiPolygon, GeometryCollection)):
            zs = []
            for part in g.geoms:
                zs.extend(extract_z_coords(part))
            return zs
        else:
            return []

    zs = extract_z_coords(geom)
    return min(zs) if zs else None


def mesh_to_compressed_obj(mesh):
    """
    Convert a PyVista mesh to a zlib compressed OBJ string.
    
    Parameters:
    -----------
    mesh : pyvista.PolyData or pyvista.UnstructuredGrid
        The PyVista mesh to convert
    
    Returns:
    --------
    bytes
        zlib compressed OBJ data
    """
    # Ensure the mesh is a PolyData object
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    
    # Create a string buffer to hold the OBJ data
    obj_buffer = io.StringIO()
    
    # Write vertex positions
    vertices = mesh.points
    for v in vertices:
        obj_buffer.write(f"v {v[0]} {v[1]} {v[2]}\n")
    
    # Write texture coordinates if they exist
    if mesh.active_texture_coordinates is not None:
        for tc in mesh.active_texture_coordinates:
            obj_buffer.write(f"vt {tc[0]} {tc[1]}\n")
    
    # Write normals if they exist
    if mesh.active_normals is not None:
        for n in mesh.active_normals:
            obj_buffer.write(f"vn {n[0]} {n[1]} {n[2]}\n")
    
    # Write face definitions
    faces = mesh.faces
    i = 0
    while i < len(faces):
        n_points = faces[i]
        i += 1
        face_str = "f"
        for j in range(n_points):
            # OBJ indices are 1-based
            vertex_index = faces[i] + 1
            i += 1
            face_str += f" {vertex_index}"
        obj_buffer.write(face_str + "\n")
    
    # Get the OBJ data as a string
    obj_str = obj_buffer.getvalue()
    
    # Compress the OBJ string using zlib
    compressed_obj = zlib.compress(obj_str.encode('utf-8'))
    
    return compressed_obj

def get_bridge_parameters_from_db(bridge_uuid, db_path="assets/bridge_parameters.db"):
    """
    Check if there are custom bridge parameters in the SQLite database.
    
    Parameters:
    -----------
    bridge_uuid : str or int
        UUID for the bridge feature to look up (UUID is from Swisstopo)
    db_path : str, optional
        Path to the SQLite database file
        
    Returns:
    --------
    dict or None
        Dictionary of bridge parameters if found, None otherwise
    """
    if not os.path.exists(db_path):
        logger.warning(f"No DB found for {db_path}")
        return None

    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bridge_parameters'")
        if not cursor.fetchone():
            logger.warning(f"table bridge_parameters doesn't seem to exist for {db_path}")
            conn.close()
            return None
            
        # Query for parameters for this specific bridge
        cursor.execute(
            "SELECT deck_width, bottom_shift_percentage, arch_fractions, pier_size, "
            "circular_arch, arch_height_fraction FROM bridge_parameters WHERE uuid = ?", 
            (str(bridge_uuid),)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Parse arch_fractions from string if it exists
            arch_fractions_str = result[2]
            if arch_fractions_str and arch_fractions_str.strip():
                try:
                    arch_fractions = [float(x) for x in arch_fractions_str.split(',')]
                except ValueError:
                    arch_fractions = None
            else:
                arch_fractions = None
                
            return {
                'deck_width': float(result[0]) if result[0] is not None else None,
                'bottom_shift_percentage': float(result[1]) if result[1] is not None else None,
                'arch_fractions': arch_fractions,
                'pier_size_meters': float(result[3]) if result[3] is not None else None,
                'circular_arch': bool(result[4]) if result[4] is not None else None,
                'arch_height_fraction': float(result[5]) if result[5] is not None else None
            }
        return None
    except Exception as e:
        print(f"Error reading bridge parameters from database: {e}")
        return None

def _process_feature(feature_data):
    """Process a single feature in a separate process"""
    feature, field_names = feature_data
    
    attributes = {}
    for field in field_names:
        attributes[field] = feature.GetField(field)
    
    geometry = feature.GetGeometryRef()
    if geometry is None:
        return None
        
    # Check if it's a line geometry
    geom_type = geometry.GetGeometryType()
    if geom_type not in [ogr.wkbLineString, ogr.wkbLineString25D, 
                         ogr.wkbMultiLineString, ogr.wkbMultiLineString25D]:
        print(f"Skipping non-line geometry with type: {geometry.GetGeometryName()}")
        return None
        
    wkt = geometry.ExportToWkt()
    shapely_geom = shapely.wkt.loads(wkt)

    # calc length
    length_3d = shapely_geom.length

    ## create bridge, first define parameters
    artificial_height = 60  # TODO: see below
    default_railway_width = 8
    default_road_width = 3
    autobahn_width = 20
    if "anzahl_spuren" in field_names: # Railway Bridges
        deck_width= default_railway_width * int(feature.GetField("anzahl_spuren"))
    elif "objektart" in field_names: # Road Bridges
        objektart = feature.GetField("objektart")
        match = re.search(r"(\d+)m\s.*", objektart)
        if match:
            deck_width = int(match.group(1)) 
            deck_width *= (
                1.3  # some factor to take real-world additional width into account
            )
        else:
            if objektart == "Autobahn":
                deck_width = autobahn_width
            else:
                deck_width = default_road_width
    else:
        raise Exception(f"Cannot determine the width of the bridge deck for {attributes}")

        
    bottom_shift_percentage = 0
    # get the min elevation based on the SwissTopo DEM service, and put in 10 m into the ground to be sure
    min_elevation = get_min_height_swissalti_service(shapely_geom) - 10
    #min_elevation = get_min_z(shapely_geom) - artificial_height # TODO: get the real minimum z
    if length_3d < 20:
        arch_fractions = None
        pier_size_meters = max(1, length_3d * 0.15)
    else:
        n = int(length_3d / 20)
        arch_fractions = [ 1/n ] * n
        pier_size_meters = 3

    circular_arch = False
    arch_height_fraction = 0.8
    
    bridge_mesh, footprint = create_bridge(
        shapely_geom,
        deck_width=deck_width,
        bottom_shift_percentage=bottom_shift_percentage,
        min_elevation=min_elevation,
        arch_fractions=arch_fractions,
        pier_size_meters=pier_size_meters,
        circular_arch=circular_arch,
        arch_height_fraction=arch_height_fraction,
    )
    bridge_obj = mesh_to_compressed_obj(bridge_mesh)
    attributes["obj"] = bridge_obj

    # save the footprint polygon 
    attributes["geometry"] = footprint

    max_z = max([coord[-1] for coord in shapely_geom.coords])
    min_z = min([coord[-1] for coord in shapely_geom.coords])
    attributes["dach_max"] = max_z
    attributes["dach_min"] = min_z
    
    return attributes

def read_ogr_dataset(input_path, parallel=False, max_workers=None):
    """
    Read a line dataset using osgeo.ogr and return a GeoDataFrame
    
    Parameters:
    -----------
    input_path : str
        Path to the input dataset (shapefile, GeoJSON, etc.)
    parallel : bool, optional
        Whether to process features in parallel using ProcessPoolExecutor
    max_workers : int, optional
        Maximum number of worker processes to use. If None, defaults to number of CPU cores.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing the line geometries and attributes
    """
    # Open the dataset with OGR
    data_source = ogr.Open(input_path)
    
    if data_source is None:
        raise ValueError(f"Could not open {input_path}")
    
    # Get the first layer
    layer = data_source.GetLayer()
    srs = layer.GetSpatialRef()
    
    # Get the field names
    layer_defn = layer.GetLayerDefn()
    field_names = []
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        field_names.append(field_defn.GetName())
    
    # Extract features
    if not parallel:
        # Original sequential processing
        features = []
        # Get total feature count for progress bar
        feature_count = layer.GetFeatureCount()
        from tqdm import tqdm
        for i, feature in enumerate(tqdm(layer, total=feature_count, desc="Processing features")):
            attributes = {}
            for field in field_names:
                attributes[field] = feature.GetField(field)
            geometry = feature.GetGeometryRef()
            if geometry is not None:
                # Check if it's a line geometry
                geom_type = geometry.GetGeometryType()
                if geom_type in [ogr.wkbLineString, ogr.wkbLineString25D, 
                                 ogr.wkbMultiLineString, ogr.wkbMultiLineString25D]:
                    wkt = geometry.ExportToWkt()
                    shapely_geom = shapely.wkt.loads(wkt)

                    # calc length
                    length_3d = shapely_geom.length

                    ## create bridges,
                    ## - let's define some sensible default parameters 
                    default_railway_width = 8
                    default_road_width = 3
                    autobahn_width = 20

                    # - let's check feature attributes for some clues
                    if "anzahl_spuren" in field_names: # Railway Bridges
                        deck_width= default_railway_width * int(feature.GetField("anzahl_spuren"))
                    elif "objektart" in field_names: # Road Bridges
                        objektart = feature.GetField("objektart")
                        match = re.search(r"(\d+)m\s.*", objektart)
                        if match:
                            deck_width = int(match.group(1)) 
                            deck_width *= (
                                1.3  # some factor to take real-world additional width into account
                            )
                        else:
                            if objektart == "Autobahn":
                                deck_width = autobahn_width
                            else:
                                deck_width = default_road_width
                    else:
                        raise Exception("Cannot determine the width of the bridge deck for {attributes}")

                    # we don't assume the piers to be larger at the bottom by default
                    bottom_shift_percentage = 0
                    
                    # get the min elevation based on the SwissTopo DEM service, and put in 10 m into the ground to be sure
                    min_elevation = get_min_height_swissalti_service(shapely_geom) - 10

                    # let's define arch_fractions and pier size based on the feature's length
                    if length_3d < 20:
                        arch_fractions = None
                        pier_size_meters = max(1, length_3d * 0.15)
                    else:
                        n = int(length_3d / 20)
                        arch_fractions = [ 1/n ] * n
                        pier_size_meters = 3

                    # by default, archs are not circular and have a certain height
                    circular_arch = False
                    arch_height_fraction = 0.8

                    # Check if there are custom parameters in the database
                    bridge_uuid = feature.GetField("UUID")  # Use feature ID or index as fallback
                    assert bridge_uuid, "Undefined UUID for a bridge feature"
                    custom_params = get_bridge_parameters_from_db(bridge_uuid)

                    if custom_params:
                        # Override defaults with custom parameters where provided
                        logger.info(f"found custom parameters for {bridge_uuid}")
                        if custom_params['deck_width'] is not None:
                            deck_width = custom_params['deck_width']
                        if custom_params['bottom_shift_percentage'] is not None:
                            bottom_shift_percentage = custom_params['bottom_shift_percentage']
                        if custom_params['arch_fractions'] is not None:
                            arch_fractions = custom_params['arch_fractions']
                        if custom_params['pier_size_meters'] is not None:
                            pier_size_meters = custom_params['pier_size_meters']
                        if custom_params['circular_arch'] is not None:
                            circular_arch = custom_params['circular_arch']
                        if custom_params['arch_height_fraction'] is not None:
                            arch_height_fraction = custom_params['arch_height_fraction']
                        logger.debug("Custom parameters: deck_width={}, bottom_shift_percentage={}, arch_fractions={}, "
                                     "pier_size_meters={}, circular_arch={}, arch_height_fraction={}".format(
                                         custom_params['deck_width'], 
                                         custom_params['bottom_shift_percentage'], 
                                         custom_params['arch_fractions'], 
                                         custom_params['pier_size_meters'], 
                                         custom_params['circular_arch'], 
                                         custom_params['arch_height_fraction']
                                    ))                        
                        # if deck_width is deliberately set to 0, we ignore this bridge feature
                        if deck_width == 0:
                            logger.info(f"the custom deck_width for {bridge_uuid} is 0, we are skipping this.")
                            continue
                        
                    bridge_mesh, footprint = create_bridge(
                        shapely_geom,
                        deck_width=deck_width,
                        bottom_shift_percentage=bottom_shift_percentage,
                        min_elevation=min_elevation,
                        arch_fractions=arch_fractions,
                        pier_size_meters=pier_size_meters,
                        circular_arch=circular_arch,
                        arch_height_fraction=arch_height_fraction,
                    )
                    bridge_obj = mesh_to_compressed_obj(bridge_mesh)
                    attributes["obj"] = bridge_obj

                    # save the footprint polygon 
                    attributes["geometry"] = footprint

                    max_z = max([coord[-1] for coord in shapely_geom.coords])
                    min_z = min([coord[-1] for coord in shapely_geom.coords])
                    attributes["dach_max"] = max_z
                    attributes["dach_min"] = min_z
                    
                    features.append(attributes)
                else:
                    print(f"Skipping non-line geometry with type: {geometry.GetGeometryName()}")
    else:
        # Parallel processing using ProcessPoolExecutor
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
            
        print(f"Processing features in parallel with {max_workers} workers...")
        
        # Collect all features first to avoid issues with OGR layer iteration
        feature_data = []
        for feature in layer:
            # Clone the feature to ensure it's safe to use in another process
            feature_clone = feature.Clone()
            feature_data.append((feature_clone, field_names))
        
        # Process features in parallel
        features = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_feature, feature_data))
            
        # Filter out None results and add valid ones to features list
        features = [result for result in results if result is not None]
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(features)
    
    # Set the CRS if available
    if srs:
        gdf.crs = srs.ExportToProj4()
    
    return gdf


def convert_to_geoparquet(input_path, output_path, parallel=False, max_workers=None):
    """
    Convert an OGR-readable line dataset to GeoParquet
    
    Parameters:
    -----------
    input_path : str
        Path to the input dataset
    output_path : str
        Path for the output GeoParquet file
    parallel : bool, optional
        Whether to process features in parallel
    max_workers : int, optional
        Maximum number of worker processes to use
    """
    try:
        # Read the data using OGR
        print(f"Reading data from {input_path}...")
        gdf = read_ogr_dataset(input_path, parallel=parallel, max_workers=max_workers)
        
        if gdf.empty:
            print("No line geometries found in the dataset.")
            return
            
        # Save to GeoParquet
        print(f"Saving to GeoParquet file: {output_path}")
        gdf.to_parquet(output_path,
                       index=False,
                       schema_version='1.1.0',
                       write_covering_bbox=True,
                       compression='snappy')
        print(f"Successfully converted {len(gdf)} features to GeoParquet!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert line geometries to 3D bridge models in GeoParquet format")
    parser.add_argument("input_path", help="Path to the input dataset")
    parser.add_argument("output_path", help="Path for the output GeoParquet file")
    parser.add_argument("--parallel", action="store_true", help="Process features in parallel")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (defaults to CPU count)")
    
    args = parser.parse_args()
    
    # Add .parquet extension if not present
    output_path = args.output_path
    if not output_path.endswith('.parquet'):
        output_path += '.parquet'
    
    convert_to_geoparquet(args.input_path, output_path, parallel=args.parallel, max_workers=args.workers)
