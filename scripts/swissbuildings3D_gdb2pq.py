import argparse
from osgeo import ogr
import geopandas as gpd
import dask_geopandas as dgpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from shapely import wkb
from tqdm import tqdm
import pandas as pd
import os
import psutil
import gc
import time
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import sys


def prepare_geometry(geom):
    """Ensure we always have MultiPolygon Z geometries with proper WKB encoding"""
    try:
        if geom is None or geom.is_empty:
            raise Exception("empty geometry")
                    
            
        # Convert everything to MultiPolygon
        if geom.geom_type == 'Polygon':
            raise Exception(f"This is not a multipolygon {geom}")
                        
        return geom
    except Exception as e:
        print(f"Error preparing geometry: {geom} ({e})")
        raise e

def safe_string(value):
    """Ensure string values are UTF-8 encoded and replace invalid characters. None should be an empty string"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if value is None:
        return ""
    return value  # Leave other types unchanged
    

def get_points(geom):
    points = []
    if geom and geom.GetPointCount() >= 3:
        points = [(geom.GetX(k), geom.GetY(k), geom.GetZ(k)) 
                  for k in range(geom.GetPointCount())]
    elif ogr.GeometryTypeToName(geom.GetGeometryType()) == "3D Triangle":
        ring = geom.GetGeometryRef(0)  # Get the ring/boundary of the triangle
        for i in range(ring.GetPointCount()):
            points.append(ring.GetPoint(i))  # This will give you (x, y, z)
    else:
        raise Exception(f"Cannot get points for {ogr.GeometryTypeToName(geom.GetGeometryType())}: {geom}")
            
    return points
    
def write_polygonz_chunks(gdb_path, layer_name, chunks_dir, chunk_size=1000):
    """
    Read large PolygonZ/TIN dataset and write chunks to parquet files
    
    Parameters:
    - gdb_path: Path to the FileGDB
    - layer_name: Name of the layer to process
    - chunk_size: Number of features to process in each chunk
    """
    gdb_path = Path(gdb_path)  # Convert to Path object
    layer_name = str(layer_name)
    
    # Extract prefix from gdb path
    prefix = str(gdb_path.parent)
    
    # Ensure prefix directory exists
    if prefix:
        os.makedirs(prefix, exist_ok=True)
    
    # Setup output paths
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Convert gdb_path back to string for ogr
    gdb_path = str(gdb_path)
    
    driver = ogr.GetDriverByName("OpenFileGDB")
    ds = driver.Open(gdb_path, 0)
    
    if ds is None:
        raise RuntimeError(f"Could not open {gdb_path}")
        
    layer = ds.GetLayer(layer_name)
    if layer is None:
        raise RuntimeError(f"Could not find layer {layer_name}")
    
    total_features = layer.GetFeatureCount()
    print(f"Total features to process: {total_features:,}")
    
    # Get field names and CRS
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() 
                  for i in range(layer_defn.GetFieldCount())]
    spatial_ref = layer.GetSpatialRef()
    
    # Process features
    features = []
    chunk_paths = []
    chunk_count = 0

    cnt=0
    print("Writing individual chunks...")
    for i in tqdm(range(total_features)):
        feat = layer.GetNextFeature()
        if feat is None:
            continue
        uuid = feat.GetField('UUID')
        if uuid is None or uuid == "":
            uuid = f"tmp-uuid-{cnt}"
            cnt += 1
            print(f"Warning: {feat} has no UUID defined, using {uuid}")                

        
        geom = feat.GetGeometryRef()
        if geom is not None:
            polygons = []

            debug = False
                
            if geom.GetGeometryName() == "TIN":
                for j in range(geom.GetGeometryCount()):
                    triangle = geom.GetGeometryRef(j)
                    points = get_points(triangle)
                    if len(points) >= 3:
                        polygons.append(Polygon(points))
                    else:
                        print(f"Warning: {uuid} pointcount insufficient")                                    
            elif geom.GetGeometryName() == "MULTIPOLYGON":
                polygons = []
                for j in range(geom.GetGeometryCount()):
                    poly = geom.GetGeometryRef(j)
                    if poly:
                        ring = poly.GetGeometryRef(0)
                        if ring and ring.GetPointCount() >= 3:
                            points = [(ring.GetX(k), ring.GetY(k), ring.GetZ(k)) 
                                    for k in range(ring.GetPointCount())]
                            if len(points) >= 3:
                                polygons.append(Polygon(points))
            else:
                print(f"Warning: {uuid} with {geom.GetGeometryName()} cannot be read")

            shp_geom = False
            if polygons:
                shp_geom = MultiPolygon(polygons)
            else:
                print(f"Warning: {uuid} with {geom.GetGeometryName()} no polygon")
            
            if shp_geom and not shp_geom.is_empty:
                attributes = {field: feat.GetField(field) for field in field_names}
                
                # Fix UUID if missing
                if attributes.get('UUID') in (None, ""):
                    attributes['UUID'] = uuid

                # Ensure all string fields are safe
                for field, value in attributes.items():
                    if isinstance(value, str) or isinstance(value, bytes):
                        attributes[field] = safe_string(value)
                        
                features.append({
                    'geometry': shp_geom,
                    **attributes
                })
            else:
                print(f"Warning: Geometry of {uuid} is empty")
                
            
        # Write chunk when reached chunk_size
        if len(features) >= chunk_size:
            try:
                start_time = time.time()
                
                chunk_gdf = gpd.GeoDataFrame(features, geometry='geometry')
                if spatial_ref:
                    chunk_gdf.set_crs(spatial_ref.ExportToWkt(), inplace=True)

                # Verify all geometries are MultiPolygon Z
                invalid_geoms = chunk_gdf[~chunk_gdf.geometry.apply(lambda g: g.geom_type == 'MultiPolygon')].index
                if len(invalid_geoms) > 0:
                    print(f"Warning: Found {len(invalid_geoms)} non-MultiPolygon geometries")
                    chunk_gdf = chunk_gdf.drop(invalid_geoms)

                # Ensure correct data types before writing
                for col in chunk_gdf.select_dtypes(include=['object']).columns:
                    if col != "geometry":
                        chunk_gdf[col] = chunk_gdf[col].astype(pd.StringDtype())  # Explicitly mark as string
                    
                # bug: https://github.com/apache/arrow/issues/44696 causes error when running in uv:
                # "Error processing final chunk: Attempted to register factory for scheme 'file' but that scheme is already registered."
                chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_count:02d}.parquet")
                chunk_gdf.to_parquet(
                    chunk_path,
                    index=False,
                    schema_version='1.1.0',
                    compression='snappy'
                )
                assert os.path.exists(chunk_path), f"Chunk {chunk_path} was not created, check for an internal bug"
                
                write_time = time.time()
                print(f"Parquet writing: {write_time - start_time:.2f} seconds")
                print(f"Features per second: {len(chunk_gdf) / (write_time - start_time):.2f}")
                        
                chunk_paths.append(chunk_path)
                
            except Exception as e:
                print(f"Error processing chunk {chunk_count}: {e}")
                raise
            finally:
                features = []
                if 'chunk_gdf' in locals():
                    del chunk_gdf
                gc.collect()
            
            chunk_count += 1
            
            if chunk_count % 5 == 0:
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                print(f"\nCurrent memory usage: {memory_usage:.2f} GB")
    
    # Process remaining features
    if features:
        try:
            chunk_gdf = gpd.GeoDataFrame(features, geometry='geometry')
            if spatial_ref:
                chunk_gdf.set_crs(spatial_ref.ExportToWkt(), inplace=True)
            
            chunk_gdf['geometry'] = chunk_gdf['geometry'].apply(prepare_geometry)
            chunk_gdf = chunk_gdf.dropna(subset=['geometry'])
            
            # Verify all geometries are MultiPolygon Z
            invalid_geoms = chunk_gdf[~chunk_gdf.geometry.apply(lambda g: g.geom_type == 'MultiPolygon')].index
            if len(invalid_geoms) > 0:
                print(f"Warning: Found {len(invalid_geoms)} non-MultiPolygon geometries")
                chunk_gdf = chunk_gdf.drop(invalid_geoms)

            # bug: https://github.com/apache/arrow/issues/44696 causes error when running in uv:
            # "Error processing final chunk: Attempted to register factory for scheme 'file' but that scheme is already registered."
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_count:02d}.parquet")
            chunk_gdf.to_parquet(
                chunk_path,
                index=False,
                schema_version='1.1.0',
                compression='snappy'
            )
            assert os.path.exists(chunk_path), f"Chunk {chunk_path} was not created, check for an internal bug"
            chunk_paths.append(chunk_path)
            
        except Exception as e:
            print(f"Error processing final chunk: {e}")
            raise
    
    ds = None
    
    return chunks_dir, chunk_paths


def concatenate_parquet_chunks_dask(chunk_paths, output_folder, npartitions=50):
    """
    Efficiently concatenate large parquet files using dask-geopandas and write output in partitions.
    """

    # Read with Dask
    ddf = dgpd.read_parquet(chunk_paths)

    # Repartition to control memory usage
    ddf = ddf.repartition(npartitions=npartitions)

    # Persist to avoid reloading files multiple times
    ddf = ddf.persist()

    # Write as partitioned output (no full memory load)
    print("Now writing...")
    ddf.to_parquet(
        output_folder,
        write_index=False,
        compression="snappy"
    )

    print("\nConcatenation complete!")
    print(f"Output saved to: {output_folder}")

    return output_folder

def concatenate_parquet_chunks(chunk_paths, output_file):
    """
    Concatenate parquet chunks into a single file
    """

    # Generator to read and concatenate in chunks
    def read_and_concat(files):
        for file in files:
            print(file)
            yield gpd.read_file(file)

    gdf = gpd.GeoDataFrame(pd.concat(read_and_concat(chunk_paths), ignore_index=True, copy=False))
    
    # Write final output
    print("now writing")
    gdf.to_parquet(
        output_file,
        index=False,
        schema_version='1.0.0',
        compression='snappy'
    )
    
    
    final_size = os.path.getsize(output_file) / (1024 * 1024 * 1024)
    print(f"\nConcatenation complete!")
    print(f"Output saved to: {output_file}")
    print(f"Final file size: {final_size:.2f} GB")
    
    return output_file

def validate_parquet(output_path):
    gdf = gpd.read_parquet(f"{output_path}")

    # Check geometry types
    print("\nGeometry types in dataset:")
    print(gdf.geometry.type.value_counts())

    # Check Z coordinates are present
    sample_geom = gdf.geometry.iloc[0]
    first_coords = list(sample_geom.geoms[0].exterior.coords)[:3]
    print("\nSample coordinates (first 3 points):")
    print(f"Each point has Z coordinate: {all(len(coord) == 3 for coord in first_coords)}")
    print(f"Sample points: {first_coords}")

    # Validate all geometries
    print("\nValidating geometries:")
    print(f"Valid geometries: {gdf.geometry.is_valid.all()}")
    print(f"Empty geometries: {gdf.geometry.is_empty.sum()}")

    # Check CRS
    print("\nCRS information:")
    print(gdf.crs)
        
    # Check to see if we can read it with other tools
    try:
        import pyogrio
        gdf_test = pyogrio.read_dataframe(output_path)
        print("\nCan read with pyogrio: Yes")
    except Exception as e:
        print(f"\nCan read with pyogrio: No {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert SwissBuildings3D GDB to Parquet format')
    parser.add_argument('gdb_path', type=str, help='Path to the FileGDB')
    parser.add_argument('layer_name', type=str, help='Name of the layer to process')
    parser.add_argument('--chunk-size', type=int, default=3000, help='Number of features per chunk')
    args = parser.parse_args()

    # Extract prefix from gdb path and create output paths
    gdb_path = Path(args.gdb_path)
    prefix = str(gdb_path.parent)
    print(f"Processing {gdb_path} with layer {args.layer_name}")
    
    # Create output filename based on input GDB name and layer
    gdb_stem = gdb_path.stem.replace('.gdb', '')
    output_folder = Path(prefix) / f"{gdb_stem}_{args.layer_name}"
    print(f"Output folder {output_folder}")

    # Run Chunks
    chunks_dir, chunk_paths = write_polygonz_chunks(
       str(gdb_path), 
       args.layer_name,
       output_folder,
       chunk_size=args.chunk_size
    )

    # only validate for single parquet files
    if len(chunk_paths) == 1:
        validate_parquet(chunks_dir)
