import argparse
from osgeo import ogr
import geopandas as gpd
# import dask_geopandas as dgpd
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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import queue

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
        points = [(geom.GetX(k), geom.GetY(k), geom.GetZ(k)) for k in range(geom.GetPointCount())]
    elif ogr.GeometryTypeToName(geom.GetGeometryType()) == "3D Triangle":
        ring = geom.GetGeometryRef(0)  # Get the ring/boundary of the triangle
        for i in range(ring.GetPointCount()):
            points.append(ring.GetPoint(i))  # This will give you (x, y, z)
    else:
        raise Exception(f"Cannot get points for {ogr.GeometryTypeToName(geom.GetGeometryType())}: {geom}")

    return points


def _process_ranges_worker(gdb_path, layer_name, assigned_ranges, worker_id, chunks_dir, progress_queue=None):
    """Worker process: open GDB, iterate over assigned (start,end,idx) ranges, write chunks.
    Reports progress via progress_queue as (worker_id, increment) messages.
    Returns list[(chunk_idx, chunk_path)].
    """
    try:
        driver = ogr.GetDriverByName("OpenFileGDB")
        ds = driver.Open(str(gdb_path), 0)
        if ds is None:
            raise RuntimeError(f"Could not open {gdb_path}")
        layer = ds.GetLayer(str(layer_name))
        if layer is None:
            raise RuntimeError(f"Could not find layer {layer_name}")

        # Collect field names and spatial ref inside the worker
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        spatial_ref = layer.GetSpatialRef()
        spatial_ref_wkt = spatial_ref.ExportToWkt() if spatial_ref else None

        produced = []
        for (start_idx, end_idx, chunk_index) in assigned_ranges:
            # Seek to start index if supported
            try:
                layer.SetNextByIndex(start_idx)
            except Exception:
                # fallback: iterate until start
                for _ in range(start_idx):
                    _ = layer.GetNextFeature()

            features = []
            local_missing_uuid = 0
            to_read = max(0, end_idx - start_idx)
            for _ in range(to_read):
                feat = layer.GetNextFeature()
                if feat is None:
                    break

                uuid = feat.GetField('UUID')
                if uuid is None or uuid == "":
                    uuid = f"tmp-uuid-{chunk_index}-{local_missing_uuid}"
                    local_missing_uuid += 1

                geom = feat.GetGeometryRef()
                if geom is None:
                    if progress_queue is not None:
                        progress_queue.put((worker_id, 1))
                    continue

                polygons = []
                gname = geom.GetGeometryName()
                if gname == "TIN":
                    for j in range(geom.GetGeometryCount()):
                        triangle = geom.GetGeometryRef(j)
                        points = get_points(triangle)
                        if len(points) >= 3:
                            polygons.append(Polygon(points))
                elif gname == "MULTIPOLYGON":
                    for j in range(geom.GetGeometryCount()):
                        poly = geom.GetGeometryRef(j)
                        if poly:
                            ring = poly.GetGeometryRef(0)
                            if ring and ring.GetPointCount() >= 3:
                                points = [(ring.GetX(k), ring.GetY(k), ring.GetZ(k)) for k in range(ring.GetPointCount())]
                                if len(points) >= 3:
                                    polygons.append(Polygon(points))
                else:
                    # unsupported; count progress and continue
                    if progress_queue is not None:
                        progress_queue.put((worker_id, 1))
                    continue

                shp_geom = MultiPolygon(polygons) if polygons else None
                if shp_geom and not shp_geom.is_empty:
                    attributes = {field: feat.GetField(field) for field in field_names}
                    if attributes.get('UUID') in (None, ""):
                        attributes['UUID'] = uuid

                    for field, value in attributes.items():
                        if isinstance(value, (str, bytes)):
                            attributes[field] = safe_string(value)

                    features.append({'geometry': shp_geom, **attributes})
                # count progress for every feature visited
                if progress_queue is not None:
                    progress_queue.put((worker_id, 1))

            if not features:
                continue

            # Build GDF and write
            chunk_gdf = gpd.GeoDataFrame(features, geometry='geometry')
            if spatial_ref_wkt:
                chunk_gdf.set_crs(spatial_ref_wkt, inplace=True)

            # Ensure geometry types
            invalid_geoms = chunk_gdf[~chunk_gdf.geometry.apply(lambda g: g.geom_type == 'MultiPolygon')].index
            if len(invalid_geoms) > 0:
                chunk_gdf = chunk_gdf.drop(invalid_geoms)

            # Normalize string columns
            for col in chunk_gdf.select_dtypes(include=['object']).columns:
                if col != 'geometry':
                    chunk_gdf[col] = chunk_gdf[col].astype(pd.StringDtype())

            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index:02d}.parquet")
            chunk_gdf.to_parquet(
                chunk_path,
                index=False,
                schema_version='1.1.0',
                compression='snappy'
            )
            assert os.path.exists(chunk_path), f"Chunk {chunk_path} was not created, check for an internal bug"
            produced.append((chunk_index, chunk_path))

        return produced
    finally:
        ds = None
        gc.collect()


def write_polygonz_chunks(gdb_path, layer_name, chunks_dir, chunk_size=1000, workers=4):
    """
    Read large PolygonZ/TIN dataset and write chunks to parquet files using 4 parallel workers by default.

    Parameters:
    - gdb_path: Path to the FileGDB
    - layer_name: Name of the layer to process
    - chunk_size: Number of features to process in each chunk
    - workers: Number of parallel worker processes (default: 4)
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

    ds = None  # close in parent; workers reopen

    # Build ranges
    ranges = []
    chunk_index = 0
    for start in range(0, total_features, chunk_size):
        end = min(start + chunk_size, total_features)
        ranges.append((start, end, chunk_index))
        chunk_index += 1

    # Partition ranges across workers (round-robin)
    assignments = {i: [] for i in range(workers)}
    totals = {i: 0 for i in range(workers)}
    for idx, r in enumerate(ranges):
        wid = idx % workers
        assignments[wid].append(r)
        totals[wid] += r[1] - r[0]

    print(f"Writing {len(ranges)} chunks with {workers} worker(s)...")

    manager = Manager()
    progress_queue = manager.Queue()

    # Create per-worker progress bars
    bars = [tqdm(total=totals[i], position=i, desc=f"worker {i+1}", leave=True) for i in range(workers)]

    chunk_paths_indexed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_process_ranges_worker, gdb_path, layer_name, assignments[i], i, chunks_dir, progress_queue)
            for i in range(workers)
        ]
        # Update progress bars as workers report
        remaining = len(futures)
        while remaining > 0:
            try:
                wid, inc = progress_queue.get(timeout=0.1)
                if 0 <= wid < workers:
                    bars[wid].update(inc)
            except queue.Empty:
                pass
            remaining = sum(1 for f in futures if not f.done())

        # Drain any remaining queued updates
        while True:
            try:
                wid, inc = progress_queue.get_nowait()
                if 0 <= wid < workers:
                    bars[wid].update(inc)
            except queue.Empty:
                break

        # Collect results
        for fut in futures:
            produced = fut.result()
            chunk_paths_indexed.extend(produced)

    for b in bars:
        b.close()

    # Sort deterministically by chunk index
    chunk_paths = [p for _, p in sorted(chunk_paths_indexed, key=lambda x: x[0])]

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
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    # Extract prefix from gdb path and create output paths
    gdb_path = Path(args.gdb_path)
    prefix = str(gdb_path.parent)
    print(f"Processing {gdb_path} with layer {args.layer_name}")

    # Create output filename based on input GDB name and layer
    gdb_stem = gdb_path.stem.replace('.gdb', '')
    output_folder = Path(prefix) / f"{gdb_stem}_{args.layer_name}"
    print(f"Output folder {output_folder}")

    # Run Chunks in parallel
    chunks_dir, chunk_paths = write_polygonz_chunks(
        str(gdb_path),
        args.layer_name,
        output_folder,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )

    # only validate for single parquet files
    if len(chunk_paths) == 1:
        validate_parquet(chunks_dir)
