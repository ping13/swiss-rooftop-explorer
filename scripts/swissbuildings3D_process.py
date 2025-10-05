import os
import sys
import io
import math
import random
import glob

import shapely
import zlib

import numpy as np
import geopandas as gpd
import pyogrio

import pyarrow as pa
import pyarrow.parquet as pq

from shapely.geometry import LineString, MultiLineString

from typing import Union, Optional
from tqdm.auto import tqdm

tqdm.pandas(desc="Building OBJ")

# Convert to OBJ format
def multipolygon_to_obj(geometry):
    obj_lines = []
    vertex_counter = 1  # OBJ vertex indices start at 1

    if geometry is None:
        return None
    try:
        for polygon in geometry.geoms:
            assert len(polygon.exterior.coords) == 4, f"Polygon exterior has {len(polygon.exterior.coords)} coordinates (not 4, {list(polygon.exterior.coords)})"
            assert polygon.exterior.coords[0] == polygon.exterior.coords[3], "Point 0 is not equal to point 3"

            for ring in [polygon.exterior] + list(polygon.interiors):
                assert len(ring.coords) == 4, f"Polygon ring has {len(ring.coords)} coordinates (not 4, {list(ring.coords)})"
                assert ring.coords[0] == ring.coords[3], f"Point 0 is not equal to point 3: {list(ring.coords)}"

                # Add vertices
                for x, y, z in ring.coords[0:3]:
                    obj_lines.append(f"v {x} {y} {z}")
                # Add faces
                face_indices = list(range(vertex_counter, vertex_counter + 3))
                obj_lines.append(f"f {' '.join(map(str, face_indices))}")
                vertex_counter += 3
    except AssertionError as e:
        print(f"*** Warning (to_obj): '{e}', return empty string")
        return zlib.compress("".encode('utf-8'))

    return zlib.compress("\n".join(obj_lines).encode('utf-8'))


def mesh_to_obj_text(mesh) -> str:
    """Convert a PyVista mesh into an OBJ string (triangles only)."""
    lines = []
    pts = np.asarray(mesh.points)
    for x, y, z in pts:
        lines.append(f"v {x} {y} {z}")
    # faces is a flat array: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    i = 0
    n = faces.size
    while i < n:
        count = int(faces[i])
        if count == 3 and i + 3 < n:
            i0 = int(faces[i+1]) + 1
            i1 = int(faces[i+2]) + 1
            i2 = int(faces[i+3]) + 1
            lines.append(f"f {i0} {i1} {i2}")
        i += count + 1
    return "\n".join(lines)


def multipolygon_to_watertight_obj(geometry):
    import pyvista as pv  # local import to avoid import-time crashes
    """Build a watertight triangular-prism mesh for each triangular roof patch and export as compressed OBJ bytes."""
    # Find a base elevation for this feature (lowest Z on any exterior vertex)
    try:
        min_z = min(pt[2] for poly in geometry.geoms for pt in list(poly.exterior.coords)[:-1])
    except Exception as e:
        print(f"*** Warning (min_z): {e}")
        return zlib.compress("".encode("utf-8"))

    feature_mesh = pv.PolyData()

    for polygon in geometry.geoms:
        exterior = polygon.exterior
        # Dataset stores triangular patches: 4 coords with last equal to first
        assert len(exterior.coords) == 4, f"Polygon exterior has {len(exterior.coords)} coordinates (not 4)"
        assert exterior.coords[0] == exterior.coords[3], "Exterior is not closed"

        ring = list(exterior.coords)[:-1]  # 3 unique vertices
        # top triangle points
        top_pts = [[x, y, z] for (x, y, z) in ring]
        # bottom triangle points at min_z (same XY)
        bot_pts = [[x, y, min_z] for (x, y, _) in ring]
        pts = np.asarray(top_pts + bot_pts, dtype=np.float64)

        # faces: top (0,1,2), bottom reversed (5,4,3), and walls around edges
        faces = []
        faces.append([3, 0, 1, 2])      # top
        faces.append([3, 5, 4, 3])      # bottom (reverse winding)
        edges = [(0, 1), (1, 2), (2, 0)]
        for i0, i1 in edges:
            b0, b1 = i0 + 3, i1 + 3
            # two triangles per quad wall
            faces.append([3, i0, i1, b1])
            faces.append([3, i0, b1, b0])
        faces = np.asarray(faces, dtype=np.int32).ravel()

        poly = pv.PolyData(pts, faces).clean().triangulate()
        if feature_mesh.n_points == 0:
            feature_mesh = poly
        else:
            feature_mesh = feature_mesh.merge(poly)

    # Weld close vertices and ensure triangulation
    feature_mesh = feature_mesh.clean(point_merging=True, tolerance=1e-3).triangulate()

    # Optional consistency checks (non-fatal in production)
    try:
        assert feature_mesh.n_points > 0
        assert feature_mesh.is_all_triangles
        # These are stricter; comment if input occasionally violates topology
        assert feature_mesh.is_manifold
        assert feature_mesh.n_open_edges == 0
    except AssertionError:
        pass

    obj_text = mesh_to_obj_text(feature_mesh)
    return zlib.compress(obj_text.encode("utf-8"))



def validate_obj_binary_column(gdf, column: str = 'obj'):
    if column not in gdf.columns:
        return
    s = gdf[column]
    # All values must be bytes or None; NaN (float) or str indicates a problem
    ok = s.map(lambda x: (x is None) or isinstance(x, (bytes,))).all()
    if ok:
        return
    # If not ok, report offending types for diagnosis
    type_counts = s.map(lambda x: type(x).__name__).value_counts(dropna=False)
    raise AssertionError(
        f"Column '{column}' must contain only bytes or None before writing. Found types: {dict(type_counts)}"
    )


def normalize_obj_binary_column(gdf, column: str = 'obj'):
    if column not in gdf.columns:
        return gdf
    def _norm(x):
        if x is None:
            return b""
        if isinstance(x, bytes):
            return x
        if isinstance(x, bytearray):
            return bytes(x)
        if isinstance(x, memoryview):
            return bytes(x)
        # Do not auto-convert str; that would change semantics (expect compressed bytes)
        return x
    gdf = gdf.copy()
    gdf[column] = gdf[column].map(_norm)
    return gdf


def process_chunk(gdf, build_watertight: bool, n_workers: int, next_id: int, skip_3d: bool):
    # Assign IDs and drop UUID
    gdf = gdf.copy()
    gdf['id'] = range(next_id, next_id + len(gdf))
    if 'UUID' in gdf.columns:
        gdf = gdf.drop(columns=['UUID'])

    # Optionally create 3D OBJ column
    if not skip_3d:
        print(f"Create 3d models ({'watertight' if build_watertight else 'raw'}) for {len(gdf)} rows")
        builder = multipolygon_to_watertight_obj if build_watertight else multipolygon_to_obj
        from concurrent.futures import ProcessPoolExecutor
        geoms = list(gdf.geometry.values)
        with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
            objs = list(tqdm(ex.map(builder, geoms, chunksize=32), total=len(geoms), desc="Building OBJ (parallel)"))
        gdf['obj'] = np.array(objs, dtype=object)
    else:
        print("Skipping 3D model creation (--skip-3d)")

    print("Create 2D boundary")
    def simplify_multipolygon(x):
        try:
            return shapely.ops.unary_union(x)
        except shapely.errors.GEOSException as e:
            print(f"Skip geometry: {e}")
            return x

    gdf.geometry = gdf.geometry.apply(simplify_multipolygon)
    gdf['boundary_2d'] = gdf.geometry.progress_apply(
        lambda geom: shapely.ops.transform(lambda x, y, z=None: (x, y), geom.boundary)
    )

    print("Set new geometry")
    del gdf['geometry']
    gdf.set_geometry("boundary_2d", inplace=True)
    return gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process SwissBUILDINGS3D into watertight OBJ + 2D boundaries")
    parser.add_argument("input", help="Path to source dataset (Parquet folder or file; GPKG/GDB ok for small runs)")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("XMIN","YMIN","XMAX","YMAX"),
                        help="Bounding box in source CRS (LV95)")
    parser.add_argument("--layer", default=None, help="OGR layer name (for GDB/GPKG)")
    parser.add_argument("--limit", type=int, default=None, help="Max features to read (single-file only)")
    parser.add_argument("--skip-3d", action="store_true", help="Skip OBJ creation to isolate read issues")
    wt_group = parser.add_mutually_exclusive_group()
    wt_group.add_argument("--watertight", dest="watertight", action="store_true", help="Build watertight meshes")
    wt_group.add_argument("--no-watertight", dest="watertight", action="store_false", help="Use raw OBJ output (faster, not watertight) (default)")
    parser.set_defaults(watertight=False)
    parser.add_argument("--out", default=None, help="Output folder (default: <input>_2d)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for OBJ creation (default: 8)")
    args = parser.parse_args()

    src_path = args.input

    bbox = tuple(args.bbox) if args.bbox else None

    print(f"Input: {src_path}")
    print(f"BBox: {bbox}")
    if args.layer:
        print(f"Layer: {args.layer}")

    # Resolve input files (prefer chunked folder of Parquet files)
    files = []
    if os.path.isdir(src_path):
        files = sorted(glob.glob(os.path.join(src_path, "*.parquet")))
        if not files:
            print(f"No parquet files found in folder: {src_path}")
            sys.exit(1)
    else:
        files = [src_path]

    # Output folders
    out_folder = args.out or f"{src_path}_2d/"
    obj_folder = f"{src_path}_obj/"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(obj_folder, exist_ok=True)

    next_id = 1
    total_rows = 0

    print(f"Processing {len(files)} chunk(s) -> {out_folder}")
    for i, fp in enumerate(tqdm(files, desc="Chunks")):
        read_kwargs = {}
        if bbox is not None:
            read_kwargs["bbox"] = bbox
        if args.layer and not fp.lower().endswith(".parquet"):
            read_kwargs["layer"] = args.layer
        if args.limit and len(files) == 1:
            read_kwargs["max_features"] = args.limit

        print(f"Reading {fp} with pyogrio.read_dataframe(...) ...")
        gdf = pyogrio.read_dataframe(fp, use_arrow=True, **read_kwargs)
        if gdf.empty:
            print(f"Skip empty chunk: {fp}")
            continue

        # Extra bbox clip (safety) if provided and engine-level bbox was not effective
        if bbox:
            try:
                gdf = gdf.cx[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            except Exception:
                pass
        if gdf.empty:
            print(f"Skip chunk (bbox filtered to empty): {fp}")
            continue

        print(f"Chunk rows: {len(gdf)}")
        out_gdf = process_chunk(gdf, build_watertight=args.watertight, n_workers=args.workers, next_id=next_id, skip_3d=args.skip_3d)
        next_id += len(out_gdf)
        total_rows += len(out_gdf)

        # Derive output filename
        base = os.path.basename(fp)
        if not base.lower().endswith('.parquet'):
            base = f"chunk_{i:02d}.parquet"
        out_path = os.path.join(out_folder, base)
        obj_path = os.path.join(obj_folder, base)

        # Prepare sidecar first (if obj exists)
        has_obj = 'obj' in out_gdf.columns and not args.skip_3d
        if has_obj:
            # Normalize and validate 'obj' to avoid VARCHAR inference
            out_gdf = normalize_obj_binary_column(out_gdf, 'obj')
            validate_obj_binary_column(out_gdf, 'obj')

            # Compute bbox from current geometry (boundary_2d)
            b = out_gdf.geometry.bounds
            b = b[['minx','miny','maxx','maxy']].astype('float64')

            # Build sidecar table with explicit types (id:int64, obj:binary, bbox:struct{xmin,ymin,xmax,ymax})
            ids = out_gdf['id'].astype('int64').tolist()
            objs = out_gdf['obj'].tolist()
            obj_arr = pa.array(objs, type=pa.binary())
            id_arr = pa.array(ids, type=pa.int64())
            xmin_arr = pa.array(b['minx'].to_numpy(), type=pa.float64())
            ymin_arr = pa.array(b['miny'].to_numpy(), type=pa.float64())
            xmax_arr = pa.array(b['maxx'].to_numpy(), type=pa.float64())
            maxy_arr = pa.array(b['maxy'].to_numpy(), type=pa.float64())
            bbox_struct = pa.StructArray.from_arrays(
                [xmin_arr, ymin_arr, xmax_arr, maxy_arr],
                fields=[
                    pa.field('xmin', pa.float64()),
                    pa.field('ymin', pa.float64()),
                    pa.field('xmax', pa.float64()),
                    pa.field('ymax', pa.float64()),
                ]
            )
            sidecar = pa.table({'id': id_arr, 'obj': obj_arr, 'bbox': bbox_struct})

            # Write sidecar via in-memory buffer to avoid uv/pyarrow filesystem issue
            sink = pa.BufferOutputStream()
            pq.write_table(sidecar, sink)
            data = sink.getvalue().to_pybytes()
            with open(obj_path, 'wb') as f:
                f.write(data)
            print(f"Export sidecar: {obj_path}")

        # Write main dataset (without obj)
        if has_obj and 'obj' in out_gdf.columns:
            main_gdf = out_gdf.drop(columns=['obj'])
        else:
            main_gdf = out_gdf

        print(f"Export 2d: {out_path}")
        pyogrio.write_dataframe(
            main_gdf,
            out_path,
            driver="Parquet",
            dataset_options={
                "GEOMETRY_ENCODING": "WKB",
                "PARQUET_COMPRESSION": "SNAPPY"
            }
        )

    print(f"Done. Total rows written: {total_rows}")
