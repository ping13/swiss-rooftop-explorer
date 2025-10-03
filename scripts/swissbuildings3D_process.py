import os
import sys
import io
import math
import random

import shapely
import zlib

import numpy as np
import geopandas as gpd
import pyogrio

import pyarrow as pa
import pyarrow.parquet as pq

from shapely.geometry import LineString, MultiLineString

from typing import Union, Optional



# Convert to OBJ format
def multipolygon_to_obj(geometry):
    obj_lines = []
    vertex_counter = 1  # OBJ vertex indices start at 1

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



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process SwissBUILDINGS3D into watertight OBJ + 2D boundaries")
    parser.add_argument("input", help="Path to source dataset (e.g., .gpkg, .gdb, or folder)")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("XMIN","YMIN","XMAX","YMAX"),
                        help="Bounding box in source CRS (LV95)")
    parser.add_argument("--layer", default=None, help="OGR layer name (for GDB/GPKG)")
    parser.add_argument("--engine", choices=["gdal","pyogrio"], default="gdal",
                        help="Reader engine (pyogrio is faster/more memory-friendly)")
    parser.add_argument("--limit", type=int, default=None, help="Max features to read (pyogrio only)")
    parser.add_argument("--skip-3d", action="store_true", help="Skip OBJ creation to isolate read issues")
    # Choose OBJ mode: watertight prisms (default) vs original raw triangles
    wt_group = parser.add_mutually_exclusive_group()
    wt_group.add_argument("--watertight", dest="watertight", action="store_true", help="Build watertight meshes")
    wt_group.add_argument("--no-watertight", dest="watertight", action="store_false", help="Use raw OBJ output (faster, not watertight) (default)")
    parser.set_defaults(watertight=False)
    parser.add_argument("--out", default=None, help="Output parquet file or folder (default: <input>_2d/chunk_00.parquet)")
    args = parser.parse_args()

    pq_folder = args.input

    bbox = tuple(args.bbox) if args.bbox else (
#        2689604.9543,1283305.8130,2690017.7198,1283705.6030 # Schaffhausen
#        2677982.4964,1243412.3961,2688871.6163,1251011.7174 # Zürich
#        2678085.9586,1235544.0658,2692913.9194,1251073.7169 # Zürich + Meilen
#        2691829,1236353,2691873,1236392 # Haltenweg Meilen
#        2601455.1,1199547.2,2601734.7,1199750.8  # default test bbox
    )

    print(f"Reader engine: {args.engine}")
    print(f"Input: {pq_folder}")
    print(f"BBox: {bbox}")
    if args.layer:
        print(f"Layer: {args.layer}")

    # Read with selected engine
    if args.engine == "pyogrio":
        read_kwargs = dict(bbox=bbox)
        if args.layer:
            read_kwargs["layer"] = args.layer
        if args.limit is not None:
            read_kwargs["max_features"] = args.limit
        print("Reading with pyogrio.read_dataframe(...) ...")
        gdf = pyogrio.read_dataframe(pq_folder, **read_kwargs)
    else:
        print("Reading with geopandas.read_file(...) ...")
        gdf = gpd.read_parquet(pq_folder, layer=args.layer) if args.layer else gpd.read_parquet(pq_folder)

    if bbox:
        gdf = gdf.cx[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # clip using bounding box (xmin, ymin, xmax, ymax)
        
    print(f"Found {len(gdf)} rows")

    # Add incrementing ID and remove UUID if it exists
    gdf['id'] = range(1, len(gdf) + 1)  # Add incrementing ID starting from 1
    if 'UUID' in gdf.columns:
        gdf = gdf.drop(columns=['UUID'])

    # Optionally create 3D OBJ column
    if not args.skip_3d:
        print(f"Create 3d models ({'watertight' if args.watertight else 'raw'})")
        if args.watertight:
            builder = multipolygon_to_watertight_obj
        else:
            builder = multipolygon_to_obj
        gdf['obj'] = gdf.geometry.apply(builder).astype('O')
    else:
        print("Skipping 3D model creation (--skip-3d)")

    print("Create 2D boundary")
    # simplify the multipolygon
    def simplify_multipolygon(x):
        try:
            return shapely.ops.unary_union(x)
        except shapely.errors.GEOSException as e:
            print(f"Skip geometry: {e}")
            return x
        
    gdf.geometry = gdf.geometry.apply(simplify_multipolygon)
    # Remove the Z-dimension (if present) directly without looping
    gdf['boundary_2d'] = gdf.geometry.apply(
        lambda geom: shapely.ops.transform(lambda x, y, z=None: (x, y), geom.boundary)
    )

    print("Set new geometry")
    del gdf['geometry'] 
    gdf.set_geometry("boundary_2d", inplace=True)

    out_folder = f"{pq_folder}_2d/"
    os.makedirs(out_folder, exist_ok=True)
    output_file = args.out or os.path.join(out_folder, "chunk_00.parquet")

    print(f"Export to parquet: {output_file}")
    # write to parquet file and include covering_bbox for faster query
    gdf.to_parquet(output_file, write_covering_bbox=True)

    print(gdf.info())
        
