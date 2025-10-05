import os
import click
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon


def multiline_to_multipolygon(geom):
    if isinstance(geom, MultiLineString):
        polygons = [Polygon(ls) for ls in geom.geoms if ls.is_ring]
        return MultiPolygon(polygons)
    elif isinstance(geom, LineString):
        # Fallback: wrap a single ring as polygon if possible
        return MultiPolygon([Polygon(geom)])
    else:
        raise Exception(f"unhandled geom {geom.geom_type}")


def normalize_obj_to_bytes(series):
    def _to_bytes(x):
        if x is None:
            return None
        if isinstance(x, bytes):
            return x
        if isinstance(x, str):
            return x.encode("utf-8")
        raise TypeError(f"Unsupported obj type: {type(x)}")
    return series.map(_to_bytes)


def validate_obj_binary_column(series, col_name="obj"):
    ok = series.map(lambda x: (x is None) or isinstance(x, bytes)).all()
    if not ok:
        raise AssertionError(
            f"Column '{col_name}' has non-bytes values; ensure bytes-only (or None)."
        )


def build_bbox_struct_from_gdf(gdf):
    bounds = gdf.geometry.bounds
    arrays = [
        pa.array(bounds["minx"].astype("float64")),
        pa.array(bounds["miny"].astype("float64")),
        pa.array(bounds["maxx"].astype("float64")),
        pa.array(bounds["maxy"].astype("float64")),
    ]
    return pa.StructArray.from_arrays(arrays, names=["xmin", "ymin", "xmax", "ymax"])


@click.command()
@click.option('--buildings-file', required=True, help='Path to the buildings parquet file')
@click.option('--roofs-file', required=True, help='Path to the roofs parquet file')
@click.option('--output-file', required=False, default=None,
              help='Optional base path used to derive default outputs when --out-2d/--out-obj are not provided')
@click.option('--out-2d', 'out_2d', required=False, default=None,
              help='Path to the 2D parquet output (no obj)')
@click.option('--out-obj', 'out_obj', required=False, default=None,
              help='Path to the OBJ sidecar parquet (id, obj, bbox)')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing outputs if present')
@click.option('--coverage-threshold', default=0.3, type=float, help='Threshold for building coverage (default: 0.3)')
def main(buildings_file, roofs_file, output_file, out_2d, out_obj, overwrite, coverage_threshold):
    # Resolve default output paths
    if out_2d is None or out_obj is None:
        base_dir = "."
        filename = "missing_buildings.parquet"
        if output_file:
            base_dir = os.path.dirname(output_file) or '.'
            filename = os.path.basename(output_file) 
        if out_2d is None:
            out_2d = os.path.join(base_dir, filename.replace('.parquet', '_2d.parquet'))
        if out_obj is None:
            out_obj = os.path.join(base_dir, filename.replace('.parquet', '_obj.parquet'))

    if (not overwrite) and (os.path.exists(out_2d) or os.path.exists(out_obj)):
        raise FileExistsError(
            f"Output exists. Use --overwrite to replace. Existing: "
            f"{out_2d if os.path.exists(out_2d) else ''} "
            f"{out_obj if os.path.exists(out_obj) else ''}"
        )

    print("read buildings")
    buildings = gpd.read_file(buildings_file)

    print("read roofs")
    roofs = gpd.read_file(roofs_file)
    roofs_copy = roofs.copy()

    print(f"found {len(buildings)} buildings and {len(roofs_copy)} roofs")
    assert buildings.crs == roofs_copy.crs

    # convert to polygons
    print("convert both to polygons")
    roofs_copy['geometry'] = roofs_copy['geometry'].apply(multiline_to_multipolygon)
    buildings['geometry'] = buildings['geometry'].apply(multiline_to_multipolygon)

    # Add unique ID to roofs (keep original roofs index/columns for selection)
    roofs_copy = roofs_copy.reset_index(drop=True)
    roofs_copy["roof_id"] = roofs_copy.index
    roofs = roofs.reset_index(drop=True)

    # Compute intersection
    print("compute intersection")
    intersection = gpd.overlay(roofs_copy, buildings, how="intersection")

    # Compute areas
    roofs_copy["area"] = roofs_copy.geometry.area
    intersection["inter_area"] = intersection.geometry.area

    # Sum intersection area per roof_id
    print("sum intersection")
    inter_area_sum = intersection.groupby("roof_id")["inter_area"].sum()

    # Map to roofs
    roofs_copy["inter_area"] = roofs_copy.index.map(inter_area_sum).fillna(0)
    roofs_copy["coverage"] = roofs_copy["inter_area"] / roofs_copy["area"]

    # Filter and use original geometry from roofs
    print("filter")
    selected_indices = roofs_copy[roofs_copy["coverage"] < coverage_threshold].index
    selected = roofs.loc[selected_indices].copy()

    # Ensure join key `id`
    selected = selected.reset_index(drop=True)
    if 'id' not in selected.columns:
        selected['id'] = selected.index.astype('int64')
    else:
        # enforce int64 if possible
        try:
            selected['id'] = selected['id'].astype('int64')
        except Exception:
            pass

    # Normalize & validate obj for sidecar
    if 'obj' in selected.columns:
        selected['obj'] = normalize_obj_to_bytes(selected['obj'])
        validate_obj_binary_column(selected['obj'], 'obj')
    else:
        # Create empty obj column if missing
        selected['obj'] = None

    # Prepare outputs
    print(f"write outputs: 2D -> {out_2d}, OBJ -> {out_obj}")

    # 1) Write 2D file (drop obj)
    out_2d_gdf = selected.drop(columns=['obj'], errors='ignore')
    # write via GDAL/pyogrio to avoid pyarrow filesystem issues
    pyogrio.write_dataframe(out_2d_gdf, out_2d, driver='Parquet')

    # 2) Write OBJ sidecar with bbox struct via PyArrow
    if len(selected) == 0:
        id_arr = pa.array([], type=pa.int64())
        obj_arr = pa.array([], type=pa.binary())
        bbox_arr = pa.StructArray.from_arrays(
            [pa.array([], type=pa.float64()), pa.array([], type=pa.float64()),
             pa.array([], type=pa.float64()), pa.array([], type=pa.float64())],
            names=["xmin", "ymin", "xmax", "ymax"]
        )
    else:
        id_arr = pa.array(selected['id'].tolist(), type=pa.int64())
        obj_arr = pa.array(selected['obj'].tolist(), type=pa.binary())
        bbox_arr = build_bbox_struct_from_gdf(selected)

    table = pa.Table.from_arrays([id_arr, obj_arr, bbox_arr], names=['id', 'obj', 'bbox'])

    # Write to in-memory buffer first to avoid filesystem registration issues
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression='snappy')
    data = sink.getvalue().to_pybytes()
    with open(out_obj, 'wb') as f:
        f.write(data)

    # Summary
    print(
        f"done. features: {len(selected)} | 2D: {out_2d} | OBJ: {out_obj}"
    )


if __name__ == "__main__":
    main()
