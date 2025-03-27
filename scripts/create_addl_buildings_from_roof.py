import click
import geopandas
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon

def multiline_to_multipolygon(geom):
    if isinstance(geom, MultiLineString):
        polygons = [Polygon(ls) for ls in geom.geoms if ls.is_ring]
        return MultiPolygon(polygons) if polygons else geom
    elif isinstance(geom, LineString):
        return Polygon(geom)
    return geom

@click.command()
@click.option('--buildings-file', required=True, help='Path to the buildings parquet file')
@click.option('--roofs-file', required=True, help='Path to the roofs parquet file')
@click.option('--output-file', required=True, help='Path to the output parquet file for missing buildings')
def main(buildings_file, roofs_file, output_file):
    buildings = geopandas.read_file(buildings_file)
    roofs = geopandas.read_file(roofs_file)
    print(f"found {len(buildings)} buildings and {len(roofs)} roofs")
    assert buildings.crs == roofs.crs

    # convert to polygons
    print(f"convert to polygons")
    roofs['geometry'] = roofs['geometry'].apply(multiline_to_multipolygon)
    buildings['geometry'] = buildings['geometry'].apply(multiline_to_multipolygon)

    # identify candidates for missing buildings based on roof
    print(f"identify missing building candidates based on roof points")
    representative_roof_points = roofs.copy()
    representative_roof_points['geometry'] = representative_roof_points.geometry.representative_point()

    joined = geopandas.sjoin(
        representative_roof_points, buildings, how='left', predicate='within'
    ).reset_index()    
    uncovered_roof_candidates = roofs.loc[joined.loc[joined['index_right'].isna(), 'index']]

    # filter candidates by
    # Keep only roofs that remain after difference (i.e., not fully covered)
    print(f"keep only roofs that have no intersection with other buildings")
    uncovered_roofs = uncovered_roof_candidates[uncovered_roof_candidates.geometry.apply(lambda g: not buildings.geometry.intersects(g).any())]

    print(f"Save file to {output_file}")    
    uncovered_roofs.to_parquet(output_file,
                               index=False,
                               schema_version='1.1.0',
                               write_covering_bbox=True,
                               compression='snappy')

if __name__ == "__main__":
    main()
