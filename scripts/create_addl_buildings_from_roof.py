import geopandas
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon

def multiline_to_multipolygon(geom):
    if isinstance(geom, MultiLineString):
        polygons = [Polygon(ls) for ls in geom.geoms if ls.is_ring]
        return MultiPolygon(polygons) if polygons else geom
    elif isinstance(geom, LineString):
        return Polygon(geom)
    return geom

# add a CLI interface to define the input files and the output file. AI!
def main():
    buildings = geopandas.read_file('assets/swissBUILDINGS3D_3-0_1112-13_Building_solid_2d/chunk_00.parquet')
    roofs = geopandas.read_file('assets/swissBUILDINGS3D_3-0_1112-13_Roof_solid_2d/chunk_00.parquet')
    buildings.drop(columns=["obj"], inplace=True)
    roofs.drop(columns=["obj"], inplace=True)
    print(f"found {len(buildings)} buildings and {len(roofs)} roofs")
    assert buildings.crs == roofs.crs

    # convert to polygons
    roofs['geometry'] = roofs['geometry'].apply(multiline_to_multipolygon)
    buildings['geometry'] = buildings['geometry'].apply(multiline_to_multipolygon)

    # identify candidates for missing buildings based on roof
    representative_roof_points = roofs.copy()
    representative_roof_points['geometry'] = representative_roof_points.geometry.representative_point()
    
    joined = geopandas.sjoin(representative_roof_points, buildings, how='left', predicate='within')
    uncovered_roof_candidates = roofs[joined['index_right'].isna().values]

    # filter candidates by
    # Keep only roofs that remain after difference (i.e., not fully covered)
    uncovered_roofs = uncovered_roof_candidates[uncovered_roof_candidates.geometry.apply(lambda g: not buildings.geometry.intersects(g).any())]

    uncovered_roofs.to_parquet("assets/missing_buildings_small.parquet",
                               index=False,
                               schema_version='1.1.0',
                               write_covering_bbox=True,
                               compression='snappy')

if __name__ == "__main__":
    main()
