import click
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon

def multiline_to_multipolygon(geom):
    if isinstance(geom, MultiLineString):
        polygons = [Polygon(ls) for ls in geom.geoms if ls.is_ring]
        return MultiPolygon(polygons)
    elif isinstance(geom, LineString):
        return MultiPolygon[Polygon(geom)]
    else:
        raise Exception("unhandled geom {geom.geom_type}")



@click.command()
@click.option('--buildings-file', required=True, help='Path to the buildings parquet file')
@click.option('--roofs-file', required=True, help='Path to the roofs parquet file')
@click.option('--output-file', required=True, help='Path to the output parquet file for missing buildings')
@click.option('--coverage-threshold', default=0.3, type=float, help='Threshold for building coverage (default: 0.3)')
def main(buildings_file, roofs_file, output_file, coverage_threshold):
    print(f"read buildings")
    buildings = gpd.read_file(buildings_file)
    
    print(f"read roofs")
    roofs = gpd.read_file(roofs_file)
    roofs_copy = roofs.copy()
    
    print(f"found {len(buildings)} buildings and {len(roofs_copy)} roofs")
    assert buildings.crs == roofs_copy.crs
    
    # convert to polygons
    print(f"convert both to polygons")
    roofs_copy['geometry'] = roofs_copy['geometry'].apply(multiline_to_multipolygon)
    buildings['geometry'] = buildings['geometry'].apply(multiline_to_multipolygon)
    
    # Add unique ID to roofs
    roofs_copy = roofs_copy.reset_index(drop=True)
    roofs_copy["roof_id"] = roofs_copy.index
    roofs = roofs.reset_index(drop=True)
    roofs["roof_id"] = roofs.index
    
    # Compute intersection
    print(f"compute intersection")
    intersection = gpd.overlay(roofs_copy, buildings, how="intersection")
    
    # Compute areas
    roofs_copy["area"] = roofs_copy.geometry.area
    intersection["inter_area"] = intersection.geometry.area
    
    # Sum intersection area per roof_id
    print(f"sum intersection")
    inter_area_sum = intersection.groupby("roof_id")["inter_area"].sum()
    
    # Map to roofs
    roofs_copy["inter_area"] = roofs_copy["roof_id"].map(inter_area_sum).fillna(0)
    roofs_copy["coverage"] = roofs_copy["inter_area"] / roofs_copy["area"]
    
    # Filter and use original geometry
    print(f"filter")
    selected_indices = roofs_copy[roofs_copy["coverage"] < coverage_threshold].index
    selected = roofs.loc[selected_indices]
    
    print(f"Save file to {output_file}")    
    selected.to_parquet(output_file,
                        index=False,
                        schema_version='1.1.0',
                        write_covering_bbox=True,
                        compression='snappy')

if __name__ == "__main__":
    main()
