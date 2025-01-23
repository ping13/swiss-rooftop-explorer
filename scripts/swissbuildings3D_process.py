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
import dask_geopandas as dgpd
from dask.distributed import Client

import pyarrow as pa
import pyarrow.parquet as pq

import pyvista as pv
import vtk


from shapely.geometry import LineString, MultiLineString

import vtk
import pyvista as pv
from typing import Union, Optional


def polydata_to_bytes(polydata: pv.PolyData) -> bytes:
    """Serialize PyVista PolyData to a VTK XML byte stream."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetWriteToOutputString(True)
    writer.SetInputData(polydata)  # Convert PyVista PolyData to VTK format
    writer.SetDataModeToBinary()   # Use binary mode for compactness
    writer.Write()
    return writer.GetOutputString().encode("utf-8")  # Convert string to bytes

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


def multipolygon_to_mesh(geometry):

    
    # retrieve the minimum z coordinate and calculate the center point
    min_elevation_feature = 9999
    x_coords = []
    y_coords = []
    z_coords = []
    
    for polygon in geometry.geoms:
        for point in polygon.exterior.coords:
            x_coords.append(point[0])
            y_coords.append(point[1])
            z_coords.append(point[2])
            if point[2] < min_elevation_feature:
                min_elevation_feature = point[2]
    
    feature_mesh = pv.PolyData()
    for polygon in geometry.geoms:
        exterior = polygon.exterior

        assert(len(exterior.coords) == 4) ## this is how its stored for swissbuildings3D
        assert(exterior.coords[0] == exterior.coords[3])

        n_points = len(exterior.coords)-1
        
        # Create vertices for roof face for this element
        points = []
        for point in exterior.coords[0:n_points]:
            x, y, z = (
                point[0],
                point[1],
                point[2],
            )
            points.append([x, y, z])  # top vertex

        
        # Create top face
        faces = [ [n_points] + list(reversed(range(0, n_points))) ]

        # Compute normal for top face
        p0 = np.array(points[0])
        p1 = np.array(points[1])
        p2 = np.array(points[2])
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize the vector
        
        # Skip this polygon if face is vertical
        z_component = math.fabs(normal[2])
        if z_component < 0.1:
            continue  
        
        # let's make sure that the top face is really a roof face.
        # skip horizontal faces close to the min_elevation_feature
        if z_component > 0.95: 
            # Check if all points are close to min_elevation_feature
            z_coords = np.array(points)[:, 2]
            if np.all(np.abs(z_coords - min_elevation_feature) < 0.1):  # 10cm threshold
                continue  # Skip this polygon if it's a horizontal face near ground level

        # create ground face, start with the points 
        for point in exterior.coords:
            x, y, z = (
                point[0],
                point[1],
                min_elevation_feature 
            )
            points.append([x, y, z])  # top vertex

        # Create ground face 
        faces.append([n_points] + list(range(n_points, n_points*2)))
        # now create the side walls
        faces.append([3, 0,1,4])
        faces.append([3, 3,4,0])

        faces.append([3, 1,2,5])
        faces.append([3, 5,4,1])

        faces.append([3, 2,0,3])
        faces.append([3, 2,3,5])
        
        # Create a new polydata for this polygon
        polygon_mesh =  pv.PolyData(np.array(points), np.array([faces])).clean().triangulate()

        assert polygon_mesh.is_all_triangles, "mesh is not all triangles"
        assert polygon_mesh.is_manifold, "mesh is not manifold"
        assert polygon_mesh.n_open_edges == 0, "mesh has open edges"
        assert polygon_mesh.n_points > 0, "mesh has no points"

        if feature_mesh.n_points == 0:
            feature_mesh = polygon_mesh
        else:
            # plotter = pv.Plotter()
            # plotter.add_mesh(feature_mesh, color="yellow", opacity=0.5, show_edges=True)  # 50% transparent
            # plotter.add_mesh(polygon_mesh, color="green", opacity=0.5, show_edges=True)  # 50% transparent
            # plotter.show()
            feature_mesh = feature_mesh.merge(polygon_mesh)
                
            #polygon_mesh.plot_normals(mag=3, faces=True, show_edges=True)

#            assert feature_mesh.is_manifold, "mesh is not manifold"
#            assert feature_mesh.n_open_edges == 0, "mesh has open edges"
            assert feature_mesh.is_all_triangles, "mesh is not all triangles"
            assert feature_mesh.n_points > 0, "mesh has no points"

#    feature_mesh = pv.UnstructuredGrid(feature_mesh).extract_surface()
#    feature_mesh.plot(show_edges=True, color=True, opacity=0.5)


    return zlib.compress(polydata_to_bytes(feature_mesh))

def process_file(file_path, output_path, bbox):
    print(f"Processing {file_path} with bbox {bbox}")
    gdf = gpd.read_file(file_path, bbox=bbox)
    print(f"Found {len(gdf)} rows")

    # Create 3D models
    gdf['obj'] = gdf.geometry.apply(lambda geom: multipolygon_to_obj(geom)).astype('O')

    # Create 2D boundary
    gdf.geometry = gdf.geometry.apply(lambda geom: shapely.ops.unary_union(geom))
    gdf['boundary_2d'] = gdf.geometry.apply(
        lambda geom: shapely.ops.transform(lambda x, y, z=None: (x, y), geom.boundary)
    )

    # Set new geometry and reproject
    del gdf['geometry']
    gdf.set_geometry("boundary_2d", inplace=True)
    gdf = gdf.to_crs(epsg=4326)

    gdf.to_parquet(output_path)
    print(gdf.info())


def transform_2d(geom):
    def remove_z(x, y, z=None):
        return (x, y)
    if geom.is_empty:
        return geom
    if hasattr(geom, "geoms"):
        return shapely.ops.transform(remove_z, geom)
    if geom.geom_type in ['LineString', 'LinearRing']:
        return type(geom)([remove_z(*coord) for coord in geom.coords])
    return shapely.ops.transform(remove_z, geom)

def process_partition(gdf):
    # Create 3D models.
    gdf['obj'] = gdf.geometry.apply(multipolygon_to_obj)
    # Simplify the geometry.
    gdf['geometry'] = gdf.geometry.apply(lambda geom: shapely.ops.unary_union(geom))
    # Create 2D boundary.
    gdf['boundary_2d'] = gdf.geometry.apply(lambda geom: transform_2d(geom.boundary))
    # Drop the original geometry and set the new one.
    gdf = gdf.drop(columns=['geometry'])
    gdf = gpd.GeoDataFrame(gdf, geometry='boundary_2d')
    # Reproject.
    gdf = gdf.to_crs(epsg=4326)
    return gdf

#client = Client(dashboard_address=':8788')

if __name__ == "__main__":
    pq_folder = sys.argv[1]
    
    bbox = (
#        2689604.9543,1283305.8130,2690017.7198,1283705.6030 # Schaffhausen
#        2677982.4964,1243412.3961,2688871.6163,1251011.7174 # Zürich
#        2678085.9586,1235544.0658,2692913.9194,1251073.7169 # Zürich + Meilen
#        2691829,1236353,2691873,1236392 # Haltenweg Meilen
        2474597,1043243,2876680,1294968 #Schweiz
    )

    # ddf = dgpd.read_parquet(os.path.join(pq_folder, 'chunk_*.parquet'))
    # bbox_poly = shapely.geometry.box(*bbox)
    # ddf = ddf[ddf.geometry.intersects(bbox_poly)]
    
    # # Build meta from the empty _meta of the input.
    # meta = ddf._meta.copy()
    # # Add expected output columns.
    # meta['obj'] = None
    # meta['boundary_2d'] = meta['geometry']
    # meta = meta.drop(columns=['geometry'])
    # meta = gpd.GeoDataFrame(meta, geometry='boundary_2d')
    
    # processed_ddf = ddf.map_partitions(process_partition, meta=meta)
    
    # output_folder = f"{pq_folder}_2d"
    # os.makedirs(output_folder, exist_ok=True)
    # processed_ddf.to_parquet(output_folder)

    ##################################################################################

    print(f"Read dataframe from {pq_folder} with bbox {bbox}")
    gdf = gpd.read_file(pq_folder, bbox=bbox)
    print(f"found {len(gdf)} rows")

    # create an obj column
    print(f"Create 3d models")
    gdf['obj'] = gdf.geometry.apply(
        lambda geom: multipolygon_to_obj(geom),
    ).astype('O')

    
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

    print(f"Export to parquet")
    os.makedirs(f"{pq_folder}_2d/", exist_ok=True)
    output_file = f"{pq_folder}_2d/chunk_00.parquet"
    # write to parquet file and include covering_bbox for faster query
    gdf.to_parquet(output_file, write_covering_bbox=True)

    print(gdf.info())
        
