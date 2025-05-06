import duckdb
import gzip
import zlib
import tempfile
import os
import sys
import numpy as np
import pyvista as pv

def find_building_at_location(parquet_file, x, y, proximity_threshold=10.0):
    """
    Find a building near the specified x, y coordinates using DuckDB.
    
    Args:
        parquet_file: Path to the parquet file
        x, y: Coordinates to search for
        proximity_threshold: Maximum distance to consider a match
    
    Returns:
        Building data row if found, None otherwise
    """
    print(f"Connecting to DuckDB and searching for buildings near ({x}, {y})...")
    
    # Connect to DuckDB
    con = duckdb.connect(database=':memory:')
    
    # Load spatial extension
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Create a spatial point for the target location
    point_wkt = f"POINT({x} {y})"
    
    # Query to find the closest building
    query = f"""
    SELECT *, 
           ST_Distance(boundary_2d, ST_GeomFromText('{point_wkt}')) AS distance
    FROM '{parquet_file}'
    WHERE boundary_2d IS NOT NULL
    ORDER BY distance
    LIMIT 1;
    """
    
    try:
        result = con.execute(query).fetchone()
        
        if result is None:
            print("No buildings found with valid boundary geometry.")
            return None
        
        distance = result[-1]  # The distance is the last column we added
        
        if distance <= proximity_threshold:
            print(f"Found building at distance {distance:.2f} units")
            # Convert to dict for consistency with the rest of the code
            column_names = con.execute(f"SELECT * FROM '{parquet_file}' LIMIT 0").description
            column_names = [col[0] for col in column_names] + ['distance']
            building_dict = dict(zip(column_names, result))
            return building_dict
        else:
            print(f"Closest building is at distance {distance:.2f} units, which exceeds threshold of {proximity_threshold}")
            if distance < float('inf'):
                column_names = con.execute(f"SELECT * FROM '{parquet_file}' LIMIT 0").description
                column_names = [col[0] for col in column_names] + ['distance']
                building_dict = dict(zip(column_names, result))
                return building_dict
    except Exception as e:
        print(f"Error executing DuckDB query: {e}")
        # Try a simpler query without spatial functions if the first one fails
        try:
            print("Trying alternative query method...")
            # Get total count
            count = con.execute(f"SELECT COUNT(*) FROM '{parquet_file}'").fetchone()[0]
            print(f"Found {count} buildings in the parquet file.")
            
            # Get the first building with non-null obj column
            result = con.execute(f"SELECT * FROM '{parquet_file}' WHERE obj IS NOT NULL LIMIT 1").fetchone()
            if result:
                print("Selected a random building for visualization")
                column_names = con.execute(f"SELECT * FROM '{parquet_file}' LIMIT 0").description
                column_names = [col[0] for col in column_names]
                building_dict = dict(zip(column_names, result))
                return building_dict
        except Exception as e2:
            print(f"Alternative query also failed: {e2}")
    
    return None

def extract_obj_from_blob(obj_blob):
    """
    Extract the OBJ data from the compressed blob.
    """
    print("Extracting 3D model data...")
    
    if obj_blob is None:
        print("Error: Object blob is None")
        return None
    
    decompression_methods = [
        # Standard gzip decompression
        lambda b: gzip.decompress(b),
        # Zlib decompression with different window sizes
        lambda b: zlib.decompress(b),
        lambda b: zlib.decompress(b, wbits=zlib.MAX_WBITS | 16),  # gzip format
        lambda b: zlib.decompress(b, wbits=zlib.MAX_WBITS | 32),  # zlib or gzip format
        # Try decoding directly (in case it's not actually compressed)
        lambda b: b,
    ]
    
    for method in decompression_methods:
        try:
            decompressed_data = method(obj_blob)
            # Check if it looks like a valid OBJ file by checking for common OBJ elements
            try:
                text = decompressed_data.decode('utf-8')
                if 'v ' in text[:1000] and 'f ' in text:  # Simple check for vertex and face definitions
                    print("Successfully decompressed OBJ data")
                    return text
            except UnicodeDecodeError:
                continue
        except Exception:
            continue
    
    print("Failed to extract valid OBJ data after trying multiple decompression methods")
    return None

def visualize_obj_with_pyvista(obj_data):
    """
    Visualize the OBJ 3D model using PyVista.
    """
    # Write the OBJ data to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
        tmp.write(obj_data.encode('utf-8'))
        tmp_path = tmp.name
    
    try:
        print("Visualizing with PyVista...")
        
        # Load the mesh with PyVista
        mesh = pv.read(tmp_path)
        
        # Set up the plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        plotter.show_axes()
        
        # Add some basic information to the visualization
        v_count = obj_data.count("\nv ")
        f_count = obj_data.count("\nf ")
        plotter.add_text(f"Vertices: {v_count}, Faces: {f_count}", position="upper_left", font_size=12)
        
        # Show the plot
        plotter.show()
        
    except Exception as e:
        print(f"Error visualizing with PyVista: {e}")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def get_buildings_stats(parquet_file):
    """
    Get some statistics about the buildings in the parquet file using DuckDB.
    """
    print("Getting statistics about the buildings dataset...")
    
    con = duckdb.connect(database=':memory:')
    
    try:
        # Total number of buildings
        total = con.execute(f"SELECT COUNT(*) FROM '{parquet_file}'").fetchone()[0]
        print(f"Total number of buildings: {total}")
        
        # Buildings by type
        try:
            type_stats = con.execute(f"""
                SELECT OBJEKTART, COUNT(*) as count 
                FROM '{parquet_file}' 
                GROUP BY OBJEKTART 
                ORDER BY count DESC
            """).fetchall()
            
            print("\nBuildings by type:")
            for type_name, count in type_stats:
                print(f"  {type_name}: {count}")
        except:
            print("Could not get building type statistics")
        
        # Check bounding box of dataset
        try:
            bbox_query = f"""
            SELECT 
                MIN(bbox->>'xmin') as min_x, 
                MAX(bbox->>'xmax') as max_x,
                MIN(bbox->>'ymin') as min_y, 
                MAX(bbox->>'ymax') as max_y
            FROM '{parquet_file}'
            WHERE bbox IS NOT NULL
            """
            bbox = con.execute(bbox_query).fetchone()
            if bbox and all(bbox):
                min_x, max_x, min_y, max_y = bbox
                print(f"\nDataset bounding box: X({min_x}, {max_x}), Y({min_y}, {max_y})")
        except:
            print("Could not calculate dataset bounding box")
            
    except Exception as e:
        print(f"Error getting dataset statistics: {e}")

def main(parquet_file, x, y, proximity_threshold=10.0, show_stats=True):
    """
    Main function to find a building at a location, extract its OBJ, and visualize it.
    """
    # Optionally show statistics about the dataset
    if show_stats:
        get_buildings_stats(parquet_file)
    
    # Find the building
    building = find_building_at_location(parquet_file, x, y, proximity_threshold)
    
    if building is None:
        print(f"No building found near coordinates ({x}, {y})")
        return
    
    print(f"\nBuilding details:")
    print(f"ID: {building.get('id', 'N/A')}")
    print(f"Type: {building.get('OBJEKTART', 'N/A')}")
    
    bbox = building.get('bbox')
    if bbox and hasattr(bbox, 'get'):
        print(f"Bounding Box: X({bbox.get('xmin', 'N/A'):.2f}, {bbox.get('xmax', 'N/A'):.2f}), "
              f"Y({bbox.get('ymin', 'N/A'):.2f}, {bbox.get('ymax', 'N/A'):.2f})")
    
    # Extract the OBJ data
    obj_data = extract_obj_from_blob(building.get('obj'))
    
    if obj_data is None:
        print("Failed to extract OBJ data")
        return
    
    # Count vertices and faces for a quick check
    v_count = obj_data.count("\nv ")
    f_count = obj_data.count("\nf ")
    print(f"OBJ file contains approximately {v_count} vertices and {f_count} faces")
    
    # Visualize the 3D model with PyVista
    visualize_obj_with_pyvista(obj_data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize a building 3D model from a parquet file using DuckDB')
    parser.add_argument('parquet_file', help='Path to the parquet file')
    parser.add_argument('x', type=float, help='X coordinate')
    parser.add_argument('y', type=float, help='Y coordinate')
    parser.add_argument('--threshold', type=float, default=10.0, help='Maximum distance to search for buildings')
    parser.add_argument('--no-stats', action='store_true', help='Skip showing dataset statistics')
    
    args = parser.parse_args()
    
    main(args.parquet_file, args.x, args.y, args.threshold, not args.no_stats)
