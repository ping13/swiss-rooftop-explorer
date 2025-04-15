#!/usr/bin/env python3
"""
Simple script to extract building center points from a parquet file.
Usage: python create_center_points input.parquet output.parquet
"""

import sys
import duckdb

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_center_points input.parquet output.parquet")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Processing {input_file}...")
    
    # Connect to DuckDB and load spatial extension
    conn = duckdb.connect(':memory:')
    conn.execute("INSTALL spatial")
    conn.execute("LOAD spatial")
    
    # Create a table with building data from the input file
    conn.execute(f"CREATE TABLE buildings AS SELECT * FROM read_parquet('{input_file}')")
    
    # Calculate building centers and create a new table.
    # Use only buildings with an area greater than 200sqm, the geometry is in buildings.boundary_2d.
    conn.execute("""
    CREATE TABLE building_centers AS 
    SELECT 
    uuid,
    ((bbox['xmin'] + bbox['xmax']) / 2)::INTEGER AS x,
    ((bbox['ymin'] + bbox['ymax']) / 2)::INTEGER AS y
    FROM buildings
    WHERE ST_Area(boundary_2d) > 200;
    """)
    
    # Write to output parquet file
    conn.execute(f"COPY building_centers TO '{output_file}' (FORMAT PARQUET)")
    
    count = conn.execute("SELECT COUNT(*) FROM building_centers").fetchone()[0]
    print(f"Created {output_file} with {count} building centers")

if __name__ == "__main__":
    main()
