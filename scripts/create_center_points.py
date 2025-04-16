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
    # Use only the 50% largest buildings (measured by the length of the boundary)
    conn.execute("""
    CREATE TABLE building_centers AS
    WITH lengths AS (
      SELECT
        ST_Length(boundary_2d) AS length,
        ((bbox['xmin'] + bbox['xmax']) / 2)::INTEGER AS x,
        ((bbox['ymin'] + bbox['ymax']) / 2)::INTEGER AS y
        FROM buildings
    ),
    median_val AS (
       SELECT median(length) AS median_length FROM lengths
    )
    SELECT x,y
    FROM lengths, median_val
    WHERE length > median_length;
    """)
    
    # Write to output parquet file
    conn.execute(f"COPY building_centers TO '{output_file}' (FORMAT PARQUET)")
    
    count = conn.execute("SELECT COUNT(*) FROM building_centers").fetchone()[0]
    print(f"Created {output_file} with {count} building centers")

if __name__ == "__main__":
    main()
