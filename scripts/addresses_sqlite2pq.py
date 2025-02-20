"""Convert Swiss address data from SQLite to Parquet format.

This script processes Swiss address data stored in an SQLite database, writes
the data to a Parquet file. The data is grouped by postal code (zipcode) for
efficient storage and querying.

The script handles:
- Coordinate transformation using pyproj
- Data type validation and conversion
- Grouping data by postal code
- Writing to Parquet with optimized row group sizes

"""

from typing import Optional, Union, Dict, List
import click
import os
import csv
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
import sqlite3
from pyproj import Transformer

@click.command()
@click.option('--sqlite-file', default='data/data.sqlite', help='Path to the input zipped CSV file.')
@click.option("--full", is_flag=True, default=False, help="write out the full set of columns")
@click.argument('output_parquet_file')
def tables_to_pq_from_sqlite(sqlite_file: str, full: bool, output_parquet_file: str) -> None:
    """
    Convert SQLite address data to Parquet format.

    Args:
        sqlite_file (str): Path to the input SQLite database file
        output_parquet_file (str): Path to the output Parquet file

    The function performs the following steps:
    1. Defines the schema for the Parquet file
    2. Reads data from the SQLite database
    3. Transforms coordinates from LV95 to WGS84
    4. Groups data by postal code (DPLZ4)
    5. Writes the data to a Parquet file with optimized row groups

    Raises:
        Exception: If any error occurs during processing
    """
    # Define schema first
    if full:
        print("using full schema")
        schema = pa.schema([
            ("EGID", pa.int32()),
            ("EDID", pa.uint8()),
            ("EGAID", pa.int32()),
            ("DEINR", pa.string()),
            ("ESID", pa.int32()),
            ("STRNAME", pa.string()),
            ("STRNAMK", pa.string()),
            ("STRINDX", pa.string()),
            ("STRSP", pa.string()),
            ("STROFFIZIEL", pa.string()),
            ("DPLZ4", pa.uint16()),
            ("DPLZZ", pa.string()),
            ("DPLZNAME", pa.string()),
            ("latitude", pa.float32()),
            ("longitude", pa.float32()),
            ("DOFFADR", pa.string()),
            ("DEXPDAT", pa.date32())
        ])
    else:
        print("using stripped down schema")
        schema = pa.schema([
            ("DEINR", pa.string()),
            ("STRNAME", pa.string()),
            ("DPLZ4", pa.uint16()),
            ("DPLZNAME", pa.string()),
            ("latitude", pa.float32()),
            ("longitude", pa.float32())
        ])
        
    # Define helper functions
    def safe_float(value: Optional[Union[str, float]]) -> Optional[float]:
        if not value:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def safe_int(value: Optional[Union[str, int]]) -> Optional[int]:
        return int(value) if value not in (None,'') else None
    
    # Initialize data structures
    grouped_data = defaultdict(list)
    
    conn = None
    try:
        # Initialize transformer
        transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326")
        
        # Connect to the SQLite database
        conn = sqlite3.connect(sqlite_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM entrance ORDER BY DPLZ4 ASC")

        # Process all rows, group data for row_groups in the parquet file
        for row in cursor:
            zipcode = safe_int(row['DPLZ4'])
            easting = safe_float(row['DKODE'])
            northing = safe_float(row['DKODN'])
            
            # Transform coordinates if both values exist
            lat, lon = None, None
            if easting is not None and northing is not None:
                lat, lon = transformer.transform(easting, northing)

            if full:
                d = {
                    "EGID": safe_int(row['EGID']),
                    "EDID": safe_int(row['EDID']),
                    "EGAID": safe_int(row['EGAID']),
                    "DEINR": row['DEINR'],
                    "ESID": safe_int(row['ESID']),
                    "STRNAME": row['STRNAME'],
                    "STRNAMK": row['STRNAMK'], 
                    "STRINDX": row['STRINDX'],
                    "STRSP": row['STRSP'],
                    "STROFFIZIEL": row['STROFFIZIEL'],
                    "DPLZ4": safe_int(row['DPLZ4']),
                    "DPLZZ": row['DPLZZ'],
                    "DPLZNAME": row['DPLZNAME'],
                    "latitude": lat,
                    "longitude": lon,
                    "DOFFADR": row['DOFFADR'],
                    "DEXPDAT": row['DEXPDAT']
                }
            else:
                d = {
                    "STRNAME": row['STRNAME'],
                    "DEINR": row['DEINR'],
                    "DPLZ4": safe_int(row['DPLZ4']),
                    "DPLZNAME": row['DPLZNAME'],
                    "latitude": lat,
                    "longitude": lon
                }
                
            grouped_data[zipcode].append(d)

        # Write the Parquet file with the row groups
        with pq.ParquetWriter(output_parquet_file, schema=schema) as writer:
            for i, (zipcode, list_of_dicts) in enumerate(sorted(grouped_data.items())):
                arrays = [
                    pa.array([d[field] if field != "DPLZ4" else zipcode for d in list_of_dicts])
                    for field in schema.names
                ]
                table = pa.Table.from_arrays(arrays, schema=schema)
                writer.write_table(table, row_group_size=min(10000, len(list_of_dicts)))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    try:
        tables_to_pq_from_sqlite()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)
