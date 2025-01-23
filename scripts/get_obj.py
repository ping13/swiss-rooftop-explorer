import pandas as pd
import pyarrow.parquet as pq
import zlib

def get_obj_for_uuid(parquet_path, uuid, output_file):
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Find the row with the specified UUID
    row = df[df['UUID'] == uuid]
    
    if len(row) == 0:
        raise ValueError(f"No building found with UUID {uuid}")
    
    # Get the compressed OBJ data
    compressed_obj = row.iloc[0]['obj']

    # Decompress the data
    obj_data = zlib.decompress(compressed_obj).decode('utf-8')
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(obj_data)

if __name__ == "__main__":
    parquet_path = "assets/swissBUILDINGS3D_3-0_1112-13_Building_solid_2d.parquet"
    uuid = "{8C6B945B-82F0-4CED-8CE6-AD5A77DEDDC7}"
    output_file = "test.obj"
    
    get_obj_for_uuid(parquet_path, uuid, output_file)
    print(f"OBJ file saved to {output_file}")
