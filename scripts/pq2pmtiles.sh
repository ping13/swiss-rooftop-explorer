SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e
set -u
set -o pipefail

# load and convert from parquet file and pipe it in tippecanoe
duckdb -c "
install spatial;
load spatial;

COPY ( 
    SELECT
        'Feature' AS type,
        st_asgeojson(ST_FlipCoordinates(ST_Transform(boundary_2d, 'EPSG:2056', 'EPSG:4326'))) AS geometry,
        json_object(
            'roof_max', DACH_MAX,
            'roof_min', DACH_MIN,
            'year', ERSTELLUNG_JAHR::INT,
            'type', 1,
            'id', ROW_NUMBER() OVER(),
            'objecttype', OBJEKTART::VARCHAR
        ) AS properties,
        ROW_NUMBER() OVER() AS id
    FROM read_parquet('$1')

    UNION ALL

    SELECT
        'Feature' AS type,
        st_asgeojson(ST_FlipCoordinates(ST_Transform(geometry, 'EPSG:2056', 'EPSG:4326'))) AS geometry,
        json_object(
            'roof_max', DACH_MAX,
            'roof_min', DACH_MIN,
            'year', ERSTELLUNG_JAHR::INT,
            'type', 2,
            'id', ROW_NUMBER() OVER(),
            'objecttype', OBJEKTART::VARCHAR
        ) AS properties,
        ROW_NUMBER() OVER() AS id
    FROM read_parquet('$2')

    UNION ALL

    SELECT
        'Feature' AS type,
        st_asgeojson(ST_FlipCoordinates(ST_Transform(geometry, 'EPSG:2056', 'EPSG:4326'))) AS geometry,
        json_object(
            'roof_max', DACH_MAX,
            'roof_min', DACH_MIN,
            'year', ERSTELLUNG_JAHR::INT,
            'type', 3,
            'id', ROW_NUMBER() OVER(),
            'objecttype', OBJEKTART::VARCHAR
        ) AS properties,
        ROW_NUMBER() OVER() AS id
    FROM read_parquet('$3')

    UNION ALL

    SELECT
        'Feature' AS type,
        st_asgeojson(ST_FlipCoordinates(ST_Transform(geometry, 'EPSG:2056', 'EPSG:4326'))) AS geometry,
        json_object(
            'roof_max', DACH_MAX,
            'roof_min', DACH_MIN,
            'year', ERSTELLUNG_JAHR::INT,
            'type', 3,
            'id', ROW_NUMBER() OVER(),
            'objecttype', OBJEKTART::VARCHAR
        ) AS properties,
        ROW_NUMBER() OVER() AS id
    FROM read_parquet('$4')

) TO STDOUT (FORMAT json)
; 
" |  tippecanoe -o $5 --force -J $SCRIPT_DIR/roofs.filter.json -zg -l roof  --progress-interval=10 --no-feature-limit --no-tile-size-limit --no-clipping  # --visvalingam --no-simplification-of-shared-nodes  --no-line-simplification --use-source-polygon-winding --no-clipping 
