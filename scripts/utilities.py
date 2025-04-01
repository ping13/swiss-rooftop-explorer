import re
import logging
from shapely.geometry import (
    LineString, MultiLineString, Point, MultiPoint,
    Polygon, MultiPolygon, GeometryCollection
)
import httpx
import json
from hishel import CacheClient, FileStorage

# Setup logger
logger = logging.getLogger(__name__)

# Add a formatter that includes timestamp, level, and module
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Add INFO level logging to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


# Create a cache storage with 30-day TTL
FILE_STORAGE_CACHE = FileStorage(ttl=60 * 60 * 24 * 100)  # 100 days in seconds

def get_min_height_swissalti_service(linestring: LineString) -> float:
    coords = [[round(x), round(y)] for x, y, *_ in linestring.coords]
    
    # If there are more than 4000 coordinates, thin the list to take only every n-th coordinate
    if len(coords) > 4000:
        # Calculate how many points to skip to get under 4094 points
        n = len(coords) // 4000 + 1
        coords = coords[::n]
    
    geom = {"type": "LineString", "coordinates": coords}
    geom_str = json.dumps(geom)

    url = "https://api3.geo.admin.ch/rest/services/profile.json"
    params = {"geom": geom_str}

    try:
        
        # Create a client with the cache
        with CacheClient(storage=FILE_STORAGE_CACHE) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()  # List of points

        heights = [pt["alts"]["COMB"] for pt in data if "alts" in pt and "COMB" in pt["alts"]]
        if not heights:
            raise httpx.HTTPStatusError("No valid elevation data returned.")
        return min(heights)
    except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
        fall_back_height = 50 
        logger.critical(f"couldn't get height for {linestring}: {e}, fall back to default (assume {fall_back_height} height)")
        if e.response.status_code >= 400:
            # If status is 400, calculate a fallback height
            # Find the smallest z-coordinate and subtract 100m
            z_coords = [coord[2] for coord in linestring.coords if len(coord) > 2]
            if z_coords:
                return min(z_coords) - fall_back_height
            else:
                # If no z-coordinates are available, use a default value
                return 0  # or some other sensible default
        else:
            # Re-raise other HTTP errors
            raise

def define_bridge_parameters(length_3d, 
                             anzahl_spuren=None,
                             objektart=None):

    default_railway_width=8
    default_road_width=3
    autobahn_width=20 
    bottom_shift_percentage=0
    arch_height_fraction=0.85
    circular_arch = False
    
    # - let's determine the deck width based on input parameters
    if anzahl_spuren is not None:  # Railway Bridges
        deck_width = str(default_railway_width * int(anzahl_spuren))
    elif objektart is not None:  # Road Bridges
        match = re.search(r"(\d+)m\s.*", objektart)
        if match:
            deck_width = match.group(1)
            deck_width = f"{(float(deck_width) * 1.3):.1f}"  # correction factor, TODO: still needed?
        else:
            if objektart == "Autobahn":
                deck_width = str(autobahn_width)
            else:
                deck_width = str(default_road_width)
    else:
        raise Exception("Cannot determine the width of the bridge deck: property `anzahl_spuren` nor `objektart` is defined.")

    pier_size_meters = min(3, float(deck_width) / 2)
    
    # let's define arch_fractions based on the feature's length
    if length_3d < 20:
        arch_fractions = None
    elif length_3d < 50:
        n = 3  # TODO: define a better default value for arches
        arch_fractions = [1 / n] * n
    elif length_3d < 100:
        n = 4  # TODO: define a better default value for arches
        arch_fractions = [1 / n] * n
    else:
        n = 6  # TODO: define a better default value for arches
        arch_fractions = [1 / n] * n

    return deck_width, bottom_shift_percentage, arch_fractions, pier_size_meters, arch_height_fraction, circular_arch
        
