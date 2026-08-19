import time
import logging
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import threading
from config.settings import GEOCODE_TIMEOUT, GEOCODE_DELAY

logger = logging.getLogger(__name__)

# Single instances
_geolocator = None
_geocode_lock = threading.Lock()
_urban_gdf = None
_cache = {}

def get_geolocator():
    global _geolocator
    if _geolocator is None:
        _geolocator = Nominatim(user_agent="urban_fire_dataset_generator_v2")
    return _geolocator

def get_bounding_box(query: str):
    """
    Forward geocodes a location query (e.g. 'Mumbai India') to get a bounding box.
    Returns: (min_lat, max_lat, min_lon, max_lon) or None if not found.
    """
    geolocator = get_geolocator()
    try:
        with _geocode_lock:
            time.sleep(GEOCODE_DELAY)
            loc = geolocator.geocode(query)
            
        if loc and loc.raw.get('boundingbox'):
            bb = loc.raw['boundingbox']
            return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        return None
    except (GeocoderTimedOut, GeocoderUnavailable, Exception) as e:
        logger.error(f"Failed to geocode bounding box for '{query}': {e}")
        return None

def get_location(lat: float, lon: float):
    """
    Reverse geocodes lat/lon to City, State, Country.
    Implements a simple in-memory cache to avoid redundant API calls.
    Returns: "City, State, Country" or "Not available" for missing parts.
    """
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in _cache:
        return _cache[cache_key]
        
    geolocator = get_geolocator()
        
    try:
        with _geocode_lock:
            time.sleep(GEOCODE_DELAY) 
            location = geolocator.reverse((lat, lon), exactly_one=True, timeout=GEOCODE_TIMEOUT)
        
        if not location or not location.raw.get('address'):
            result = "Not available, Not available, Not available"
        else:
            address = location.raw['address']
            city = address.get('city', address.get('town', address.get('village', address.get('county', 'Not available'))))
            state = address.get('state', 'Not available')
            country = address.get('country', 'Not available')
            result = f"{city}, {state}, {country}"
            
        _cache[cache_key] = result
        return result
        
    except (GeocoderTimedOut, GeocoderUnavailable, Exception) as e:
        logger.error(f"Geocoding failed for {lat},{lon}: {e}")
        return "Not available, Not available, Not available"



def filter_urban_events(events):
    """
    Batched spatial filter. Returns a list of events that strictly intersect urban polygons.
    """
    global _urban_gdf
    if _urban_gdf is None:
        logger.warning("Offline filter not initialized, skipping fast filter.")
        return events
        
    if not events:
        return []
        
    import pandas as pd
    
    df = pd.DataFrame(events)
    events_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    
    # Vectorized spatial join
    joined = gpd.sjoin(events_gdf, _urban_gdf, how="inner", predicate="intersects")
    
    # Deduplicate in case a point intersects multiple overlapping polygons
    if 'event_id' in joined.columns:
        joined = joined.drop_duplicates(subset=['event_id'])
    else:
        joined = joined.drop_duplicates(subset=['latitude', 'longitude', 'date', 'time'])
        
    # Drop geometry and sjoin columns
    out_cols = [c for c in df.columns if c != 'geometry']
    filtered_df = joined[out_cols]
    
    return filtered_df.to_dict('records')
