import os
import logging
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Global cache for the shapefile so we only load it once
_urban_gdf = None

def init_offline_filter():
    """Loads the Natural Earth Urban Areas shapefile into a Geopandas spatial index."""
    global _urban_gdf
    if _urban_gdf is not None:
        return True
        
    shapefile_path = "data/shapefiles/ne_10m_urban_areas.shp"
    if not os.path.exists(shapefile_path):
        logger.error(f"Shapefile not found at {shapefile_path}. Did you download it?")
        return False
        
    try:
        logger.info("Loading offline urban boundaries into memory...")
        _urban_gdf = gpd.read_file(shapefile_path)
        # Ensure it has a fast spatial index (sindex is built automatically on first query in geopandas)
        _urban_gdf.sindex
        logger.info(f"Successfully loaded {_urban_gdf.shape[0]} global urban polygons.")
        return True
    except Exception as e:
        logger.error(f"Failed to load urban shapefile: {e}")
        return False

def is_in_urban_area(lat, lon):
    """
    Checks if a coordinate is strictly within a known urban polygon.
    Returns True if it's in a city, False if it's rural/ocean/forest.
    """
    global _urban_gdf
    if _urban_gdf is None:
        # If shapefile isn't loaded, default to True so we don't accidentally drop valid fires.
        # It will just fall back to the slow WorldCover API check.
        logger.warning("Offline filter not initialized, skipping fast filter.")
        return True
        
    # Create a shapely point (lon, lat)
    point = Point(lon, lat)
    
    # Query spatial index for intersecting bounding boxes first (super fast)
    possible_matches_index = list(_urban_gdf.sindex.intersection(point.bounds))
    if not possible_matches_index:
        return False
        
    # Then do precise exact intersection check
    possible_matches = _urban_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(point)]
    
    return not precise_matches.empty

