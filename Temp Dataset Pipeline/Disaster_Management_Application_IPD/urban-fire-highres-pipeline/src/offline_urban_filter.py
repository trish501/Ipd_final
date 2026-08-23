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
    required_files = [
        "data/shapefiles/ne_10m_urban_areas.shp",
        "data/shapefiles/ne_10m_urban_areas.shx",
        "data/shapefiles/ne_10m_urban_areas.dbf",
        "data/shapefiles/ne_10m_urban_areas.prj"
    ]
    for rf in required_files:
        if not os.path.exists(rf):
            logger.error(f"Shapefile companion missing: {rf}")
            return False
        
    try:
        logger.info("Loading offline urban boundaries into memory...")
        _urban_gdf = gpd.read_file(shapefile_path)
        if _urban_gdf is None or _urban_gdf.empty:
            logger.error("Shapefile loaded but is empty or corrupted.")
            _urban_gdf = None
            return False
            
        if _urban_gdf.crs is None:
            logger.error("Shapefile missing CRS.")
            _urban_gdf = None
            return False
            
        # Ensure it's in WGS84 since FIRMS provides lat/lon
        if _urban_gdf.crs.to_string() != "EPSG:4326":
            logger.info(f"Transforming shapefile CRS from {_urban_gdf.crs} to EPSG:4326")
            _urban_gdf = _urban_gdf.to_crs("EPSG:4326")

        # Ensure it has a fast spatial index (sindex is built automatically on first query in geopandas)
        _urban_gdf.sindex
        logger.info(f"Successfully loaded {_urban_gdf.shape[0]} global urban polygons.")
        return True
    except Exception as e:
        logger.error(f"Failed to load urban shapefile: {e}")
        _urban_gdf = None
        return False

def is_in_urban_area(lat, lon):
    """
    Checks if a coordinate is strictly within a known urban polygon.
    Returns True if it's in a city, False if it's rural/ocean/forest.
    """
    global _urban_gdf
    if _urban_gdf is None:
        logger.error("Offline filter not initialized, fail closed.")
        return "URBAN_FILTER_VALIDATION_FAILED"
        
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

