import os
import logging
import urllib.request
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Global cache for the industrial dataset so we only load it once
_industrial_gdf = None

# Using a buffer of roughly 2km (~0.02 degrees) around power plants
INDUSTRIAL_BUFFER_DEG = 0.02 

def init_industrial_filter():
    """Loads the Global Power Plant Database into a Geopandas spatial index.
    Downloads it automatically if it doesn't exist."""
    global _industrial_gdf
    if _industrial_gdf is not None:
        return True
        
    csv_dir = "data/csv"
    csv_path = os.path.join(csv_dir, "global_power_plant_database.csv")
    download_url = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
    
    if not os.path.exists(csv_path):
        os.makedirs(csv_dir, exist_ok=True)
        print("\n[INFO] WRI Global Power Plant Database not found. Downloading (this may take a moment)...")
        try:
            urllib.request.urlretrieve(download_url, csv_path)
            print("[INFO] Successfully downloaded Global Power Plant Database.\n")
        except Exception as e:
            logger.error(f"Failed to download Global Power Plant Database: {e}")
            print(f"\n[ERROR] Failed to download Global Power Plant Database: {e}\n")
            return False

    try:
        logger.info("Loading offline industrial filter into memory...")
        df = pd.read_csv(csv_path, usecols=['latitude', 'longitude'])
        
        # Convert to Geopandas DataFrame
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        _industrial_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Ensure it has a fast spatial index (sindex is built automatically on first query in geopandas)
        _industrial_gdf.sindex
        logger.info(f"Successfully loaded {_industrial_gdf.shape[0]} global industrial power plants.")
        return True
    except Exception as e:
        logger.error(f"Failed to load industrial shapefile: {e}")
        _industrial_gdf = None
        return False

def is_near_industrial(lat, lon, buffer_deg=INDUSTRIAL_BUFFER_DEG):
    """
    Checks if a coordinate is within `buffer_deg` of a known industrial site (power plant).
    Returns True if it's too close to a power plant, False otherwise.
    """
    global _industrial_gdf
    if _industrial_gdf is None:
        logger.error("Offline industrial filter not initialized, fail closed.")
        # Fail closed to avoid generating false positives if requested
        return "INDUSTRIAL_FILTER_VALIDATION_FAILED"
        
    # Create a bounding box around the target point
    minx, miny = lon - buffer_deg, lat - buffer_deg
    maxx, maxy = lon + buffer_deg, lat + buffer_deg
    
    # Query spatial index for intersecting bounding boxes (super fast)
    possible_matches_index = list(_industrial_gdf.sindex.intersection((minx, miny, maxx, maxy)))
    
    # If any power plants are within this bounding box, we consider it a match
    # Since we only buffered slightly, a bounding box check is sufficient for filtering.
    if possible_matches_index:
        return True
        
    return False
