import logging
import time
from datetime import datetime, timedelta, timezone
import pystac
import pystac_client
import planetary_computer
import threading
from shapely.geometry import shape, Point
from src.cache import get_grid_search_cache, set_grid_search_cache
from src.spatial_grid import get_grid_cell

logger = logging.getLogger(__name__)

_catalog = None
_grid_locks = {}
_grid_locks_lock = threading.Lock()

def get_catalog():
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    return _catalog

def retry_on_exception(retries=3, backoff=2.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise e
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
        return wrapper
    return decorator

def get_bounding_box(lat: float, lon: float, offset_deg: float = 0.05):
    return [lon - offset_deg, lat - offset_deg, lon + offset_deg, lat + offset_deg]

def format_time_window(center_datetime: datetime, search_days: int) -> str:
    start_dt = center_datetime - timedelta(days=search_days)
    end_dt = center_datetime + timedelta(days=search_days)
    return f"{start_dt.isoformat()}/{end_dt.isoformat()}"

@retry_on_exception(retries=3, backoff=2.0)
def search_satellite_imagery(lat: float, lon: float, date_str: str, time_str: str, search_days: int = 3, max_cloud: float = 30.0):
    grid_x, grid_y = get_grid_cell(lat, lon)
    grid_key = f"{grid_x}_{grid_y}_{date_str}"
    
    with _grid_locks_lock:
        if grid_key not in _grid_locks:
            _grid_locks[grid_key] = threading.Lock()
        glock = _grid_locks[grid_key]
        
    with glock:
        cached_item = get_grid_search_cache(grid_x, grid_y, date_str)
        if cached_item is not None:
            if cached_item == {}: 
                return None
            item = pystac.Item.from_dict(cached_item)
            return planetary_computer.sign(item)
            
        try:
            if time_str:
                dt_str = f"{date_str} {time_str}"
                # Handle HH:MM or HHMM depending on FIRMS data format
                if ":" in time_str:
                    event_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                else:
                    event_dt = datetime.strptime(dt_str, "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
            else:
                event_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.error(f"Invalid date/time format for search: {e}")
            return None
            
        bbox = get_bounding_box(lat, lon)
        time_window = format_time_window(event_dt, search_days)
        
        try:
            catalog = get_catalog()
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=time_window,
                query={"eo:cloud_cover": {"lt": max_cloud}}
            )
            items = list(search.items())
            
            if not items:
                set_grid_search_cache(grid_x, grid_y, date_str, {})
                return None
                
            best_item = None
            min_time_diff = None
            event_point = Point(lon, lat)
            required_bands = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
            
            for item in items:
                # 1. Temporal Window
                item_dt = item.datetime
                if not item_dt.tzinfo:
                    item_dt = item_dt.replace(tzinfo=timezone.utc)
                    
                time_diff = abs((item_dt - event_dt).total_seconds())
                if time_diff > search_days * 86400:
                    logger.debug(f"REJECTED {item.id}: outside temporal window")
                    continue
                    
                # 2. Cloud Cover Verification
                cloud_cover = item.properties.get("eo:cloud_cover", 100.0)
                if cloud_cover >= max_cloud:
                    logger.debug(f"REJECTED {item.id}: cloud cover >= {max_cloud}%")
                    continue
                    
                # 3. Geometric Coverage
                geom = shape(item.geometry)
                if not geom.covers(event_point):
                    logger.debug(f"REJECTED {item.id}: fire coordinate outside Sentinel-2 footprint")
                    continue
                    
                # 4. Required Band Validation
                missing_bands = [b for b in required_bands if b not in item.assets]
                if missing_bands:
                    logger.debug(f"REJECTED {item.id}: missing required band(s) {missing_bands}")
                    continue
                    
                # 5. Temporal Ranking
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_item = item
                elif time_diff == min_time_diff:
                    if best_item is None or item.id > best_item.id:
                        best_item = item
                    
            if best_item:
                logger.debug(f"SELECTED {best_item.id}: valid candidate with minimum temporal difference")
                set_grid_search_cache(grid_x, grid_y, date_str, best_item.to_dict())
                return planetary_computer.sign(best_item)
            else:
                logger.warning(f"No Sentinel-2 scene satisfies temporal, cloud, coordinate-coverage, and required-band requirements for {date_str} at {lat}, {lon}.")
                
            return best_item
            
        except Exception as e:
            logger.error(f"Sentinel-2 search failed: {e}")
            return None

