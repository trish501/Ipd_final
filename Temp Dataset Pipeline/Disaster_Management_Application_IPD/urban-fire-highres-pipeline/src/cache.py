import logging
logger = logging.getLogger(__name__)
import os
import json
import threading
from config.settings import URBAN_CACHE_DIR, SEARCH_CACHE_DIR
from src.utils import get_hash_key

def _atomic_write_json(path, data):
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)

def init_cache():
    os.makedirs(URBAN_CACHE_DIR, exist_ok=True)
    os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)

def get_urban_cache(lat: float, lon: float):
    # Round to 4 decimal places (~11m accuracy) for caching geographical lookups
    key = f"{round(lat, 4)}_{round(lon, 4)}"
    path = os.path.join(URBAN_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read cache {path}: {e}")
            return None
    return None

def set_urban_cache(lat: float, lon: float, data: dict):
    if data is None:
        return
    key = f"{round(lat, 4)}_{round(lon, 4)}"
    path = os.path.join(URBAN_CACHE_DIR, f"{key}.json")
    _atomic_write_json(path, data)

def get_search_cache(lat: float, lon: float, date_str: str, time_str: str):
    key = get_hash_key(f"{round(lat, 4)}_{round(lon, 4)}_{date_str}_{time_str}")
    path = os.path.join(SEARCH_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read cache {path}: {e}")
            return None
    return None

def set_search_cache(lat: float, lon: float, date_str: str, time_str: str, item_dict: dict):
    if item_dict is None:
        return
    key = get_hash_key(f"{round(lat, 4)}_{round(lon, 4)}_{date_str}_{time_str}")
    path = os.path.join(SEARCH_CACHE_DIR, f"{key}.json")
    _atomic_write_json(path, item_dict)

def get_grid_search_cache(grid_x: int, grid_y: int, date_str: str):
    key = get_hash_key(f"grid_{grid_x}_{grid_y}_{date_str}")
    path = os.path.join(SEARCH_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read cache {path}: {e}")
            return None
    return None

def set_grid_search_cache(grid_x: int, grid_y: int, date_str: str, item_dict: dict):
    if item_dict is None:
        return
    key = get_hash_key(f"grid_{grid_x}_{grid_y}_{date_str}")
    path = os.path.join(SEARCH_CACHE_DIR, f"{key}.json")
    _atomic_write_json(path, item_dict)
