import os

# Base paths
DATASET_DIR = "dataset"
DATA_DIR = "data"
CSV_DIR = os.path.join(DATA_DIR, "csv")
SHAPEFILE_DIR = os.path.join(DATA_DIR, "shapefiles")

CACHE_DIR = os.path.join(DATASET_DIR, "cache")
URBAN_CACHE_DIR = os.path.join(CACHE_DIR, "urban")
SEARCH_CACHE_DIR = os.path.join(CACHE_DIR, "search")

METADATA_DIR = os.path.join(DATASET_DIR, "metadata")
EVENTS_DIR = os.path.join(DATASET_DIR, "unreviewed", "events")

# API Configs
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

MAX_RETRIES = 3
API_BACKOFF = 2.0
GEOCODE_TIMEOUT = 5
GEOCODE_DELAY = 1.1 # Strict Nominatim 1-req-per-sec limit

# Satellite Search Config
MAX_CLOUD_COVER = 30.0
SEARCH_DAYS = 3
GRID_SIZE_KM = 5.0
IMAGE_CROP_KM = 2.0

# Multithreading (can be overwritten by CLI)
DEFAULT_MAX_WORKERS = 4
