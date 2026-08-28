import logging
import math
import requests
import io
import os

from PIL import Image
from pyproj import Transformer
from datetime import datetime, timezone
import rasterio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.warp import reproject, Resampling
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared_models.canonical_aoi import CanonicalAOI

logger = logging.getLogger(__name__)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def _download_tile(zoom, x, y, cache_dir, provider="esri"):
    cache_path = os.path.join(cache_dir, f"{zoom}_{x}_{y}.jpg")
    if os.path.exists(cache_path):
        try:
            return x, y, Image.open(cache_path).convert('RGB')
        except Exception:
            pass # Invalid cache file, redownload

    if provider == "esri":
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    else:
        url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
        
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            if provider == "esri":
                arr = np.array(img)
                # Esri "Map data not yet available" tiles are uniformly grey with very low standard deviation
                if np.std(arr) < 15.0:
                    return x, y, None
            os.makedirs(cache_dir, exist_ok=True)
            img.save(cache_path)
            return x, y, img
        elif resp.status_code == 404:
            # Missing tile
            return x, y, None
    except Exception as e:
        logger.error(f"Failed to fetch high-res tile {x},{y} via {provider}: {e}")
    return x, y, None

def fetch_high_res_basemap(bounds_wgs84, zoom=18, canonical_aoi: 'CanonicalAOI' = None):
    """
    bounds_wgs84: (minx, miny, maxx, maxy) -> (min_lon, min_lat, max_lon, max_lat)
    Downloads Esri World Imagery (approx 0.6m resolution). Fallback to Google Maps Satellite if missing.
    """
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    
    for provider in ["esri", "google"]:
        cache_dir = f"cache/imagery/{provider}"
        provider_name = "Esri World Imagery" if provider == "esri" else "Google Maps Satellite"
        scene_prefix = "Esri_World_Imagery" if provider == "esri" else "Google_Satellite"
        
        # Attempt zoom 18, if too many 404s, fallback to zoom 17, 16
        for attempt_zoom in [zoom, zoom - 1, zoom - 2]:
            x_min, y_max = deg2num(min_lat, min_lon, attempt_zoom)
            x_max, y_min = deg2num(max_lat, max_lon, attempt_zoom)
            
            width = (x_max - x_min + 1) * 256
            height = (y_max - y_min + 1) * 256
            
            tasks = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        tasks.append(executor.submit(_download_tile, attempt_zoom, x, y, cache_dir, provider))
                        
            tiles = []
            missing = 0
            for future in as_completed(tasks):
                x, y, img = future.result()
                if img:
                    tiles.append((x, y, img))
                else:
                    missing += 1
                    
            total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
            if missing > total_tiles * 0.5:
                logger.warning(f"Over 50% tiles missing for {provider_name} at zoom {attempt_zoom}. Trying lower zoom...")
                continue # Try next zoom level
                
            stitched = Image.new('RGB', (width, height))
            for x, y, img in tiles:
                stitched.paste(img, ((x - x_min) * 256, (y - y_min) * 256))
                
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            
            tl_lat, tl_lon = num2deg(x_min, y_min, attempt_zoom)
            br_lat, br_lon = num2deg(x_max + 1, y_max + 1, attempt_zoom)
            
            tl_x, tl_y = transformer.transform(tl_lon, tl_lat)
            br_x, br_y = transformer.transform(br_lon, br_lat)
            
            t_min_x, t_min_y = transformer.transform(min_lon, min_lat)
            t_max_x, t_max_y = transformer.transform(max_lon, max_lat)
            
            px_min_x = int((t_min_x - tl_x) / (br_x - tl_x) * width)
            px_max_x = int((t_max_x - tl_x) / (br_x - tl_x) * width)
            px_min_y = int((t_max_y - tl_y) / (br_y - tl_y) * height)
            px_max_y = int((t_min_y - tl_y) / (br_y - tl_y) * height)
            
            cropped = stitched.crop((px_min_x, px_min_y, px_max_x, px_max_y))
            arr = np.array(cropped).transpose(2, 0, 1)
            
            crop_width = px_max_x - px_min_x
            crop_height = px_max_y - px_min_y
            
            transform_matrix = rasterio.transform.from_bounds(t_min_x, t_min_y, t_max_x, t_max_y, crop_width, crop_height)
            res_m = 156543.03 * math.cos(math.radians((min_lat+max_lat)/2)) / (2 ** attempt_zoom)
            
            if canonical_aoi:
                target_crs = rasterio.crs.CRS.from_string(canonical_aoi.crs_epsg)
                
                # Approximate provider GSD
                target_res = 0.6
                target_width = int(canonical_aoi.width_m / target_res)
                target_height = int(canonical_aoi.height_m / target_res)
                
                target_transform = rasterio.transform.from_bounds(
                    canonical_aoi.min_x, canonical_aoi.min_y, 
                    canonical_aoi.max_x, canonical_aoi.max_y, 
                    target_width, target_height
                )
                
                source_crs = rasterio.crs.CRS.from_epsg(3857)
                
                warped_data = np.zeros((3, target_height, target_width), dtype=arr.dtype)
                
                reproject(
                    source=arr,
                    destination=warped_data,
                    src_transform=transform_matrix,
                    src_crs=source_crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
                
                return {
                    "data": warped_data,
                    "transform": target_transform,
                    "crs": target_crs,
                    "bounds": (canonical_aoi.min_x, canonical_aoi.min_y, canonical_aoi.max_x, canonical_aoi.max_y),
                    "resolution": target_res,
                    "metadata": {
                        "provider": provider_name,
                        "scene_id": f"{scene_prefix}_Z{attempt_zoom}_Aligned",
                        "acquisition_datetime": datetime.now(timezone.utc).isoformat(),
                        "crs": canonical_aoi.crs_epsg,
                        "cloud_cover": 0.0
                    }
                }
                
            return {
                "data": arr,
                "transform": transform_matrix,
                "crs": rasterio.crs.CRS.from_epsg(3857),
                "bounds": (t_min_x, t_min_y, t_max_x, t_max_y),
                "resolution": res_m,
                "metadata": {
                    "provider": provider_name,
                    "scene_id": f"{scene_prefix}_Z{attempt_zoom}",
                    "acquisition_datetime": datetime.now(timezone.utc).isoformat(),
                    "crs": "EPSG:3857",
                    "cloud_cover": 0.0
                }
            }
    raise Exception("Failed to retrieve imagery from both Esri and Google Maps Satellite.")
