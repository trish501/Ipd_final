import os
import gzip
import csv
import json
import requests
import hashlib
from requests.exceptions import RequestException
from typing import Dict, Optional, Tuple, List
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree
import math

from src.models.output import Building
from src.utils.logger import logger

def latlon_to_quadkey(lat: float, lon: float, level: int = 9) -> str:
    """Calculates the Bing Maps QuadKey for a given lat/lon at a specific level."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    sin_lat = math.sin(lat * math.pi / 180)
    
    pixel_x = ((lon + 180) / 360) * 256 * (2 ** level)
    pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * 256 * (2 ** level)
    
    tile_x = int(pixel_x / 256)
    tile_y = int(pixel_y / 256)
    
    quadkey = ""
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey

class MicrosoftBuildingFootprintsBatchSource:
    def __init__(self, cache_dir: str = "cache/microsoft"):
        self.name = "MicrosoftBuildingFootprints"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.links_cache_path = os.path.join(self.cache_dir, "dataset-links.csv")
        self._ensure_dataset_links()

    def _ensure_dataset_links(self):
        if not os.path.exists(self.links_cache_path) or os.path.getsize(self.links_cache_path) == 0:
            logger.info("Downloading Microsoft dataset-links.csv...")
            url = "https://bfppub.blob.core.windows.net/$web/2026-07-24/dataset-links.csv"
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                with open(self.links_cache_path, "w") as f:
                    f.write(resp.text)
            except RequestException as e:
                logger.error(f"Failed to download Microsoft links: {e}")

    def resolve_partitions(self, lat: float, lon: float) -> List[str]:
        """Returns all partition URLs for the given coordinate."""
        target_quadkey = latlon_to_quadkey(lat, lon, level=9)
        urls = []
        
        if not os.path.exists(self.links_cache_path):
            return urls
            
        with open(self.links_cache_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qk = row.get("QuadKey", "")
                loc = row.get("Location", "")
                url = row.get("Url", "")
                
                if (qk and target_quadkey.startswith(qk)) or ("Hawaii" in loc and -161 <= lon <= -154 and 18 <= lat <= 23):
                    urls.append(url)
        return urls
        
    def _get_filename_from_url(self, url: str) -> str:
        # e.g. "part-0000.csv.gz". We use a hash to be safe against collisions if multiple partitions have same name
        h = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"ms_{h}.geojsonl.gz"

    def partition_is_cached(self, url: str) -> bool:
        cache_path = os.path.join(self.cache_dir, self._get_filename_from_url(url))
        return os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
        
    def download_partition(self, url: str) -> bool:
        """Downloads the partition if missing. Returns True if available, False if error."""
        if self.partition_is_cached(url):
            return True
            
        cache_path = os.path.join(self.cache_dir, self._get_filename_from_url(url))
        tmp_path = f"{cache_path}.tmp"
        
        logger.info(f"Downloading Microsoft partition from {url}")
        try:
            resp = requests.get(url, stream=True, timeout=(5, 30))
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        
            os.rename(tmp_path, cache_path)
            return True
            
        except RequestException as e:
            logger.error(f"Network error downloading Microsoft partition {url}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    def parse_and_index(self, url: str, batch_bounds: Tuple[float, float, float, float]) -> Tuple[Optional[STRtree], List[Building]]:
        """
        Parses the cached partition, keeping only buildings that intersect the `batch_bounds`.
        Returns an STRtree and a list of valid Building objects.
        """
        cache_path = os.path.join(self.cache_dir, self._get_filename_from_url(url))
        if not self.partition_is_cached(url):
            return None, []
            
        buildings = []
        geometries = []
        minx, miny, maxx, maxy = batch_bounds
        batch_poly = Polygon.from_bounds(minx, miny, maxx, maxy)
        
        logger.info(f"Parsing Microsoft partition {self._get_filename_from_url(url)} within bounds {batch_bounds}...")
        
        with gzip.open(cache_path, mode='rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    feat = json.loads(line)
                    geom = shape(feat.get("geometry", {}))
                    
                    if not geom.is_valid or geom.area == 0:
                        continue
                        
                    # Fast bounding box check first
                    g_minx, g_miny, g_maxx, g_maxy = geom.bounds
                    if g_minx > maxx or g_maxx < minx or g_miny > maxy or g_maxy < miny:
                        continue
                        
                    # Exact intersection
                    if not geom.intersects(batch_poly):
                        continue
                        
                    props = feat.get("properties", {})
                    
                    b = Building(
                        building_id=f"m_{url[-10:]}_{len(buildings)}",
                        geometry=geom,
                        centroid=geom.centroid,
                        source=self.name,
                        source_identifier=props.get("id"),
                        confidence=props.get("confidence"),
                        attributes=props
                    )
                    buildings.append(b)
                    geometries.append(geom)
                except Exception as e:
                    pass

        if not buildings:
            logger.info(f"No buildings found in Microsoft partition {self._get_filename_from_url(url)} within batch bounds.")
            return None, []
            
        logger.info(f"Building STRtree for {len(buildings)} Microsoft buildings in partition.")
        tree = STRtree(geometries)
        return tree, buildings
