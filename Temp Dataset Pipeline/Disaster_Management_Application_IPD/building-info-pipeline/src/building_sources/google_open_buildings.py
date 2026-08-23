import os
import gzip
import csv
import requests
from requests.exceptions import RequestException
from typing import Dict, Optional, Tuple, List
from shapely import wkt
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import s2sphere

from src.models.output import Building
from src.utils.logger import logger

class GoogleOpenBuildingsBatchSource:
    def __init__(self, cache_dir: str = "cache/google"):
        self.name = "GoogleOpenBuildings"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def resolve_partition(self, lat: float, lon: float) -> str:
        """Returns the S2 Level 4 cell token for the given coordinate."""
        p = s2sphere.LatLng.from_degrees(lat, lon)
        cell = s2sphere.CellId.from_lat_lng(p).parent(4)
        return cell.to_token()
        
    def partition_is_cached(self, cell_token: str) -> bool:
        cache_path = os.path.join(self.cache_dir, f"{cell_token}_buildings.csv.gz")
        return os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
        
    def download_partition(self, cell_token: str) -> bool:
        """Downloads the partition if missing. Returns True if available, False if 404/error."""
        if self.partition_is_cached(cell_token):
            return True
            
        url = f"https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/{cell_token}_buildings.csv.gz"
        cache_path = os.path.join(self.cache_dir, f"{cell_token}_buildings.csv.gz")
        tmp_path = f"{cache_path}.tmp"
        
        logger.info(f"Downloading Google partition {cell_token} from {url}")
        try:
            resp = requests.get(url, stream=True, timeout=(5, 30))
            if resp.status_code == 404:
                logger.warning(f"Google S2 cell {cell_token} has no data (404).")
                return False
            resp.raise_for_status()
            
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        
            os.rename(tmp_path, cache_path)
            return True
            
        except RequestException as e:
            logger.error(f"Network error downloading Google partition {cell_token}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    def parse_and_index(self, cell_token: str, batch_bounds: Tuple[float, float, float, float]) -> Tuple[Optional[STRtree], List[Building]]:
        """
        Parses the cached partition, keeping only buildings that intersect the `batch_bounds`.
        Returns an STRtree and a list of valid Building objects (where index in tree == index in list).
        """
        cache_path = os.path.join(self.cache_dir, f"{cell_token}_buildings.csv.gz")
        if not self.partition_is_cached(cell_token):
            return None, []
            
        buildings = []
        geometries = []
        minx, miny, maxx, maxy = batch_bounds
        
        logger.info(f"Parsing Google partition {cell_token} within bounds {batch_bounds}...")
        
        with gzip.open(cache_path, mode='rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Stage 1: Fast bounding box check based on coordinates if available
                # GOB has latitude/longitude of centroid. If it's way outside our batch bounds, skip it
                try:
                    c_lat = float(row["latitude"])
                    c_lon = float(row["longitude"])
                    # Provide a small buffer (e.g., 0.05 deg ~ 5km) for the bounding box check to account for large buildings
                    if c_lat < miny - 0.05 or c_lat > maxy + 0.05 or c_lon < minx - 0.05 or c_lon > maxx + 0.05:
                        continue
                except (KeyError, ValueError):
                    pass # Fallback to full wkt parsing if lat/lon missing or malformed
                    
                wkt_geom = row.get("geometry")
                if not wkt_geom:
                    continue
                    
                try:
                    poly = wkt.loads(wkt_geom)
                    if not poly.is_valid or poly.area == 0:
                        continue
                        
                    # Stage 2 bbox intersection via Shapely
                    if not poly.intersects(Polygon.from_bounds(minx, miny, maxx, maxy)):
                        continue
                        
                    confidence = float(row.get("confidence", 0.0))
                    
                    b = Building(
                        building_id=f"g_{cell_token}_{len(buildings)}",
                        geometry=poly,
                        centroid=poly.centroid,
                        source=self.name,
                        source_identifier=cell_token,
                        confidence=confidence,
                        attributes=row
                    )
                    buildings.append(b)
                    geometries.append(poly)
                except Exception as e:
                    pass

        if not buildings:
            logger.info(f"No buildings found in Google partition {cell_token} within batch bounds.")
            return None, []
            
        logger.info(f"Building STRtree for {len(buildings)} Google buildings in partition {cell_token}.")
        tree = STRtree(geometries)
        return tree, buildings
