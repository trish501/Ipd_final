import os
import logging
import cv2
import numpy as np
import rasterio
from typing import List, Tuple
from shapely.geometry import Polygon

from dimensions.aoi import AOI
from dimensions.models import BuildingSearchResult, Building

logger = logging.getLogger(__name__)

class CVSegmentationSource:
    """
    Dynamically extracts building footprints using OpenCV (Classical CV logic).
    """
    
    @property
    def name(self) -> str:
        return "CVSegmentation"

    def get_buildings_from_image(self, rgb_image_path: str, bounds_wgs84: Tuple[float, float, float, float]) -> List[Building]:
        if not rgb_image_path or not os.path.exists(rgb_image_path):
            logger.warning(f"CVSegmentation requires a valid rgb_image_path, got: {rgb_image_path}")
            return []
            
        logger.info("Running classical CV algorithm to dynamically detect buildings...")
        
        # Open the image using rasterio to get the spatial transform
        with rasterio.open(rgb_image_path) as src:
            transform = src.transform
            crs = src.crs
            
        # Load the image using OpenCV
        img = cv2.imread(rgb_image_path)
        if img is None:
            logger.error("CV could not read the rgb image!")
            return []
            
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Bilateral Filter for noise reduction while preserving edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 3. Edge Detection (Canny)
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Morphological Closing to fill gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 5. Find Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buildings = []
        import pyproj
        
        # Set up transformers
        # Rasterio transform maps pixel coords to the image's native CRS (usually Web Mercator EPSG:3857)
        # We need to project that to WGS84 (EPSG:4326)
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        for i, contour in enumerate(contours):
            # Geometric Filtering
            area = cv2.contourArea(contour)
            # Filter out tiny noise and massive background shapes
            if area < 50 or area > 50000:
                continue
                
            # Filter by rectangularity or compactness
            peri = cv2.arcLength(contour, True)
            if peri == 0:
                continue
                
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity < 0.1:
                # Highly irregular or long/thin (likely a road or noise)
                continue
                
            # Convert contour points to WGS84 geographic coordinates
            geo_coords = []
            for pt in contour:
                px, py = pt[0]
                # Convert pixel to native spatial coordinate
                native_x, native_y = transform * (px, py)
                # Convert to WGS84
                lon, lat = transformer.transform(native_x, native_y)
                geo_coords.append((lon, lat))
                
            if len(geo_coords) >= 3:
                poly = Polygon(geo_coords)
                if poly.is_valid and not poly.is_empty:
                    bldg = Building(
                        building_id=f"CV_{i}",
                        geometry=poly,
                        centroid={"latitude": poly.centroid.y, "longitude": poly.centroid.x},
                        source=self.name
                    )
                    buildings.append(bldg)
                    
        logger.info(f"CVSegmentation detected {len(buildings)} buildings.")
        return buildings
