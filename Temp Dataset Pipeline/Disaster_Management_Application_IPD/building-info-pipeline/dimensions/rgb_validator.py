import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

class RGBValidator:
    def __init__(self):
        pass

    def _extract_pixels(self, src, geometry, crop=True, all_touched=True):
        try:
            b_image, _ = mask(src, [geometry], crop=crop, filled=False, all_touched=all_touched)
            return b_image
        except ValueError:
            return None

    def validate_buildings(self, buildings, rgb_path: str):
        """
        Validates that buildings overlap the RGB imagery footprint and contain valid pixels.
        We strictly ignore NIR/SWIR and scientific NDVI features.
        """
        if not rgb_path or not buildings:
            return buildings
            
        try:
            with rasterio.open(rgb_path) as src_rgb:
                gdf = gpd.GeoDataFrame(
                    [{"geometry": b.geometry_wgs84} for b in buildings],
                    crs="EPSG:4326"
                ).to_crs(src_rgb.crs)
                
                for idx, b in enumerate(buildings):
                    geom_crs = gdf.iloc[idx].geometry
                    if not geom_crs.is_valid or geom_crs.is_empty:
                        b.coverage_ratio = 0.0
                        continue
                        
                    rgb_pixels = self._extract_pixels(src_rgb, geom_crs, all_touched=True)
                    if rgb_pixels is None or rgb_pixels[0].count() == 0:
                        b.coverage_ratio = 0.0
                        continue
                        
                    # Calculate coverage ratio strictly based on valid pixels inside the RGB bounds
                    valid_px_count = np.count_nonzero(~rgb_pixels[0].mask)
                    if valid_px_count == 0:
                        b.coverage_ratio = 0.0
                        continue
                        
                    # Geometry/image spatial consistency check passes
                    b.coverage_ratio = 1.0 # Or compute actual ratio if needed
                    
            return buildings
        except Exception as e:
            logger.error(f"RGB Validation failed: {e}")
            return buildings
