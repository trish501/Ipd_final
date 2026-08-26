import logging
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared_models.canonical_aoi import CanonicalAOI

logger = logging.getLogger(__name__)

@dataclass
class MultispectralData:
    b02: np.ndarray  # Blue, scaled to surface reflectance
    b03: np.ndarray  # Green, scaled to surface reflectance
    b04: np.ndarray  # Red, scaled to surface reflectance
    b08: np.ndarray  # Broad NIR, scaled to surface reflectance
    b11: np.ndarray  # SWIR1, scaled to surface reflectance
    b12: np.ndarray  # SWIR2, scaled to surface reflectance
    valid_mask: np.ndarray  # Boolean mask of valid data (True = valid)
    cloud_mask: np.ndarray  # Boolean mask of clouds/cirrus/shadows (True = cloud)
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    resolution: float  # e.g., 20.0
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    metadata: Dict[str, Any]  # logs like scene_id, processing_level, masked_percentage

class S2Preprocessor:
    def __init__(self):
        pass

    def get_asset_href(self, item, asset_name: str) -> str:
        return item.assets[asset_name].href

    def process(self, item, lat: float, lon: float, crop_km: float = 2.0) -> MultispectralData:
        """
        Loads required Sentinel-2 bands, applies physical scaling, cloud masking, 
        and spatial alignment on a common 20m grid.
        """
        processing_baseline = item.properties.get("s2:processing_baseline", "00.00")
        scale_factor = 0.0001
        offset = 0.0
        
        # Baseline >= 04.00 introduced a -1000 DN offset
        if float(processing_baseline) >= 4.0:
            offset = -1000.0

        bands_needed = {
            "B02": 10.0, # Blue
            "B03": 10.0, # Green
            "B04": 10.0, # Red
            "B08": 10.0, # Broad NIR
            "B11": 20.0, # SWIR1
            "B12": 20.0, # SWIR2
            "SCL": 20.0  # Scene Classification Layer
        }
        
        band_data = {}
        common_res = 20.0
        
        target_size = int((crop_km * 1000) / common_res)
        half_size = (crop_km * 1000) / 2.0
        
        # Get bounds using a reference band to establish CRS
        ref_href = self.get_asset_href(item, "B12")
        with rasterio.open(ref_href) as src:
            crs = src.crs
            xs, ys = transform('EPSG:4326', crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
            left, bottom, right, top = cx - half_size, cy - half_size, cx + half_size, cy + half_size
            common_transform = rasterio.transform.from_bounds(left, bottom, right, top, target_size, target_size)
            common_bounds = (left, bottom, right, top)
            
            canonical_aoi = CanonicalAOI(
                center_lat=lat,
                center_lon=lon,
                width_m=crop_km * 1000,
                height_m=crop_km * 1000,
                rotation_angle_deg=0.0,
                crs_epsg=str(crs),
                min_x=left,
                max_x=right,
                min_y=bottom,
                max_y=top
            )

        for band, native_res in bands_needed.items():
            href = self.get_asset_href(item, band)
            with rasterio.open(href) as src:
                window = from_bounds(left, bottom, right, top, src.transform)
                
                # If natively 10m (B04, B08), we downsample to 20m via bilinear to match IFOV correctly.
                # Natively 20m bands are read using nearest to preserve exact native values.
                resampling_method = rasterio.enums.Resampling.bilinear if native_res < common_res and band != "SCL" else rasterio.enums.Resampling.nearest
                
                arr = src.read(
                    window=window,
                    out_shape=(1, target_size, target_size),
                    resampling=resampling_method,
                    boundless=True,
                    fill_value=0
                )
                
                raw_data = arr[0].astype(np.float32)
                
                if band != "SCL":
                    # Apply scaling to get physical surface reflectance
                    valid = raw_data > 0
                    scaled_data = np.zeros_like(raw_data)
                    scaled_data[valid] = (raw_data[valid] + offset) * scale_factor
                    band_data[band] = scaled_data
                else:
                    band_data[band] = raw_data

        # Valid Mask (Pixels where all required bands are valid/non-zero)
        valid_mask = (band_data["B02"] > 0) & (band_data["B03"] > 0) & (band_data["B04"] > 0) & (band_data["B08"] > 0) & (band_data["B11"] > 0) & (band_data["B12"] > 0)
        
        # Cloud Mask from SCL
        # Classes: 3=Cloud Shadows, 8=Cloud Medium Prob, 9=Cloud High Prob, 10=Thin Cirrus
        scl = band_data["SCL"]
        cloud_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
        
        masked_percentage = (np.sum(cloud_mask) / cloud_mask.size) * 100.0
        
        metadata_logs = {
            "scene_id": item.id,
            "acquisition_datetime": item.datetime.isoformat() if hasattr(item.datetime, 'isoformat') else str(item.datetime),
            "processing_level": "Level-2A",
            "bands_loaded": list(bands_needed.keys()),
            "original_resolutions": bands_needed,
            "analysis_resolution_m": common_res,
            "crs": str(crs),
            "dimensions": (target_size, target_size),
            "masked_pixel_percentage": float(masked_percentage),
            "canonical_aoi": canonical_aoi.to_dict()
        }

        return MultispectralData(
            b02=band_data["B02"],
            b03=band_data["B03"],
            b04=band_data["B04"],
            b08=band_data["B08"],
            b11=band_data["B11"],
            b12=band_data["B12"],
            valid_mask=valid_mask,
            cloud_mask=cloud_mask,
            transform=common_transform,
            crs=crs,
            resolution=common_res,
            bounds=common_bounds,
            metadata=metadata_logs
        )
