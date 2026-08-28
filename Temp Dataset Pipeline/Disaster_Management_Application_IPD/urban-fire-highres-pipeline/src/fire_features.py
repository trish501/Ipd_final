import numpy as np
from dataclasses import dataclass
import rasterio
from typing import Tuple, Dict, Any
from src.s2_preprocessing import MultispectralData

@dataclass
class FireFeatures:
    """
    Scientifically meaningful spectral features for active-fire candidate detection.
    These features are candidate indicators, not automatic proof of fire.
    """
    # Base bands (Surface Reflectance)
    b12: np.ndarray
    b11: np.ndarray
    b04: np.ndarray
    b08: np.ndarray
    b08a: np.ndarray

    
    # Derived Ratios and Indices
    swir_ratio: np.ndarray         # B12 / B11
    swir_red_ratio: np.ndarray     # B12 / B04
    swir_red_diff: np.ndarray      # B12 - B04
    norm_swir_diff: np.ndarray     # (B12 - B11) / (B12 + B11)
    red_swir_contrast: np.ndarray  # 0.734 * B12 - B04
    ndvi: np.ndarray               # (B08 - B04) / (B08 + B04)
    b12_b8a_ratio: np.ndarray    # B12 / B8A
    b11_b8a_ratio: np.ndarray    # B11 / B8A
    swir21_b8a_contrast: np.ndarray # (B12 - B11) / B8A
    ndvi_b8a: np.ndarray           # (B8A - B04) / (B8A + B04)
    nbr: np.ndarray                # (B8A - B12) / (B8A + B12)

    
    # Geospatial and Validity context preserved from preprocessing
    valid_mask: np.ndarray
    cloud_mask: np.ndarray
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    resolution: float
    bounds: Tuple[float, float, float, float]
    metadata: Dict[str, Any]

class FeatureGenerator:
    """
    Generates spectral fire features using robust vectorized numerical operations.
    Guarantees no NaN or Inf values by safely handling division-by-zero scenarios.
    """
    def __init__(self):
        pass
        
    def _safe_divide(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        """
        Safely divides two arrays. Returns 0.0 where the denominator is 0.0 to prevent NaNs/Infs.
        """
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=(den != 0))

    def generate_features(self, ms_data: MultispectralData) -> FireFeatures:
        b12 = ms_data.b12
        b11 = ms_data.b11
        b04 = ms_data.b04
        b08 = ms_data.b08
        
        # 5. SWIR ratio: B12 / B11
        swir_ratio = self._safe_divide(b12, b11)
        
        # 6. SWIR-Red ratio: B12 / B4
        swir_red_ratio = self._safe_divide(b12, b04)
        
        # 7. SWIR-Red difference: B12 - B4
        swir_red_diff = b12 - b04
        
        # 8. Normalized SWIR difference: (B12 - B11) / (B12 + B11)
        swir_sum = b12 + b11
        norm_swir_diff = self._safe_divide(b12 - b11, swir_sum)
        
        # 9. Red-SWIR contrast: 0.734 * B12 - B4
        red_swir_contrast = (0.734 * b12) - b04
        
        # 10. NDVI: (B8 - B4) / (B8 + B4)
        nir_red_sum = b08 + b04
        ndvi = self._safe_divide(b08 - b04, nir_red_sum)
        
        # 11. B8A Auxiliary Features
        b08a = ms_data.b08a
        b12_b8a_ratio = self._safe_divide(b12, b08a)
        b11_b8a_ratio = self._safe_divide(b11, b08a)
        swir21_b8a_contrast = self._safe_divide(b12 - b11, b08a)
        
        nir_narrow_red_sum = b08a + b04
        ndvi_b8a = self._safe_divide(b08a - b04, nir_narrow_red_sum)
        
        # 12. Normalized Burn Ratio: (B8A - B12) / (B8A + B12)
        nbr_sum = b08a + b12
        nbr = self._safe_divide(b08a - b12, nbr_sum)
        
        return FireFeatures(
            b12=b12,
            b11=b11,
            b04=b04,
            b08=b08,
            b08a=b08a,
            swir_ratio=swir_ratio,
            swir_red_ratio=swir_red_ratio,
            swir_red_diff=swir_red_diff,
            norm_swir_diff=norm_swir_diff,
            red_swir_contrast=red_swir_contrast,
            ndvi=ndvi,
            b12_b8a_ratio=b12_b8a_ratio,
            b11_b8a_ratio=b11_b8a_ratio,
            swir21_b8a_contrast=swir21_b8a_contrast,
            ndvi_b8a=ndvi_b8a,
            nbr=nbr,
            valid_mask=ms_data.valid_mask,
            cloud_mask=ms_data.cloud_mask,
            transform=ms_data.transform,
            crs=ms_data.crs,
            resolution=ms_data.resolution,
            bounds=ms_data.bounds,
            metadata=ms_data.metadata
        )
