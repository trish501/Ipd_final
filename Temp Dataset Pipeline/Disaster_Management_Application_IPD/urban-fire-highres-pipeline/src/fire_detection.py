import numpy as np
import logging
from dataclasses import dataclass
from src.fire_features import FireFeatures

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FireCandidateConfig:
    """
    Configuration for the Phase 3 baseline fire-candidate detector.
    Thresholds must be explicitly set and are kept in physical reflectance units.
    """
    swir2_abs_thresh: float
    swir_ratio_thresh: float
    swir_red_ratio_thresh: float
    b04_bright_reject_thresh: float
    retained_features: tuple[str, ...]

@dataclass
class FireDetectionResult:
    candidate_mask: np.ndarray
    valid_mask: np.ndarray
    b12_absolute_mask: np.ndarray
    b12_b11_ratio_mask: np.ndarray
    b12_b4_ratio_mask: np.ndarray
    b04_brightness_rejection_mask: np.ndarray
    diagnostics: dict
    config: FireCandidateConfig

def detect_fire_candidate(
    features: FireFeatures,
    config: FireCandidateConfig,
) -> FireDetectionResult:
    """
    Baseline pixel-level fire-candidate detector.
    """
    
    # 1. Require every input array to have the same shape
    shape = features.b12.shape
    if not (
        features.b11.shape == shape and
        features.b04.shape == shape and
        features.swir_ratio.shape == shape and
        features.swir_red_ratio.shape == shape and
        features.valid_mask.shape == shape
    ):
        raise ValueError("Mismatched feature shapes in input.")

    # 2. Check that all features are finite before applying thresholds
    # We will compute a finite mask
    finite_mask = (
        np.isfinite(features.b12) & 
        np.isfinite(features.b11) & 
        np.isfinite(features.b04) & 
        np.isfinite(features.swir_ratio) & 
        np.isfinite(features.swir_red_ratio)
    )

    # 3. Use the exact logic, with clear intermediate variables
    ρ12 = features.b12
    ρ11 = features.b11
    ρ4  = features.b04

    R12_11 = features.swir_ratio
    R12_4  = features.swir_red_ratio

    # Force invalid/non-finite pixels to false by using bitwise AND with finite_mask
    with np.errstate(invalid='ignore'):
        C1 = (ρ12 >= config.swir2_abs_thresh) & finite_mask
        C2 = (R12_11 >= config.swir_ratio_thresh) & finite_mask
        C3 = (R12_4 >= config.swir_red_ratio_thresh) & finite_mask
        C4 = (ρ4 < config.b04_bright_reject_thresh) & finite_mask

    V = features.valid_mask & finite_mask

    candidate_mask = V & C1 & C2 & C3 & C4

    # 4. Create Diagnostics
    diagnostics = {
        "total_pixels": int(features.valid_mask.size),
        "valid_pixels": int(np.sum(V)),
        "b12_criterion_pixels": int(np.sum(C1)),
        "b12_b11_criterion_pixels": int(np.sum(C2)),
        "b12_b4_criterion_pixels": int(np.sum(C3)),
        "b4_brightness_rejection_pixels": int(np.sum(C4)),
        "final_candidate_pixels": int(np.sum(candidate_mask)),
        "threshold_values": {
            "swir2_abs_thresh": config.swir2_abs_thresh,
            "swir_ratio_thresh": config.swir_ratio_thresh,
            "swir_red_ratio_thresh": config.swir_red_ratio_thresh,
            "b04_bright_reject_thresh": config.b04_bright_reject_thresh
        },
        "feature_names_actually_used": ["b12", "b11", "b04", "swir_ratio", "swir_red_ratio", "valid_mask"],
        "feature_names_explicitly_retained_but_unused": list(config.retained_features),
        "baseline_detector_uses_b8": False,
        "baseline_detector_uses_ndvi": False,
        "detector_output_semantics": "spectral fire candidate mask",
        "baseline_bands_used": ["B04", "B11", "B12"],
        "experimental_features_excluded": ["B08", "NDVI"],
        "scientific_limitations": [
            "FIRMS is a spatial/temporal cue, not a pixel-level label.",
            "Candidate pixels are not confirmed active fire.",
            "No standalone cloud mask is applied in Phase 3.",
            "Thresholds require real-event calibration."
        ]
    }

    return FireDetectionResult(
        candidate_mask=candidate_mask,
        valid_mask=V,
        b12_absolute_mask=C1,
        b12_b11_ratio_mask=C2,
        b12_b4_ratio_mask=C3,
        b04_brightness_rejection_mask=C4,
        diagnostics=diagnostics,
        config=config
    )

def compute_candidate_bounding_box(candidate_mask: np.ndarray, transform_20m):
    """
    Computes the geographic and pixel bounding box of the detected fire candidate.
    
    Args:
        candidate_mask: Boolean mask at 20m resolution.
        transform_20m: affine.Affine raster transform of the 20m grid.
        
    Returns:
        dict: {'geographic': {...}, 'pixel': {...}} or None
    """
    fire_indices = np.argwhere(candidate_mask)
    if len(fire_indices) == 0:
        return None
        
    min_row_20m, max_row_20m = fire_indices[:, 0].min(), fire_indices[:, 0].max()
    min_col_20m, max_col_20m = fire_indices[:, 1].min(), fire_indices[:, 1].max()
    
    # Compute geographic coordinates using the 20m transform.
    tl_lon, tl_lat = transform_20m * (min_col_20m, min_row_20m)
    br_lon, br_lat = transform_20m * (max_col_20m + 1, max_row_20m + 1.0)
    
    min_lon = min(tl_lon, br_lon)
    max_lon = max(tl_lon, br_lon)
    min_lat = min(tl_lat, br_lat)
    max_lat = max(tl_lat, br_lat)
    
    return {
        'geographic': {
            'min_lon': float(min_lon),
            'max_lon': float(max_lon),
            'min_lat': float(min_lat),
            'max_lat': float(max_lat)
        },
        'pixel': {
            'min_col': int(min_col_20m),
            'max_col': int(max_col_20m),
            'min_row': int(min_row_20m),
            'max_row': int(max_row_20m)
        }
    }
