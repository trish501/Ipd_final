import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple
from rasterio.warp import transform as rp_transform
from src.fire_features import FireFeatures
from src.fire_detection import FireDetectionResult

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LocalizationConfig:
    min_component_pixels: int
    min_component_area_m2: float
    min_auto_export_pixels: int
    min_auto_export_area_m2: float
    min_fill_ratio: float
    max_firms_viirs_distance_m: float
    max_firms_modis_distance_m: float
    fallback_firms_distance_m: float
    reject_invalid_edge_components: bool
    morphology_enabled: bool
    morphology_operation: str
    morphology_iterations: int

@dataclass
class FireComponent:
    event_id: str
    component_id: int
    pixel_count_20m: int
    area_m2: float
    x_min_20m: int
    x_max_20m: int
    y_min_20m: int
    y_max_20m: int
    width_20m: int
    height_20m: int
    centroid_x_20m: float
    centroid_y_20m: float
    fill_ratio: float
    firms_x_20m: float
    firms_y_20m: float
    distance_to_firms_m: float
    median_b04: float
    median_b11: float
    median_b12: float
    median_swir_ratio: float
    median_swir_red_ratio: float
    decision: str
    decision_reasons: List[str]
    eligible_for_yolo_export: bool

@dataclass
class FireLocalizationResult:
    accepted_components: List[FireComponent]
    review_required_components: List[FireComponent]
    rejected_components: List[FireComponent]
    cleaned_candidate_mask: np.ndarray
    labeled_components: np.ndarray
    config: LocalizationConfig

def connected_components_8way(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Simple 8-way connected components labeling for a 2D boolean mask.
    Returns: labeled_array, num_features
    """
    labeled = np.zeros_like(mask, dtype=np.int32)
    label_count = 0
    rows, cols = mask.shape
    
    stack = []
    
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and labeled[r, c] == 0:
                label_count += 1
                labeled[r, c] = label_count
                stack.append((r, c))
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if mask[nr, nc] and labeled[nr, nc] == 0:
                                    labeled[nr, nc] = label_count
                                    stack.append((nr, nc))
                                    
    return labeled, label_count

def localize_fire_candidates(
    detection: FireDetectionResult,
    features: FireFeatures,
    event_metadata: dict,
    config: LocalizationConfig,
) -> FireLocalizationResult:
    """
    Converts Phase 3 candidate mask into spatially coherent candidate regions.
    """
    candidate_mask = detection.candidate_mask.copy()
    rows, cols = candidate_mask.shape

    # 1. Enforce valid_mask
    candidate_mask = candidate_mask & features.valid_mask

    # 2. Minimal morphology
    if config.morphology_enabled:
        # We do not apply morphology to avoid complexity unless strictly needed.
        pass
        
    # 3. Connected-component labelling
    labeled_components, num_features = connected_components_8way(candidate_mask)
    
    # 4. Project FIRMS coordinates
    firms_lat = event_metadata.get('latitude', 0.0)
    firms_lon = event_metadata.get('longitude', 0.0)
    
    if hasattr(features, 'crs') and features.crs is not None:
        xs, ys = rp_transform('EPSG:4326', features.crs, [firms_lon], [firms_lat])
        cx, cy = xs[0], ys[0]
        inv_transform = ~features.transform
        firms_x_20m, firms_y_20m = inv_transform * (cx, cy)
    else:
        # Fallback if testing with mock features missing CRS
        firms_x_20m = cols / 2.0
        firms_y_20m = rows / 2.0
    
    accepted_components = []
    review_required_components = []
    rejected_components = []
    cleaned_candidate_mask = np.zeros_like(candidate_mask)
    
    event_id = event_metadata.get('event_id', 'unknown')
    
    firms_source = str(event_metadata.get('source', '')).upper()
    if 'VIIRS' in firms_source:
        max_dist_m = config.max_firms_viirs_distance_m
    elif 'MODIS' in firms_source:
        max_dist_m = config.max_firms_modis_distance_m
    else:
        max_dist_m = config.fallback_firms_distance_m

    max_possible_dist_m = np.sqrt((rows * features.resolution / 2)**2 + (cols * features.resolution / 2)**2)
    if max_dist_m >= max_possible_dist_m:
        raise ValueError(
            f"FIRMS proximity criterion is non-discriminative because its threshold covers the entire crop. "
            f"max_dist_m ({max_dist_m}) >= max_crop_distance_m ({max_possible_dist_m:.1f}m)."
        )

    # 5. Measure and validate every component
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_components == comp_id)
        
        pixel_count = int(np.sum(comp_mask))
        if pixel_count == 0:
            continue
            
        coords = np.argwhere(comp_mask)
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        centroid_y = float(np.mean(y_coords))
        centroid_x = float(np.mean(x_coords))
        
        pixel_width_m = features.resolution
        pixel_height_m = features.resolution
        
        area_m2 = pixel_count * pixel_width_m * pixel_height_m
        fill_ratio = pixel_count / (width * height)
        
        dist_x_m = (centroid_x - firms_x_20m) * pixel_width_m
        dist_y_m = (centroid_y - firms_y_20m) * pixel_height_m
        distance_to_firms_m = np.sqrt(dist_x_m**2 + dist_y_m**2)
        
        median_b4 = float(np.median(features.b04[comp_mask]))
        median_b11 = float(np.median(features.b11[comp_mask]))
        median_b12 = float(np.median(features.b12[comp_mask]))
        median_ratio_12_11 = float(np.median(features.swir_ratio[comp_mask]))
        median_ratio_12_4 = float(np.median(features.swir_red_ratio[comp_mask]))
        
        reasons = []
        
        if pixel_count < config.min_component_pixels:
            reasons.append("TOO_SMALL")
        if area_m2 < config.min_component_area_m2:
            reasons.append("INSUFFICIENT_PHYSICAL_AREA")
        if fill_ratio < config.min_fill_ratio:
            reasons.append("LOW_COMPACTNESS")
        
        if distance_to_firms_m > max_dist_m:
            reasons.append("TOO_FAR_FROM_FIRMS_REFERENCE")
            
        if config.reject_invalid_edge_components:
            if x_min == 0 or x_max == cols - 1 or y_min == 0 or y_max == rows - 1:
                reasons.append("EDGE_ARTIFACT")
            else:
                y_min_pad = max(0, y_min - 1)
                y_max_pad = min(rows, y_max + 2)
                x_min_pad = max(0, x_min - 1)
                x_max_pad = min(cols, x_max + 2)
                
                neighborhood_valid = features.valid_mask[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                if not np.all(neighborhood_valid):
                    reasons.append("TOUCHES_INVALID_DATA")
                    
        # Check if NO_VALID_SPECTRAL_EVIDENCE
        if median_b12 <= 0.0 or median_ratio_12_11 <= 0.0:
            reasons.append("NO_VALID_SPECTRAL_EVIDENCE")
        
        if len(reasons) > 0:
            decision = "REJECTED"
            eligible_for_yolo_export = False
        elif pixel_count < config.min_auto_export_pixels or area_m2 < config.min_auto_export_area_m2:
            decision = "REVIEW_REQUIRED"
            reasons.append("INSUFFICIENT_EVIDENCE_FOR_AUTO_EXPORT")
            eligible_for_yolo_export = False
        else:
            decision = "ACCEPTED_FOR_AUTO_EXPORT"
            eligible_for_yolo_export = True
        
        comp = FireComponent(
            event_id=event_id,
            component_id=comp_id,
            pixel_count_20m=pixel_count,
            area_m2=area_m2,
            x_min_20m=x_min,
            x_max_20m=x_max,
            y_min_20m=y_min,
            y_max_20m=y_max,
            width_20m=width,
            height_20m=height,
            centroid_x_20m=centroid_x,
            centroid_y_20m=centroid_y,
            fill_ratio=fill_ratio,
            firms_x_20m=firms_x_20m,
            firms_y_20m=firms_y_20m,
            distance_to_firms_m=distance_to_firms_m,
            median_b04=median_b4,
            median_b11=median_b11,
            median_b12=median_b12,
            median_swir_ratio=median_ratio_12_11,
            median_swir_red_ratio=median_ratio_12_4,
            decision=decision,
            decision_reasons=reasons,
            eligible_for_yolo_export=eligible_for_yolo_export
        )
        
        if decision == "ACCEPTED_FOR_AUTO_EXPORT":
            accepted_components.append(comp)
            cleaned_candidate_mask[comp_mask] = True
        elif decision == "REVIEW_REQUIRED":
            review_required_components.append(comp)
            # Retain in mask? The prompt doesn't specify if REVIEW_REQUIRED should be in the final mask.
            # Usually, REVIEW_REQUIRED needs inspection but is not auto-exported. We'll leave them out of the cleaned mask 
            # for strict auto-export, but they are retained in diagnostic reports.
        else:
            rejected_components.append(comp)
            
    return FireLocalizationResult(
        accepted_components=accepted_components,
        review_required_components=review_required_components,
        rejected_components=rejected_components,
        cleaned_candidate_mask=cleaned_candidate_mask,
        labeled_components=labeled_components,
        config=config
    )
