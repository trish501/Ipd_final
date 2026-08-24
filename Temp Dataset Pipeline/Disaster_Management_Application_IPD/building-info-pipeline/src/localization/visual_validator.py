import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from typing import List

from src.models.output import BuildingInfo, ImageryResult
from src.models.enums import VisualStatus
from src.utils.logger import logger

class VisualValidator:
    def __init__(self):
        pass

    def _extract_pixels(self, src, geometry, crop=True, all_touched=True):
        try:
            b_image, _ = mask(src, [geometry], crop=crop, filled=False, all_touched=all_touched)
            return b_image
        except ValueError:
            return None

    def validate_batch(self, candidates: List[BuildingInfo], imagery: ImageryResult) -> None:
        """
        Validates candidates against RGB, NIR, and SWIR imagery using a precise multi-stage evidence model.
        """
        if not imagery.rgb_image_path or not os.path.exists(imagery.rgb_image_path):
            for c in candidates:
                c.visual_status = VisualStatus.REJECTED_NON_BUILDING
                c.rejection_reason = "No RGB image available for verification"
            return

        try:
            with rasterio.open(imagery.rgb_image_path) as src_rgb:
                src_nir = rasterio.open(imagery.nir_image_path) if imagery.nir_image_path and os.path.exists(imagery.nir_image_path) else None
                src_swir = rasterio.open(imagery.swir_image_path) if imagery.swir_image_path and os.path.exists(imagery.swir_image_path) else None

                gdf = gpd.GeoDataFrame(
                    [{"geometry": c.original_geometry_wgs84} for c in candidates],
                    crs="EPSG:4326"
                ).to_crs(src_rgb.crs)
                
                for idx, candidate in enumerate(candidates):
                    # Initial default state
                    candidate.visual_status = VisualStatus.UNRESOLVED
                    
                    native_gsd = imagery.native_resolution_m if imagery.native_resolution_m and imagery.native_resolution_m > 0 else 10.0
                    pixels_short_axis = candidate.short_axis_meters / native_gsd
                    
                    
                    candidate.attributes["native_gsd_m"] = native_gsd
                    candidate.attributes["estimated_native_pixels_short_axis"] = pixels_short_axis
                    
                    # 1. GEOMETRIC CANDIDATE VALIDATION
                    geom_crs = gdf.iloc[idx].geometry
                    if not geom_crs.is_valid or geom_crs.is_empty:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = "Invalid footprint geometry in image projection"
                        continue
                        
                    if candidate.coverage_ratio < 0.1:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = "Candidate is outside image footprint"
                        continue

                    # 2. PIXEL EXTRACTION
                    # Using all_touched=True ensures we gather any pixel intersected by the polygon
                    rgb_pixels = self._extract_pixels(src_rgb, geom_crs, all_touched=True)
                    if rgb_pixels is None or rgb_pixels[0].count() == 0:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = "Insufficient valid pixels extracted (count = 0)"
                        continue
                        
                    r = rgb_pixels[0].compressed().astype(np.float32)
                    g = rgb_pixels[1].compressed().astype(np.float32)
                    b = rgb_pixels[2].compressed().astype(np.float32)
                    
                    total_pixels = len(r)
                    if total_pixels == 0:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = "0 valid interior pixels"
                        continue
                        
                    candidate.attributes["interior_valid_pixels"] = total_pixels
                    
                    # 3. PIXEL SUPPORT CLASSIFICATION
                    if total_pixels >= 4:
                        candidate.pixel_support = "HIGH"
                    elif total_pixels >= 2:
                        candidate.pixel_support = "MEDIUM"
                    elif total_pixels == 1:
                        candidate.pixel_support = "LOW"
                    else:
                        candidate.pixel_support = "INSUFFICIENT"
                        
                    if candidate.pixel_support == "INSUFFICIENT":
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = "INSUFFICIENT_PIXEL_SUPPORT"
                        continue
                    
                    # Multispectral extraction
                    nir = None
                    swir = None
                    
                    if src_nir:
                        nir_pixels = self._extract_pixels(src_nir, geom_crs, all_touched=True)
                        if nir_pixels is not None and nir_pixels[0].count() > 0:
                            nir_mask = ~nir_pixels[0].mask
                            rgb_mask = ~rgb_pixels[0].mask
                            common_mask = nir_mask & rgb_mask
                            if np.any(common_mask):
                                nir = nir_pixels[0][common_mask].astype(np.float32)
                                r_com = rgb_pixels[0][common_mask].astype(np.float32)
                            else:
                                nir = nir_pixels[0].compressed().astype(np.float32)
                            
                    if src_swir:
                        geom_swir = gpd.GeoDataFrame([{"geometry": geom_crs}], crs=src_rgb.crs).to_crs(src_swir.crs).iloc[0].geometry
                        swir_pixels = self._extract_pixels(src_swir, geom_swir, all_touched=True)
                        if swir_pixels is not None and swir_pixels[0].count() > 0:
                            swir = swir_pixels[0].compressed().astype(np.float32)

                    # 4. FRACTIONAL MULTISPECTRAL EVIDENCE
                    
                    # Shadow Evidence (using a relative offset accounting for Sentinel-2 scale)
                    shadow_pixels_cnt = np.sum((r < 400) & (g < 400) & (b < 400))
                    shadow_fraction = shadow_pixels_cnt / total_pixels
                    candidate.attributes["shadow_fraction"] = float(shadow_fraction)
                    
                    # Vegetation Evidence (NDVI > 0.3)
                    vegetation_fraction = 0.0
                    if nir is not None and len(nir) == len(r_com) and len(nir) > 0:
                        ndvi_arr = (nir - r_com) / (nir + r_com + 1e-5)
                        veg_pixels = np.sum(ndvi_arr > 0.3)
                        vegetation_fraction = veg_pixels / len(nir)
                    else:
                        exg_arr = 2 * g - r - b
                        veg_pixels = np.sum((exg_arr > 300) & (g > r) & (g > b))
                        vegetation_fraction = veg_pixels / total_pixels
                        
                    candidate.attributes["vegetation_fraction"] = float(vegetation_fraction)
                    
                    # Water Evidence (MNDWI > 0)
                    water_fraction = 0.0
                    if swir is not None and len(swir) > 0:
                        med_g = np.median(g)
                        med_swir = np.median(swir)
                        mndwi = (med_g - med_swir) / (med_g + med_swir + 1e-5)
                        candidate.attributes["mndwi_median"] = float(mndwi)
                        if mndwi > 0.0:
                            water_fraction = 1.0 # Approximation
                    candidate.attributes["water_fraction"] = float(water_fraction)
                    
                    candidate.attributes["SHADOW_REJECTION"] = bool(shadow_fraction > 0.5)
                    candidate.attributes["VEGETATION_REJECTION"] = bool(vegetation_fraction > 0.5)
                    candidate.attributes["WATER_REJECTION"] = bool(water_fraction > 0.5)

                    # 5. CONTEXT ANALYSIS
                    buffer_geom = geom_crs.buffer(15.0).difference(geom_crs)
                    bg_rgb_pixels = self._extract_pixels(src_rgb, buffer_geom, all_touched=True)
                    
                    color_diff = 0.0
                    if bg_rgb_pixels is not None and bg_rgb_pixels[0].count() > 0:
                        bg_r = bg_rgb_pixels[0].compressed().astype(np.float32)
                        bg_g = bg_rgb_pixels[1].compressed().astype(np.float32)
                        bg_b = bg_rgb_pixels[2].compressed().astype(np.float32)
                        
                        med_r, med_g, med_b = np.median(r), np.median(g), np.median(b)
                        bg_med_r, bg_med_g, bg_med_b = np.median(bg_r), np.median(bg_g), np.median(bg_b)
                        
                        color_diff = np.sqrt((med_r - bg_med_r)**2 + (med_g - bg_med_g)**2 + (med_b - bg_med_b)**2)
                        candidate.attributes["boundary_contrast"] = float(color_diff)
                        
                    candidate.attributes["BACKGROUND_REJECTION"] = bool(color_diff < 150.0) # Statistically indistinguishable

                    # 6. STRUCTURAL EVIDENCE
                    # High variance might mean mixed pixels, but low variance with zero contrast is bare land.
                    variance_r = np.var(r)
                    candidate.attributes["STRUCTURAL_EVIDENCE"] = bool(color_diff >= 300.0 or variance_r > 500)

                    # 7. STRICT BUILDING EVIDENCE RULES
                    
                    # Base Rejections
                    if candidate.attributes["VEGETATION_REJECTION"]:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = f"Dominated by vegetation ({vegetation_fraction*100:.0f}%)"
                        continue
                        
                    if candidate.attributes["WATER_REJECTION"]:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = "Dominated by water"
                        continue
                        
                    if candidate.attributes["SHADOW_REJECTION"]:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = f"Dominated by shadow ({shadow_fraction*100:.0f}%)"
                        continue

                    if candidate.attributes["BACKGROUND_REJECTION"]:
                        candidate.visual_status = VisualStatus.REJECTED_NON_BUILDING
                        candidate.rejection_reason = f"No meaningful contrast with background (diff={color_diff:.0f})"
                        continue

                    # VERIFIED_VISIBLE_BUILDING CLASSIFICATION (STAGE 10)
                    has_source_agreement = (candidate.source_agreement == "BOTH")
                    
                    if candidate.pixel_support in ["HIGH", "MEDIUM"]:
                        if candidate.attributes["STRUCTURAL_EVIDENCE"] or has_source_agreement:
                            candidate.visual_status = VisualStatus.VERIFIED_VISIBLE_BUILDING
                            candidate.rejection_reason = "None"
                            candidate.attributes["decision"] = "VERIFIED_VISIBLE_BUILDING"
                        else:
                            candidate.visual_status = VisualStatus.PROBABLE_BUILDING
                            candidate.rejection_reason = "Medium/High support but lacks structural or source agreement."
                            
                    elif candidate.pixel_support == "LOW":
                        # Require unusually strong independent evidence
                        if candidate.attributes["STRUCTURAL_EVIDENCE"] and has_source_agreement:
                            candidate.visual_status = VisualStatus.VERIFIED_VISIBLE_BUILDING
                            candidate.rejection_reason = "None"
                        else:
                            candidate.visual_status = VisualStatus.UNRESOLVED
                            candidate.rejection_reason = "LOW pixel support without overwhelming structural+source evidence"
                    
                if src_nir:
                    src_nir.close()
                if src_swir:
                    src_swir.close()
                    
        except Exception as e:
            logger.error(f"Failed to run strict visual verification: {e}")
            for c in candidates:
                c.visual_status = VisualStatus.UNRESOLVED
                c.rejection_reason = f"Verification crash: {e}"
