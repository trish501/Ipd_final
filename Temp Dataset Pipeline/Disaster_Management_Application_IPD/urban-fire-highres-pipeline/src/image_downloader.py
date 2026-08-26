import os
import logging
from PIL import Image, ImageDraw
import numpy as np
import json
from datetime import datetime
import shutil

from src.filters import get_location
from src.s2_preprocessing import S2Preprocessor
from src.fire_features import FeatureGenerator
from src.fire_detection import detect_fire_candidate, compute_candidate_bounding_box, FireCandidateConfig
from src.fire_localization import localize_fire_candidates, LocalizationConfig
from dataclasses import asdict
from src.yolo_ground_truth import process_yolo_export

logger = logging.getLogger(__name__)

def write_image_metadata(txt_path, image_type, bands_text, band_order_list, item, event_meta, crs_val, res_val, preprocessor_metadata):
    if not event_meta: event_meta = {}
    
    lat = event_meta.get('latitude')
    lon = event_meta.get('longitude')
    
    location_str = "Not available, Not available, Not available"
    if lat is not None and lon is not None:
        location_str = get_location(lat, lon)
    
    parts = [p.strip() for p in location_str.split(",")]
    if len(parts) >= 3:
        city, state, country = parts[0], parts[1], parts[2]
    else:
        city, state, country = "Not available", "Not available", "Not available"

    with open(txt_path, 'w') as f:
        f.write("-" * 50 + "\n")
        f.write("IMAGE METADATA\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Image Type: {image_type}\n")
        f.write("Satellite: Sentinel-2\n")
        f.write(f"Scene ID: {item.id}\n")
        f.write(f"Processing Level: {preprocessor_metadata.get('processing_level', 'Unknown')}\n")
        f.write(f"Processing Baseline: {preprocessor_metadata.get('processing_baseline', 'Unknown')}\n\n")
        
        f.write("Bands:\n")
        for b in bands_text:
            f.write(f"{b}\n")
            
        f.write("\nBand Order:\n")
        f.write(f"{', '.join(band_order_list)}\n\n")
        
        sat_dt = item.datetime
        f.write(f"Acquisition Date: {sat_dt.strftime('%Y-%m-%d') if sat_dt else 'Not available'}\n")
        f.write(f"Acquisition Time: {sat_dt.strftime('%H:%M:%S') if sat_dt else 'Not available'}\n\n")
        
        f.write(f"Fire Acquisition Date: {event_meta.get('date', 'Not available')}\n")
        f.write(f"Fire Acquisition Time: {event_meta.get('time', 'Not available')}\n\n")
        
        f.write(f"Latitude: {lat if lat is not None else 'Not available'}\n")
        f.write(f"Longitude: {lon if lon is not None else 'Not available'}\n\n")
        
        f.write(f"Country: {country}\n")
        f.write(f"State: {state}\n")
        f.write(f"City: {city}\n\n")
        
        f.write(f"Event ID: {event_meta.get('event_id', 'Not available')}\n\n")
        
        f.write(f"Coordinate Reference System: {crs_val}\n\n")
        f.write(f"Spatial Resolution: {res_val}\n\n")
        f.write(f"Cloud/Cirrus Masked Percentage: {preprocessor_metadata.get('masked_pixel_percentage', 0.0):.2f}%\n\n")
        
        f.write("Source: NASA FIRMS\n")
        source = event_meta.get('source_file', '')
        if "VIIRS" in source:
            f.write("Fire Data Source: VIIRS\n")
        elif "MODIS" in source:
            f.write("Fire Data Source: MODIS\n")
        else:
            f.write("Fire Data Source: Not available\n")
            
        candidate_bbox = event_meta.get('detected_fire_region_bbox')
        if candidate_bbox and 'geographic' in candidate_bbox:
            geo = candidate_bbox['geographic']
            f.write("\nDetected Fire Region Bounding Box (Multispectral Anomaly):\n")
            f.write(f"Min Lon: {geo['min_lon']}\n")
            f.write(f"Max Lon: {geo['max_lon']}\n")
            f.write(f"Min Lat: {geo['min_lat']}\n")
            f.write(f"Max Lat: {geo['max_lat']}\n")
            
        f.write("\n" + "-" * 50 + "\n")

def scale_to_8bit(arr):
    """
    Scales physical surface reflectance (where 1.0 is max nominal reflection) to 0-255.
    We cap at 1.0 (some fires go above 1.0 reflectance in SWIR, so they will saturate at 255).
    """
    scaled = arr * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def download_and_crop_image(item, lat: float, lon: float, event_id: str, out_dir: str, crop_km: float = 2.0, output_size: int = 1024, event_meta: dict = None):
    """
    Downloads native Sentinel-2 bands using S2Preprocessor, generates active-fire analysis, 
    and outputs a false-color visualization.
    """
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
        
    try:
        # 1. PREPROCESSING (Phase 1)
        preprocessor = S2Preprocessor()
        ms_data = preprocessor.process(item, lat, lon, crop_km)
        
        # 2. VALIDITY GATE
        valid_percentage = (np.sum(ms_data.valid_mask) / ms_data.valid_mask.size) * 100.0
        if valid_percentage < 25.0:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": f"Crop contains insufficient valid imagery overall (< 25%). Actual: {valid_percentage:.1f}%"}

        center_y, center_x = ms_data.valid_mask.shape[0] // 2, ms_data.valid_mask.shape[1] // 2
        if not ms_data.valid_mask[center_y, center_x]:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": "Event coordinate maps to invalid/NoData Sentinel-2 raster area."}
            
        if ms_data.metadata['masked_pixel_percentage'] > 75.0:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": f"Crop is too cloudy to analyze. Cloud/cirrus masking: {ms_data.metadata['masked_pixel_percentage']:.1f}%"}

        # 2. FEATURE GENERATION (Phase 2)
        feature_generator = FeatureGenerator()
        features = feature_generator.generate_features(ms_data)
        
        # 3. MULTISPECTRAL ACTIVE FIRE DETECTION
        config = FireCandidateConfig(
            swir2_abs_thresh=0.8,
            swir_ratio_thresh=1.0,
            swir_red_ratio_thresh=1.5,
            b04_bright_reject_thresh=0.3,
            retained_features=('b08', 'swir_red_diff', 'norm_swir_diff', 'red_swir_contrast', 'ndvi')
        )
        detection_result = detect_fire_candidate(features, config)
        
        # 3.5 MULTISPECTRAL SPATIAL LOCALIZATION
        loc_config = LocalizationConfig(
            min_component_pixels=1,
            min_component_area_m2=400.0,
            min_auto_export_pixels=2,
            min_auto_export_area_m2=800.0,
            min_fill_ratio=0.05,
            max_firms_viirs_distance_m=375.0,
            max_firms_modis_distance_m=1000.0,
            fallback_firms_distance_m=1000.0,
            reject_invalid_edge_components=True,
            morphology_enabled=False,
            morphology_operation="none",
            morphology_iterations=0
        )
        if event_meta is None:
            event_meta = {}
            
        localization_result = localize_fire_candidates(detection_result, features, event_meta, loc_config)
        
        # Save diagnostics
        diagnostics_dir = os.path.join(out_dir, "diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)
        with open(os.path.join(diagnostics_dir, "phase3_fire_candidate_diagnostics.json"), "w") as f:
            json.dump(detection_result.diagnostics, f, indent=4)
            
        loc_report = {
            "event_id": event_meta.get("event_id", "unknown"),
            "firms_source": event_meta.get("source", "unknown"),
            "firms_coordinate_lon": event_meta.get("longitude"),
            "firms_coordinate_lat": event_meta.get("latitude"),
            "firms_projected_x_20m": event_meta.get("firms_x_20m"),
            "firms_projected_y_20m": event_meta.get("firms_y_20m"),
            "crop_size_rows": ms_data.b04.shape[0],
            "crop_size_cols": ms_data.b04.shape[1],
            "resolution_m": ms_data.transform.a,
            "candidate_pixel_count": int(np.sum(detection_result.candidate_mask)),
            "config": asdict(localization_result.config),
            "components_found": len(localization_result.accepted_components) + len(localization_result.review_required_components) + len(localization_result.rejected_components),
            "accepted_regions_count": len(localization_result.accepted_components),
            "review_required_count": len(localization_result.review_required_components),
            "rejected_regions_count": len(localization_result.rejected_components),
            "accepted_region_ids": [c.component_id for c in localization_result.accepted_components],
            "review_required_ids": [c.component_id for c in localization_result.review_required_components],
            "rejected_region_ids": [c.component_id for c in localization_result.rejected_components],
            "all_components": (
                [asdict(c) for c in localization_result.accepted_components] + 
                [asdict(c) for c in localization_result.review_required_components] + 
                [asdict(c) for c in localization_result.rejected_components]
            )
        }
        with open(os.path.join(diagnostics_dir, "phase4_localization_report.json"), "w") as f:
            json.dump(loc_report, f, indent=4)
            
        # Create visual overlays natively
        def save_mask_png(mask, name):
            img = Image.fromarray((mask * 255).astype(np.uint8), 'L')
            img.save(os.path.join(diagnostics_dir, name))
            
        save_mask_png(detection_result.candidate_mask, "phase4_candidate_mask.png")
        save_mask_png(localization_result.cleaned_candidate_mask, "phase4_cleaned_mask.png")
        
        labeled_norm = (localization_result.labeled_components > 0)
        save_mask_png(labeled_norm, "phase4_connected_components.png")
        
        # We can reconstruct accepted/rejected/review masks
        acc_mask = np.zeros_like(detection_result.candidate_mask, dtype=bool)
        rev_mask = np.zeros_like(detection_result.candidate_mask, dtype=bool)
        rej_mask = np.zeros_like(detection_result.candidate_mask, dtype=bool)
        
        for comp in localization_result.accepted_components:
            acc_mask = acc_mask | (localization_result.labeled_components == comp.component_id)
        for comp in localization_result.review_required_components:
            rev_mask = rev_mask | (localization_result.labeled_components == comp.component_id)
        for comp in localization_result.rejected_components:
            rej_mask = rej_mask | (localization_result.labeled_components == comp.component_id)
            
        save_mask_png(acc_mask, "phase4_accepted_regions.png")
        save_mask_png(rev_mask, "phase4_review_required_regions.png")
        save_mask_png(rej_mask, "phase4_rejected_regions.png")
        
        fire_mask = localization_result.cleaned_candidate_mask
        fire_candidate_bbox = compute_candidate_bounding_box(fire_mask, ms_data.transform)
        
        # 4. VISUALIZATION (False Color B12-B08-B04)
        fc_arr = np.stack([ms_data.b12, ms_data.b08, ms_data.b04], axis=-1)
        fc_img = Image.fromarray(scale_to_8bit(fc_arr), 'RGB')
        
        # --- PHASE 5: YOLO EXPORT ---
        process_yolo_export(
            localization_result=localization_result,
            clean_fc_img=fc_img,
            ms_data=ms_data,
            event_meta=event_meta,
            dataset_root="YOLO_dataset"
        )
        
        if fire_candidate_bbox is None:
            # We no longer delete the out_dir entirely so the user can inspect diagnostics
            # shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": "No fire candidate detected after 20m multispectral analysis and spatial localization (rejected false positive)."}
            
        event_meta['detected_fire_region_bbox'] = fire_candidate_bbox

        draw = ImageDraw.Draw(fc_img)
        pix = fire_candidate_bbox['pixel']
        draw.rectangle([pix['min_col'], pix['min_row'], pix['max_col'], pix['max_row']], outline="red", width=1)
        
        fc_path = os.path.join(vis_dir, "B12-B8-B4.jpg")
        fc_img.save(fc_path, quality=90)
        
        write_image_metadata(
            txt_path=os.path.join(vis_dir, "B12-B8-B4.txt"),
            image_type="False Color (SWIR2-NIR-Red)",
            bands_text=["B12 - SWIR2", "B08 - NIR", "B04 - Red"],
            band_order_list=["B12", "B08", "B04"],
            item=item,
            event_meta=event_meta,
            crs_val=str(ms_data.crs),
            res_val=f"{ms_data.resolution}m (Analysis Grid)",
            preprocessor_metadata=ms_data.metadata
        )
        
        metadata = {
            "analysis_grid_resolution": f"{ms_data.resolution}m",
            "native_resolution": "10m (Red, NIR), 20m (SWIR)",
            "bands_available": ", ".join(ms_data.metadata["bands_loaded"]),
            "false_color_path": fc_path,
            "detected_fire_region_bounding_box": fire_candidate_bbox['geographic'],
            "generation_timestamp": datetime.utcnow().isoformat(),
            "canonical_aoi": ms_data.metadata.get("canonical_aoi")
        }
        
        with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return metadata
            
    except Exception as e:
        logger.error(f"Failed to process image for {event_id}: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None
