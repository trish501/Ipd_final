import os
import json
import csv
import logging
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np
import hashlib

logger = logging.getLogger(__name__)

def calculate_component_square(x_min_20m, x_max_20m, y_min_20m, y_max_20m, grid_cols, grid_rows):
    width = x_max_20m - x_min_20m
    height = y_max_20m - y_min_20m
    side = max(width, height)
    
    center_x = (x_min_20m + x_max_20m) / 2.0
    center_y = (y_min_20m + y_max_20m) / 2.0
    
    sq_x_min = center_x - side / 2.0
    sq_x_max = center_x + side / 2.0
    sq_y_min = center_y - side / 2.0
    sq_y_max = center_y + side / 2.0
    
    # Handle edges by shifting to preserve side length
    if sq_x_min < 0:
        shift = 0 - sq_x_min
        sq_x_min += shift
        sq_x_max += shift
    if sq_x_max > grid_cols:
        shift = sq_x_max - grid_cols
        sq_x_min -= shift
        sq_x_max -= shift
        
    if sq_y_min < 0:
        shift = 0 - sq_y_min
        sq_y_min += shift
        sq_y_max += shift
    if sq_y_max > grid_rows:
        shift = sq_y_max - grid_rows
        sq_y_min -= shift
        sq_y_max -= shift
        
    # Clip to bounds if side > max size (should not occur unless image is very thin)
    sq_x_min = max(0, sq_x_min)
    sq_x_max = min(grid_cols, sq_x_max)
    sq_y_min = max(0, sq_y_min)
    sq_y_max = min(grid_rows, sq_y_max)
    
    return int(round(sq_x_min)), int(round(sq_x_max)), int(round(sq_y_min)), int(round(sq_y_max))

def transform_20m_to_rgb(x_20m, y_20m, grid_transform, rgb_transform):
    """
    Transforms pixel coordinate from 20m grid to RGB grid using geospatial transforms.
    """
    # Grid pixel to Spatial
    spatial_x, spatial_y = grid_transform * (x_20m, y_20m)
    # Spatial to RGB pixel
    rgb_transform_inv = ~rgb_transform
    x_rgb, y_rgb = rgb_transform_inv * (spatial_x, spatial_y)
    
    return float(x_rgb), float(y_rgb)

def generate_yolo_label(x_min_rgb, x_max_rgb, y_min_rgb, y_max_rgb, image_width, image_height, class_id=0):
    x_center = (x_min_rgb + x_max_rgb) / 2.0
    y_center = (y_min_rgb + y_max_rgb) / 2.0
    box_width = x_max_rgb - x_min_rgb
    box_height = y_max_rgb - y_min_rgb
    
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height
    
    return {
        "x_center_norm": x_center_norm,
        "y_center_norm": y_center_norm,
        "width_norm": width_norm,
        "height_norm": height_norm,
        "label_line": f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    }

def get_deterministic_split(event_id, train_ratio=0.7, val_ratio=0.15):
    # Deterministic split based on event_id hash
    h = int(hashlib.md5(event_id.encode('utf-8')).hexdigest(), 16)
    val = (h % 100) / 100.0
    if val < train_ratio:
        return "train"
    elif val < train_ratio + val_ratio:
        return "val"
    else:
        return "test"

def scale_to_8bit(arr):
    scaled = arr * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def process_yolo_export(localization_result, clean_fc_img, ms_data, event_meta, dataset_root="YOLO_dataset"):
    """
    Main Phase 5 entry point.
    """
    event_id = event_meta.get("event_id", "unknown")
    grid_rows, grid_cols = ms_data.b04.shape
    image_width, image_height = clean_fc_img.size
    
    grid_transform = ms_data.transform
    rgb_transform = ms_data.transform
    
    accepted_comps = []
    for comp in localization_result.accepted_components + localization_result.review_required_components + localization_result.rejected_components:
        eligible = getattr(comp, 'eligible_for_yolo_export', False)
        decision = comp.decision
        
        if eligible is True and decision != "ACCEPTED_FOR_AUTO_EXPORT":
            error_msg = f"Strict export gate validation failed: Component {comp.component_id} has eligible_for_yolo_export=True but decision='{decision}'. Export blocked."
            logger.error(error_msg)
            error_path = os.path.join(dataset_root, "diagnostics", event_id, "export_validation_error.json")
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            with open(error_path, "w") as f:
                json.dump({"event_id": event_id, "component_id": comp.component_id, "error": error_msg}, f)
            raise ValueError(error_msg)
            
        if eligible is True and decision == "ACCEPTED_FOR_AUTO_EXPORT":
            accepted_comps.append(comp)
            
    metadata = {
        "schema_version": "1.0",
        "event_id": event_id,
        "pipeline_run_id": "auto_export_run",
        "firms_source": event_meta.get("source", ""),
        "firms_lat": event_meta.get("latitude"),
        "firms_lon": event_meta.get("longitude"),
        "firms_date": event_meta.get("date", ""),
        "firms_time": event_meta.get("time", ""),
        "scene_id": ms_data.metadata.get("scene_id", ""),
        "acquisition_datetime": ms_data.metadata.get("acquisition_datetime", ""),
        "cloud_masked_percentage": ms_data.metadata.get("masked_pixel_percentage", 0.0),
        "crs": str(ms_data.crs),
        "grid_resolution_m": ms_data.resolution,
        "image_width": image_width,
        "image_height": image_height,
        "export_status": "REJECTED" if len(accepted_comps) == 0 else "EXPORTED",
        "generation_timestamp": datetime.utcnow().isoformat(),
        "image_path": None,
        "label_path": None,
        "yolo_boxes": [],
        "components": []
    }
    
    split = get_deterministic_split(event_id)
    os.makedirs(os.path.join(dataset_root, "manifests"), exist_ok=True)
    splits_csv = os.path.join(dataset_root, "manifests", "splits.csv")
    splits_exists = os.path.exists(splits_csv)
    with open(splits_csv, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "split"])
        if not splits_exists:
            writer.writeheader()
        writer.writerow({"event_id": event_id, "split": split})
    
    if len(accepted_comps) == 0:
        os.makedirs(os.path.join(dataset_root, "diagnostics", event_id), exist_ok=True)
        metadata_path = os.path.join(dataset_root, "diagnostics", event_id, "event_ground_truth.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        return metadata 
        
    os.makedirs(os.path.join(dataset_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "diagnostics", event_id), exist_ok=True)
    
    swir_dir = os.path.join(dataset_root, "diagnostics", "swir_overlays")
    rgb_dir = os.path.join(dataset_root, "diagnostics", "rgb_overlays")
    os.makedirs(swir_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    
    img_filename = f"{event_id}.jpg"
    lbl_filename = f"{event_id}.txt"
    img_path = os.path.join(dataset_root, "images", split, img_filename)
    lbl_path = os.path.join(dataset_root, "labels", split, lbl_filename)
    
    metadata["image_path"] = img_path
    metadata["label_path"] = lbl_path
    
    # Save clean image
    clean_fc_img.save(img_path, quality=95)
    
    # Prepare diagnostic overlays
    diag_img = clean_fc_img.copy() # original behavior
    draw_orig = ImageDraw.Draw(diag_img)
    
    # Generate SWIR and RGB images
    swir_arr = np.stack([ms_data.b12, ms_data.b11, ms_data.b04], axis=-1)
    rgb_arr = np.stack([ms_data.b04, ms_data.b03, ms_data.b02], axis=-1)
    
    swir_img = Image.fromarray(scale_to_8bit(swir_arr), 'RGB')
    rgb_img = Image.fromarray(scale_to_8bit(rgb_arr), 'RGB')
    
    draw_swir = ImageDraw.Draw(swir_img)
    draw_rgb = ImageDraw.Draw(rgb_img)
    
    swir_out_path = os.path.join(swir_dir, f"{event_id}_swir_yolo_overlay.jpg")
    rgb_out_path = os.path.join(rgb_dir, f"{event_id}_rgb_yolo_overlay.jpg")
    
    swir_metadata_list = []
    rgb_metadata_list = []
    
    labels = []
    manifest_rows = []
    
    for comp in accepted_comps:
        c_x_min = comp.x_min_20m
        c_x_max = comp.x_max_20m + 1
        c_y_min = comp.y_min_20m
        c_y_max = comp.y_max_20m + 1
        
        sq_x_min, sq_x_max, sq_y_min, sq_y_max = calculate_component_square(
            c_x_min, c_x_max, c_y_min, c_y_max, grid_cols, grid_rows
        )
        
        rgb_x_min, rgb_y_min = transform_20m_to_rgb(sq_x_min, sq_y_min, grid_transform, rgb_transform)
        rgb_x_max, rgb_y_max = transform_20m_to_rgb(sq_x_max, sq_y_max, grid_transform, rgb_transform)
        
        rgb_x_min = max(0, int(round(rgb_x_min)))
        rgb_x_max = min(image_width, int(round(rgb_x_max)))
        rgb_y_min = max(0, int(round(rgb_y_min)))
        rgb_y_max = min(image_height, int(round(rgb_y_max)))
        
        yolo_data = generate_yolo_label(
            rgb_x_min, rgb_x_max, rgb_y_min, rgb_y_max, 
            image_width, image_height, class_id=0
        )
        labels.append(yolo_data["label_line"])
        
        box_coords = [rgb_x_min, rgb_y_min, rgb_x_max, rgb_y_max]
        
        # Draw on old diagnostic overlay (backward compat)
        draw_orig.rectangle(box_coords, outline=(0, 120, 255), width=1)
        
        # Draw on new overlays
        draw_swir.rectangle(box_coords, outline=(0, 120, 255), width=1)
        draw_rgb.rectangle(box_coords, outline=(0, 120, 255), width=1)
        
        # Record metadata for dual overlays
        gen_time = datetime.utcnow().isoformat()
        swir_metadata_list.append({
            "event_id": event_id,
            "component_id": comp.component_id,
            "overlay_type": "SWIR_B12_B11_B04",
            "source_image_path": img_path,
            "overlay_image_path": swir_out_path,
            "box_coordinates_used": box_coords,
            "box_colour": [0, 120, 255],
            "line_thickness": 1,
            "image_width": image_width,
            "image_height": image_height,
            "generation_time": gen_time
        })
        
        rgb_metadata_list.append({
            "event_id": event_id,
            "component_id": comp.component_id,
            "overlay_type": "RGB_B04_B03_B02",
            "source_image_path": img_path,
            "overlay_image_path": rgb_out_path,
            "box_coordinates_used": box_coords,
            "box_colour": [0, 120, 255],
            "line_thickness": 1,
            "image_width": image_width,
            "image_height": image_height,
            "generation_time": gen_time
        })
        
        comp_meta = {
            "component_id": comp.component_id,
            "x_min_20m": c_x_min,
            "x_max_20m": c_x_max,
            "y_min_20m": c_y_min,
            "y_max_20m": c_y_max,
            "square_x_min_20m": sq_x_min,
            "square_x_max_20m": sq_x_max,
            "square_y_min_20m": sq_y_min,
            "square_y_max_20m": sq_y_max,
            "x_min_rgb": rgb_x_min,
            "x_max_rgb": rgb_x_max,
            "y_min_rgb": rgb_y_min,
            "y_max_rgb": rgb_y_max,
            "box_width_rgb": rgb_x_max - rgb_x_min,
            "box_height_rgb": rgb_y_max - rgb_y_min,
            "yolo_x_center": yolo_data["x_center_norm"],
            "yolo_y_center": yolo_data["y_center_norm"],
            "yolo_width": yolo_data["width_norm"],
            "yolo_height": yolo_data["height_norm"]
        }
        metadata["components"].append(comp_meta)
        metadata["yolo_boxes"].append(yolo_data["label_line"])
        
        manifest_rows.append({
            "event_id": event_id,
            "component_id": comp.component_id,
            "image_path": img_path,
            "label_path": lbl_path,
            "x_min_rgb": rgb_x_min,
            "x_max_rgb": rgb_x_max,
            "y_min_rgb": rgb_y_min,
            "y_max_rgb": rgb_y_max,
            "box_width_rgb": rgb_x_max - rgb_x_min,
            "box_height_rgb": rgb_y_max - rgb_y_min,
            "x_center_norm": yolo_data["x_center_norm"],
            "y_center_norm": yolo_data["y_center_norm"],
            "width_norm": yolo_data["width_norm"],
            "height_norm": yolo_data["height_norm"],
            "class_id": 0,
            "class_name": "fire_candidate",
            "split": split
        })
        
    with open(lbl_path, "w") as f:
        f.write("\n".join(labels) + "\n")
        
    diag_img.save(os.path.join(dataset_root, "diagnostics", event_id, "yolo_overlay.jpg"), quality=90)
    swir_img.save(swir_out_path, quality=90)
    rgb_img.save(rgb_out_path, quality=90)
    
    with open(os.path.join(swir_dir, f"{event_id}_swir_metadata.json"), "w") as f:
        json.dump(swir_metadata_list, f, indent=4)
    with open(os.path.join(rgb_dir, f"{event_id}_rgb_metadata.json"), "w") as f:
        json.dump(rgb_metadata_list, f, indent=4)
    
    metadata_path = os.path.join(dataset_root, "diagnostics", event_id, "event_ground_truth.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    csv_path = os.path.join(dataset_root, "manifests", "annotations.csv")
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerows(manifest_rows)
        
    jsonl_path = os.path.join(dataset_root, "manifests", "annotations.jsonl")
    with open(jsonl_path, "a") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
            
    return metadata

