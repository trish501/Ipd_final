import os
import json
from typing import List, Optional
from datetime import datetime, timezone

from src.models.input import InputEvent
from src.models.aoi import AOI
from src.models.output import BuildingInfo, ImageryResult
from src.models.enums import PipelineState, VisualStatus
from src.utils.logger import logger
from shapely.geometry import mapping, box
import geopandas as gpd
import rasterio

class EventExporter:
    def __init__(self, base_output_dir: str = "outputs"):
        self.base_output_dir = base_output_dir

    def _val(self, value):
        """Helper to format missing values explicitly"""
        if value is None or value == "":
            return "NOT_AVAILABLE"
        return str(value)

    def export(self, event: InputEvent, aoi: AOI, imagery: ImageryResult, results: List[BuildingInfo]) -> str:
        """
        Exports the entire event dataset strictly to text/JSON/PNG formats.
        """
        if imagery.status == PipelineState.SUCCESS:
            if imagery.image_width <= 0 or imagery.image_height <= 0 or imagery.native_resolution_m <= 0:
                raise ValueError("IMAGE_QUALITY_VALIDATION_FAILED: Invalid native dimensions or GSD")
                
        event_dir = os.path.join(self.base_output_dir, event.event_id)
        os.makedirs(event_dir, exist_ok=True)
        
        txt_path = os.path.join(event_dir, "building_info.txt")
        json_path = os.path.join(event_dir, "provenance.json")
        geojson_path = os.path.join(event_dir, "buildings.geojson")
        overlay_path = os.path.join(event_dir, "building_overlay.png")
        report_path = os.path.join(event_dir, "verification_report.json")

        # Split into verified and rejected
        verified_results = [r for r in results if r.visual_status == VisualStatus.VERIFIED_BUILDING]
        rejected_results = [r for r in results if r.visual_status != VisualStatus.VERIFIED_BUILDING]

        # 1. Verification Report
        report_data = []
        for idx, r in enumerate(results, start=1):
            report_data.append({
                "candidate_id": f"C{idx:04d}",
                "source": r.source,
                "geometry_area_m2": round(r.area_sq_meters, 1),
                "length_m": round(r.long_axis_meters, 1),
                "width_m": round(r.short_axis_meters, 1),
                "coverage_ratio": round(r.coverage_ratio, 2) if r.coverage_ratio else None,
                "pixel_support": getattr(r, "pixel_support", "UNKNOWN"),
                "valid_pixel_count": r.attributes.get("interior_valid_pixels", 0),
                "vegetation_fraction": round(r.attributes.get("vegetation_fraction", 0.0), 2),
                "water_fraction": round(r.attributes.get("water_fraction", 0.0), 2),
                "shadow_fraction": round(r.attributes.get("shadow_fraction", 0.0), 2),
                "contextual_contrast": round(r.attributes.get("boundary_contrast", 0.0), 1),
                "structural_evidence": r.attributes.get("STRUCTURAL_EVIDENCE", False),
                "source_agreement": getattr(r, "source_agreement", "UNKNOWN"),
                "final_visual_status": r.visual_status.value if r.visual_status else "UNKNOWN",
                "rejection_reason": r.rejection_reason
            })
        with open(report_path, "w") as rf:
            json.dump(report_data, rf, indent=2)

        # 2. Building Info TXT
        with open(txt_path, "w") as txt_file:
            txt_file.write(f"EVENT ID:\n{event.event_id}\n\n")
            txt_file.write(f"FIRE COORDINATE:\nLatitude: {event.latitude}\nLongitude: {event.longitude}\n\n")
            
            txt_file.write("RGB IMAGE:\n")
            provider = self._val(imagery.provider)
            if imagery.status != PipelineState.SUCCESS:
                txt_file.write(f"{provider} (STATUS: {imagery.status.value})\n")
            else:
                txt_file.write(f"{provider}\n")
            txt_file.write(f"Image CRS: EPSG:{imagery.crs_epsg}\n")
            txt_file.write(f"Image width: {imagery.image_width} px\n")
            txt_file.write(f"Image height: {imagery.image_height} px\n")
            txt_file.write(f"Ground sampling distance: {imagery.native_resolution_m} m/px\n\n")
            
            bounds = imagery.image_footprint_wgs84.bounds if imagery.image_footprint_wgs84 and not imagery.image_footprint_wgs84.is_empty else (0,0,0,0)
            txt_file.write("IMAGE FOOTPRINT:\n")
            txt_file.write(f"North: {bounds[3]}\nSouth: {bounds[1]}\nEast: {bounds[2]}\nWest: {bounds[0]}\n\n")
            
            txt_file.write("VERIFIED VISIBLE BUILDINGS\n==========================\n\n")

            if not verified_results:
                txt_file.write("NO_VISUALLY_VERIFIED_BUILDINGS\n\n")
            else:
                for idx, b_info in enumerate(verified_results, start=1):
                    building_id = f"V{idx:03d}"
                    
                    txt_file.write(f"{building_id}\n----------------\n")
                    txt_file.write(f"Source: {b_info.source}\n")
                    cov_str = b_info.coverage_status.value if b_info.coverage_status else "UNKNOWN"
                    txt_file.write(f"Coverage: {cov_str}\n")
                    txt_file.write(f"Visual Status: {b_info.visual_status.value}\n")
                    txt_file.write(f"Latitude/Longitude centroid: {b_info.centroid_lat}, {b_info.centroid_lon}\n")
                    txt_file.write(f"Area: {b_info.area_sq_meters:.1f} m²\n")
                    txt_file.write(f"Length: {b_info.long_axis_meters:.1f} m\n")
                    txt_file.write(f"Breadth: {b_info.short_axis_meters:.1f} m\n\n")

            txt_file.write(f"TOTAL VERIFIED VISIBLE BUILDINGS: {len(verified_results)}\n\n")

            txt_file.write("CANDIDATE SUMMARY\n=================\n\n")
            txt_file.write(f"Total candidates: {len(results)}\n")
            txt_file.write(f"Verified: {len(verified_results)}\n")
            txt_file.write(f"Rejected: {len(rejected_results)}\n")

        # 3. GeoJSON (ONLY VERIFIED)
        features = []
        for idx, b_info in enumerate(verified_results, start=1):
            building_id = f"V{idx:03d}"
            geom_dict = mapping(b_info.original_geometry_wgs84)
            feature = {
                "type": "Feature",
                "geometry": geom_dict,
                "properties": {
                    "building_id": building_id,
                    "source": b_info.source,
                    "area_m2": round(b_info.area_sq_meters, 1),
                    "length_m": round(b_info.long_axis_meters, 1),
                    "width_m": round(b_info.short_axis_meters, 1),
                    "coverage_status": b_info.coverage_status.value if b_info.coverage_status else "UNKNOWN",
                    "visual_status": b_info.visual_status.value if b_info.visual_status else "UNKNOWN",
                    "centroid_lat": b_info.centroid_lat,
                    "centroid_lon": b_info.centroid_lon
                }
            }
            features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        with open(geojson_path, "w") as gf:
            json.dump(geojson_data, gf, indent=2)

        # 4. Provenance
        acq_dt = imagery.acquisition_datetime.isoformat() if imagery.acquisition_datetime else "UNKNOWN"
        provenance = {
            "event_id": event.event_id,
            "input_lat": event.latitude,
            "input_lon": event.longitude,
            "image_source": imagery.provider if imagery.provider else "UNKNOWN",
            "image_acquisition_datetime": acq_dt,
            "image_gsd": imagery.native_resolution_m if imagery.native_resolution_m else "UNKNOWN",
            "image_crs": f"EPSG:{imagery.crs_epsg}" if imagery.crs_epsg else "UNKNOWN",
            "image_footprint": mapping(imagery.image_footprint_wgs84) if imagery.image_footprint_wgs84 and not imagery.image_footprint_wgs84.is_empty else "UNKNOWN",
            "native_width_px": imagery.image_width,
            "native_height_px": imagery.image_height,
            "display_scale": imagery.metadata.get("display_scale", 1),
            "resampling_method": imagery.metadata.get("resampling_method", "UNKNOWN"),
            "contrast_stretch": imagery.metadata.get("contrast_stretch", "UNKNOWN"),
            "building_sources": ["Google Open Buildings", "Microsoft Global ML Building Footprints"],
            "number_of_buildings": len(verified_results),
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_status": "USED"
        }
        with open(json_path, "w") as mf:
            json.dump(provenance, mf, indent=2)

        # 5. Overlay Image generation with PIL
        diagnostic_overlay_path = os.path.join(event_dir, "verification_overlay.png")
        png_path = os.path.join(event_dir, "rgb.png")
        
        import time
        from PIL import Image, ImageDraw, ImageFont
        import pyproj
        from src.config import settings

        if imagery.rgb_image_path and os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            try:
                # FAST-PATH FOR ZERO BUILDINGS
                if len(verified_results) == 0:
                    t0 = time.perf_counter()
                    import shutil
                    shutil.copy(png_path, overlay_path)
                    print(f"\nOVERLAY PERFORMANCE\n-------------------\nBuildings: 0\nTOTAL: {(time.perf_counter()-t0):.4f}s\n")
                    return event_dir
                
                t_total_start = time.perf_counter()
                
                # A. Image loading
                t_img_start = time.perf_counter()
                prod_img = Image.open(png_path).convert("RGB")
                prod_draw = ImageDraw.Draw(prod_img, "RGBA")
                scale = imagery.metadata.get("display_scale", 1)
                
                if settings.generate_debug_overlay:
                    diag_img = Image.open(png_path).convert("RGB")
                    diag_draw = ImageDraw.Draw(diag_img, "RGBA")
                else:
                    diag_img = None
                    diag_draw = None
                t_img = time.perf_counter() - t_img_start
                
                # C. Coordinate transformation (init)
                t_transform_start = time.perf_counter()
                with rasterio.open(imagery.rgb_image_path) as src:
                    src_crs = src.crs
                    inv_transform = ~src.transform
                    width = src.width
                    height = src.height
                
                project_to_src = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True).transform
                t_transform = time.perf_counter() - t_transform_start
                
                # Bounding box of the image in WGS84 for fast rejection
                img_bounds = imagery.image_footprint_wgs84.bounds if imagery.image_footprint_wgs84 else None
                
                def to_pixels(x, y):
                    col, row = inv_transform * (x, y)
                    return (col * scale, row * scale)

                # Initialize font once
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

                t_poly = 0
                t_draw = 0
                t_labels = 0
                
                # Determine label mode
                label_mode = settings.label_mode
                if len(verified_results) > 500:
                    label_mode = "NONE" # Force fast path for many buildings
                
                # Process verified results to find index for production labels
                verified_map = {id(r): f"V{(i+1):03d}" for i, r in enumerate(verified_results)}
                
                # If only production overlay is needed, only loop through verified
                bldgs_to_process = results if settings.generate_debug_overlay else verified_results
                
                for r in bldgs_to_process:
                    # F. Bounding box rejection
                    t_p_start = time.perf_counter()
                    b_bounds = r.original_geometry_wgs84.bounds
                    if img_bounds:
                        if (b_bounds[2] < img_bounds[0] or b_bounds[0] > img_bounds[2] or
                            b_bounds[3] < img_bounds[1] or b_bounds[1] > img_bounds[3]):
                            t_poly += (time.perf_counter() - t_p_start)
                            continue
                    
                    # Convert to pixel geometry
                    from shapely.ops import transform as shapely_transform
                    geom_src = shapely_transform(project_to_src, r.original_geometry_wgs84)
                    polys = [geom_src] if geom_src.geom_type == 'Polygon' else (list(geom_src.geoms) if geom_src.geom_type == 'MultiPolygon' else [])
                    
                    pixel_polys = []
                    for poly in polys:
                        pixel_polys.append([to_pixels(x, y) for x, y in poly.exterior.coords])
                    t_poly += (time.perf_counter() - t_p_start)
                    
                    # G. PIL drawing
                    t_d_start = time.perf_counter()
                    status = r.visual_status.value if r.visual_status else "UNKNOWN"
                    is_verified = status == "VERIFIED_BUILDING"
                    
                    for exterior_coords in pixel_polys:
                        if diag_draw:
                            if status in ["REJECTED_NON_BUILDING", "UNRESOLVED", "UNKNOWN"]:
                                diag_draw.polygon(exterior_coords, outline=(255, 0, 0, 255), width=2)
                            elif status == "PROBABLE_BUILDING":
                                diag_draw.polygon(exterior_coords, outline=(255, 255, 0, 255), width=2)
                            elif is_verified:
                                diag_draw.polygon(exterior_coords, outline=(0, 255, 0, 255), width=2)
                                
                        if is_verified:
                            prod_draw.polygon(exterior_coords, outline=(255, 0, 255, 255), width=2)
                    t_draw += (time.perf_counter() - t_d_start)
                    
                    # F. Labels (only for verified)
                    if is_verified and label_mode != "NONE" and polys:
                        t_l_start = time.perf_counter()
                        poly = polys[0]
                        cx, cy = poly.centroid.x, poly.centroid.y
                        px, py = to_pixels(cx, cy)
                        b_id = verified_map.get(id(r), "")
                        
                        if label_mode == "FULL":
                            text_str = f"{b_id}\nL:{r.long_axis_meters:.1f}m\nW:{r.short_axis_meters:.1f}m"
                        else:
                            text_str = b_id
                            
                        bbox = prod_draw.textbbox((px + 5, py + 5), text_str, font=font)
                        if bbox:
                            prod_draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(255, 0, 255, 128))
                        prod_draw.text((px + 5, py + 5), text_str, fill="white", font=font)
                        t_labels += (time.perf_counter() - t_l_start)

                # H/I. PNG encoding and File write
                t_save_start = time.perf_counter()
                if diag_img:
                    diag_img.save(diagnostic_overlay_path, compress_level=1)
                if verified_results:
                    prod_img.save(overlay_path, compress_level=1)
                t_save = time.perf_counter() - t_save_start
                
                t_total = time.perf_counter() - t_total_start
                
                print(f"\\nOVERLAY PERFORMANCE\\n-------------------")
                print(f"Buildings: {len(bldgs_to_process)}")
                print(f"Image loading: {t_img:.4f}s")
                print(f"Coordinate transform init: {t_transform:.4f}s")
                print(f"Polygon conversion: {t_poly:.4f}s")
                print(f"Drawing: {t_draw:.4f}s")
                print(f"Labels: {t_labels:.4f}s")
                print(f"PNG encoding / File write: {t_save:.4f}s")
                print(f"TOTAL: {t_total:.4f}s\\n")
                            
            except Exception as e:
                logger.error(f"Failed to generate building overlays with PIL: {e}")

        return event_dir
