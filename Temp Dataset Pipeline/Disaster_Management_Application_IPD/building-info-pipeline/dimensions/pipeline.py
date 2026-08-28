import logging
import os
import time
import rasterio
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared_models.canonical_aoi import CanonicalAOI


from dimensions.models import SIHInput, SIHResult, SIHImagerySummary, SIHInstitutionSummary
from dimensions.highres_rgb_retrieval import fetch_high_res_basemap
from dimensions.institution_selector import InstitutionSelector
from dimensions.building_measurement import measure_building, calculate_union_area
from dimensions.rgb_validator import RGBValidator
from dimensions.exporters import SIHExporter
from dimensions.image_renderer import render_annotated_image
from dimensions.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

class DimensionsPipeline:
    def __init__(self):
        self.selector = InstitutionSelector()
        self.validator = RGBValidator()
        self.exporter = SIHExporter()
        
    def run(self, input_data: SIHInput):
 
        logger.info(f"Running pipeline for lat={input_data.latitude}, lon={input_data.longitude}")
        
        start_time = time.time()
        timings = {}
        
        result = SIHResult(
            status="RUNNING",
            input=input_data,
            imagery=SIHImagerySummary(),
            institution=SIHInstitutionSummary(),
            buildings=[]
        )
        
        # 1. Define initial AOI for image fetch
        t0 = time.time()

        canonical_aoi = None
        if input_data.event_dir and os.path.exists(os.path.join(input_data.event_dir, "metadata.json")):
            try:
                with open(os.path.join(input_data.event_dir, "metadata.json"), "r") as f:
                    meta = json.load(f)
                    if "canonical_aoi" in meta and meta["canonical_aoi"]:
                        canonical_aoi = CanonicalAOI.from_dict(meta["canonical_aoi"])
            except Exception as e:
                logger.error(f"Failed to load CanonicalAOI from metadata: {e}")

        if canonical_aoi:
            vis_bounds_wgs84 = canonical_aoi.get_wgs84_bounds()
            class DummyAOI:
                def __init__(self, area):
                    self.area_sq_meters = area
            aoi = DummyAOI(canonical_aoi.width_m * canonical_aoi.height_m)
        else:
            radius = 200.0 if input_data.event_id else 250.0
            from dimensions.aoi import AOI
            aoi = AOI.create_from_point(input_data.latitude, input_data.longitude, radius)
            vis_bounds_wgs84 = aoi.geometry_wgs84.bounds
            
        timings["AOI creation"] = time.time() - t0
        
        # 2. Search Satellite Imagery FIRST
        t4 = time.time()
        if input_data.event_dir:
            out_folder = input_data.event_dir
        elif input_data.event_id:
            out_folder = os.path.join(OUTPUTS_DIR, input_data.event_id)
        else:
            inst_name = input_data.institution_name.replace(' ', '_') if input_data.institution_name else "Location"
            out_folder = os.path.join(OUTPUTS_DIR, f"{inst_name}_{input_data.latitude}_{input_data.longitude}")
        os.makedirs(out_folder, exist_ok=True)
        temp_rgb_path = os.path.join(out_folder, "temp_rgb.tif")
        
        sat_warning = None
        try:
            logger.info("Attempting to fetch high-res satellite imagery...")
            img_data = fetch_high_res_basemap(vis_bounds_wgs84, zoom=18, canonical_aoi=canonical_aoi)
            result.imagery.provider = img_data["metadata"]["provider"]
            result.imagery.product_id = img_data["metadata"]["scene_id"]
            result.imagery.acquisition_datetime = img_data["metadata"]["acquisition_datetime"]
            result.imagery.bands = ["Red", "Green", "Blue"]
            result.imagery.native_resolution_m = img_data["resolution"]
            result.imagery.image_width = img_data["data"].shape[2]
            result.imagery.image_height = img_data["data"].shape[1]
            result.imagery.image_bounds = list(img_data["bounds"])
            
            with rasterio.open(
                temp_rgb_path, 'w', driver='GTiff',
                height=img_data["data"].shape[1], width=img_data["data"].shape[2],
                count=3, dtype=img_data["data"].dtype,
                crs=img_data["crs"], transform=img_data["transform"]
            ) as dst:
                dst.write(img_data["data"])
        except Exception as e:
            logger.error(f"High-res fetch failed: {e}")
            result.status = "HIGH_RES_RGB_IMAGERY_UNAVAILABLE"
            result.error = f"Could not find any high-resolution RGB imagery: {e}"
            self.exporter.export_all(result, out_folder)
            if os.path.exists(temp_rgb_path):
                os.remove(temp_rgb_path)
            return result
        timings["Esri request preparation and download"] = time.time() - t4
        
        # 3. Fetch Buildings in Image Bounds
        t1 = time.time()
        logger.info("Fetching all building footprints within image extent...")
        try:
            image_bounds_wgs84 = vis_bounds_wgs84
            candidates = self.selector.get_candidate_buildings(image_bounds_wgs84, rgb_image_path=temp_rgb_path)
            
            google_c = sum(1 for b in candidates if b.source == "GoogleOpenBuildings")
            ms_c = sum(1 for b in candidates if b.source == "MicrosoftBuildingFootprints")
            
            result.institution.candidate_buildings = len(candidates)
            result.institution.google_candidates = google_c
            result.institution.microsoft_candidates = ms_c
            
            if aoi:
                result.institution.institution_aoi_area_sq_m = aoi.area_sq_meters
                from dimensions.area_calculator import sq_meters_to_sq_ft
                result.institution.institution_aoi_area_sq_ft = sq_meters_to_sq_ft(aoi.area_sq_meters)
            
            if not candidates:
                result.status = "NO_BUILDINGS_FOUND"
                # Keep going to generate PNG anyway (user request: no event should be empty)
                # self.exporter.export_all(result, out_folder)
                # if os.path.exists(temp_rgb_path):
                #     os.remove(temp_rgb_path)
                # return result
                
        except Exception as e:
            result.status = "BUILDING_RETRIEVAL_FAILED"
            result.error = str(e)
            self.exporter.export_all(result, out_folder)
            if os.path.exists(temp_rgb_path):
                os.remove(temp_rgb_path)
            return result
        timings["Partition lookup and download"] = time.time() - t1
        
        # 4. Geometry Deduplication (IoU)
        t2 = time.time()
        candidates.sort(key=lambda b: b.geometry.area, reverse=True)
        
        from shapely.strtree import STRtree
        from shapely.validation import make_valid
        
        valid_candidates = []
        for b in candidates:
            if not b.geometry.is_valid:
                b.geometry = make_valid(b.geometry)
            if not b.geometry.is_empty:
                valid_candidates.append(b)
                
        candidates = valid_candidates
        
        unique_candidates = []
        duplicates = 0
        
        if candidates:
            tree = STRtree([b.geometry for b in candidates])
            retained_indices = set()
            
            for i, b in enumerate(candidates):
                is_dup = False
                idx = tree.query(b.geometry)
                for j in idx:
                    if j in retained_indices:
                        other_geom = candidates[j].geometry
                        if other_geom.intersects(b.geometry):
                            intersection = other_geom.intersection(b.geometry).area
                            union = other_geom.union(b.geometry).area
                            iou = intersection / union if union > 0 else 0
                            if iou > 0.5:
                                is_dup = True
                                break
                
                if not is_dup:
                    retained_indices.add(i)
                    unique_candidates.append(b)
                else:
                    duplicates += 1
                    
        candidates = unique_candidates
        result.institution.duplicates_removed = duplicates
        
        selected = candidates
        result.institution.selected_buildings = len(selected)
        result.institution.anchor_latitude = input_data.latitude
        result.institution.anchor_longitude = input_data.longitude
        timings["Geometry processing"] = time.time() - t2
        
        # 5. Union overlapping/touching buildings to form canonical physical footprints
        t5 = time.time()
        from shapely.ops import unary_union
        from dimensions.models import Building
        
        merged_geom = unary_union([b.geometry for b in selected])
        canonical_geoms = []
        if merged_geom.geom_type == 'Polygon':
            canonical_geoms.append(merged_geom)
        elif merged_geom.geom_type == 'MultiPolygon':
            canonical_geoms.extend(list(merged_geom.geoms))
        elif merged_geom.geom_type == 'GeometryCollection':
            canonical_geoms.extend([g for g in merged_geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')])
            
        sih_buildings = []
        invalid_geom_count = 0
        outside_aoi_count = 0
        
        from shapely.geometry import box
        img_bounds_box = box(*image_bounds_wgs84)
        
        import pyproj
        utm_zone = int((input_data.longitude + 180) / 6) + 1
        hemisphere = 'north' if input_data.latitude >= 0 else 'south'
        proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', **({} if hemisphere == 'north' else {'south': True}))
        project_transformer = pyproj.Transformer.from_proj(pyproj.Proj('epsg:4326'), proj, always_xy=True).transform

        for i, geom in enumerate(canonical_geoms):
            if geom.is_empty:
                invalid_geom_count += 1
                continue
                
            # If it's completely outside the image bounds, reject it
            if not geom.intersects(img_bounds_box):
                outside_aoi_count += 1
                continue
                
            canonical_b = Building(
                building_id=f"B{len(sih_buildings)+1:03d}",
                geometry=geom,
                centroid=geom.centroid,
                source="Merged Canonical"
            )
            measured = measure_building(canonical_b, project_transformer)
            sih_buildings.append(measured)
            
        result.institution.invalid_geometries = invalid_geom_count
        result.institution.outside_aoi = outside_aoi_count
        
        # Validate RGB
        sih_buildings = self.validator.validate_buildings(sih_buildings, temp_rgb_path)
        
        # Remove buildings with 0.0 coverage
        filtered_buildings = []
        for b in sih_buildings:
            if b.coverage_ratio > 0.0:
                filtered_buildings.append(b)
            else:
                outside_aoi_count += 1
        sih_buildings = filtered_buildings
        result.institution.outside_aoi = outside_aoi_count
        
        result.buildings = sih_buildings
        result.institution.measured_buildings = len(sih_buildings)
        timings["Measurement"] = time.time() - t5
        
        # 6. Union Area
        total_area_m2, total_area_ft2, final_union = calculate_union_area(sih_buildings, input_data.latitude, input_data.longitude, project_transformer)
        result.institution.total_building_footprint_area_sq_m = total_area_m2
        result.institution.total_building_footprint_area_sq_ft = total_area_ft2
        
        warnings = []
        if sat_warning: warnings.append(sat_warning)
        if warnings:
            result.error = "WARNINGS: " + ", ".join(warnings)
        
        # 7. Render PNG Image
        t6 = time.time()
        png_path = os.path.join(out_folder, "institution_measurement.png")
        try:
            logger.info("Rendering annotated PNG image...")
            loc_str = input_data.institution_name if input_data.institution_name else f"Fire Event {input_data.event_id}" if input_data.event_id else "Location"
            render_annotated_image(temp_rgb_path, sih_buildings, result.institution, result.imagery, png_path, loc_str, warnings, final_union, input_data.latitude, input_data.longitude)
            result.imagery.preprocessing = "PNG Annotation"
            result.institution.rendered_buildings = len(sih_buildings)
        except Exception as e:
            logger.error(f"PNG rendering failed: {e}")
            result.status = "PNG_RENDER_FAILED"
            result.error = str(e)
            result.institution.rendered_buildings = 0
        timings["Rendering"] = time.time() - t6
            
        # 8. Export
        t7 = time.time()
        try:
            result.status = "SUCCESS"
            result.institution.exported_buildings = len(sih_buildings)
            out_dir = self.exporter.export_all(result, out_folder, png_path, final_union, candidates, selected)
            logger.info(f"Pipeline complete. Outputs saved to {out_dir}")
        except Exception as e:
            result.status = "EXPORT_FAILED"
            result.error = str(e)
            result.institution.exported_buildings = 0
        timings["Export"] = time.time() - t7
            
        # Clean up temp
        if os.path.exists(temp_rgb_path):
            os.remove(temp_rgb_path)
            
        total_time = time.time() - start_time
        timings["TOTAL"] = total_time
        logger.info("\n" + "-"*32 + "\nPERFORMANCE AUDIT\n" + "-"*32)
        for k, v in timings.items():
            logger.info(f"{k:<30} {v:.2f} sec")
        logger.info("-" * 32)
            
        return result
