import json
import os
import shutil
from shapely.geometry import mapping

from dimensions.models import SIHResult

class SIHExporter:
    @staticmethod
    def export_all(result: SIHResult, out_dir: str, png_image_path: str = None, union_geom = None, candidates = None, selected = None):
        # Create directories
        for d in ["imagery", "buildings", "measurements", "metadata"]:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)
            
        # 1. GeoJSONs
        selected_geojson_path = os.path.join(out_dir, "buildings", "selected_buildings.geojson")
        
        features = []
        for b in result.buildings:
            feat = {
                "type": "Feature",
                "geometry": mapping(b.geometry_wgs84),
                "properties": {
                    "building_id": b.building_id,
                    "source": b.source,
                    "footprint_area_sq_m": b.footprint_area_sq_m,
                    "footprint_area_sq_ft": b.footprint_area_sq_ft,
                    "bounding_rectangle_area_sq_m": b.bounding_rectangle_area_sq_m,
                    "bounding_rectangle_area_sq_ft": b.bounding_rectangle_area_sq_ft,
                    "perimeter_m": b.perimeter_m,
                    "long_axis_m": b.length_m,
                    "short_axis_m": b.width_m,
                    "centroid_lat": b.centroid["latitude"],
                    "centroid_lon": b.centroid["longitude"],
                    "coverage_ratio": b.coverage_ratio
                }
            }
            features.append(feat)
            
        with open(selected_geojson_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
            
        if union_geom:
            union_geojson_path = os.path.join(out_dir, "buildings", "institution_building_union.geojson")
            with open(union_geojson_path, "w") as f:
                json.dump({
                    "type": "FeatureCollection", 
                    "features": [{
                        "type": "Feature",
                        "geometry": mapping(union_geom),
                        "properties": {
                            "total_area_sq_m": result.institution.total_building_footprint_area_sq_m
                        }
                    }]
                }, f)
                
        if candidates and selected:
            rejected_geojson_path = os.path.join(out_dir, "buildings", "rejected_buildings.geojson")
            sel_ids = {b.building_id for b in selected}
            rej_features = []
            for b in candidates:
                if b.building_id not in sel_ids:
                    rej_features.append({
                        "type": "Feature",
                        "geometry": mapping(b.geometry),
                        "properties": {"building_id": b.building_id}
                    })
            with open(rejected_geojson_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": rej_features}, f)
                
        # 2. Measurements CSV
        import csv
        csv_path = os.path.join(out_dir, "measurements", "measurements.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "building_id", "source", "footprint_area_sq_m", "footprint_area_sq_ft", 
                "bounding_rectangle_area_sq_m", "bounding_rectangle_area_sq_ft",
                "perimeter_m", "long_axis_m", "short_axis_m", "centroid_lat", "centroid_lon"
            ])
            for b in result.buildings:
                if b.building_id:
                    writer.writerow([
                        b.building_id, b.source, 
                        b.footprint_area_sq_m, b.footprint_area_sq_ft, 
                        b.bounding_rectangle_area_sq_m, b.bounding_rectangle_area_sq_ft,
                        b.perimeter_m, b.length_m, b.width_m, 
                        b.centroid['latitude'], b.centroid['longitude']
                    ])
                
        # 3. Copy PNG image
        if png_image_path and os.path.exists(png_image_path):
            dest_png = os.path.join(out_dir, "imagery", os.path.basename(png_image_path))
            shutil.copy(png_image_path, dest_png)
            
            # also enhanced if it exists
            enh_path = png_image_path.replace(".png", "_enhanced.png")
            if os.path.exists(enh_path):
                shutil.copy(enh_path, os.path.join(out_dir, "imagery", os.path.basename(enh_path)))
            
        # 4. Summary JSON
        summary_path = os.path.join(out_dir, "summary.json")
        summary_data = {
            "event_id": result.input.event_id if result.input.event_id else "",
            "fire_latitude": result.input.latitude if result.input.event_id else None,
            "fire_longitude": result.input.longitude if result.input.event_id else None,
            "imagery_source": getattr(result.imagery, "provider", ""),
            "imagery_resolution_m": getattr(result.imagery, "native_resolution_m", 10.0),
            "institution": result.input.institution_name,
            "city": getattr(result.input, "city", ""),
            "input_latitude": result.input.latitude,
            "input_longitude": result.input.longitude,
            "candidate_buildings": result.institution.candidate_buildings,
            "google_candidates": result.institution.google_candidates,
            "microsoft_candidates": result.institution.microsoft_candidates,
            "duplicates_removed": result.institution.duplicates_removed,
            "outside_aoi": result.institution.outside_aoi,
            "invalid_geometries": result.institution.invalid_geometries,
            "selected_institution_buildings": result.institution.selected_buildings,
            "measured_building_count": result.institution.measured_buildings,
            "rendered_building_count": result.institution.rendered_buildings,
            "exported_building_count": result.institution.exported_buildings,
            "final_building_count": len(result.buildings),
            "selection": {
                "method": result.institution.selection_method,
                "cluster_count": result.institution.cluster_count,
                "selected_cluster": result.institution.selected_cluster_id,
                "confidence": result.institution.confidence_score
            },
            "selected_bounds": result.institution.selected_bounds,
            "measurements": {
                "total_building_area_m2": result.institution.total_building_footprint_area_sq_m,
                "total_building_area_ft2": result.institution.total_building_footprint_area_sq_ft
            },
            "imagery": {
                "source": getattr(result.imagery, "provider", ""),
                "product_id": getattr(result.imagery, "product_id", ""),
                "acquisition_datetime": getattr(result.imagery, "acquisition_datetime", ""),
                "bands": getattr(result.imagery, "bands", [])
            },
            "visualization": {
                "width_m": 500.0,
                "height_m": 500.0,
                "image_width": getattr(result.imagery, "image_width", 0),
                "image_height": getattr(result.imagery, "image_height", 0),
                "image_bounds": getattr(result.imagery, "image_bounds", []),
                "image_format": "PNG",
                "imagery_source": getattr(result.imagery, "provider", ""),
                "native_resolution_m": getattr(result.imagery, "native_resolution_m", 10.0),
                "visual_resolution_m": 500.0 / 4800.0,
                "image_path": os.path.join(out_dir, "imagery", os.path.basename(png_image_path)) if png_image_path else ""
            },
            "status": result.status,
            "errors_or_warnings": result.error if result.error else ""
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=4)
            
        return out_dir
