import argparse
import os
import csv
import sys
import time
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.config import settings
from src.utils.logger import logger
from src.models.input import InputEvent
from src.models.aoi import AOI
from src.models.output import Building, ImageryResult
from src.models.enums import PipelineState, CoverageStatus
from src.building_sources.google_open_buildings import GoogleOpenBuildingsBatchSource
from src.building_sources.microsoft_footprints import MicrosoftBuildingFootprintsBatchSource
from src.building_sources.imagery import Sentinel2STACSource
from src.localization.building_manager import BuildingManager
from src.localization.visual_validator import VisualValidator
from src.exporter.event_exporter import EventExporter

def parse_args():
    parser = argparse.ArgumentParser(description="High-Speed Building Discovery Pipeline")
    parser.add_argument("--batch", type=str, help="Path to CSV containing event_id,lat,lon")
    parser.add_argument("--event-id", type=str, help="Single event ID")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--radius", type=float, default=100.0, help="Search radius in meters")
    parser.add_argument("--workers", type=int, default=4, help="Max download workers")
    parser.add_argument("--verbose", action="store_true", help="Print detailed timings")
    parser.add_argument("--benchmark", action="store_true", help="Print benchmark summary")
    return parser.parse_args()

class BatchPipelineOrchestrator:
    def __init__(self, radius: float, workers: int, verbose: bool):
        self.radius = radius
        self.workers = workers
        self.verbose = verbose
        self.benchmark = False
        self.mode = settings.building_source_mode
        self.google_source = GoogleOpenBuildingsBatchSource()
        self.ms_source = MicrosoftBuildingFootprintsBatchSource()
        self.imagery_source = Sentinel2STACSource()
        self.visual_validator = VisualValidator()
        self.exporter = EventExporter()
        self.building_manager = BuildingManager([]) # Using solely for _deduplicate logic

    def _get_batch_bounds(self, events: List[InputEvent], im_results: Dict[str, ImageryResult]) -> Tuple[float, float, float, float]:
        minx, miny, maxx, maxy = float('inf'), float('inf'), -float('inf'), -float('inf')
        for e in events:
            bounds = im_results[e.event_id].image_footprint_wgs84.bounds
            minx = min(minx, bounds[0])
            miny = min(miny, bounds[1])
            maxx = max(maxx, bounds[2])
            maxy = max(maxy, bounds[3])
        return minx, miny, maxx, maxy

    def run(self, events: List[InputEvent]) -> Dict[str, dict]:
        t0 = time.time()
        timings = defaultdict(float)
        
        # 1. AOI Generation
        t_start = time.time()
        aois = {e.event_id: AOI.create_from_point(e.latitude, e.longitude, self.radius) for e in events}
        timings["aoi_generation"] = time.time() - t_start

        # 1.5. Imagery Retrieval
        t_start = time.time()
        imagery_results = {}
        
        def download_imagery(e: InputEvent):
            out_dir = os.path.join(self.exporter.base_output_dir, e.event_id)
            return e.event_id, self.imagery_source.retrieve_imagery(aois[e.event_id], e.event_id, out_dir)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_e = {executor.submit(download_imagery, e): e for e in events}
            for future in concurrent.futures.as_completed(future_to_e):
                eid, res = future.result()
                imagery_results[eid] = res
        
        # Filter to only successful imagery events
        successful_events = [e for e in events if imagery_results[e.event_id].status == PipelineState.SUCCESS]
        if len(successful_events) < len(events):
            logger.warning(f"{len(events) - len(successful_events)} events failed RGB retrieval and will be skipped.")
            
        # Write failure outputs for failed events immediately
        for e in events:
            if imagery_results[e.event_id].status != PipelineState.SUCCESS:
                self.exporter.export(e, aois[e.event_id], imagery_results[e.event_id], [])
                
        events = successful_events
        if not events:
            return {e.event_id: 0 for e in events}
            
        timings["imagery_retrieval"] = time.time() - t_start

        # 2. Partition Resolution (Google)
        t_start = time.time()
        google_partitions = defaultdict(list)
        for e in events:
            token = self.google_source.resolve_partition(e.latitude, e.longitude)
            google_partitions[token].append(e)
        unique_google_tokens = list(google_partitions.keys())
        timings["google_resolution"] = time.time() - t_start

        # 3. Download Google Partitions
        t_start = time.time()
        def download_google(token):
            return token, self.google_source.download_partition(token)
            
        google_download_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            for token, success in executor.map(download_google, unique_google_tokens):
                google_download_results[token] = success
        timings["google_download"] = time.time() - t_start

        # 4. Parse & Index Google, Query Events
        t_start = time.time()
        results_by_event = {e.event_id: [] for e in events}
        google_success_events = set()
        
        for token in unique_google_tokens:
            if not google_download_results.get(token):
                continue
            token_events = google_partitions[token]
            # Only include successful events
            token_events = [e for e in token_events if e.event_id in [se.event_id for se in successful_events]]
            if not token_events:
                continue
            bounds = self._get_batch_bounds(token_events, imagery_results)
            tree, partition_buildings = self.google_source.parse_and_index(token, bounds)
            
            if tree and partition_buildings:
                for e in token_events:
                    im_geom = imagery_results[e.event_id].image_footprint_wgs84
                    indices = tree.query(im_geom)
                    found_any = False
                    for i in indices:
                        b = partition_buildings[i]
                        # Exact intersection check with IMAGE footprint
                        if b.geometry.intersects(im_geom):
                            results_by_event[e.event_id].append(b)
                            found_any = True
                    if found_any:
                        google_success_events.add(e.event_id)
        timings["google_parse_and_query"] = time.time() - t_start

        # 5. Microsoft Fallback
        t_start = time.time()
        events_needing_ms = []
        if self.mode == "FASTEST":
            events_needing_ms = [e for e in events if e.event_id not in google_success_events]
        else: # MAX_COVERAGE
            events_needing_ms = events
            
        if events_needing_ms:
            ms_partitions = defaultdict(list)
            for e in events_needing_ms:
                urls = self.ms_source.resolve_partitions(e.latitude, e.longitude)
                for url in urls:
                    ms_partitions[url].append(e)
            
            unique_ms_urls = list(ms_partitions.keys())
            
            def download_ms(url):
                return url, self.ms_source.download_partition(url)
                
            ms_download_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                for url, success in executor.map(download_ms, unique_ms_urls):
                    ms_download_results[url] = success
                    
            for url in unique_ms_urls:
                if not ms_download_results.get(url):
                    continue
                    
                url_events = ms_partitions[url]
                # Filter successful
                url_events = [e for e in url_events if e.event_id in [se.event_id for se in successful_events]]
                if not url_events:
                    continue
                bounds = self._get_batch_bounds(url_events, imagery_results)
                tree, partition_buildings = self.ms_source.parse_and_index(url, bounds)
                
                if tree and partition_buildings:
                    for e in url_events:
                        im_geom = imagery_results[e.event_id].image_footprint_wgs84
                        indices = tree.query(im_geom)
                        for i in indices:
                            b = partition_buildings[i]
                            if b.geometry.intersects(im_geom):
                                results_by_event[e.event_id].append(b)
        timings["microsoft_fallback"] = time.time() - t_start

        # 6. Deduplicate & Export
        t_start = time.time()
        final_counts = {}
        for e in events:
            buildings = results_by_event[e.event_id]
            
            # Map Building -> BuildingInfo for export
            from src.models.output import BuildingInfo
            b_infos = []
            
            im_geom = imagery_results[e.event_id].image_footprint_wgs84
            
            for b in buildings:
                # Calculate dimensions FIRST so visual validator can use them
                import pyproj
                from shapely.ops import transform
                utm_zone = int((e.longitude + 180) / 6) + 1
                hemisphere = 'north' if e.latitude >= 0 else 'south'
                proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', **({} if hemisphere == 'north' else {'south': True}))
                project = pyproj.Transformer.from_proj(pyproj.Proj('epsg:4326'), proj, always_xy=True).transform
                geom_m = transform(project, b.geometry)
                
                mbb = geom_m.minimum_rotated_rectangle
                coords = list(mbb.exterior.coords)
                import math
                edge1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
                edge2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
                length = max(edge1, edge2)
                width = min(edge1, edge2)
                
                cov_status = CoverageStatus.FULLY_WITHIN_IMAGE if im_geom.contains(b.geometry) else CoverageStatus.PARTIALLY_WITHIN_IMAGE

                bi = BuildingInfo(
                    source=b.source,
                    source_identifier=b.source_identifier,
                    confidence=b.confidence,
                    original_geometry_wgs84=b.geometry,
                    centroid_lat=b.centroid.y,
                    centroid_lon=b.centroid.x,
                    area_sq_meters=geom_m.area,
                    perimeter_meters=geom_m.length,
                    min_rotated_rect_wgs84=b.geometry.minimum_rotated_rectangle,
                    long_axis_meters=length,
                    short_axis_meters=width,
                    geometry_valid=True,
                    coverage_status=cov_status,
                    coverage_ratio=1.0 if cov_status == CoverageStatus.FULLY_WITHIN_IMAGE else (b.geometry.intersection(im_geom).area / b.geometry.area)
                )
                b_infos.append(bi)
                
            if len(b_infos) > 1:
                b_infos = self.building_manager._deduplicate_infos(b_infos)
                
            # Perform Visual Validation on candidates
            if b_infos:
                self.visual_validator.validate_batch(b_infos, imagery_results[e.event_id])
                
            # Stage 17: Diagnostic Summary
            total_cand = len(buildings)
            dedup_cand = len(b_infos)
            
            insuf_px = sum(1 for b in b_infos if getattr(b, "pixel_support", "") == "INSUFFICIENT")
            veg_rej = sum(1 for b in b_infos if b.attributes.get("VEGETATION_REJECTION", False))
            water_rej = sum(1 for b in b_infos if b.attributes.get("WATER_REJECTION", False))
            shadow_rej = sum(1 for b in b_infos if b.attributes.get("SHADOW_REJECTION", False))
            no_struct = sum(1 for b in b_infos if getattr(b, "pixel_support", "") != "INSUFFICIENT" and not b.attributes.get("STRUCTURAL_EVIDENCE", True) and not b.attributes.get("VEGETATION_REJECTION", False) and not b.attributes.get("WATER_REJECTION", False) and not b.attributes.get("SHADOW_REJECTION", False))
            
            from src.models.enums import VisualStatus
            v_bld = sum(1 for b in b_infos if b.visual_status == VisualStatus.VERIFIED_BUILDING)
            p_bld = sum(1 for b in b_infos if b.visual_status == VisualStatus.PROBABLE_BUILDING)
            u_bld = sum(1 for b in b_infos if b.visual_status == VisualStatus.UNRESOLVED)
            r_bld = sum(1 for b in b_infos if b.visual_status == VisualStatus.REJECTED_NON_BUILDING)
            
            print(f"\nEVENT {e.event_id}")
            print(f"Candidates: {total_cand}")
            print(f"Deduplicated: {dedup_cand}")
            print(f"Insufficient pixels: {insuf_px}")
            print(f"Vegetation: {veg_rej}")
            print(f"Water: {water_rej}")
            print(f"Shadow: {shadow_rej}")
            print(f"Insufficient structural evidence: {no_struct}")
            print()
            print(f"VERIFIED_BUILDING: {v_bld}")
            print(f"PROBABLE_BUILDING: {p_bld}")
            print(f"UNRESOLVED: {u_bld}")
            print(f"REJECTED: {r_bld}")
                
            self.exporter.export(e, aois[e.event_id], imagery_results[e.event_id], b_infos)
            final_counts[e.event_id] = v_bld
        timings["dedup_and_export"] = time.time() - t_start
        
        timings["total"] = time.time() - t0
        
        if self.verbose or self.benchmark:
            print("\n" + "="*40)
            print(f"BATCH PERFORMANCE SUMMARY ({len(events)} events)")
            print("="*40)
            print(f"Total time:       {timings['total']:.2f}s")
            print(f"AOI gen:          {timings['aoi_generation']:.2f}s")
            print(f"Imagery fetch:    {timings['imagery_retrieval']:.2f}s")
            print(f"Google resolve:   {timings['google_resolution']:.2f}s")
            print(f"Google download:  {timings['google_download']:.2f}s")
            print(f"Google process:   {timings['google_parse_and_query']:.2f}s")
            print(f"Microsoft fallback:{timings['microsoft_fallback']:.2f}s")
            print(f"Dedup & Export:   {timings['dedup_and_export']:.2f}s")
            print("="*40)
            print(f"Total Buildings found across batch: {sum(final_counts.values())}")
            
        return final_counts

def main():
    args = parse_args()
    events = []
    
    if args.batch:
        with open(args.batch, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(InputEvent(
                    event_id=row["event_id"],
                    latitude=float(row["lat"]),
                    longitude=float(row["lon"])
                ))
    elif args.event_id and args.lat is not None and args.lon is not None:
        events.append(InputEvent(
            event_id=args.event_id,
            latitude=args.lat,
            longitude=args.lon
        ))
    else:
        print("Error: Must provide either --batch or (--event-id, --lat, --lon)")
        sys.exit(1)
        
    print(f"Processing {len(events)} events (Radius: {args.radius}m, Mode: {settings.building_source_mode})")
    
    orchestrator = BatchPipelineOrchestrator(
        radius=args.radius,
        workers=args.workers,
        verbose=args.verbose or args.benchmark
    )
    
    orchestrator.benchmark = args.benchmark
    orchestrator.run(events)

if __name__ == "__main__":
    main()
