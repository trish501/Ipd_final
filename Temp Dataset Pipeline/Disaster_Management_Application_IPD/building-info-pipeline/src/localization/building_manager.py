from src.models.output import BuildingInfo
from src.models.output import Building
from typing import List, Tuple
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from src.models.aoi import AOI
from src.models.enums import BuildingRetrievalState
from src.building_sources.base import BaseBuildingSource
from src.building_sources.google_open_buildings import GoogleOpenBuildingsBatchSource
from src.building_sources.microsoft_footprints import MicrosoftBuildingFootprintsBatchSource
from src.utils.logger import logger

class BuildingManager:
    def __init__(self, providers: List[BaseBuildingSource] = None):
        if providers is None:
            self.providers = [
                GoogleOpenBuildingsBatchSource(),
                MicrosoftBuildingFootprintsBatchSource()
            ]
        else:
            self.providers = providers

    def _deduplicate_infos(self, buildings: List['BuildingInfo'], iou_threshold: float = 0.3) -> List['BuildingInfo']:
        """
        Deduplicate BuildingInfo objects using IoU on projected geometries.
        Sorts by coverage ratio, then area (descending) so better candidates are kept.
        """
        if not buildings:
            return []

        logger.info(f"Deduplicating {len(buildings)} buildings...")
        
        # Sort buildings: higher coverage first, then larger area, then prefer Google
        def sort_key(b):
            is_google = 1 if 'Google' in b.source else 0
            return (b.coverage_ratio, is_google, b.area_sq_meters)
            
        sorted_buildings = sorted(buildings, key=sort_key, reverse=True)
        
        retained = []
        geometries = [b.original_geometry_wgs84 for b in sorted_buildings]
        tree = STRtree(geometries)
        duplicate_indices = set()
        
        for i, b in enumerate(sorted_buildings):
            if i in duplicate_indices:
                continue
                
            # Default to its own source
            has_google = "Google" in b.source
            has_ms = "Microsoft" in b.source
                
            intersecting_indices = tree.query(b.original_geometry_wgs84)
            for j in intersecting_indices:
                if i == j:
                    continue
                    
                other = sorted_buildings[j]
                intersection = b.original_geometry_wgs84.intersection(other.original_geometry_wgs84).area
                union = b.original_geometry_wgs84.union(other.original_geometry_wgs84).area
                
                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        if j not in duplicate_indices:
                            duplicate_indices.add(j)
                        if "Google" in other.source:
                            has_google = True
                        if "Microsoft" in other.source:
                            has_ms = True
                            
            if has_google and has_ms:
                b.source_agreement = "BOTH"
            elif has_google:
                b.source_agreement = "GOOGLE_ONLY"
            elif has_ms:
                b.source_agreement = "MICROSOFT_ONLY"
            else:
                b.source_agreement = "UNKNOWN"
                
            retained.append(b)
            
        logger.info(f"Deduplication complete. Retained {len(retained)} out of {len(buildings)} buildings.")
        return retained

    def retrieve_buildings(self, image_footprint: Polygon) -> Tuple[BuildingRetrievalState, List[Building]]:
        """
        Queries all available providers using a bounding box derived from the image footprint.
        Merges results and deduplicates overlapping geometries.
        """
        # Create an AOI exactly bounding the image footprint
        minx, miny, maxx, maxy = image_footprint.bounds
        
        # The underlying providers expect an AOI object for search logic
        # We manually craft one that bounds the footprint
        aoi = AOI.create_from_point(
            lat=(miny + maxy) / 2.0,
            lon=(minx + maxx) / 2.0,
            buffer_meters=10  # Dummy buffer, we'll override the geometry
        )
        aoi.geometry_wgs84 = Polygon.from_bounds(minx, miny, maxx, maxy)
        
        all_buildings = []
        last_failure_state = BuildingRetrievalState.SOURCE_COVERAGE_UNAVAILABLE
        
        any_success = False
        
        for provider in self.providers:
            logger.info(f"Querying {provider.name} for image footprint bounds...")
            result = provider.search(aoi)
            
            if result.status == BuildingRetrievalState.BUILDINGS_FOUND:
                logger.info(f"Found {len(result.buildings)} intersecting buildings from {provider.name}.")
                all_buildings.extend(result.buildings)
                any_success = True
                
            elif result.status == BuildingRetrievalState.NO_BUILDINGS_FOUND:
                logger.info(f"{provider.name} query succeeded but returned zero buildings.")
                any_success = True
                
            else:
                logger.warning(f"{provider.name} failed with state {result.status.value}: {result.failure_reason}")
                if result.status in (BuildingRetrievalState.SOURCE_ACCESS_FAILED, BuildingRetrievalState.SOURCE_QUERY_FAILED):
                    last_failure_state = result.status

        if not any_success:
            logger.error(f"All building providers failed. Last failure: {last_failure_state.value}")
            return last_failure_state, []
            
        if not all_buildings:
            return BuildingRetrievalState.NO_BUILDINGS_FOUND, []

        # Deduplicate
        deduplicated = self._deduplicate(all_buildings)
        return BuildingRetrievalState.BUILDINGS_FOUND, deduplicated
