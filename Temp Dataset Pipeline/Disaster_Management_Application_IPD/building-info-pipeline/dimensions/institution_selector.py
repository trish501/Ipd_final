import logging
from typing import List, Tuple

from dimensions.building_sources.google_open_buildings import GoogleOpenBuildingsBatchSource
from dimensions.building_sources.microsoft_footprints import MicrosoftBuildingFootprintsBatchSource
from dimensions.building_sources.cv_segmentation import CVSegmentationSource

logger = logging.getLogger(__name__)

class InstitutionSelector:
    def __init__(self):
        self.google_source = GoogleOpenBuildingsBatchSource()
        self.ms_source = MicrosoftBuildingFootprintsBatchSource()
        self.cv_source = CVSegmentationSource()
        
    def get_candidate_buildings(self, image_bounds: Tuple[float, float, float, float], rgb_image_path: str = None) -> List:
        from shapely.geometry import box
        bounds_geom = box(*image_bounds)
        # Use centroid of bounds for partition resolution
        lat = (image_bounds[1] + image_bounds[3]) / 2.0
        lon = (image_bounds[0] + image_bounds[2]) / 2.0
        
        logger.info(f"Retrieving partitions for image bounds {image_bounds}...")
        all_candidates_tree, all_candidates_list = self._fetch_buildings(lat, lon, image_bounds)
        
        buildings = []
        if all_candidates_tree and all_candidates_list:
            indices = all_candidates_tree.query(bounds_geom)
            for i in indices:
                if all_candidates_list[i].geometry.intersects(bounds_geom):
                    buildings.append(all_candidates_list[i])
                    
        # Apply the CV segmentation source dynamically
        if rgb_image_path:
            cv_buildings = self.cv_source.get_buildings_from_image(rgb_image_path, image_bounds)
            if cv_buildings:
                buildings.extend(cv_buildings)
                
        return buildings
        
    def _fetch_buildings(self, lat, lon, bounds):
        """Fetches all candidates from Google and MS within the max bounds and returns a unified STRtree/list."""
        buildings = []
        try:
            token = self.google_source.resolve_partition(lat, lon)
            if self.google_source.download_partition(token):
                tree, partition_buildings = self.google_source.parse_and_index(token, bounds)
                if partition_buildings:
                    buildings.extend(partition_buildings)
        except Exception as e:
            logger.error(f"Google retrieval failed: {e}")
            
        try:
            urls = self.ms_source.resolve_partitions(lat, lon)
            for url in urls:
                if self.ms_source.download_partition(url):
                    tree, partition_buildings = self.ms_source.parse_and_index(url, bounds)
                    if partition_buildings:
                        buildings.extend(partition_buildings)
        except Exception as e:
            logger.error(f"Microsoft retrieval failed: {e}")
            
        if not buildings:
            return None, []
            
        # Deduplicate using a dictionary keyed by wkt or centroid string
        # Actually MS and Google might overlap slightly, but we just return them all and deduplicate later
        # Or better yet, build a single STRtree
        from shapely.strtree import STRtree
        geometries = [b.geometry for b in buildings]
        tree = STRtree(geometries)
        return tree, buildings

