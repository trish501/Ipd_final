from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from shapely.geometry import Polygon, Point

@dataclass
class Building:
    building_id: str
    geometry: Polygon
    centroid: Point
    source: str
    source_identifier: Optional[str] = None
    confidence: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BuildingSearchResult:
    status: Any
    source: str
    buildings: List[Building] = field(default_factory=list)
    query_metadata: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

@dataclass
class SIHInput:
    latitude: float
    longitude: float
    event_id: Optional[str] = None
    institution_name: Optional[str] = None
    city: Optional[str] = None
    event_dir: Optional[str] = None

@dataclass
class SIHBuilding:
    building_id: str
    perimeter_m: float
    length_m: float
    width_m: float
    footprint_area_sq_m: float
    footprint_area_sq_ft: float
    bounding_rectangle_area_sq_m: float
    bounding_rectangle_area_sq_ft: float
    centroid: Dict[str, float]
    coverage_ratio: float
    source: str
    geometry_wgs84: Any = None
    
@dataclass
class SIHImagerySummary:
    provider: str = ""
    product_id: str = ""
    acquisition_datetime: str = ""
    cloud_cover: float = 0.0
    bands: List[str] = field(default_factory=list)
    preprocessing: str = ""
    native_resolution_m: float = 10.0
    image_width: int = 0
    image_height: int = 0
    image_bounds: List[float] = field(default_factory=list)

@dataclass
class SIHInstitutionSummary:
    candidate_buildings: int = 0
    google_candidates: int = 0
    microsoft_candidates: int = 0
    duplicates_removed: int = 0
    outside_aoi: int = 0
    invalid_geometries: int = 0
    selected_buildings: int = 0
    measured_buildings: int = 0
    rendered_buildings: int = 0
    exported_buildings: int = 0
    total_building_footprint_area_sq_m: float = 0.0
    total_building_footprint_area_sq_ft: float = 0.0
    institution_aoi_area_sq_m: float = 0.0
    institution_aoi_area_sq_ft: float = 0.0
    
    # New Selection Diagnostics
    anchor_latitude: float = 0.0
    anchor_longitude: float = 0.0
    cluster_count: int = 0
    selected_cluster_id: str = ""
    selection_method: str = ""
    confidence_score: float = 0.0
    selected_bounds: Dict[str, float] = field(default_factory=dict)
    cluster_warning: Optional[str] = None

@dataclass
class SIHResult:
    status: str
    input: SIHInput
    imagery: SIHImagerySummary
    institution: SIHInstitutionSummary
    buildings: List[SIHBuilding]
    error: Optional[str] = None
