from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from shapely.geometry import Polygon, Point
from src.models.enums import CoverageStatus, VisualStatus

@dataclass
class Building:
    """
    Represents building information retrieved from a geographic data source.
    """
    building_id: str
    geometry: Polygon  # shapely Polygon representing the footprint
    centroid: Point    # shapely Point representing the centroid
    source: str
    source_identifier: Optional[str] = None
    confidence: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BuildingSearchResult:
    status: Any  # "BuildingRetrievalState"
    source: str
    buildings: List[Building] = field(default_factory=list)
    query_metadata: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

@dataclass
class ImageryResult:
    status: Any  # "PipelineState"
    provider: str
    product_id: str
    acquisition_datetime: Optional[datetime]
    cloud_cover: Optional[float]
    crs_epsg: int
    image_width: int
    image_height: int
    image_footprint_wgs84: Polygon
    native_resolution_m: float
    rgb_image_path: str
    nir_image_path: Optional[str] = None
    swir_image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildingInfo:
    """
    Analyzed geometric and source information for a building footprint.
    Does not contain any estimated occupancy/population/floors to maintain strict adherence.
    """
    source: str
    source_identifier: Optional[str]
    confidence: Optional[float]
    original_geometry_wgs84: Polygon
    centroid_lat: float
    centroid_lon: float
    area_sq_meters: float
    perimeter_meters: float
    min_rotated_rect_wgs84: Polygon
    long_axis_meters: float
    short_axis_meters: float
    geometry_valid: bool
    
    # Image coverage fields
    coverage_status: Optional[CoverageStatus] = None
    coverage_ratio: float = 0.0
    
    # Visual Verification fields
    visual_status: Optional[VisualStatus] = None
    rejection_reason: Optional[str] = None
    visual_evidence_score: float = 0.0
    pixel_support: str = "UNKNOWN"
    source_agreement: str = "UNKNOWN"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import math
        if not self.geometry_valid:
            raise ValueError("Geometry must be valid to instantiate BuildingInfo")
        if self.area_sq_meters < 0:
            raise ValueError(f"Area must be non-negative, got {self.area_sq_meters}")
        if self.perimeter_meters < 0:
            raise ValueError(f"Perimeter must be non-negative, got {self.perimeter_meters}")
        if not math.isfinite(self.long_axis_meters) or self.long_axis_meters < 0:
            raise ValueError(f"Long axis must be finite and non-negative, got {self.long_axis_meters}")
        if not math.isfinite(self.short_axis_meters) or self.short_axis_meters < 0:
            raise ValueError(f"Short axis must be finite and non-negative, got {self.short_axis_meters}")
        if not math.isfinite(self.centroid_lat) or not (-90.0 <= self.centroid_lat <= 90.0):
            raise ValueError(f"Invalid centroid_lat: {self.centroid_lat}")
        if not math.isfinite(self.centroid_lon) or not (-180.0 <= self.centroid_lon <= 180.0):
            raise ValueError(f"Invalid centroid_lon: {self.centroid_lon}")
        if not (0.0 <= self.coverage_ratio <= 1.0):
            raise ValueError(f"Invalid coverage_ratio: {self.coverage_ratio}")
