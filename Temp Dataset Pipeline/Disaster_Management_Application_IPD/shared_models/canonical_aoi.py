import math
import pyproj
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class CanonicalAOI:
    """
    A single authoritative geometry definition for both Sentinel-2 and Esri imagery.
    Ensures both imagery systems cover the exact same physical area, orientation,
    rotation, and spatial extent.
    """
    center_lat: float
    center_lon: float
    width_m: float
    height_m: float
    rotation_angle_deg: float
    crs_epsg: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalAOI':
        return cls(**data)

    def get_wgs84_bounds(self):
        """
        Calculates the WGS84 bounding box (min_lon, min_lat, max_lon, max_lat)
        that completely covers this exact projected geometry.
        """
        wgs84 = pyproj.CRS("EPSG:4326")
        target_crs = pyproj.CRS(self.crs_epsg)
        transformer = pyproj.Transformer.from_crs(target_crs, wgs84, always_xy=True)

        # The 4 corners in projected space
        corners = [
            (self.min_x, self.min_y),
            (self.min_x, self.max_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y)
        ]

        lons = []
        lats = []
        for x, y in corners:
            lon, lat = transformer.transform(x, y)
            lons.append(lon)
            lats.append(lat)

        return min(lons), min(lats), max(lons), max(lats)
