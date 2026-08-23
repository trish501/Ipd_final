import pyproj
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from dataclasses import dataclass

@dataclass
class AOI:
    """
    Represents an Area of Interest (AOI) strictly constructed using metric projection,
    then converted back to WGS84 for standardized querying.
    """
    geometry_wgs84: Polygon
    area_sq_meters: float

    @classmethod
    def create_from_point(cls, lat: float, lon: float, buffer_meters: float) -> 'AOI':
        """
        Creates a square/polygon AOI around a point by buffering in a local projected CRS,
        ensuring distances and areas are metrically accurate, avoiding lat/lon degree distortion.
        """
        # 1. Define WGS84
        wgs84 = pyproj.CRS("EPSG:4326")
        
        # 2. Define a local Azimuthal Equidistant projection centered exactly on the input point.
        # This provides highly accurate metric distance for the buffer operation locally.
        aeqd_proj = pyproj.CRS(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        
        # 3. Create transformers
        # always_xy=True forces output to be (lon, lat) to match shapely x, y convention
        project_to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd_proj, always_xy=True).transform
        project_to_wgs84 = pyproj.Transformer.from_crs(aeqd_proj, wgs84, always_xy=True).transform

        # 4. Construct point (lon, lat) in WGS84
        point_wgs84 = Point(lon, lat)
        
        # 5. Project to local metric CRS
        point_aeqd = transform(project_to_aeqd, point_wgs84)
        
        # 6. Create buffer in meters.
        # cap_style=3 (square) forms a square bounding box instead of a circle, which is common for tile-based AOIs.
        # If a true circle is desired, cap_style=1 (round) is default. We'll use a circular buffer
        # unless specifically instructed otherwise, as it accurately represents "distance from point".
        # We will use a square (envelope) to make generic bounding box queries simpler.
        aoi_aeqd = point_aeqd.buffer(buffer_meters).envelope
        
        # 7. Calculate exact metric area
        area_sqm = aoi_aeqd.area
        
        # 8. Project back to WGS84 for the final model
        aoi_wgs84 = transform(project_to_wgs84, aoi_aeqd)
        
        return cls(geometry_wgs84=aoi_wgs84, area_sq_meters=area_sqm)
