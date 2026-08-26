import math
from shapely.ops import transform, unary_union


from dimensions.models import SIHBuilding
from dimensions.area_calculator import sq_meters_to_sq_ft

def measure_building(building, project_transformer) -> SIHBuilding:
    geom = building.geometry
    if geom.is_empty:
        raise ValueError("Geometry is empty")
    if not geom.is_valid:
        from shapely.validation import make_valid
        geom = make_valid(geom)
        if geom.geom_type == 'GeometryCollection':
            from shapely.ops import unary_union
            # extract only polygons
            polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            geom = unary_union(polys)

    geom_m = transform(project_transformer, geom)
    
    mbb = geom_m.minimum_rotated_rectangle
    coords = list(mbb.exterior.coords)
    edge1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    edge2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    length = max(edge1, edge2)
    width = min(edge1, edge2)
    
    footprint_area_m2 = geom_m.area
    footprint_area_sq_ft = sq_meters_to_sq_ft(footprint_area_m2)
    
    bounding_rectangle_area_m2 = length * width
    bounding_rectangle_area_sq_ft = sq_meters_to_sq_ft(bounding_rectangle_area_m2)
    
    # Coverage is footprint / rectangle
    cov_ratio = footprint_area_m2 / bounding_rectangle_area_m2 if bounding_rectangle_area_m2 > 0 else 0.0
            

    return SIHBuilding(
        building_id=building.building_id,
        footprint_area_sq_m=footprint_area_m2,
        footprint_area_sq_ft=footprint_area_sq_ft,
        bounding_rectangle_area_sq_m=bounding_rectangle_area_m2,
        bounding_rectangle_area_sq_ft=bounding_rectangle_area_sq_ft,
        perimeter_m=geom_m.length,
        length_m=length,
        width_m=width,
        centroid={"latitude": building.centroid.y, "longitude": building.centroid.x},
        coverage_ratio=cov_ratio,
        source=building.source,
        geometry_wgs84=geom
    )

def calculate_union_area(buildings: list, lat: float, lon: float, project_transformer):
    if not buildings:
        return 0.0, 0.0, None
        
    geoms = [b.geometry_wgs84 for b in buildings]
    union_geom = unary_union(geoms)
    
    union_geom_m = transform(project_transformer, union_geom)
    
    area_m2 = union_geom_m.area
    return area_m2, sq_meters_to_sq_ft(area_m2), union_geom
