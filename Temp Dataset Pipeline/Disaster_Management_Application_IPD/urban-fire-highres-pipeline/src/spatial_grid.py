def get_grid_cell(lat: float, lon: float, grid_size_km: float = 5.0) -> tuple:
    """
    Converts continuous coordinates into discrete spatial buckets based on the V1 algorithm.
    1 degree latitude is approximately 111km.
    """
    cell_size_deg = (grid_size_km * 1000.0) / 111_000.0
    cell_x = int(lon / cell_size_deg)
    cell_y = int(lat / cell_size_deg)
    return (cell_x, cell_y)

