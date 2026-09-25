import time
import pandas as pd
import numpy as np
import geopandas as gpd
from src.offline_urban_filter import init_offline_filter, _urban_gdf
from src.offline_industrial_filter import init_industrial_filter, _industrial_gdf, INDUSTRIAL_BUFFER_DEG

init_offline_filter()
init_industrial_filter()

N = 100000 # Test with 100k
lats = np.random.uniform(-90, 90, N)
lons = np.random.uniform(-180, 180, N)

start = time.time()
points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
joined = gpd.sjoin(points, _urban_gdf, how="left", predicate="intersects")
is_urban = joined['index_right'].notna().groupby(joined.index).any().tolist()
print("Urban check time:", time.time() - start)

start = time.time()
ind_buffered = gpd.GeoDataFrame(geometry=_industrial_gdf.geometry.buffer(INDUSTRIAL_BUFFER_DEG), crs="EPSG:4326")
joined = gpd.sjoin(points, ind_buffered, how="left", predicate="intersects")
is_industrial = joined['index_right'].notna().groupby(joined.index).any().tolist()
print("Industrial check time:", time.time() - start)

print("Lens:", len(is_urban), len(is_industrial))
