import geopandas as gpd
import pandas as pd
import numpy as np
from src.offline_urban_filter import init_offline_filter
import src.offline_urban_filter as filter

init_offline_filter()
print("Loaded:", filter._urban_gdf.shape if filter._urban_gdf is not None else "None")

N = 1000000
lats = np.random.uniform(-90, 90, N)
lons = np.random.uniform(-180, 180, N)

import time
start = time.time()
points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
joined = gpd.sjoin(points, filter._urban_gdf, how="left", predicate="intersects")
is_urban = joined['index_right'].notna().groupby(joined.index).any().tolist()
print("Time taken for 1M points:", time.time() - start)
