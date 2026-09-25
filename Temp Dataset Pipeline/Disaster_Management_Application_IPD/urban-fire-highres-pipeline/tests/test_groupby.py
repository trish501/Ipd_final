import pandas as pd
import numpy as np

# Create a dataframe with 5 points
df = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
# Simulate a left join where point 2 matches twice, point 3 matches zero, point 0 matches once
# Index: 0, 1, 1, 2, 3, 4
joined = pd.DataFrame({'index_right': [10, 20, 30, np.nan, 40, np.nan]}, index=[0, 1, 1, 2, 3, 4])
res = joined['index_right'].notna().groupby(joined.index).any().values
print("Values:", res)
