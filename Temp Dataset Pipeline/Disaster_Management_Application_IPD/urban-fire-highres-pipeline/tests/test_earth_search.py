import pystac_client
from datetime import datetime, timezone
from shapely.geometry import Point

try:
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[70, 10, 70.1, 10.1],
        datetime="2025-01-01T00:00:00Z/2025-01-10T00:00:00Z",
        query={"eo:cloud_cover": {"lt": 30}}
    )
    items = list(search.items())
    if items:
        item = items[0]
        print("Found items:", len(items))
        print("B04 href:", item.assets["red"].href)
    else:
        print("No items found.")
except Exception as e:
    print("Error:", e)
