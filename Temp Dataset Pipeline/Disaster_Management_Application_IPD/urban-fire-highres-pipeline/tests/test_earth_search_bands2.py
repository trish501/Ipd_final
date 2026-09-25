import pystac_client
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(collections=["sentinel-2-l2a"], max_items=50)
for item in search.items():
    if "swir22" not in item.assets:
        print("Missing swir22 in:", item.id)
        print("Keys:", item.assets.keys())
print("Done")
