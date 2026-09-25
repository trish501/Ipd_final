import pystac_client
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(collections=["sentinel-2-l2a"], max_items=1)
items = list(search.items())
if items:
    print(list(items[0].assets.keys()))
