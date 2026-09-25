import pystac_client
import rasterio
import rasterio.env

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(collections=["sentinel-2-l2a"], max_items=1)
items = list(search.items())
if items:
    item = items[0]
    href = item.assets["red"].href
    print("Href:", href)
    
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        try:
            with rasterio.open(href) as src:
                print("Read shape:", src.shape)
        except Exception as e:
            print("Error with NO_SIGN_REQUEST:", e)
