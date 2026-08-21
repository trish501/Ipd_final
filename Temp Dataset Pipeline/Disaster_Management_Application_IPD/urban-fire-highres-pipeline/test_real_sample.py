import os
from pystac_client import Client
import planetary_computer
from src.image_downloader import download_and_crop_image

def main():
    print("Connecting to Planetary Computer...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # 2021 Dixie Fire in California (approximate coordinates)
    lat, lon = 40.08, -121.15
    
    # Search for Sentinel-2 L2A images
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime="2021-08-01/2021-08-10",
        query={"eo:cloud_cover": {"lt": 20}}
    )
    
    items = list(search.items())
    if not items:
        print("No items found.")
        return
        
    item = items[0]
    print(f"Found item: {item.id}")
    
    event_meta = {
        "latitude": lat,
        "longitude": lon,
        "event_id": "test_dixie_fire",
        "date": "2021-08-05",
        "time": "18:00:00"
    }
    
    out_dir = "dataset/test_dixie_fire"
    print(f"Processing image and saving to {out_dir}...")
    
    metadata = download_and_crop_image(
        item=item,
        lat=lat,
        lon=lon,
        event_id="test_dixie_fire",
        out_dir=out_dir,
        event_meta=event_meta
    )
    
    if metadata and "error" not in metadata:
        print("Success! Metadata:")
        print(metadata)
    else:
        print("Failed or rejected:", metadata)

if __name__ == "__main__":
    main()
