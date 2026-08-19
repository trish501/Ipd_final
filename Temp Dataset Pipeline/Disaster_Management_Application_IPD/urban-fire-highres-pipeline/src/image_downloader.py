import os
import logging
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform
from PIL import Image
import numpy as np
import json
from datetime import datetime
from src.filters import get_location

import shutil

logger = logging.getLogger(__name__)



def get_asset_href(item, asset_name):
    """Returns signed href for an asset (assuming item is already signed by planetary_computer)."""
    return item.assets[asset_name].href

def write_image_metadata(txt_path, image_type, bands_text, band_order_list, item, event_meta, crs_val, res_val):
    if not event_meta: event_meta = {}
    
    lat = event_meta.get('latitude')
    lon = event_meta.get('longitude')
    
    location_str = "Not available, Not available, Not available"
    if lat is not None and lon is not None:
        location_str = get_location(lat, lon)
    
    # get_location returns "City, State, Country"
    parts = [p.strip() for p in location_str.split(",")]
    if len(parts) >= 3:
        city, state, country = parts[0], parts[1], parts[2]
    else:
        city, state, country = "Not available", "Not available", "Not available"

    with open(txt_path, 'w') as f:
        f.write("-" * 50 + "\n")
        f.write("IMAGE METADATA\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Image Type: {image_type}\n")
        f.write("Satellite: Sentinel-2\n")
        f.write(f"Scene ID: {item.id}\n\n")
        
        f.write("Bands:\n")
        for b in bands_text:
            f.write(f"{b}\n")
            
        f.write("\nBand Order:\n")
        f.write(f"{', '.join(band_order_list)}\n\n")
        
        sat_dt = item.datetime
        f.write(f"Acquisition Date: {sat_dt.strftime('%Y-%m-%d') if sat_dt else 'Not available'}\n")
        f.write(f"Acquisition Time: {sat_dt.strftime('%H:%M:%S') if sat_dt else 'Not available'}\n\n")
        
        # Fire Event Dates explicitly populated from event_meta correctly
        f.write(f"Fire Acquisition Date: {event_meta.get('date', 'Not available')}\n")
        f.write(f"Fire Acquisition Time: {event_meta.get('time', 'Not available')}\n\n")
        
        f.write(f"Latitude: {lat if lat is not None else 'Not available'}\n")
        f.write(f"Longitude: {lon if lon is not None else 'Not available'}\n\n")
        
        f.write(f"Country: {country}\n")
        f.write(f"State: {state}\n")
        f.write(f"City: {city}\n\n")
        
        f.write(f"Event ID: {event_meta.get('event_id', 'Not available')}\n\n")
        
        f.write(f"Coordinate Reference System: {crs_val}\n\n")
        f.write(f"Spatial Resolution: {res_val}\n\n")
        
        f.write("Source: NASA FIRMS\n")
        source = event_meta.get('source_file', '')
        if "VIIRS" in source:
            f.write("Fire Data Source: VIIRS\n")
        elif "MODIS" in source:
            f.write("Fire Data Source: MODIS\n")
        else:
            f.write("Fire Data Source: Not available\n")
        f.write("\n" + "-" * 50 + "\n")

def download_and_crop_image(item, lat: float, lon: float, event_id: str, out_dir: str, crop_km: float = 2.0, output_size: int = 1024, event_meta: dict = None):
    """
    Downloads native Sentinel-2 bands, extracts them to a common 10m analysis grid, and generates RGB, SWIR composites.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "aligned_10m")
    rgb_dir = os.path.join(out_dir, "rgb")
    swir_dir = os.path.join(out_dir, "swir")
    swir_nir_dir = os.path.join(out_dir, "swir_nir")
    
    for d in [raw_dir, rgb_dir, swir_dir, swir_nir_dir]:
        os.makedirs(d, exist_ok=True)
        
    try:
        bands_to_fetch = ["B02", "B03", "B04", "B08", "B08a", "B11", "B12"]
        band_data = {}
        
        # Download native bands
        for band in bands_to_fetch:
            # Planet Computer Sentinel-2 STAC uses B02, B03, B04, B08, B8A, B11, B12
            # Notice the casing for B8A might be "B8A" or "B08a". PC uses "B08", "B8A".
            asset_name = "B8A" if band == "B08a" else band
            
            href = get_asset_href(item, asset_name)
            with rasterio.open(href) as src:
                crs = src.crs
                # Convert lat/lon to image CRS
                xs, ys = transform('EPSG:4326', crs, [lon], [lat])
                cx, cy = xs[0], ys[0]
                
                # Calculate bounds in meters
                half_size = (crop_km * 1000) / 2.0
                left, bottom, right, top = cx - half_size, cy - half_size, cx + half_size, cy + half_size
                
                # Establish 10m analysis grid exactly
                target_size = int((crop_km * 1000) / 10.0)
                
                # Get the pixel window for these bounds
                window = from_bounds(left, bottom, right, top, src.transform)
                
                # 10m bands get nearest (preserves raw reflectance), 20m bands get bilinear (interpolation)
                resampling_method = rasterio.enums.Resampling.nearest if band in ["B02", "B03", "B04", "B08"] else rasterio.enums.Resampling.bilinear
                
                # Read the data within the window onto the 10m grid
                arr = src.read(
                    window=window, 
                    out_shape=(1, target_size, target_size), 
                    resampling=resampling_method,
                    boundless=True,
                    fill_value=0
                )
                band_data[band] = arr[0]
                
                # Create a new transform for the cropped output
                new_transform = rasterio.transform.from_bounds(left, bottom, right, top, target_size, target_size)
                
                # Save 16-bit TIFF on common 10m grid
                tiff_path = os.path.join(raw_dir, f"{band}.tif")
                with rasterio.open(
                    tiff_path,
                    'w',
                    driver='GTiff',
                    height=target_size,
                    width=target_size,
                    count=1,
                    dtype=band_data[band].dtype,
                    crs=crs,
                    transform=new_transform,
                ) as dst:
                    dst.write(band_data[band], 1)
                
        def scale_to_8bit(arr):
            # Linearly scale 0-10000 to 0-255. No contrast enhancement.
            scaled = (arr / 10000.0) * 255.0
            return np.clip(scaled, 0, 255).astype(np.uint8)

        # DATA-QUALITY VALIDITY GATE
        b04_arr = band_data["B04"]
        b04_size = b04_arr.shape[0]
        total_pixels = b04_arr.size
        valid_pixels = np.sum(b04_arr > 0)
        valid_percentage = (valid_pixels / total_pixels) * 100.0
        
        center_y, center_x = b04_size // 2, b04_size // 2
        center_valid = b04_arr[center_y, center_x] > 0
        
        window_size = 20
        half_w = window_size // 2
        local_window = b04_arr[
            max(0, center_y - half_w) : min(b04_size, center_y + half_w), 
            max(0, center_x - half_w) : min(b04_size, center_x + half_w)
        ]
        local_valid_percentage = (np.sum(local_window > 0) / local_window.size) * 100.0
        
        if not center_valid:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": "Event coordinate maps to invalid/NoData Sentinel-2 raster area."}
            
        if local_valid_percentage < 75.0:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": f"Insufficient valid imagery around event coordinate (local valid < 75%). Actual: {local_valid_percentage:.1f}%"}
            
        if valid_percentage < 25.0:
            shutil.rmtree(out_dir, ignore_errors=True)
            return {"error": f"Crop contains insufficient valid imagery overall (< 25%). Actual: {valid_percentage:.1f}%"}

        # Generate RGB Composite (B04, B03, B02)
        rgb_arr = np.stack([band_data["B04"], band_data["B03"], band_data["B02"]], axis=-1)
        rgb_8bit = scale_to_8bit(rgb_arr)
        
        rgb_img = Image.fromarray(rgb_8bit, 'RGB')
        rgb_path = os.path.join(rgb_dir, "B4-B3-B2.jpg")
        rgb_img.save(rgb_path, quality=90)
        
        write_image_metadata(
            txt_path=os.path.join(rgb_dir, "B4-B3-B2.txt"),
            image_type="RGB",
            bands_text=["B04 - Red", "B03 - Green", "B02 - Blue"],
            band_order_list=["B04", "B03", "B02"],
            item=item,
            event_meta=event_meta,
            crs_val=str(crs),
            res_val="10m (Native)"
        )
        
        # Generate SWIR Composite (B12, B11, B04)
        swir_arr = np.stack([band_data["B12"], band_data["B11"], band_data["B04"]], axis=-1)
        swir_img = Image.fromarray(scale_to_8bit(swir_arr), 'RGB')
        swir_path = os.path.join(swir_dir, "B12-B11-B4.jpg")
        swir_img.save(swir_path, quality=90)
        
        write_image_metadata(
            txt_path=os.path.join(swir_dir, "B12-B11-B4.txt"),
            image_type="SWIR",
            bands_text=["B12", "B11", "B04"],
            band_order_list=["B12", "B11", "B04"],
            item=item,
            event_meta=event_meta,
            crs_val=str(crs),
            res_val="10m (20m Native represented on 10m analysis grid)"
        )
        
        # Generate SWIR/NIR Composite (B12, B08a, B04)
        swir_nir_arr = np.stack([band_data["B12"], band_data["B08a"], band_data["B04"]], axis=-1)
        swir_nir_img = Image.fromarray(scale_to_8bit(swir_nir_arr), 'RGB')
        swir_nir_path = os.path.join(swir_nir_dir, "B12-B8A-B4.jpg")
        swir_nir_img.save(swir_nir_path, quality=90)
        
        write_image_metadata(
            txt_path=os.path.join(swir_nir_dir, "B12-B8A-B4.txt"),
            image_type="SWIR+NIR",
            bands_text=["B12", "B08A", "B04"],
            band_order_list=["B12", "B08A", "B04"],
            item=item,
            event_meta=event_meta,
            crs_val=str(crs),
            res_val="10m (20m Native represented on 10m analysis grid)"
        )
        
        metadata = {
            "analysis_grid_resolution": "10m",
            "native_resolution": "10m (RGB, NIR), 20m (SWIR, Narrow NIR)",
            "bands_available": ", ".join(bands_to_fetch),
            "rgb_path": rgb_path,
            "swir_path": swir_path,
            "swir_nir_path": swir_nir_path,
            "generation_timestamp": datetime.utcnow().isoformat()
        }
        
        with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return metadata
            
    except Exception as e:
        logger.error(f"Failed to download/crop image for {event_id}: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None

