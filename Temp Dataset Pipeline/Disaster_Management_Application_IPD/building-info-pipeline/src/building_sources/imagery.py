import os
import urllib.request
from typing import Optional
from datetime import datetime
import concurrent.futures

from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import array_bounds
import numpy as np
from shapely.geometry import Polygon
import pyproj
from shapely.ops import transform
from PIL import Image

from src.models.aoi import AOI
from src.models.output import ImageryResult
from src.models.enums import PipelineState
from src.utils.logger import logger
from src.config import settings

class Sentinel2STACSource:
    def __init__(self, stac_url: str = "https://earth-search.aws.element84.com/v1", cache_dir: str = "cache/imagery"):
        self.stac_url = stac_url
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_asset_window(self, asset_href: str, bounds: tuple, cache_path: str, output_path: str, padding: int = 100):
        """Downloads a windowed subset of an asset (like NIR or SWIR)."""
        import shutil
        if os.path.exists(cache_path):
            shutil.copy(cache_path, output_path)
            return True
            
        try:
            with rasterio.open(asset_href) as src:
                src_crs = src.crs
                wgs84 = pyproj.CRS("EPSG:4326")
                project_to_src = pyproj.Transformer.from_crs(wgs84, src_crs, always_xy=True).transform
                
                min_lon, min_lat, max_lon, max_lat = bounds
                x1, y1 = project_to_src(min_lon, min_lat)
                x2, y2 = project_to_src(max_lon, max_lat)
                
                src_minx = min(x1, x2)
                src_miny = min(y1, y2)
                src_maxx = max(x1, x2)
                src_maxy = max(y1, y2)
                
                width_m = settings.image_width_meters
                height_m = settings.image_height_meters
                
                center_x = (src_minx + src_maxx) / 2.0
                center_y = (src_miny + src_maxy) / 2.0
                
                window = from_bounds(
                    center_x - (width_m / 2.0),
                    center_y - (height_m / 2.0),
                    center_x + (width_m / 2.0),
                    center_y + (height_m / 2.0),
                    transform=src.transform
                )
                window = window.round_shape()
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                data = src.read(window=window)
                win_transform = src.window_transform(window)
                
                meta = src.meta.copy()
                meta.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": win_transform,
                    "driver": "GTiff"
                })
                
                with rasterio.open(cache_path, "w", **meta) as dest:
                    dest.write(data)
                    
                shutil.copy(cache_path, output_path)
                return True
        except Exception as e:
            logger.error(f"Failed to download asset {asset_href}: {e}")
            return False

    def retrieve_imagery(self, aoi: AOI, event_id: str, output_dir: str) -> ImageryResult:
        """
        Retrieves Sentinel-2 imagery (Visual, NIR, SWIR16) covering the AOI.
        """
        try:
            client = Client.open(self.stac_url)
        except Exception as e:
            logger.error(f"Failed to connect to STAC API {self.stac_url}: {e}")
            return ImageryResult(status=PipelineState.NO_FREE_IMAGERY_FOUND, provider="", product_id="",
                                 acquisition_datetime=None, cloud_cover=None, crs_epsg=4326,
                                 image_width=0, image_height=0, image_footprint_wgs84=Polygon(),
                                 native_resolution_m=0.0, rgb_image_path="")

        # Check cache
        import hashlib
        bounds = aoi.geometry_wgs84.bounds
        bounds_str = f"{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}"
        cache_key = hashlib.md5(bounds_str.encode()).hexdigest()
        cache_img_dir = os.path.join(self.cache_dir, cache_key)
        
        cached_result_path = os.path.join(cache_img_dir, "result.json")
        
        # Prepare output paths
        os.makedirs(output_dir, exist_ok=True)
        dest_rgb_path = os.path.join(output_dir, "rgb.tif")
        dest_png_path = os.path.join(output_dir, "rgb.png")
        dest_nir_path = os.path.join(output_dir, "nir.tif")
        dest_swir_path = os.path.join(output_dir, "swir.tif")

        if os.path.exists(cached_result_path):
            import json
            try:
                with open(cached_result_path, "r") as f:
                    data = json.load(f)
                
                import shutil
                shutil.copy(os.path.join(cache_img_dir, "rgb.tif"), dest_rgb_path)
                shutil.copy(os.path.join(cache_img_dir, "rgb.png"), dest_png_path)
                if os.path.exists(os.path.join(cache_img_dir, "nir.tif")):
                    shutil.copy(os.path.join(cache_img_dir, "nir.tif"), dest_nir_path)
                if os.path.exists(os.path.join(cache_img_dir, "swir.tif")):
                    shutil.copy(os.path.join(cache_img_dir, "swir.tif"), dest_swir_path)
                
                from shapely.geometry import shape
                geom = shape(data["image_footprint_wgs84"])
                
                return ImageryResult(
                    status=PipelineState.SUCCESS,
                    provider=data["provider"],
                    product_id=data["product_id"],
                    acquisition_datetime=datetime.fromisoformat(data["acquisition_datetime"]) if data["acquisition_datetime"] else None,
                    cloud_cover=data["cloud_cover"],
                    crs_epsg=data["crs_epsg"],
                    image_width=data["image_width"],
                    image_height=data["image_height"],
                    image_footprint_wgs84=geom,
                    native_resolution_m=data["native_resolution_m"],
                    rgb_image_path=dest_rgb_path,
                    nir_image_path=dest_nir_path if os.path.exists(dest_nir_path) else None,
                    swir_image_path=dest_swir_path if os.path.exists(dest_swir_path) else None,
                    metadata=data.get("metadata", {})
                )
            except Exception as e:
                logger.warning(f"Cache read failed, re-fetching: {e}")

        # Use the AOI bounds to query STAC
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bounds,
            query={"eo:cloud_cover": {"lt": 30}},
            max_items=10
        )
        
        items = list(search.items())
        if not items:
            logger.error("No free imagery found for AOI.")
            return ImageryResult(status=PipelineState.NO_FREE_IMAGERY_FOUND, provider="Sentinel-2", product_id="",
                                 acquisition_datetime=None, cloud_cover=None, crs_epsg=4326,
                                 image_width=0, image_height=0, image_footprint_wgs84=Polygon(),
                                 native_resolution_m=0.0, rgb_image_path="")

        best_item = items[0]
        product_id = best_item.id
        dt = best_item.datetime
        cloud_cover = best_item.properties.get("eo:cloud_cover", 100.0)
        
        red_href = best_item.assets["red"].href if "red" in best_item.assets else None
        green_href = best_item.assets["green"].href if "green" in best_item.assets else None
        blue_href = best_item.assets["blue"].href if "blue" in best_item.assets else None
        nir_href = best_item.assets["nir"].href if "nir" in best_item.assets else None
        swir_href = best_item.assets["swir16"].href if "swir16" in best_item.assets else None
        
        if not (red_href and green_href and blue_href):
            logger.error(f"Missing raw red/green/blue assets in product {product_id}.")
            return ImageryResult(status=PipelineState.IMAGE_DOWNLOAD_FAILED, provider="Sentinel-2", product_id=product_id,
                                 acquisition_datetime=dt, cloud_cover=cloud_cover, crs_epsg=4326,
                                 image_width=0, image_height=0, image_footprint_wgs84=Polygon(),
                                 native_resolution_m=0.0, rgb_image_path="")

        os.makedirs(cache_img_dir, exist_ok=True)
        cache_rgb_path = os.path.join(cache_img_dir, "rgb.tif")
        cache_nir_path = os.path.join(cache_img_dir, "nir.tif")
        cache_swir_path = os.path.join(cache_img_dir, "swir.tif")

        try:
            # Parallel download of all bands
            cache_r = os.path.join(cache_img_dir, "r.tif")
            cache_g = os.path.join(cache_img_dir, "g.tif")
            cache_b = os.path.join(cache_img_dir, "b.tif")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                r_fut = executor.submit(self._download_asset_window, red_href, bounds, cache_r, cache_r + "_out")
                g_fut = executor.submit(self._download_asset_window, green_href, bounds, cache_g, cache_g + "_out")
                b_fut = executor.submit(self._download_asset_window, blue_href, bounds, cache_b, cache_b + "_out")
                nir_future = executor.submit(self._download_asset_window, nir_href, bounds, cache_nir_path, dest_nir_path) if nir_href else None
                swir_future = executor.submit(self._download_asset_window, swir_href, bounds, cache_swir_path, dest_swir_path) if swir_href else None
                
                # Wait for RGB to finish so we can stack them
                if not (r_fut.result() and g_fut.result() and b_fut.result()):
                    raise Exception("Failed to download raw R, G, or B bands")
                    
                # Stack them into 16-bit RGB
                with rasterio.open(cache_r) as src_r:
                    r_data = src_r.read(1)
                    meta = src_r.meta.copy()
                    
                    window_height = src_r.height
                    window_width = src_r.width
                    win_transform = src_r.transform
                    src_crs = src_r.crs
                    
                with rasterio.open(cache_g) as src_g:
                    g_data = src_g.read(1)
                with rasterio.open(cache_b) as src_b:
                    b_data = src_b.read(1)
                    
                meta.update({"count": 3})
                with rasterio.open(cache_rgb_path, "w", **meta) as dest:
                    dest.write(r_data, 1)
                    dest.write(g_data, 2)
                    dest.write(b_data, 3)
                    
                import shutil
                shutil.copy(cache_rgb_path, dest_rgb_path)

                # Use src_r for footprint metadata
                left, bottom, right, top = array_bounds(window_height, window_width, win_transform)
                footprint_src = Polygon([(left, bottom), (right, bottom), (right, top), (left, top), (left, bottom)])
                wgs84 = pyproj.CRS("EPSG:4326")
                project_to_wgs84 = pyproj.Transformer.from_crs(src_crs, wgs84, always_xy=True).transform
                from shapely.ops import transform
                footprint_wgs84 = transform(project_to_wgs84, footprint_src)
                
                epsg_code = src_crs.to_epsg() or 4326
                native_res = win_transform[0] # assuming square pixels
                
                cache_png_path = os.path.join(cache_img_dir, "rgb.png")
                def stretch(band):
                    p2, p98 = np.percentile(band, (2, 98))
                    if p98 == p2:
                        return np.zeros_like(band, dtype=np.uint8)
                    return np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                
                rgb_array = np.dstack((stretch(r_data), stretch(g_data), stretch(b_data)))
                img = Image.fromarray(rgb_array)
                
                scale = settings.display_upscaling_factor
                if scale > 1:
                    new_width = img.width * scale
                    new_height = img.height * scale
                    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
                    
                img.save(cache_png_path)
                shutil.copy(cache_png_path, dest_png_path)
                
                # Wait for other bands to finish downloading
                nir_success = nir_future.result() if nir_future else False
                swir_success = swir_future.result() if swir_future else False

                import json
                from shapely.geometry import mapping
                cache_data = {
                    "provider": "Sentinel-2 Element84 AWS",
                    "product_id": product_id,
                    "acquisition_datetime": dt.isoformat() if dt else None,
                    "cloud_cover": cloud_cover,
                    "crs_epsg": epsg_code,
                    "image_width": window_width,
                    "image_height": window_height,
                    "image_footprint_wgs84": mapping(footprint_wgs84),
                    "native_resolution_m": native_res,
                    "metadata": {
                        "red_href": red_href,
                        "display_scale": scale,
                        "resampling_method": "nearest",
                        "contrast_stretch": "percentile_2_98"
                    }
                }
                with open(cached_result_path, "w") as f:
                    json.dump(cache_data, f)

                return ImageryResult(
                    status=PipelineState.SUCCESS,
                    provider="Sentinel-2 Element84 AWS",
                    product_id=product_id,
                    acquisition_datetime=dt,
                    cloud_cover=cloud_cover,
                    crs_epsg=epsg_code,
                    image_width=window_width,
                    image_height=window_height,
                    image_footprint_wgs84=footprint_wgs84,
                    native_resolution_m=native_res,
                    rgb_image_path=dest_rgb_path,
                    nir_image_path=dest_nir_path if nir_success else None,
                    swir_image_path=dest_swir_path if swir_success else None,
                    metadata={
                        "red_href": red_href,
                        "display_scale": scale,
                        "resampling_method": "nearest",
                        "contrast_stretch": "percentile_2_98"
                    }
                )

        except Exception as e:
            logger.error(f"Failed to download and crop images: {e}")
            return ImageryResult(status=PipelineState.IMAGE_DOWNLOAD_FAILED, provider="Sentinel-2", product_id=product_id,
                                 acquisition_datetime=dt, cloud_cover=cloud_cover, crs_epsg=4326,
                                 image_width=0, image_height=0, image_footprint_wgs84=Polygon(),
                                 native_resolution_m=0.0, rgb_image_path="")
