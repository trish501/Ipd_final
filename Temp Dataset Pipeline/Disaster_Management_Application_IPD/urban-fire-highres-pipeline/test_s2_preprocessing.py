import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from src.s2_preprocessing import S2Preprocessor, MultispectralData

def create_synthetic_tiff(data, transform, crs='EPSG:32610'):
    memfile = MemoryFile()
    with memfile.open(
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dataset:
        dataset.write(data, 1)
    return memfile

class DummyItem:
    def __init__(self, baseline="04.00"):
        self.id = "S2_DUMMY_SCENE"
        self.datetime = "2023-01-01T00:00:00Z"
        self.properties = {"s2:processing_baseline": baseline}
        
        # We will mock get_asset_href anyway, but let's provide a structure
        class Asset:
            def __init__(self, href):
                self.href = href
        
        self.assets = {
            "B04": Asset("mock_b04"),
            "B08": Asset("mock_b08"),
            "B11": Asset("mock_b11"),
            "B12": Asset("mock_b12"),
            "SCL": Asset("mock_scl")
        }

class TestS2Preprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = S2Preprocessor()
        
    @patch('src.s2_preprocessing.rasterio.open')
    def test_processing_and_scaling(self, mock_rasterio_open):
        # We want to test a baseline >= 04.00 (which has a -1000 offset)
        item = DummyItem(baseline="04.00")
        
        # We need to simulate rasterio.open returning our memory files.
        # Let's create dummy 10m and 20m rasters.
        # Center lat/lon is somewhat irrelevant if we just mock the CRS/transform matching.
        # But we need valid transform math.
        
        # Let's say our STAC item covers 0,0 to 1000,1000 in EPSG:32610
        transform_10m = from_origin(0, 1000, 10, 10)
        transform_20m = from_origin(0, 1000, 20, 20)
        
        # 10m arrays (100x100 = 1km x 1km)
        b04_data = np.full((100, 100), 5000, dtype=np.uint16)
        b08_data = np.full((100, 100), 6000, dtype=np.uint16)
        # Put some NoData (0) in B04
        b04_data[0:10, 0:10] = 0
        
        # 20m arrays (50x50 = 1km x 1km)
        b11_data = np.full((50, 50), 7000, dtype=np.uint16)
        b12_data = np.full((50, 50), 8000, dtype=np.uint16)
        scl_data = np.full((50, 50), 4, dtype=np.uint8) # 4 = Vegetation (valid)
        scl_data[40:50, 40:50] = 9 # 9 = Cloud High Prob
        
        # Keep references to prevent garbage collection closing the memfiles
        self.memfiles = {
            "mock_b04": create_synthetic_tiff(b04_data, transform_10m),
            "mock_b08": create_synthetic_tiff(b08_data, transform_10m),
            "mock_b11": create_synthetic_tiff(b11_data, transform_20m),
            "mock_b12": create_synthetic_tiff(b12_data, transform_20m),
            "mock_scl": create_synthetic_tiff(scl_data, transform_20m)
        }
        
        # We need to intercept the `with rasterio.open(href) as src:`
        # By making mock_rasterio_open side_effect return the opened datasets.
        def mock_open_func(href):
            return self.memfiles[href].open()
            
        mock_rasterio_open.side_effect = mock_open_func
        
        # Mocking rasterio.warp.transform since our inputs are not EPSG:4326 lat/lon
        # We just want to extract a 400m x 400m crop at the center
        with patch('src.s2_preprocessing.transform', return_value=([500.0], [500.0])):
            ms_data = self.preprocessor.process(item, lat=0.0, lon=0.0, crop_km=0.4)
            
        # Target size for 400m at 20m resolution is 20x20
        self.assertEqual(ms_data.b04.shape, (20, 20))
        self.assertEqual(ms_data.b12.shape, (20, 20))
        self.assertEqual(ms_data.valid_mask.shape, (20, 20))
        self.assertEqual(ms_data.cloud_mask.shape, (20, 20))
        
        # Test Scaling (baseline 04.00 has -1000 offset, scale 0.0001)
        # B12 original was 8000. Expected: (8000 - 1000) * 0.0001 = 0.7
        self.assertAlmostEqual(ms_data.b12[10, 10], 0.7, places=5)
        
        # B04 original was 5000. Expected: (5000 - 1000) * 0.0001 = 0.4
        self.assertAlmostEqual(ms_data.b04[10, 10], 0.4, places=5)
        
        # Test No-Data handling (B04 had 0s at the top left)
        # Since we are centering at 500,500, we missed the top left.
        # But we know that any 0s shouldn't be scaled (they remain 0).
        
        # Check cloud masking
        # SCL was 9 at the bottom right. Depending on crop, it might be included.
        # Center is 500. Crop is 400x400. So bounds are 300 to 700.
        # In 20m array (0 to 1000), 300 to 700 corresponds to indices 15 to 35.
        # Our SCL cloud was at index 40 to 50, so it is outside the crop!
        self.assertFalse(np.any(ms_data.cloud_mask))

if __name__ == '__main__':
    unittest.main()
