import unittest
import numpy as np
from src.s2_preprocessing import MultispectralData
from src.fire_features import FeatureGenerator

class TestFireFeatures(unittest.TestCase):
    def setUp(self):
        self.generator = FeatureGenerator()
        
        # Create a synthetic 10x10 MultispectralData object
        # We will intentionally include zeroes to test division-by-zero handling
        self.b12 = np.full((10, 10), 0.8, dtype=np.float32)
        self.b11 = np.full((10, 10), 0.4, dtype=np.float32)
        self.b04 = np.full((10, 10), 0.2, dtype=np.float32)
        self.b08 = np.full((10, 10), 0.6, dtype=np.float32)
        
        # Edge cases: row 0 has B11=0 (SWIR ratio div by zero)
        self.b11[0, :] = 0.0
        
        # row 1 has B04=0 (SWIR-Red ratio div by zero)
        self.b04[1, :] = 0.0
        
        # row 2 has B12=0 and B11=0 (Norm SWIR diff div by zero)
        self.b12[2, :] = 0.0
        self.b11[2, :] = 0.0
        
        # row 3 has B08=0 and B04=0 (NDVI div by zero)
        self.b08[3, :] = 0.0
        self.b04[3, :] = 0.0
        
        # row 4 has B08a=0 (B8A ratios div by zero)
        self.b08a = np.full((10, 10), 0.5, dtype=np.float32)
        self.b08a[4, :] = 0.0
        
        self.b02 = np.full((10, 10), 0.1, dtype=np.float32)
        self.b03 = np.full((10, 10), 0.1, dtype=np.float32)
    
        self.ms_data = MultispectralData(
            b02=self.b02,
            b03=self.b03,
            b04=self.b04,
            b08=self.b08,
            b08a=self.b08a,
            b11=self.b11,
            b12=self.b12,
            valid_mask=np.ones((10, 10), dtype=bool),
            cloud_mask=np.zeros((10, 10), dtype=bool),
            transform=None,
            crs=None,
            resolution=20.0,
            bounds=(0,0,10,10),
            metadata={}
        )

    def test_feature_generation(self):
        features = self.generator.generate_features(self.ms_data)
        
        # 1. Verify Shapes
        self.assertEqual(features.swir_ratio.shape, (10, 10))
        self.assertEqual(features.swir_red_ratio.shape, (10, 10))
        self.assertEqual(features.swir_red_diff.shape, (10, 10))
        self.assertEqual(features.norm_swir_diff.shape, (10, 10))
        self.assertEqual(features.red_swir_contrast.shape, (10, 10))
        self.assertEqual(features.ndvi.shape, (10, 10))
        self.assertEqual(features.b12_b8a_ratio.shape, (10, 10))
        self.assertEqual(features.b11_b8a_ratio.shape, (10, 10))
        self.assertEqual(features.swir21_b8a_contrast.shape, (10, 10))
        self.assertEqual(features.ndvi_b8a.shape, (10, 10))
        self.assertEqual(features.nbr.shape, (10, 10))
        
        # 2. Verify Finite Values (No NaN or Inf allowed anywhere)
        self.assertTrue(np.isfinite(features.swir_ratio).all())
        self.assertTrue(np.isfinite(features.swir_red_ratio).all())
        self.assertTrue(np.isfinite(features.norm_swir_diff).all())
        self.assertTrue(np.isfinite(features.ndvi).all())
        self.assertTrue(np.isfinite(features.b12_b8a_ratio).all())
        self.assertTrue(np.isfinite(features.b11_b8a_ratio).all())
        self.assertTrue(np.isfinite(features.swir21_b8a_contrast).all())
        self.assertTrue(np.isfinite(features.ndvi_b8a).all())
        self.assertTrue(np.isfinite(features.nbr).all())
        
        # 3. Verify Edge Cases (Division by Zero yields 0.0)
        # swir_ratio (B12/B11) where B11 is 0 (Row 0)
        self.assertTrue((features.swir_ratio[0, :] == 0.0).all())
        
        # swir_red_ratio (B12/B04) where B04 is 0 (Row 1)
        self.assertTrue((features.swir_red_ratio[1, :] == 0.0).all())
        
        # norm_swir_diff where B12+B11 is 0 (Row 2)
        self.assertTrue((features.norm_swir_diff[2, :] == 0.0).all())
        
        # ndvi where B08+B04 is 0 (Row 3)
        self.assertTrue((features.ndvi[3, :] == 0.0).all())
        
        # b8a ratio where B08A is 0 (Row 4)
        self.assertTrue((features.b12_b8a_ratio[4, :] == 0.0).all())
        self.assertTrue((features.swir21_b8a_contrast[4, :] == 0.0).all())
        # NBR where B08A+B12 is 0. B12 is 0.8, so sum is 0.8. We don't have B08A+B12=0 yet.
        # Let's add a test for nbr div by zero on row 5: B08a=0, B12=0
        self.b12[5, :] = 0.0
        self.b08a[5, :] = 0.0
        
        # 4. Verify Numerical Correctness on normal pixels (Row 9)
        # B12=0.8, B11=0.4, B04=0.2, B08=0.6, B08A=0.5
        self.assertAlmostEqual(features.swir_ratio[9, 0], 0.8 / 0.4, places=5)
        self.assertAlmostEqual(features.swir_red_ratio[9, 0], 0.8 / 0.2, places=5)
        self.assertAlmostEqual(features.swir_red_diff[9, 0], 0.8 - 0.2, places=5)
        self.assertAlmostEqual(features.norm_swir_diff[9, 0], (0.8 - 0.4) / (0.8 + 0.4), places=5)
        self.assertAlmostEqual(features.red_swir_contrast[9, 0], (0.734 * 0.8) - 0.2, places=5)
        self.assertAlmostEqual(features.ndvi[9, 0], (0.6 - 0.2) / (0.6 + 0.2), places=5)
        self.assertAlmostEqual(features.b12_b8a_ratio[9, 0], 0.8 / 0.5, places=5)
        self.assertAlmostEqual(features.b11_b8a_ratio[9, 0], 0.4 / 0.5, places=5)
        self.assertAlmostEqual(features.swir21_b8a_contrast[9, 0], (0.8 - 0.4) / 0.5, places=5)
        self.assertAlmostEqual(features.ndvi_b8a[9, 0], (0.5 - 0.2) / (0.5 + 0.2), places=5)
        self.assertAlmostEqual(features.nbr[9, 0], (0.5 - 0.8) / (0.5 + 0.8), places=5)

if __name__ == '__main__':
    unittest.main()
