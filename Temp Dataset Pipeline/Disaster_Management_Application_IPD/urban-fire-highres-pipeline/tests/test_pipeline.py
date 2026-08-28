import pytest
import numpy as np
from src.s2_preprocessing import MultispectralData
from src.fire_features import FeatureGenerator
from src.fire_detection import detect_fire_candidate, FireCandidateConfig
from src.fire_localization import FireComponent, FireLocalizationResult, LocalizationConfig
from src.yolo_ground_truth import process_yolo_export
from PIL import Image

def test_safe_divide():
    fg = FeatureGenerator()
    num = np.array([1.0, 0.0, -1.0, 5.0], dtype=np.float32)
    den = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    
    result = fg._safe_divide(num, den)
    
    # 1.0 / 0.0 -> 0.0
    assert result[0] == 0.0
    # 0.0 / 0.0 -> 0.0
    assert result[1] == 0.0
    # -1.0 / 0.0 -> 0.0
    assert result[2] == 0.0
    # 5.0 / 2.0 -> 2.5
    assert result[3] == 2.5

def get_mock_ms_data(b08a_val=0.5):
    shape = (10, 10)
    b02 = np.ones(shape, dtype=np.float32) * 0.1
    b03 = np.ones(shape, dtype=np.float32) * 0.1
    b04 = np.ones(shape, dtype=np.float32) * 0.1
    b08 = np.ones(shape, dtype=np.float32) * 0.2
    b08a = np.ones(shape, dtype=np.float32) * b08a_val
    b11 = np.ones(shape, dtype=np.float32) * 0.5
    b12 = np.ones(shape, dtype=np.float32) * 0.9 # High B12 to trigger candidate
    
    valid_mask = np.ones(shape, dtype=bool)
    cloud_mask = np.zeros(shape, dtype=bool)
    
    import rasterio.transform
    transform = rasterio.transform.from_origin(0, 0, 20, 20)
    
    return MultispectralData(
        b02=b02, b03=b03, b04=b04, b08=b08, b08a=b08a, b11=b11, b12=b12,
        valid_mask=valid_mask, cloud_mask=cloud_mask,
        transform=transform, crs=None, resolution=20.0, bounds=(0,0,200,200),
        metadata={}
    )



def test_yolo_strict_export_gate(tmp_path):
    # Test that a component with eligible_for_yolo_export=True but decision='REVIEW_REQUIRED' fails
    # Create mock component
    comp = FireComponent(
        event_id="test_event",
        component_id=1,
        pixel_count_20m=10,
        area_m2=4000.0,
        x_min_20m=0, x_max_20m=5,
        y_min_20m=0, y_max_20m=5,
        width_20m=5, height_20m=5,
        centroid_x_20m=2.5, centroid_y_20m=2.5,
        fill_ratio=1.0,
        firms_x_20m=2.5, firms_y_20m=2.5,
        distance_to_firms_m=0.0,
        median_b04=0.1, median_b11=0.5, median_b12=0.9, median_b08a=0.5,
        median_swir_ratio=1.8, median_swir_red_ratio=9.0,
        median_b12_b8a_ratio=1.8, median_ndvi_b8a=0.5,
        decision="REVIEW_REQUIRED", # Contradictory state!
        decision_reasons=["HIGH_NDVI_B8A"],
        eligible_for_yolo_export=True # Contradictory state!
    )
    
    loc_result = FireLocalizationResult(
        accepted_components=[],
        review_required_components=[comp],
        rejected_components=[],
        cleaned_candidate_mask=np.zeros((10,10), dtype=bool),
        labeled_components=np.zeros((10,10), dtype=int),
        config=LocalizationConfig(
            min_component_pixels=1, min_component_area_m2=400,
            min_auto_export_pixels=2, min_auto_export_area_m2=800,
            min_fill_ratio=0.05, max_firms_viirs_distance_m=375,
            max_firms_modis_distance_m=1000, fallback_firms_distance_m=1000,
            reject_invalid_edge_components=True, morphology_enabled=False,
            morphology_operation="none", morphology_iterations=0,
            mode="B8A_AUXILIARY", min_b12_b8a_ratio=0.5, max_ndvi_b8a=0.3
        )
    )
    
    clean_fc_img = Image.new('RGB', (100, 100))
    ms_data = get_mock_ms_data()
    event_meta = {"event_id": "test_event"}
    
    with pytest.raises(ValueError, match="Strict export gate validation failed"):
        process_yolo_export(loc_result, clean_fc_img, ms_data, event_meta, dataset_root=str(tmp_path))
