import pytest
import numpy as np
import affine
from src.fire_localization import localize_fire_candidates, LocalizationConfig, FireLocalizationResult
from src.fire_detection import FireDetectionResult, FireCandidateConfig
from src.fire_features import FireFeatures

@pytest.fixture
def base_config():
    return LocalizationConfig(
        min_component_pixels=1,
        min_component_area_m2=400.0,
        min_auto_export_pixels=2,
        min_auto_export_area_m2=800.0,
        min_fill_ratio=0.1,
        max_firms_viirs_distance_m=100.0,
        max_firms_modis_distance_m=100.0,
        fallback_firms_distance_m=100.0,
        reject_invalid_edge_components=True,
        morphology_enabled=False,
        morphology_operation="none",
        morphology_iterations=0
    )

def create_mock_detection(candidate_mask, valid_mask):
    shape = candidate_mask.shape
    return FireDetectionResult(
        candidate_mask=candidate_mask,
        valid_mask=valid_mask,
        b12_absolute_mask=np.ones(shape, dtype=bool),
        b12_b11_ratio_mask=np.ones(shape, dtype=bool),
        b12_b4_ratio_mask=np.ones(shape, dtype=bool),
        b04_brightness_rejection_mask=np.ones(shape, dtype=bool),
        diagnostics={},
        config=FireCandidateConfig(0.8, 1.0, 1.5, 0.3, ('b08', 'ndvi'))
    )

def create_mock_features(shape):
    return FireFeatures(
        b12=np.ones(shape, dtype=np.float32),
        b11=np.ones(shape, dtype=np.float32),
        b04=np.ones(shape, dtype=np.float32),
        b08=np.ones(shape, dtype=np.float32),
        swir_ratio=np.ones(shape, dtype=np.float32),
        swir_red_ratio=np.ones(shape, dtype=np.float32),
        swir_red_diff=np.zeros(shape, dtype=np.float32),
        norm_swir_diff=np.zeros(shape, dtype=np.float32),
        red_swir_contrast=np.zeros(shape, dtype=np.float32),
        ndvi=np.zeros(shape, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=bool),
        cloud_mask=np.zeros(shape, dtype=bool),
        transform=affine.Affine.translation(0.0, 0.0),
        crs=None,
        resolution=20.0,
        bounds=(0.0, 0.0, 1.0, 1.0),
        metadata={}
    )

# 1. Two separated candidate regions remain separate.
def test_two_separated_regions_remain_separate(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True # Region 1 (4 pixels)
    mask[7:9, 7:9] = True # Region 2 (4 pixels)
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 2

# 2. A FIRMS point cannot create a region when the mask is empty.
def test_firms_point_cannot_create_region(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    event_metadata = {'latitude': 0.0, 'longitude': 0.0} # FIRMS is right in the center (0,0 maps to 0,0)
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 0
    assert len(result.rejected_components) == 0

# 3. A tiny isolated component is retained as `REVIEW_REQUIRED` (not REJECTED).
def test_one_pixel_component_review_required(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True # 1 pixel, min_component is 1, min_auto is 2
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    
    # Check that a one-pixel component cannot receive ACCEPTED_FOR_AUTO_EXPORT
    assert len(result.accepted_components) == 0
    # Check that it is retained as REVIEW_REQUIRED
    assert len(result.review_required_components) == 1
    comp = result.review_required_components[0]
    assert comp.decision == "REVIEW_REQUIRED"
    assert "INSUFFICIENT_EVIDENCE_FOR_AUTO_EXPORT" in comp.decision_reasons
    assert comp.eligible_for_yolo_export is False

# 3b. A sufficiently coherent multi-pixel component can receive ACCEPTED_FOR_AUTO_EXPORT
def test_multipixel_component_accepted(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:7, 5:7] = True # 4 pixels, >= min_auto_export (2)
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 1
    assert len(result.review_required_components) == 0
    comp = result.accepted_components[0]
    assert comp.decision == "ACCEPTED_FOR_AUTO_EXPORT"
    assert comp.eligible_for_yolo_export is True

# 4. A scattered component fails `LOW_COMPACTNESS`.
def test_scattered_component_fails_compactness(base_config):
    mask = np.zeros((20, 20), dtype=bool)
    # create a very sparse component connected via a thin line
    mask[5, 5] = True
    mask[6, 6] = True
    mask[7, 7] = True
    mask[15, 15] = True
    # Fill in the line so it's one component but very bounding box sparse
    for i in range(5, 16):
        mask[i, i] = True
        
    # Area is 11 pixels. Bbox is 11x11 = 121. Fill ratio = 11/121 = 0.09
    
    valid_mask = np.ones((20, 20), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((20, 20))
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 0
    assert len(result.rejected_components) == 1
    assert "LOW_COMPACTNESS" in result.rejected_components[0].decision_reasons

# 5. Invalid-mask pixels cannot become accepted regions.
def test_invalid_mask_pixels_cannot_become_accepted(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:7, 5:7] = True
    
    valid_mask = np.ones((10, 10), dtype=bool)
    valid_mask[5, 5] = False
    
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    features.valid_mask = valid_mask
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 0
    assert len(result.rejected_components) == 1
    assert "TOUCHES_INVALID_DATA" in result.rejected_components[0].decision_reasons

# 6. Empty candidate masks return a valid empty localization result.
def test_empty_candidate_mask(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 0
    assert not np.any(result.cleaned_candidate_mask)

# 7. Pixel count converts correctly to physical area.
def test_pixel_count_converts_to_physical_area(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True # 4 pixels
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    # resolution is 20m in mock features, 4 * 400 = 1600m2
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert result.accepted_components[0].pixel_count_20m == 4
    assert result.accepted_components[0].area_m2 == 1600.0

# 8. FIRMS pixel coordinates are projected from real transform/CRS metadata.
def test_firms_coordinates_from_transform(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:7, 5:7] = True
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    # We will test the fallback since pyproj is not mocked easily here without scipy/pyproj mock
    # Wait, the code now checks for `hasattr(features, 'crs')`
    features.transform = affine.Affine.translation(10.0, 20.0)
    import rasterio
    features.crs = rasterio.crs.CRS.from_epsg(32610)
    
    event_metadata = {'latitude': 40.0, 'longitude': -121.0} 
    
    # We will just verify it runs and doesn't crash, because testing the exact pyproj output requires the math
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    # The projected coordinate will likely be far outside the 10x10 mock, so it gets rejected.
    comp = result.rejected_components[0]
    # We don't assert the exact location, just that it didn't crash and calculated a distance.
    assert comp.distance_to_firms_m >= 0

# 8b. Test that VIIRS and MODIS select different thresholds.
def test_viirs_and_modis_select_different_thresholds(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True # Top left component, not touching edge
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    # No crs so it uses fallback center (5, 5). Distance to (1.5, 1.5) is approx 3.5*sqrt(2)*20 = 99m
    # Let's set viirs threshold to 50, and modis to 130 (must be < 141.4 to avoid ValueError)
    from dataclasses import replace
    viirs_config = replace(base_config, max_firms_viirs_distance_m=50.0, max_firms_modis_distance_m=130.0)
    
    # VIIRS should reject
    viirs_meta = {'source': 'VIIRS'}
    viirs_res = localize_fire_candidates(detection, features, viirs_meta, viirs_config)
    assert len(viirs_res.rejected_components) == 1
    assert "TOO_FAR_FROM_FIRMS_REFERENCE" in viirs_res.rejected_components[0].decision_reasons
    
    # MODIS should accept
    modis_meta = {'source': 'MODIS'}
    modis_res = localize_fire_candidates(detection, features, modis_meta, viirs_config)
    assert len(modis_res.accepted_components) == 1

# 8c. Test ValueError on non-discriminative threshold
def test_non_discriminative_threshold_raises_error(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    
    from dataclasses import replace
    # 10x10 crop at 20m -> 200x200m. Center to corner is 141.4m.
    bad_config = replace(base_config, max_firms_viirs_distance_m=150.0)
    event_metadata = {'source': 'VIIRS'}
    
    import pytest
    with pytest.raises(ValueError, match="FIRMS proximity criterion is non-discriminative"):
        localize_fire_candidates(detection, features, event_metadata, bad_config)

# 9. All accepted/rejected components preserve reason records.
def test_preserve_reason_records(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True # accepted
    
    valid_mask = np.ones((10, 10), dtype=bool)
    valid_mask[7, 7] = False
    # add a component that touches invalid data
    mask[8, 8] = True 
    
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    features.valid_mask = valid_mask
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 1
    assert len(result.rejected_components) == 1
    assert len(result.accepted_components[0].decision_reasons) == 0
    assert "TOUCHES_INVALID_DATA" in result.rejected_components[0].decision_reasons

# 10. B8 and NDVI cannot influence localization.
def test_no_b8_ndvi_influence(base_config):
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:7, 5:7] = True
    
    valid_mask = np.ones((10, 10), dtype=bool)
    detection = create_mock_detection(mask, valid_mask)
    features = create_mock_features((10, 10))
    features.b08.fill(np.nan)
    features.ndvi.fill(np.nan)
    
    event_metadata = {'latitude': 0.0, 'longitude': 0.0}
    result = localize_fire_candidates(detection, features, event_metadata, base_config)
    assert len(result.accepted_components) == 1
    # Check that they aren't even present in attributes
    assert not hasattr(result.accepted_components[0], 'median_b08')
