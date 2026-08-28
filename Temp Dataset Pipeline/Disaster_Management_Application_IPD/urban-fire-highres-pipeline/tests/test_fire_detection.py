import pytest
import numpy as np
from src.fire_detection import detect_fire_candidate, FireCandidateConfig, compute_candidate_bounding_box
from src.fire_features import FireFeatures

@pytest.fixture
def base_config():
    return FireCandidateConfig(
        swir2_abs_thresh=0.8,
        swir_ratio_thresh=1.0,
        swir_red_ratio_thresh=1.5,
        b04_bright_reject_thresh=0.3,
        min_b12_b8a_ratio_thresh=1.5,
        max_nbr_thresh=-0.1,
        retained_features=('b08', 'swir_red_diff', 'norm_swir_diff', 'red_swir_contrast', 'ndvi')
    )

def create_mock_features(b12, b11, b04, b08, swir_ratio, swir_red_ratio, ndvi, valid_mask, b12_b8a_ratio=2.0, nbr=-0.5):
    return FireFeatures(
        b12=np.array([[b12]], dtype=np.float32),
        b11=np.array([[b11]], dtype=np.float32),
        b04=np.array([[b04]], dtype=np.float32),
        b08=np.array([[b08]], dtype=np.float32),
        b08a=np.array([[b08]], dtype=np.float32),
        swir_ratio=np.array([[swir_ratio]], dtype=np.float32),
        swir_red_ratio=np.array([[swir_red_ratio]], dtype=np.float32),
        swir_red_diff=np.array([[0.0]], dtype=np.float32),
        norm_swir_diff=np.array([[0.0]], dtype=np.float32),
        red_swir_contrast=np.array([[0.0]], dtype=np.float32),
        ndvi=np.array([[ndvi]], dtype=np.float32),
        b12_b8a_ratio=np.array([[b12_b8a_ratio]], dtype=np.float32),
        b11_b8a_ratio=np.array([[swir_ratio]], dtype=np.float32),
        swir21_b8a_contrast=np.array([[0.0]], dtype=np.float32),
        ndvi_b8a=np.array([[ndvi]], dtype=np.float32),
        nbr=np.array([[nbr]], dtype=np.float32),
        valid_mask=np.array([[valid_mask]], dtype=bool),
        cloud_mask=np.array([[False]], dtype=bool),
        transform=None,
        crs=None,
        resolution=20.0,
        bounds=(0.0, 0.0, 1.0, 1.0),
        metadata={}
    )

def test_valid_synthetic_pixel_becomes_candidate(base_config):
    # Meets all criteria
    f = create_mock_features(
        b12=1.0, b11=0.8, b04=0.2, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=5.0, ndvi=0.4,
        valid_mask=True
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == True

def test_failing_c1_prevents_candidacy(base_config):
    # Fails C1: swir2_abs_thresh (b12 < 0.8)
    f = create_mock_features(
        b12=0.5, b11=0.4, b04=0.2, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=2.5, ndvi=0.4,
        valid_mask=True
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == False

def test_failing_c2_prevents_candidacy(base_config):
    # Fails C2: swir_ratio_thresh (swir_ratio < 1.0)
    f = create_mock_features(
        b12=1.0, b11=1.2, b04=0.2, b08=0.5,
        swir_ratio=0.83, swir_red_ratio=5.0, ndvi=0.4,
        valid_mask=True
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == False

def test_failing_c3_prevents_candidacy(base_config):
    # Fails C3: swir_red_ratio_thresh (swir_red_ratio < 1.5)
    f = create_mock_features(
        b12=1.0, b11=0.8, b04=0.8, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=1.25, ndvi=0.4,
        valid_mask=True
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == False

def test_failing_c4_prevents_candidacy(base_config):
    # Fails C4: b04_bright_reject_thresh (b04 >= 0.3)
    f = create_mock_features(
        b12=1.0, b11=0.8, b04=0.5, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=2.0, ndvi=0.4,
        valid_mask=True
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == False

def test_invalid_mask_pixels_never_become_candidates(base_config):
    f = create_mock_features(
        b12=1.0, b11=0.8, b04=0.2, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=5.0, ndvi=0.4,
        valid_mask=False
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == False

def test_non_finite_inputs_never_create_nan_inf(base_config):
    f = create_mock_features(
        b12=np.nan, b11=np.inf, b04=0.2, b08=0.5,
        swir_ratio=np.nan, swir_red_ratio=np.inf, ndvi=0.4,
        valid_mask=True
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask[0, 0] == False
    assert np.all(~np.isnan(result.candidate_mask))
    assert np.all(~np.isinf(result.candidate_mask))

def test_b8_and_ndvi_do_not_affect_candidate_mask(base_config):
    f1 = create_mock_features(
        b12=1.0, b11=0.8, b04=0.2, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=5.0, ndvi=0.4,
        valid_mask=True
    )
    res1 = detect_fire_candidate(f1, base_config).candidate_mask[0, 0]
    
    f2 = create_mock_features(
        b12=1.0, b11=0.8, b04=0.2, b08=99.0,
        swir_ratio=1.25, swir_red_ratio=5.0, ndvi=-99.0,
        valid_mask=True
    )
    res2 = detect_fire_candidate(f2, base_config).candidate_mask[0, 0]
    
    assert res1 == True
    assert res2 == True

def test_diagnostics_report_exclusions(base_config):
    f = create_mock_features(
        b12=1.0, b11=0.8, b04=0.2, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=5.0, ndvi=0.4,
        valid_mask=True
    )
    diagnostics = detect_fire_candidate(f, base_config).diagnostics
    assert "B08" in diagnostics["experimental_features_excluded"]
    assert "NDVI" in diagnostics["experimental_features_excluded"]
    assert "detector_output_semantics" in diagnostics
    assert diagnostics["detector_output_semantics"] == "spectral fire candidate mask"
    assert "B04" in diagnostics["baseline_bands_used"]
    assert "B11" in diagnostics["baseline_bands_used"]
    assert "B12" in diagnostics["baseline_bands_used"]
    assert len(diagnostics["scientific_limitations"]) == 4

def test_mismatched_feature_shapes_raise_error(base_config):
    f = create_mock_features(
        b12=1.0, b11=0.8, b04=0.2, b08=0.5,
        swir_ratio=1.25, swir_red_ratio=5.0, ndvi=0.4,
        valid_mask=True
    )
    # Tamper with shape
    f.b11 = np.array([0.8, 0.8], dtype=np.float32)
    
    with pytest.raises(ValueError, match="Mismatched feature shapes"):
        detect_fire_candidate(f, base_config)

def test_empty_candidate_results_are_valid():
    import affine
    candidate_mask = np.zeros((10, 10), dtype=bool)
    transform = affine.Affine.translation(0.0, 0.0)
    bbox = compute_candidate_bounding_box(candidate_mask, transform)
    assert bbox is None

def test_all_output_masks_match_phase1_shape(base_config):
    shape = (5, 5)
    f = FireFeatures(
        b12=np.ones(shape, dtype=np.float32),
        b11=np.ones(shape, dtype=np.float32),
        b04=np.ones(shape, dtype=np.float32),
        b08=np.ones(shape, dtype=np.float32),
        b08a=np.ones(shape, dtype=np.float32),
        swir_ratio=np.ones(shape, dtype=np.float32),
        swir_red_ratio=np.ones(shape, dtype=np.float32),
        swir_red_diff=np.zeros(shape, dtype=np.float32),
        norm_swir_diff=np.zeros(shape, dtype=np.float32),
        red_swir_contrast=np.zeros(shape, dtype=np.float32),
        ndvi=np.zeros(shape, dtype=np.float32),
        b12_b8a_ratio=np.ones(shape, dtype=np.float32),
        b11_b8a_ratio=np.ones(shape, dtype=np.float32),
        swir21_b8a_contrast=np.zeros(shape, dtype=np.float32),
        ndvi_b8a=np.zeros(shape, dtype=np.float32),
        nbr=np.full(shape, -0.5, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=bool),
        cloud_mask=np.zeros(shape, dtype=bool),
        transform=None, crs=None, resolution=20.0, bounds=(0,0,10,10), metadata={}
    )
    result = detect_fire_candidate(f, base_config)
    assert result.candidate_mask.shape == shape
    assert result.valid_mask.shape == shape
    assert result.b12_absolute_mask.shape == shape
    assert result.b12_b11_ratio_mask.shape == shape
    assert result.b12_b4_ratio_mask.shape == shape
    assert result.b04_brightness_rejection_mask.shape == shape
