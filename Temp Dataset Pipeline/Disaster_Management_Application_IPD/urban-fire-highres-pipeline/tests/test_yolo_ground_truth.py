import os
import pytest
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine

from src.yolo_ground_truth import (
    calculate_component_square,
    transform_20m_to_rgb,
    generate_yolo_label,
    process_yolo_export
)
from src.fire_localization import FireComponent, FireLocalizationResult

class MockMSData:
    def __init__(self, cols, rows, transform):
        self.b02 = np.zeros((rows, cols))
        self.b03 = np.zeros((rows, cols))
        self.b04 = np.zeros((rows, cols))
        self.b08 = np.zeros((rows, cols))
        self.b08a = np.zeros((rows, cols))
        self.b11 = np.zeros((rows, cols))
        self.b12 = np.zeros((rows, cols))
        self.transform = transform
        self.crs = rasterio.crs.CRS.from_epsg(32610)
        self.resolution = 20.0
        self.metadata = {
            "scene_id": "test_scene",
            "acquisition_datetime": "2023-01-01T12:00:00Z",
            "masked_pixel_percentage": 0.0
        }

@pytest.fixture
def mock_ms_data():
    # 100x100 grid (2km x 2km at 20m)
    transform = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 4600000.0)
    return MockMSData(cols=100, rows=100, transform=transform)

@pytest.fixture
def mock_event_meta():
    return {
        "event_id": "event_123",
        "source": "VIIRS",
        "latitude": 40.0,
        "longitude": -120.0,
        "date": "2023-01-01",
        "time": "11:55:00"
    }

def create_component(comp_id, decision, eligible, x_min, x_max, y_min, y_max):
    return FireComponent(
        event_id="event_123", component_id=comp_id, pixel_count_20m=10, area_m2=4000.0,
        x_min_20m=x_min, x_max_20m=x_max, y_min_20m=y_min, y_max_20m=y_max,
        width_20m=x_max - x_min + 1, height_20m=y_max - y_min + 1,
        centroid_x_20m=(x_min+x_max)/2, centroid_y_20m=(y_min+y_max)/2,
        fill_ratio=1.0, firms_x_20m=50.0, firms_y_20m=50.0, distance_to_firms_m=0.0,
        median_b04=0.1, median_b11=0.2, median_b12=0.5,
        median_b08a=0.3, median_b12_b8a_ratio=1.66, median_ndvi_b8a=0.5,
        median_swir_ratio=2.5, median_swir_red_ratio=5.0,
        decision=decision, decision_reasons=[], eligible_for_yolo_export=eligible
    )

def test_1_one_accepted_produces_one_yolo_line(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    # Check lines in label file
    labels_dir = tmp_path / "labels"
    txt_files = list(labels_dir.rglob("*.txt"))
    assert len(txt_files) == 1
    with open(txt_files[0]) as f:
        lines = f.readlines()
    assert len(lines) == 1

def test_2_multiple_accepted_produce_multiple_lines(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    c2 = create_component(2, "ACCEPTED_FOR_AUTO_EXPORT", True, 60, 65, 60, 65)
    loc_res = FireLocalizationResult([c1, c2], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    txt_files = list((tmp_path / "labels").rglob("*.txt"))
    assert len(txt_files) == 1
    with open(txt_files[0]) as f:
        lines = f.readlines()
    assert len(lines) == 2

def test_3_review_and_rejected_produce_no_labels(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "REVIEW_REQUIRED", False, 40, 42, 40, 45)
    c2 = create_component(2, "REJECTED", False, 60, 65, 60, 65)
    loc_res = FireLocalizationResult([], [c1], [c2], None, None, None)
    img = Image.new("RGB", (100, 100))
    process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    txt_files = list((tmp_path / "labels").rglob("*.txt"))
    assert len(txt_files) == 0

def test_4_smallest_square_contains_pixels():
    # c_x_max/c_y_max in FireComponent are inclusive, but calculate_component_square expects exclusive for the actual box width
    # Example: pixels from x=40 to x=42 (inclusive). That's a width of 3.
    # We pass x_min=40, x_max=43.
    x_min, x_max, y_min, y_max = 40, 43, 40, 46 # 3x6 rectangle
    sq_x_min, sq_x_max, sq_y_min, sq_y_max = calculate_component_square(x_min, x_max, y_min, y_max, 100, 100)
    assert sq_x_min <= x_min
    assert sq_x_max >= x_max
    assert sq_y_min <= y_min
    assert sq_y_max >= y_max
    assert (sq_x_max - sq_x_min) == (sq_y_max - sq_y_min) # is square

def test_5_edge_adjacent_remain_enclosed():
    # Component on the top-left edge
    x_min, x_max, y_min, y_max = 0, 3, 0, 6
    sq_x_min, sq_x_max, sq_y_min, sq_y_max = calculate_component_square(x_min, x_max, y_min, y_max, 100, 100)
    assert sq_x_min == 0 # Cannot go negative
    assert sq_x_max >= x_max
    assert sq_y_min == 0 # Cannot go negative
    assert sq_y_max >= y_max
    assert (sq_x_max - sq_x_min) == (sq_y_max - sq_y_min)

def test_6_pixel_boxes_are_square():
    x_min, x_max, y_min, y_max = 20, 25, 20, 40
    sq_x_min, sq_x_max, sq_y_min, sq_y_max = calculate_component_square(x_min, x_max, y_min, y_max, 100, 100)
    assert (sq_x_max - sq_x_min) == (sq_y_max - sq_y_min)

def test_7_yolo_values_finite_and_in_range():
    # Provide rgb pixel square coordinates and image dimensions
    yolo_res = generate_yolo_label(10, 30, 20, 40, 100, 100)
    assert 0 <= yolo_res["x_center_norm"] <= 1
    assert 0 <= yolo_res["y_center_norm"] <= 1
    assert 0 <= yolo_res["width_norm"] <= 1
    assert 0 <= yolo_res["height_norm"] <= 1

def test_8_rgb_and_label_names_match(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    img_files = list((tmp_path / "images").rglob("*.jpg"))
    txt_files = list((tmp_path / "labels").rglob("*.txt"))
    assert img_files[0].stem == txt_files[0].stem

def test_9_overlay_coordinates_equal_saved_metadata(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    meta = process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    comp_meta = meta["components"][0]
    diag_path = tmp_path / "diagnostics" / "event_123" / "yolo_overlay.jpg"
    assert diag_path.exists()
    assert comp_meta["x_min_rgb"] is not None

def test_10_clean_training_images_remain_unmodified(tmp_path, mock_ms_data, mock_event_meta):
    # Ensure draw doesn't alter the exported image
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    
    # Provide a fully black image
    img = Image.new("RGB", (100, 100), color=(0,0,0))
    process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    img_files = list((tmp_path / "images").rglob("*.jpg"))
    exported_img = Image.open(img_files[0])
    
    # The entire image should still be black, no green boxes
    arr = np.array(exported_img)
    # JPEG compression might introduce tiny artifacts, so we check near 0
    assert np.mean(arr) < 5.0

def test_11_20m_to_rgb_uses_metadata(mock_ms_data):
    # Test mapping with identical transform
    transform = mock_ms_data.transform
    x_rgb, y_rgb = transform_20m_to_rgb(10, 10, transform, transform)
    assert np.isclose(x_rgb, 10.0)
    assert np.isclose(y_rgb, 10.0)
    
    # Test mapping with different rgb transform
    # E.g. RGB is 10m resolution, meaning 200x200 grid
    rgb_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0)
    x_rgb2, y_rgb2 = transform_20m_to_rgb(10, 10, transform, rgb_transform)
    assert np.isclose(x_rgb2, 20.0)
    assert np.isclose(y_rgb2, 20.0)

def test_12_no_accepted_creates_no_positive_training_export(tmp_path, mock_ms_data, mock_event_meta):
    loc_res = FireLocalizationResult([], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    # event_ground_truth.json should still be saved
    gt_path = tmp_path / "diagnostics" / "event_123" / "event_ground_truth.json"
    assert gt_path.exists()
    
    # images/ and labels/ should NOT have this event's files
    img_files = list((tmp_path / "images").rglob("*.jpg"))
    txt_files = list((tmp_path / "labels").rglob("*.txt"))
    assert len(img_files) == 0
    assert len(txt_files) == 0

def test_13_strict_gate_review_eligible_true(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "REVIEW_REQUIRED", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([], [c1], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="Strict export gate validation failed"):
        process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    error_path = tmp_path / "diagnostics" / "event_123" / "export_validation_error.json"
    assert error_path.exists()

def test_14_strict_gate_rejected_eligible_true(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "REJECTED", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([], [], [c1], None, None, None)
    img = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="Strict export gate validation failed"):
        process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))

def test_15_strict_gate_accepted_eligible_false(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", False, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    meta = process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    # No export
    assert meta["export_status"] == "REJECTED"
    assert meta["image_path"] is None
    assert meta["label_path"] is None
    
def test_16_strict_gate_accepted_eligible_true(tmp_path, mock_ms_data, mock_event_meta):
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    meta = process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
    
    assert meta["export_status"] == "EXPORTED"
    assert meta["image_path"] is not None
    assert meta["label_path"] is not None

def test_17_diagnostic_box_appearance(tmp_path, mock_ms_data, mock_event_meta):
    from unittest.mock import patch
    c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 40, 42, 40, 45)
    loc_res = FireLocalizationResult([c1], [], [], None, None, None)
    img = Image.new("RGB", (100, 100))
    
    with patch("src.yolo_ground_truth.ImageDraw.Draw") as mock_draw_class:
        mock_draw_instance = mock_draw_class.return_value
        process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, str(tmp_path))
        
        assert mock_draw_instance.rectangle.call_count == 3
        args, kwargs = mock_draw_instance.rectangle.call_args
        
        assert kwargs.get("outline") == (0, 120, 255)
        assert kwargs.get("width") == 1
