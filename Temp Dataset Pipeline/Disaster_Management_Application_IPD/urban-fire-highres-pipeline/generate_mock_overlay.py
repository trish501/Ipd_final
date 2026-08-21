import os
import sys
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.yolo_ground_truth import process_yolo_export
from src.fire_localization import FireLocalizationResult, FireComponent

class MockMSData:
    def __init__(self, cols, rows, transform):
        self.b02 = np.zeros((rows, cols))
        self.b03 = np.zeros((rows, cols))
        self.b04 = np.zeros((rows, cols))
        self.b08 = np.zeros((rows, cols))
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

def create_component(comp_id, decision, eligible, x_min, x_max, y_min, y_max):
    return FireComponent(
        event_id="test_event", component_id=comp_id, pixel_count_20m=10, area_m2=4000.0,
        x_min_20m=x_min, x_max_20m=x_max, y_min_20m=y_min, y_max_20m=y_max,
        width_20m=x_max - x_min + 1, height_20m=y_max - y_min + 1,
        centroid_x_20m=(x_min+x_max)/2, centroid_y_20m=(y_min+y_max)/2,
        fill_ratio=1.0, firms_x_20m=50.0, firms_y_20m=50.0, distance_to_firms_m=0.0,
        median_b04=0.1, median_b11=0.2, median_b12=0.5,
        median_swir_ratio=2.5, median_swir_red_ratio=5.0,
        decision=decision, decision_reasons=[], eligible_for_yolo_export=eligible
    )

transform = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 4600000.0)

# Make the image large (1024x1024) so a 1-pixel line looks correctly thin
mock_ms_data = MockMSData(cols=1024, rows=1024, transform=transform)
mock_event_meta = {
    "event_id": "test_event",
    "source": "VIIRS",
    "latitude": 40.0,
    "longitude": -120.0,
    "date": "2023-01-01",
    "time": "11:55:00"
}

# Make a larger component so it's visible in a 1024x1024 image
c1 = create_component(1, "ACCEPTED_FOR_AUTO_EXPORT", True, 400, 450, 400, 450)
loc_res = FireLocalizationResult([c1], [], [], None, None, None)

# Generate a visually distinct background (gradient) so it doesn't look empty
img_data = np.zeros((1024, 1024, 3), dtype=np.uint8)
for i in range(1024):
    img_data[i, :, 1] = int(255 * (i / 1024.0)) # Green gradient
    img_data[:, i, 2] = int(255 * (i / 1024.0)) # Blue gradient
img = Image.fromarray(img_data)

dataset_root = "YOLO_dataset"
process_yolo_export(loc_res, img, mock_ms_data, mock_event_meta, dataset_root)
print("Saved to", os.path.abspath(os.path.join(dataset_root, "diagnostics", "test_event", "yolo_overlay.jpg")))
