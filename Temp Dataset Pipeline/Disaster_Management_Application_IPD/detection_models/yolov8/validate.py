import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataset_utils import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to validate on")
    parser.add_argument("--dataset_root", type=str, default="D:/DisasterManagementDatasets")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Please install it first.")
        sys.exit(1)

    print(f"\n--- Validating YOLOv8 on {args.dataset} ---")
    ds_info = get_dataset(args.dataset, args.dataset_root)
    
    model = YOLO(args.weights)
    
    project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "yolov8_val")
    
    metrics = model.val(
        data=ds_info.yaml_path,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=project_dir,
        name=args.dataset,
        exist_ok=True
    )
    
    print("\n--- Validation Results ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

if __name__ == '__main__':
    main()
