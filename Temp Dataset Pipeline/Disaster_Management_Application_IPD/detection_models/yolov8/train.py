import argparse
import os
import sys

# Ensure common is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataset_utils import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on complex images")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name(s), comma-separated for sequential")
    parser.add_argument("--dataset_root", type=str, default="D:/DisasterManagementDatasets")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=str, help="Path to best.pt to fine-tune from")
    parser.add_argument("--sequential", action="store_true", help="Train sequentially on provided datasets")
    return parser.parse_args()

def train_model(model_weights, dataset_name, args, project_dir):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Please install it first.")
        sys.exit(1)

    print(f"\\n--- Training YOLOv8 on {dataset_name} ---")
    ds_info = get_dataset(dataset_name, args.dataset_root)
    
    # Initialize model
    model = YOLO(model_weights)
    
    # Train
    # We use augmentations suited for complex fire and industrial environments
    results = model.train(
        data=ds_info.yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        workers=args.workers,
        project=project_dir,
        name=dataset_name,
        exist_ok=True,
        # Augmentations for complex/small object environments
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.1,
        fliplr=0.5,
        # Optimize for precision and false positive reduction
        patience=20,
        save=True
    )
    
    # The output checkpoint is saved at {project_dir}/{dataset_name}/weights/best.pt
    best_weights = os.path.join(project_dir, dataset_name, "weights", "best.pt")
    return best_weights

def main():
    args = parse_args()
    project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "yolov8")
    os.makedirs(project_dir, exist_ok=True)
    
    datasets = [d.strip() for d in args.dataset.split(",")]
    
    if args.sequential and len(datasets) > 1:
        current_weights = args.resume if args.resume else args.model
        for ds in datasets:
            print(f"Starting sequential step for: {ds} using weights: {current_weights}")
            current_weights = train_model(current_weights, ds, args, project_dir)
            print(f"Finished {ds}. Best weights: {current_weights}")
    else:
        # Train on single dataset or just the first one
        train_model(args.resume if args.resume else args.model, datasets[0], args, project_dir)

if __name__ == '__main__':
    main()
