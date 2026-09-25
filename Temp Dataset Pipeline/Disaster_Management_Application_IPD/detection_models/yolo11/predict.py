import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Predict with YOLO11 on an image or directory")
    parser.add_argument("--source", type=str, required=True, help="Path to image, video, or directory")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Please install it first.")
        sys.exit(1)

    print(f"\n--- Running YOLO11 Prediction on {args.source} ---")
    
    model = YOLO(args.weights)
    
    project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "yolo11_predict")
    
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device if args.device else None,
        project=project_dir,
        save=True,
        exist_ok=True
    )
    
    print(f"\nPredictions saved to: {project_dir}")

if __name__ == '__main__':
    main()
