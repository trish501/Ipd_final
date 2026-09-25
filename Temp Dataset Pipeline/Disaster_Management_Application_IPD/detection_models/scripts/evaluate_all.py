import argparse
import os
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple YOLO models")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to evaluate on")
    parser.add_argument("--dataset_root", type=str, default="D:/DisasterManagementDatasets")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of models to evaluate (e.g. yolov8,yolo11)")
    parser.add_argument("--weights", type=str, required=True, help="Comma-separated list of weight paths corresponding to models")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="File to save results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    models = [m.strip() for m in args.models.split(',')]
    weights = [w.strip() for w in args.weights.split(',')]
    
    if len(models) != len(weights):
        print("Error: Number of models must match number of weights provided.")
        sys.exit(1)
        
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Please install it first.")
        sys.exit(1)
        
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.dataset_utils import get_dataset
    ds_info = get_dataset(args.dataset, args.dataset_root)
        
    results = {}
    
    for model_name, weight_path in zip(models, weights):
        print(f"\n--- Evaluating {model_name} with {weight_path} ---")
        if model_name in ['yolov8', 'yolo11']:
            model = YOLO(weight_path)
            project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", f"{model_name}_eval")
            metrics = model.val(
                data=ds_info.yaml_path,
                project=project_dir,
                name=args.dataset,
                exist_ok=True
            )
            results[model_name] = {
                "mAP50-95": float(metrics.box.map),
                "mAP50": float(metrics.box.map50),
                "mAP75": float(metrics.box.map75),
            }
        else:
            print(f"Evaluation for {model_name} is not implemented in this script yet.")
            results[model_name] = "Not Implemented"
            
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation complete. Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
