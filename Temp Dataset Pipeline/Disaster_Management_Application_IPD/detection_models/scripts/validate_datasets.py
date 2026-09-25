import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataset_utils import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO datasets")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root folder containing datasets")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated list of dataset names")
    args = parser.parse_args()
    
    dataset_names = args.datasets.split(",")
    for name in dataset_names:
        name = name.strip()
        print(f"Validating dataset: {name}")
        try:
            ds = get_dataset(name, args.dataset_root)
            print(f"  Root: {ds.root}")
            print(f"  Classes ({ds.num_classes}): {ds.class_names}")
            
            # Check important classes
            if 'urban_fire' not in ds.class_names or 'industrial_false_positives' not in ds.class_names:
                print("  WARNING: Missing critical classes ('urban_fire', 'industrial_false_positives')")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == '__main__':
    main()
