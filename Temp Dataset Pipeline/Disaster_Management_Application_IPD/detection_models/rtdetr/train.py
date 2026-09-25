import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="D:/DisasterManagementDatasets")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--resume", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Initializing RT-DETR pipeline...")
    try:
        from ultralytics import RTDETR
        print(f"RTDETR class imported successfully.")
    except ImportError:
        print("Ultralytics not installed or doesn't support RTDETR.")
        return
    print(f"Arguments: {args}")
    print("Smoke test passed.")

if __name__ == '__main__':
    main()
