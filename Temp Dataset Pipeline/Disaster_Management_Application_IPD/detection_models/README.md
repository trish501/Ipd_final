# Detection Models Evaluation

This module provides a unified training and evaluation pipeline for comparing multiple object detection models:
1. YOLOv8
2. YOLO11
3. RT-DETR
4. YOLO-NAS
5. Faster R-CNN

## A. Folder Structure
The folder structure follows the specification, keeping datasets external and having separate output folders for each model.

## B. Dataset Setup
Do not put datasets inside `detection_models/`. Keep them outside:
```text
D:/DisasterManagementDatasets/
├── dataset_2023/
├── dataset_2024/
└── dataset_2025/
```

## C. Configuration
Use `--dataset` to specify the external dataset. You can configure roots via `--dataset_root`.

## D. Dataset Validation
```bash
python detection_models/scripts/validate_datasets.py --dataset_root D:/DisasterManagementDatasets --datasets dataset_2023,dataset_2024,dataset_2025
```

## E. Environment Installation
```bash
pip install -r detection_models/requirements.txt
```

## F. Individual Model Training
YOLOv8:
```bash
python detection_models/yolov8/train.py --dataset dataset_2023 --model yolov8m.pt
```
YOLO11:
```bash
python detection_models/yolo11/train.py --dataset dataset_2023 --model yolo11m.pt
```
RT-DETR:
```bash
python detection_models/rtdetr/train.py --dataset dataset_2023 --model rtdetr-l.pt
```
YOLO-NAS:
```bash
python detection_models/yolonas/train.py --dataset dataset_2023
```
Faster R-CNN:
```bash
python detection_models/faster_rcnn/train.py --dataset dataset_2023
```

## G. Sequential Training
```bash
python detection_models/yolo11/train.py --sequential --dataset dataset_2023,dataset_2024,dataset_2025
```

## H. Resume Training
```bash
python detection_models/yolo11/train.py --dataset dataset_2024 --resume path/to/last.pt
```

## I. Validation
To validate a trained model on a validation dataset:
```bash
python detection_models/yolov8/validate.py --dataset dataset_2023 --weights outputs/yolov8/dataset_2023/weights/best.pt
```

## J. Prediction / Inference
To run inference on an image or directory and save the results:
```bash
python detection_models/yolov8/predict.py --source path/to/image_or_video.jpg --weights outputs/yolov8/dataset_2023/weights/best.pt --conf 0.25
```

## K. Evaluation across multiple models
To evaluate multiple models and save their metrics to a JSON file:
```bash
python detection_models/scripts/evaluate_all.py --dataset dataset_2023 --models yolov8,yolo11 --weights outputs/yolov8/dataset_2023/weights/best.pt,outputs/yolo11/dataset_2023/weights/best.pt --output_file evaluation_results.json
```

## L. Model Comparison
To generate a bar chart comparing the mAP scores of the models from the evaluation JSON:
```bash
python detection_models/scripts/compare_models.py --results evaluation_results.json --output comparison_chart.png
```
