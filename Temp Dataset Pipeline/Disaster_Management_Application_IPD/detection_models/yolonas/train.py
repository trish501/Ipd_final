import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataset_utils import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO-NAS on complex images")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name(s), comma-separated for sequential")
    parser.add_argument("--dataset_root", type=str, default="D:/DisasterManagementDatasets")
    parser.add_argument("--model", type=str, default="yolo_nas_m")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=str, help="Path to checkpoint.pth to fine-tune from")
    parser.add_argument("--sequential", action="store_true")
    return parser.parse_args()

def train_model(checkpoint_path, dataset_name, args, project_dir):
    try:
        from super_gradients.training import Trainer, models
        from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
        from super_gradients.training.losses import PPYoloELoss
        from super_gradients.training.metrics import DetectionMetrics_050
        from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    except ImportError:
        print("SuperGradients is not installed. Please install it first.")
        sys.exit(1)

    print(f"\\n--- Training YOLO-NAS on {dataset_name} ---")
    ds_info = get_dataset(dataset_name, args.dataset_root)
    
    # Initialize trainer
    # SuperGradients uses ckpt_root_dir to save outputs inside an experiment_name subfolder
    trainer = Trainer(experiment_name=dataset_name, ckpt_root_dir=project_dir)
    
    # Setup DataLoaders
    # dataset_params needs dataset_dir, images_dir, labels_dir
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': ds_info.root,
            'images_dir': ds_info.train_path.replace(ds_info.root + os.sep, '') if ds_info.train_path.startswith(ds_info.root) else ds_info.train_path,
            'labels_dir': ds_info.train_path.replace(ds_info.root + os.sep, '').replace('images', 'labels'),
            'classes': ds_info.class_names
        },
        dataloader_params={
            'batch_size': args.batch,
            'num_workers': args.workers
        }
    )
    
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': ds_info.root,
            'images_dir': ds_info.val_path.replace(ds_info.root + os.sep, '') if ds_info.val_path.startswith(ds_info.root) else ds_info.val_path,
            'labels_dir': ds_info.val_path.replace(ds_info.root + os.sep, '').replace('images', 'labels'),
            'classes': ds_info.class_names
        },
        dataloader_params={
            'batch_size': args.batch,
            'num_workers': args.workers
        }
    )

    # Initialize model
    if checkpoint_path and checkpoint_path.endswith('.pth') and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model = models.get(args.model, num_classes=ds_info.num_classes, checkpoint_path=checkpoint_path)
    else:
        print(f"Loading pretrained COCO weights for {args.model}")
        model = models.get(args.model, pretrained_weights="coco")

    # Training parameters
    train_params = {
        "max_epochs": args.epochs,
        "initial_lr": 5e-4,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.0001},
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=ds_info.num_classes, reg_max=16),
        "valid_metrics_list": [
            DetectionMetrics_050(score_thres=0.1, top_k_predictions=300, num_cls=ds_info.num_classes, normalize_targets=True, post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7))
        ],
        "metric_to_watch": "mAP@0.50",
    }

    trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)
    
    best_weights = os.path.join(project_dir, dataset_name, "ckpt_best.pth")
    return best_weights

def main():
    args = parse_args()
    project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "yolonas")
    os.makedirs(project_dir, exist_ok=True)
    
    datasets = [d.strip() for d in args.dataset.split(",")]
    
    if args.sequential and len(datasets) > 1:
        current_weights = args.resume
        for ds in datasets:
            print(f"Starting sequential step for: {ds} using weights: {current_weights}")
            current_weights = train_model(current_weights, ds, args, project_dir)
            print(f"Finished {ds}. Best weights: {current_weights}")
    else:
        train_model(args.resume, datasets[0], args, project_dir)

if __name__ == '__main__':
    main()
