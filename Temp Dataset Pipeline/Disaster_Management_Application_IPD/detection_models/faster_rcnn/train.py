import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dataset_utils import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on complex images")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name(s), comma-separated for sequential")
    parser.add_argument("--dataset_root", type=str, default="D:/DisasterManagementDatasets")
    parser.add_argument("--model", type=str, default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--epochs", type=int, default=10) # Less epochs typically for R-CNN
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, help="Path to checkpoint.pth to fine-tune from")
    parser.add_argument("--sequential", action="store_true")
    return parser.parse_args()

def train_model(checkpoint_path, dataset_name, args, project_dir):
    try:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import torchvision.transforms as T
        from dataset import YOLODataset
    except ImportError:
        print("PyTorch or Torchvision is not installed.")
        sys.exit(1)

    print(f"\\n--- Training Faster R-CNN on {dataset_name} ---")
    ds_info = get_dataset(dataset_name, args.dataset_root)
    
    device = torch.device('cuda') if torch.cuda.is_available() and args.device == 'cuda' else torch.device('cpu')

    # Load Model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, ds_info.num_classes + 1)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint {checkpoint_path}")
    else:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the head (+1 for background class in PyTorch FasterRCNN)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, ds_info.num_classes + 1)

    model.to(device)

    # Dataloader
    def custom_collate(batch):
        return tuple(zip(*batch))
        
    def get_transform():
        return T.Compose([T.ToTensor()]) # Add Augmentations here for complex images

    images_dir = ds_info.train_path
    labels_dir = images_dir.replace('images', 'labels') if 'images' in images_dir else os.path.join(os.path.dirname(images_dir), 'labels')

    train_dataset = YOLODataset(images_dir, labels_dir, ds_info.class_names, transforms=None) # We use basic transform wrapper below
    
    # Simple transform wrapper for PyTorch
    def transform(img, target):
        return T.ToTensor()(img), target
    train_dataset.transforms = transform

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=custom_collate
    )

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.train()
    
    dataset_output_dir = os.path.join(project_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            
            # Filter empty targets for Faster RCNN
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            valid_targets = []
            valid_images = []
            for img, target in zip(images, targets):
                if len(target["boxes"]) > 0:
                    valid_targets.append(target)
                    valid_images.append(img)
            
            if len(valid_targets) == 0:
                continue

            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Iter: {i}, Loss: {losses.item()}")
                
        lr_scheduler.step()

    best_weights = os.path.join(dataset_output_dir, "best.pth")
    torch.save(model.state_dict(), best_weights)
    print(f"Saved best model to {best_weights}")
    
    return best_weights

def main():
    args = parse_args()
    project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "faster_rcnn")
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
