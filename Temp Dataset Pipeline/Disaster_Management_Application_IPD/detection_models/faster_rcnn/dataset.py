import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    # Convert YOLO format (normalized) to Pascal VOC (absolute x1, y1, x2, y2)
                    x1 = (x_center - w/2) * width
                    y1 = (y_center - h/2) * height
                    x2 = (x_center + w/2) * width
                    y2 = (y_center + h/2) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id))

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            # Using torchvision v2 transforms, assuming img and target can be passed
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_files)
