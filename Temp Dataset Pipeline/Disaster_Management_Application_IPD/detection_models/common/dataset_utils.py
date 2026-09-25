import os
import yaml
from typing import Dict, List, Any

class DatasetInfo:
    def __init__(self, name: str, root: str):
        self.name = name
        self.root = root
        self.yaml_path = os.path.join(root, "data.yaml")
        
        self.train_images = []
        self.val_images = []
        self.test_images = []
        self.class_names = []
        self.num_classes = 0
        
        if os.path.exists(self.yaml_path):
            self.load_yaml()
            
    def load_yaml(self):
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.class_names = data.get('names', [])
            if isinstance(self.class_names, dict):
                # Handle dict format: {0: 'classA', 1: 'classB'}
                self.class_names = [self.class_names[k] for k in sorted(self.class_names.keys())]
            self.num_classes = data.get('nc', len(self.class_names))
            
            # paths could be relative to dataset root
            self.train_path = os.path.join(self.root, data.get('train', 'train'))
            self.val_path = os.path.join(self.root, data.get('val', 'val'))
            self.test_path = os.path.join(self.root, data.get('test', 'test'))

def get_dataset(name: str, root_dir: str) -> DatasetInfo:
    dataset_path = os.path.join(root_dir, name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {name} not found at {dataset_path}")
    return DatasetInfo(name, dataset_path)
