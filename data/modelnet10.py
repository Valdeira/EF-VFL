from torchvision import transforms
from data.utils import DataModule
from torch.utils.data.dataset import Dataset
from pathlib import Path
import os
import pickle
from PIL import Image
import torch


class ModelNet10DataModule(DataModule):
    def __init__(self, data_dir, batch_size, num_workers=1, val_test_split=0.5):
        train_transform =transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        super().__init__(dataset_class=ModelNet10, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, val_test_split=val_test_split,
                        train_transform=train_transform, test_transform=test_transform)


class ModelNet10(Dataset):
    def __init__(self, data_dir, train, transform=None, download=False):
        # 'download' argument is unused but included for interface consistency
        self.raw_data_path = Path(data_dir + "/modelnet10")
        self.processed_data_path = Path(data_dir + "/modelnet10_processed")
        self.data_type = "train" if train == True else "test"
        self.transform = transform
        self.x = []
        self.y = []
        
        self.classes, self.class_to_idx = self.find_classes(self.raw_data_path)
        
        self.load_or_process_data()

    def load_or_process_data(self):
        """Loads processed data from disk if available, otherwise processes the raw data and saves it."""
        x_file = self.processed_data_path / f'x_{self.data_type}.pkl'
        y_file = self.processed_data_path / f'y_{self.data_type}.pkl'

        # Check if previously processed data exists
        if x_file.exists() and y_file.exists():
            # Load data
            with x_file.open('rb') as f:
                self.x = pickle.load(f)
            with y_file.open('rb') as f:
                self.y = pickle.load(f)
        else:
            # Process raw data
            for label in os.listdir(self.raw_data_path):
                label_dir = self.raw_data_path / label / self.data_type
                if not label_dir.is_dir():
                    continue

                for item in os.listdir(label_dir):
                    item_dir = label_dir / item
                    if not item_dir.is_dir():
                        continue

                    views = []
                    for view in os.listdir(item_dir):
                        view_path = item_dir / view
                        if not view_path.is_file():
                            continue

                        img = Image.open(view_path).convert("RGB")
                        img = transforms.Resize(64)(img)
                        views.append(img)

                    self.x.append(views)
                    self.y.append(self.class_to_idx[label])

            self.processed_data_path.mkdir(parents=True, exist_ok=True)

            # Save the processed data
            with x_file.open('wb') as f:
                pickle.dump(self.x, f)
            with y_file.open('wb') as f:
                pickle.dump(self.y, f)

    def __getitem__(self, index):
        original_views = self.x[index]
        views = [self.transform(view) for view in original_views] if self.transform else original_views

        label = self.y[index]

        return torch.stack(views), label

    def __len__(self):
        return len(self.x)

    def find_classes(self, dir_path):
        """Finds the class folders in a directory and creates a mapping from class to index."""
        dir_path = Path(dir_path)
        classes = sorted([d.name for d in dir_path.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
