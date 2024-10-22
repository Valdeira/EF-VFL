from torchvision import transforms
from torchvision.datasets import CIFAR10
from data.utils import DataModule


class CIFAR10DataModule(DataModule):
    def __init__(self, data_dir="../data", batch_size=None, num_workers=4, val_test_split=0.5):
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        super().__init__(dataset_class=CIFAR10, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, val_test_split=val_test_split, transform=cifar10_transform)
