from torchvision import transforms
from torchvision.datasets import CIFAR100
from data.utils import DataModule


class CIFAR100DataModule(DataModule):
    def __init__(self, data_dir="../data", batch_size=None, num_workers=4, val_test_split=0.5):
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                        ])
        test_transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                        ])

        super().__init__(dataset_class=CIFAR100, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, val_test_split=val_test_split,
                        train_transform=train_transform, test_transform=test_transform)
