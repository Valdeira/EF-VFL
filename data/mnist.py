from torchvision import transforms
from torchvision.datasets import MNIST
from data.utils import DataModule


class MNISTDataModule(DataModule):
    def __init__(self, data_dir, batch_size, num_workers=1, val_test_split=0.5):
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        super().__init__(dataset_class=MNIST, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, val_test_split=val_test_split, transform=mnist_transform)

