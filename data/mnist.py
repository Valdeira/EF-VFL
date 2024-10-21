from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as L


class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data_sample, target = self.dataset[index]
        return data_sample, target, index

    def __len__(self):
        return len(self.dataset)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="../data", batch_size=None, num_workers=4, val_test_split=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_test_split = val_test_split
        self.num_train_samples = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetWithIndex(MNIST(self.data_dir, train=True, transform=self.transform, download=False))
            self.num_train_samples = len(self.train_dataset)

        if stage == 'validate' or stage == 'test' or stage is None:
            test_data = MNIST(self.data_dir, train=False, transform=self.transform, download=False)
            test_size = int(len(test_data) * self.val_test_split)
            val_size = len(test_data) - test_size
            val_dataset, test_dataset = random_split(test_data, [val_size, test_size])
            self.val_dataset = DatasetWithIndex(val_dataset)
            self.test_dataset = DatasetWithIndex(test_dataset)

    def train_dataloader(self):
        batch_size = len(self.train_dataset) if self.batch_size is None else self.batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        batch_size = len(self.val_dataset) if self.batch_size is None else self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        batch_size = len(self.test_dataset) if self.batch_size is None else self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=self.num_workers)