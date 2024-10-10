import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy


class RepresentationModel(nn.Module):
    def __init__(self, input_size=196, cut_size=16):
        super().__init__()
        self.fc = nn.Linear(input_size, cut_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc(x))


class FusionModel(nn.Module):
    def __init__(self, cut_size=16, output_size=10):
        super().__init__()
        self.fc = nn.Linear(cut_size, output_size)
    
    def forward(self, x):
        return self.fc(torch.stack(x).sum(dim=0))


class ShallowSplitNN(L.LightningModule):
    def __init__(self, input_size=784, cut_size=16, num_classes=10, lr=1.0, num_clients=4):
        super().__init__()
        self.num_clients = num_clients
        local_input_size = input_size // num_clients
        
        self.representation_models = nn.ModuleList([RepresentationModel(local_input_size, cut_size) for _ in range(self.num_clients)])
        self.fusion_model = FusionModel(cut_size, num_classes)

        self.save_hyperparameters()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        representations = [model(x[:, i::self.num_clients]) for i, model in enumerate(self.representation_models)]
        return self.fusion_model(representations)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.train_accuracy.update(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.test_accuracy.update(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        return optimizer
