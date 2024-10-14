import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy
from models.compressors import TopKCompressor, QSGDCompressor


compressors_d = {"topk": TopKCompressor, "qsgd": QSGDCompressor}


class RepresentationModel(nn.Module):
    def __init__(self, input_size=196, cut_size=16, compressor=None, compression_parameter=None):
        super().__init__()
        self.fc = nn.Linear(input_size, cut_size)
        self.sigmoid = nn.Sigmoid()
        if compressor is None:
            self.compression_layer = None
        else:
            print(f"compressor: {compressor}")
            print(f"type(compressor): {type(compressor)}")
            if compression_parameter is None:
                raise ValueError("compression_parameter must be provided when a compressor is.")
            self.compression_layer = compressors_d[compressor](compression_parameter)

    def forward(self, x, apply_compression=False):
        x = self.sigmoid(self.fc(x))
        if apply_compression and self.compression_layer is not None:
            x = self.compression_layer(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, cut_size=16, output_size=10):
        super().__init__()
        self.fc = nn.Linear(cut_size, output_size)

    def forward(self, x):
        return self.fc(torch.stack(x).mean(dim=0))


class ShallowSplitNN(L.LightningModule):
    def __init__(self, input_size=784, cut_size=16, num_classes=10, lr=1.0, momentum=0.0, num_clients=4, 
                 compressor=None, compression_parameter=None, private_labels=False):
        super().__init__()
        self.num_clients = num_clients
        self.private_labels = private_labels
        self.automatic_optimization = False
        self.local_input_size = input_size // num_clients

        self.representation_models = nn.ModuleList([
            RepresentationModel(self.local_input_size, cut_size, compressor=compressor, compression_parameter=compression_parameter) 
            for _ in range(self.num_clients)
        ])
        self.fusion_model = FusionModel(cut_size, num_classes)

        self.save_hyperparameters()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x, apply_compression=False):
        representations = [model(self.get_feature_block(x, i), apply_compression=apply_compression) 
                           for i, model in enumerate(self.representation_models)]
        return self.fusion_model(representations)

    def training_step(self, batch, batch_idx):
        x, y = batch

        optimizers = self.optimizers()
        client_optimizers = optimizers[:self.num_clients]
        fusion_optimizer = optimizers[self.num_clients]

        compressed_representations = [
            model(self.get_feature_block(x, i), apply_compression=True) 
            for i, model in enumerate(self.representation_models)
        ]

        # Case 1: Use compressed representations for training all models
        if self.private_labels:
            y_hat = self.fusion_model(compressed_representations)
            total_loss = F.cross_entropy(y_hat, y)
            self.manual_backward(total_loss)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

        # Case 2: Each client uses its own non-compressed representation and compressed representations for others
        else:
            client_losses = []
            for i, (model, optimizer) in enumerate(zip(self.representation_models, client_optimizers)):
                non_compressed_representation = model(self.get_feature_block(x, i), apply_compression=False)
                mixed_representations = [rep.detach() for rep in compressed_representations]
                mixed_representations[i] = non_compressed_representation

                client_output = self.fusion_model(mixed_representations)
                client_loss = F.cross_entropy(client_output, y)
                client_losses.append(client_loss)

                self.manual_backward(client_loss)
                optimizer.step()
                optimizer.zero_grad()

            y_hat = self.fusion_model(compressed_representations)
            fusion_loss = F.cross_entropy(y_hat, y)

            self.manual_backward(fusion_loss)
            fusion_optimizer.step()
            fusion_optimizer.zero_grad()

            total_loss = (sum(client_losses) + fusion_loss) / (len(client_losses) + 1)

        self.train_accuracy.update(y_hat, y)
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, apply_compression=False)
        loss = F.cross_entropy(y_hat, y)
        
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, apply_compression=False)
        loss = F.cross_entropy(y_hat, y)

        self.test_accuracy.update(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        client_optimizers = [torch.optim.SGD(model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum) for model in self.representation_models]
        fusion_optimizer = torch.optim.SGD(self.fusion_model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        return client_optimizers + [fusion_optimizer]
    
    def get_feature_block(self, x, i):
        B, C, H, W = x.shape
        
        if i < 0 or i > 3:
            raise ValueError("Invalid index i: Choose i from 0, 1, 2, or 3.")
        if self.num_clients != 4:
            raise ValueError("Assuming 4 clients, each with a quadrant.")

        if i == 0:  # Top-left
            quadrant = x[:, :, :H//2, :W//2]
        elif i == 1:  # Top-right
            quadrant = x[:, :, :H//2, W//2:]
        elif i == 2:  # Bottom-left
            quadrant = x[:, :, H//2:, :W//2]
        elif i == 3:  # Bottom-right
            quadrant = x[:, :, H//2:, W//2:]

        flat_quadrant = quadrant.reshape(quadrant.size(0), -1)

        return flat_quadrant
