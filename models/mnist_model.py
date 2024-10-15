import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy
from models.compressors import EFCompressor, TopKCompressor, QSGDCompressor


compressors_d = {"topk": TopKCompressor, "qsgd": QSGDCompressor}


class RepresentationModel(nn.Module):
    def __init__(self, input_size=196, cut_size=16, compressor=None, compression_parameter=None, compression_type=None, num_samples=None):
        super().__init__()
        self.fc = nn.Linear(input_size, cut_size)
        self.sigmoid = nn.Sigmoid()
        self.compressor = compressor
        self.compression_parameter = compression_parameter
        self.compression_type = compression_type
        if compressor is None:
            self.compression_layer = None
        else:
            if compression_parameter is None:
                raise ValueError("compression_parameter must be provided when a compressor is.")
            if compression_type is None:
                raise ValueError("compression_type must be provided when a compressor is.")
            if compression_type == "direct":
                self.compression_layer = compressors_d[compressor](compression_parameter)
            elif compression_type == "ef":
                self.compression_layer = EFCompressor(compressors_d[compressor](compression_parameter), cut_size, num_samples)

    def forward(self, x, apply_compression=False, indices=None, epoch=None):
        x = self.sigmoid(self.fc(x))
        if apply_compression and self.compression_layer is not None:
            if self.compression_type == "ef":
                x = self.compression_layer(x, indices, epoch)
            elif self.compression_type == "direct":
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
                 compressor=None, compression_parameter=None, compression_type=None, private_labels=False, num_samples=None):
        super().__init__()
        self.num_clients = num_clients
        self.private_labels = private_labels
        self.automatic_optimization = False
        self.local_input_size = input_size // num_clients
        self.cut_size = cut_size

        self.representation_models = nn.ModuleList([
            RepresentationModel(self.local_input_size, cut_size, compressor=compressor,
                                compression_parameter=compression_parameter, compression_type=compression_type, num_samples=num_samples) 
            for _ in range(self.num_clients)
        ])
        self.fusion_model = FusionModel(cut_size, num_classes)

        self.bits_per_element = torch.zeros(1).element_size() * 8
        self.n_bits_per_round_per_client = self._calculate_n_bits(compressor, compression_parameter)
        self.cumulative_batches = 0

        self.initial_grad_norm = None
        
        self.save_hyperparameters()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """forward pass for inference with the SplitNN model (no compression)"""
        representations = [model(self.get_feature_block(x, i)) for i, model in enumerate(self.representation_models)]
        return self.fusion_model(representations)

    def training_step(self, batch, batch_idx):
        """Perform training using compressed representations, no metrics are logged here."""
        x, y, indices = batch

        optimizers = self.optimizers()
        client_optimizers = optimizers[:self.num_clients]
        fusion_optimizer = optimizers[self.num_clients]

        compressed_representations = [
            model(self.get_feature_block(x, i), apply_compression=True, indices=indices, epoch=self.current_epoch)
            for i, model in enumerate(self.representation_models)
        ]

        if self.private_labels:
            y_hat = self.fusion_model(compressed_representations)
            total_loss = F.cross_entropy(y_hat, y)
            self.manual_backward(total_loss)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
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
        
        n_mbytes = (self.n_bits_per_round_per_client * self.cumulative_batches * self.num_clients) / 8e6

        self.log('communication_cost', n_mbytes, prog_bar=True)

        return total_loss

    def _calculate_n_bits(self, compressor, compression_parameter):
        if compressor is None:
            return self.cut_size * self.bits_per_element
        elif compressor == 'topk':
            return self.cut_size * self.bits_per_element * compression_parameter
        elif compressor == 'qsgd':
            return self.bits_per_element + self.cut_size * (1 + compression_parameter)
        else:
            raise ValueError(f"Unknown compressor: {compressor}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Tracks the cumulative batches for communication cost logging."""
        self.cumulative_batches += 1

    def on_train_epoch_end(self):
        """Compute and log metrics on noncompressed forward pass for the training data."""

        train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes).to(self.device)
        total_loss = 0.0
        total_grad_squared_norm = 0.0

        for batch in self.trainer.train_dataloader:
            x_train, y_train, _ = batch

            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            y_train_hat = self(x_train)
            loss = F.cross_entropy(y_train_hat, y_train)

            train_acc.update(y_train_hat, y_train)
            total_loss += loss.item()

            self.zero_grad()
            loss.backward()

            grad_squared_norm = 0.0
            for param in self.parameters():
                if param.grad is not None:
                    grad_squared_norm += (param.grad.norm() ** 2).item()

            total_grad_squared_norm += grad_squared_norm

        avg_loss = total_loss / len(self.trainer.train_dataloader)
        train_acc_value = train_acc.compute()

        if self.current_epoch == 0:
            self.initial_grad_norm = total_grad_squared_norm

        normalized_grad_squared_norm = total_grad_squared_norm / self.initial_grad_norm

        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc_value, on_epoch=True, prog_bar=True)
        self.log("grad_squared_norm", normalized_grad_squared_norm, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Compute validation loss and accuracy."""
        x, y, _ = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Compute test loss and accuracy."""
        x, y, _ = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.test_accuracy.update(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        client_optimizers = [torch.optim.SGD(model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum) for model in self.representation_models]
        fusion_optimizer = torch.optim.SGD(self.fusion_model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        return client_optimizers + [fusion_optimizer]

    def get_feature_block(self, x, i):
        """Get a quadrant from the input data corresponding to a specific client."""
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

