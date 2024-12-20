import torch
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy


optimizers_d = {"sgd": torch.optim.SGD}
schedulers_d = {"cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR}


class SplitNN(L.LightningModule):
    def __init__(self, representation_models, fusion_model, lr, momentum, weight_decay,
                optimizer, eta_min_ratio, scheduler, num_epochs,
                private_labels, batch_size, compute_grad_sqd_norm):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.initial_grad_norm = None
        self.num_clients = len(representation_models)
        self.n_mbytes = 0

        self.representation_models = representation_models
        self.fusion_model = fusion_model

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes)
        self.compute_grad_sqd_norm = compute_grad_sqd_norm

    def forward(self, x):
        """Uncompressed forward pass"""
        representations = [model(self.get_feature_block(x, i)) for i, model in enumerate(self.representation_models)]
        return self.fusion_model(representations)

    def training_step(self, batch):
        x, y, indices = batch
        optimizers = self.optimizers()
        with torch.no_grad():
            compressed_representations = [model(self.get_feature_block(x, i), apply_compression=True, indices=indices, epoch=self.current_epoch)
                                        for i, model in enumerate(self.representation_models)]

        if self.hparams.private_labels:
            y_hat = self.fusion_model(compressed_representations)
            loss = F.cross_entropy(y_hat, y)
            self.manual_backward(loss)
            for optimizer in optimizers:
                optimizer.zero_grad()
                optimizer.step()
        
        else:
            for i, model in enumerate(self.representation_models + [self.fusion_model]):
                if i < self.num_clients: # representation model
                    uncompressed_representation = model(self.get_feature_block(x, i), apply_compression=False)
                    mixed_representations = [rep.detach() for rep in compressed_representations]
                    mixed_representations[i] = uncompressed_representation
                    output = self.fusion_model(mixed_representations)
                else: # fusion model
                    compressed_representations = [rep.detach() for rep in compressed_representations]
                    output = self.fusion_model(compressed_representations)

                loss = F.cross_entropy(output, y)
                optimizers[i].zero_grad()
                optimizers[-1].zero_grad()
                self.manual_backward(loss)
                optimizers[i].step()
        
        self.n_mbytes += self._calculate_n_bits(compressed_representations) / 8e6

    def _calculate_n_bits(self, compressed_representations):
        """assuming compressor and compression_parameter do not vary across representation_models"""
        compressor = self.representation_models[0].compression_module.compressor
        bits_per_element = torch.zeros(1).element_size() * 8
        
        n_bits = 0
        for compressed_representation in compressed_representations:
            if compressor is None:
                n_bits += compressed_representation.numel() * bits_per_element
            elif compressor == 'topk':
                n_bits += compressed_representation.numel() * bits_per_element * self.representation_models[0].compression_module.compression_parameter
            elif compressor == 'qsgd':
                n_bits += bits_per_element + compressed_representation.numel() * (1 + self.representation_models[0].compression_module.compression_parameter)
            else:
                raise ValueError(f"Unknown compressor: {self.representation_models[0].compression_module.compressor}")
        
        return n_bits

    def on_train_epoch_end(self):
        """(lr scheduler step and) uncompressed forward pass on training data to compute metrics"""
        if self.hparams.scheduler is not None:
            for scheduler in self.lr_schedulers():
                scheduler.step()

        total_loss = 0.0
        total_grad_squared_norm = 0.0
        train_acc = Accuracy(task="multiclass", num_classes=self.fusion_model.num_classes).to(self.device)

        for batch in self.trainer.train_dataloader:
            x_train, y_train, _ = batch
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            y_train_hat = self(x_train)
            
            loss = F.cross_entropy(y_train_hat, y_train)
            train_acc.update(y_train_hat, y_train)
            total_loss += loss.item()

            if self.compute_grad_sqd_norm:
                self.zero_grad()
                loss.backward()
                grad_squared_norm = 0.0
                for param in self.parameters():
                    if param.grad is not None:
                        grad_squared_norm += (param.grad.norm() ** 2).item()
                total_grad_squared_norm += grad_squared_norm
                self.zero_grad()

        avg_loss = total_loss / len(self.trainer.train_dataloader)
        train_acc_value = train_acc.compute()
        
        if self.compute_grad_sqd_norm:
            if self.current_epoch == 0:
                self.initial_grad_norm = total_grad_squared_norm
            total_grad_squared_norm /= self.initial_grad_norm
            self.log("grad_squared_norm", total_grad_squared_norm, on_epoch=True, prog_bar=True)
        
        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc_value, on_epoch=True, prog_bar=True)
        self.log('comm_cost', self.n_mbytes, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log('comm_cost', self.n_mbytes, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log('comm_cost', self.n_mbytes, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        Optimizer = optimizers_d[self.hparams.optimizer]
        client_optimizers = [Optimizer(model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
                            for model in self.representation_models]
        fusion_optimizer = Optimizer(self.fusion_model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        optimizers = client_optimizers + [fusion_optimizer]

        if self.hparams.scheduler is None:
            return optimizers
        else:
            Scheduler = schedulers_d[self.hparams.scheduler]
            schedulers = [Scheduler(opt, T_max=self.hparams.num_epochs, eta_min=self.hparams.lr * self.hparams.eta_min_ratio) for opt in optimizers]
            return optimizers, schedulers

    def get_feature_block(self):
        raise NotImplementedError("Subclasses must implement this method")
