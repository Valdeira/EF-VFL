import torch
import torch.nn as nn

from models.lightning_splitnn import SplitNN
from models.compressors import CompressionModule


class RepresentationModel(nn.Module):
    def __init__(self, cut_size, compressor=None, compression_parameter=None, compression_type=None, num_samples=None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 6 * 6, cut_size),)
        
        self.compression_module = CompressionModule(compressor, compression_parameter, compression_type, num_samples, cut_size)
    
    def forward(self, x, apply_compression=False, indices=None, epoch=None):
        
        B, V, C, H, W = x.shape # V are the views at this client

        # merge the batch and view dimensions for parallelism
        x = x.reshape(B * V, C, H, W)

        x = self.features(x)
        x = x.view(B * V, -1) # flatten
        x = x.view(B, V, -1) # un-merge to allow for pooling across views
        x = torch.max(x, dim=1)[0]
        x = self.classifier(x)

        return self.compression_module(x, apply_compression, indices, epoch)


class FusionModel(nn.Module):
    def __init__(self, cut_size, num_classes, aggregation_mechanism, num_clients):
        super().__init__()
        self.num_classes = num_classes
        self.aggregation_mechanism = aggregation_mechanism
        fusion_input_size = cut_size * num_clients if aggregation_mechanism == "conc" else cut_size
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fusion_input_size, fusion_input_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fusion_input_size, num_classes),
        )

    def forward(self, x):
        if self.aggregation_mechanism == "conc":
            aggregated_x = torch.cat(x, dim=1)
        elif self.aggregation_mechanism == "mean":
            aggregated_x = torch.stack(x).mean(dim=0)
        elif self.aggregation_mechanism == "sum":
            aggregated_x = torch.stack(x).sum(dim=0)
        return self.classifier(aggregated_x)


class CNNSplitNN(SplitNN):
    def __init__(self, cut_size, aggregation_mechanism, num_classes, private_labels,
                compressor, compression_parameter, compression_type, num_samples, batch_size,
                optimizer, eta_min_ratio, scheduler, num_epochs, dataset,
                lr, momentum, weight_decay, num_clients, compute_grad_sqd_norm):
        
        representation_model_parameters = cut_size, compressor, compression_parameter, compression_type, num_samples
        representation_models = nn.ModuleList([RepresentationModel(*representation_model_parameters) for _ in range(num_clients)])
        
        fusion_model_parameters = cut_size, num_classes, aggregation_mechanism, num_clients
        fusion_model = FusionModel(*fusion_model_parameters)
        
        super().__init__(representation_models, fusion_model, lr, momentum, weight_decay, optimizer, eta_min_ratio, scheduler,
                        num_epochs, private_labels, batch_size, compute_grad_sqd_norm)
        
    def get_feature_block(self, x, i):
        """Get block i of the views of input x."""
        
        num_views = x.shape[1]
        if not 0 <= i < num_views:
            raise ValueError
        if num_views % self.num_clients != 0:
            raise ValueError("Assuming num_views is divisible by num_clients.")
        
        # sample shape [12, 3, 32, 32]
        # batch shape [128, 12, 3, 32, 32]
        views_per_client = num_views // self.num_clients
        block = x[:, i * views_per_client : (i+1) * views_per_client, ...]

        return block
