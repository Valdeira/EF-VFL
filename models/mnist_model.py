import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy
from models.compressors import EFCompressor, TopKCompressor, QSGDCompressor
from models.splitnn import SplitNN


compressors_d = {"topk": TopKCompressor, "qsgd": QSGDCompressor}


class RepresentationModel(nn.Module):
    def __init__(self, input_size, cut_size, compressor=None, compression_parameter=None, compression_type=None, num_samples=None):
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
            elif compression_type is None:
                raise ValueError("compression_type must be provided when a compressor is.")
            elif compression_type == "direct":
                self.compression_layer = compressors_d[compressor](compression_parameter)
            elif compression_type == "ef":
                self.compression_layer = EFCompressor(compressors_d[compressor](compression_parameter), (num_samples, cut_size))

    def forward(self, x, apply_compression=False, indices=None, epoch=None):
        x = self.sigmoid(self.fc(x))
        if apply_compression and self.compression_layer is not None:
            if self.compression_type == "direct":
                x = self.compression_layer(x)
            elif self.compression_type == "ef":
                x = self.compression_layer(x, indices, epoch)
        return x


class FusionModel(nn.Module):
    def __init__(self, cut_size, num_classes, aggregation_mechanism, num_clients):
        super().__init__()
        self.num_classes = num_classes
        self.aggregation_mechanism = aggregation_mechanism
        fusion_input_size = cut_size * num_clients if aggregation_mechanism == "conc" else cut_size
        self.fc = nn.Linear(fusion_input_size, num_classes)

    def forward(self, x):
        if self.aggregation_mechanism == "conc":
            aggregated_x = torch.cat(x, dim=1)
        elif self.aggregation_mechanism == "mean":
            aggregated_x = torch.stack(x).mean(dim=0)
        elif self.aggregation_mechanism == "sum":
            aggregated_x = torch.stack(x).sum(dim=0)
        return self.fc(aggregated_x)


class ShallowSplitNN(SplitNN):
    def __init__(self, input_size=784, cut_size=16, num_classes=10, lr=1.0, momentum=0.0, num_clients=4, aggregation_mechanism="conc",
                 compressor=None, compression_parameter=None, compression_type=None, private_labels=False, num_samples=None, batch_size=None,
                 compute_grad_sqd_norm=False):
        
        local_input_size = input_size // num_clients
        representation_model_parameters = local_input_size, cut_size, compressor, compression_parameter, compression_type, num_samples
        representation_models = nn.ModuleList([RepresentationModel(*representation_model_parameters) for _ in range(num_clients)])

        fusion_model_parameters = cut_size, num_classes, aggregation_mechanism, num_clients
        fusion_model = FusionModel(*fusion_model_parameters)
        # compute_grad_sqd_norm: true
        super().__init__(representation_models, fusion_model, lr, momentum, private_labels, batch_size, compute_grad_sqd_norm)
    
    def get_feature_block(self, x, i):
        """Get the quadrant i of the input image x."""
        _, _, H, W = x.shape
        if not 0 <= i <= 3:
            raise ValueError("Invalid index i: Choose i from 0, 1, 2, or 3.")
        if self.num_clients != 4:
            raise ValueError("Assuming 4 clients, each holding a quadrant.")

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
