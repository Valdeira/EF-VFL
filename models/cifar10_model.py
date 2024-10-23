import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import BasicBlock

from models.splitnn import SplitNN
from models.compressors import CompressionModule


class RepresentationModel(nn.Module):
    def __init__(self, cut_size, compressor=None, compression_parameter=None, compression_type=None, num_samples=None):
        """torchvision resnet18, but smaller inital kernel_size and remove maxpool due to CIFAR size, instead of ImageNet,
        and number of channels reduced in half throughout."""
        super().__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.bn1 = nn.BatchNorm2d(32)
        self.resnet18.maxpool = nn.Identity()
        self.resnet18.inplanes = 32
        self.resnet18.layer1 = self.resnet18._make_layer(BasicBlock, 32, 2)
        self.resnet18.layer2 = self.resnet18._make_layer(BasicBlock, 64, 2, stride=2)
        self.resnet18.layer3 = self.resnet18._make_layer(BasicBlock, 128, 2, stride=2)
        self.resnet18.layer4 = self.resnet18._make_layer(BasicBlock, 256, 2, stride=2)
        self.resnet18.fc = nn.Linear(256 * BasicBlock.expansion, cut_size)
        
        self.compressor = compressor
        self.compression_parameter = compression_parameter
        self.compression_type = compression_type

        compressor_parameters = compressor, compression_parameter, compression_type, num_samples, cut_size
        self.compression_module = CompressionModule(*compressor_parameters) if compressor is not None else None
    
    def forward(self, x, apply_compression=False, indices=None, epoch=None):
        x = self.resnet18(x)
        if self.compression_module is not None:
            x = self.compression_module.apply_compression(x, apply_compression, indices, epoch)
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


class ResNetSplitNN(SplitNN):
    def __init__(self, input_size, cut_size, aggregation_mechanism, num_classes, private_labels: bool,
                compressor=None, compression_parameter=None, compression_type=None, num_samples=None, batch_size=None, 
                lr=1.0, momentum=0.0, weight_decay=0.0, num_clients=4, compute_grad_sqd_norm=False):
        
        representation_model_parameters = cut_size, compressor, compression_parameter, compression_type, num_samples
        representation_models = nn.ModuleList([RepresentationModel(*representation_model_parameters) for _ in range(num_clients)])

        fusion_model_parameters = cut_size, num_classes, aggregation_mechanism, num_clients
        fusion_model = FusionModel(*fusion_model_parameters)
        
        super().__init__(representation_models, fusion_model, lr, momentum, weight_decay,
                        private_labels, batch_size, compute_grad_sqd_norm)

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

        return quadrant
