import torch
import torch.nn as nn


class EFCompressor(nn.Module):
    
    def __init__(self, direct_compressor, shape):
        super().__init__()
        self.direct_compressor = direct_compressor
        self.shape = shape
        self.state = None
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x, indices, epoch):
        if self.state is None:
            self.state = torch.zeros(*self.shape, requires_grad=False, device=x.device)

        state_detached = self.state.detach()
        updated_state = state_detached.clone()
        if epoch == 0:
            updated_state[indices] = self.direct_compressor(x)
        else:
            updated_state[indices] = state_detached[indices] + self.direct_compressor(x - state_detached[indices])
        self.state = updated_state.detach()

        return updated_state[indices]

    def _backward_hook(self, module, grad_input, grad_output):
        return (grad_output[0], None, None)


class TopKCompressor(nn.Module):
    
    def __init__(self, compression_ratio):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x):
        k = max(1, int(x.numel() * self.compression_ratio))
        abs_x = x.abs().view(-1)
        _, topk_indices = torch.topk(abs_x, k)
        mask = torch.zeros_like(abs_x)
        mask[topk_indices] = 1.0
        compressed_x = (x.view(-1) * mask).view_as(x)
        return compressed_x
    
    def _backward_hook(self, module, grad_input, grad_output):
        return (grad_output[0],)


class QSGDCompressor(nn.Module):
    '''A biased (normalized) version of the QSGD compressor'''

    def __init__(self, n_bits):
        super().__init__()
        if not isinstance(n_bits, int) or n_bits < 1:
            raise ValueError('n_bits must be an integer >= 1')
        n_quantization_levels = 2 ** n_bits
        self.s = n_quantization_levels - 1
        self.register_full_backward_hook(self._backward_hook)
    
    def forward(self, x):
        tau = 1 + torch.min(torch.tensor(x.numel() / self.s ** 2), torch.sqrt(torch.tensor(x.numel())) / self.s)
        x_norm = torch.norm(x)
        x_in_quant_interval = self.s * torch.abs(x) / x_norm
        xi = torch.floor(x_in_quant_interval + torch.rand_like(x_in_quant_interval))
        return torch.sign(x) * x_norm * xi / (self.s * tau)
    
    def _backward_hook(self, module, grad_input, grad_output):
        return (grad_output[0],)


compressors_d = {"topk": TopKCompressor, "qsgd": QSGDCompressor}


class CompressionModule(nn.Module):
    def __init__(self, compressor=None, compression_parameter=None, compression_type=None, num_samples=None, cut_size=None):
        super().__init__()
        self.compressor = compressor
        self.compression_parameter = compression_parameter
        self.compression_type = compression_type
        self.num_samples = num_samples
        self.cut_size = cut_size

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
        if apply_compression and self.compression_layer is not None:
            if self.compression_type == "direct":
                x = self.compression_layer(x)
            elif self.compression_type == "ef":
                x = self.compression_layer(x, indices, epoch)
        return x
