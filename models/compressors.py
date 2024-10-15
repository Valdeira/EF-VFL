import torch
import torch.nn as nn


class EFCompressor(nn.Module):
    
    def __init__(self, compressor, cut_dim, num_samples):
        super().__init__()
        self.compressor = compressor
        self.cut_dim = cut_dim
        self.num_samples = num_samples
        self.state = None
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x, indices, epoch):
        
        if self.state is None:
            self.state = torch.zeros(self.num_samples, self.cut_dim, requires_grad=False, device=x.device)

        state_detached = self.state.detach()
        updated_state = state_detached.clone()
        if epoch == 0:
            updated_state[indices] = self.compressor(x)
        else:
            updated_state[indices] = state_detached[indices] + self.compressor(x - state_detached[indices])
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
    '''Stochastic quantization -> a (biased) normalized version of the QSGD compressor'''

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
